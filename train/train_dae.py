import numpy as np
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from dataset.dataset import get_datasets
from models import UNet
from utils.loss import DiceLoss, dice_score
from utils.io import load_config, dump_config, print_config, save_checkpoint, write_to_csv, deep_get
from utils.utils import seed_everything


def save_atlas(train_dataset, num_workers, logdir):
    train_dataset.set_augmentation(False)
    atlas_dataloader = DataLoader(dataset=train_dataset, batch_size=1,
        shuffle=False, num_workers=num_workers, drop_last=False)
    
    atlas = None
    for _, y, _, _, _ in atlas_dataloader:
        if atlas is None:
            atlas = y.clone()
        else:
            atlas += y

    atlas = atlas.float()
    atlas /= len(atlas_dataloader)

    save_checkpoint(
        path=os.path.join(logdir, 'atlas.h5py'),
        atlas=atlas,
    )


if __name__ == '__main__':

    print(f'Running {__file__}')

    # Loading Parameters
    params = load_config('config/params.yaml')
    logdir = deep_get(params, 'training', 'dae', 'logdir', default='logs')
    wandb_log = deep_get(params, 'training', 'wandb_log', default=False)
    resume = deep_get(params, 'training', 'resume', default=True)

    is_resumed = os.path.exists(os.path.join(logdir, 'params.yaml')) and resume
    print(f'training resumed: {is_resumed}')

    if is_resumed:
        params = load_config(os.path.join(logdir, 'params.yaml'))
    else:
        os.makedirs(logdir, exist_ok=True)
        dump_config(os.path.join(logdir, 'params.yaml'), params)

    print_config(params, keys=['training', 'model', 'datasets'])

    seed = deep_get(params, 'seed', default=0)
    seed_everything(seed)

    if wandb_log:
        wandb_params_path = os.path.join(logdir, 'wandb.yaml')
        if os.path.exists(wandb_params_path):
            wandb_params = load_config(wandb_params_path)
        else:
            wandb_params = {'id': wandb.util.generate_id()}

        wandb.init(
            project=deep_get(params, 'training', 'dae', 'wandb_project'),
            config=params,
            resume='allow',
            id=wandb_params['id']
        )
        wandb_dir = wandb.run.dir

        wandb_params['name'] = wandb.run.name
        wandb_params['project'] = wandb.run.project
        wandb_params['link'] = f'https://wandb.ai/{wandb.run.path}'

        dump_config(wandb_params_path, wandb_params)

    device = deep_get(params, 'training', 'device', default='cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print('No GPU available, using CPU instead')
    else:
        device = torch.device(device)

    print(f'Using Device {device}')

    dataset         = deep_get(params, 'training', 'dae', 'dataset')
    batch_size      = deep_get(params, 'training', 'dae', 'batch_size')
    num_workers     = deep_get(params, 'training', 'dae', 'num_workers', default=0)
    epochs          = deep_get(params, 'training', 'dae', 'epochs')
    validate_every  = deep_get(params, 'training', 'dae', 'validate_every')
    learning_rate   = deep_get(params, 'training', 'dae', 'learning_rate')
    n_classes       = deep_get(params, 'model', 'dae', 'n_classes')
    checkpoint_best = deep_get(params, 'training', 'checkpoint_best', default='checkpoint_best.pth')
    checkpoint_last = deep_get(params, 'training', 'checkpoint_last', default='checkpoint_last.pth')
    validation_set_multiplier = deep_get(params, 'training', 'dae', 'validation_set_multiplier', default=1)

    # Model definition
    dae = UNet(
        n_classes,
        n_classes,
        deep_get(params, 'model', 'dae', 'channel_size'),
        deep_get(params, 'model', 'dae', 'channels_bottleneck'),
        deep_get(params, 'model', 'dae', 'skips'),
        deep_get(params, 'model', 'dae', 'n_dimensions'),
    ).to(device)

    optimizer = torch.optim.Adam(
        dae.parameters(),
        lr=learning_rate
    )

    loss_func = DiceLoss()

    print('Models, optimizer and Loss defined')

    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        paths           = deep_get(params, 'datasets', dataset, 'paths_processed'),
        paths_original  = deep_get(params, 'datasets', dataset, 'paths_original'),
        splits          = ['train', 'val'],
        image_size      = deep_get(params, 'training', 'dae', 'image_size'),
        resolution_proc = deep_get(params, 'datasets', dataset, 'resolution_proc'),
        rescale_factor  = deep_get(params, 'training', 'dae', 'rescale_factor', default=None),
        dim_proc        = deep_get(params, 'datasets', dataset, 'dim'),
        n_classes       = n_classes,
        aug_params      = deep_get(params, 'training', 'dae', 'augmentation', default=None),
        deformation     = deep_get(params, 'training', 'dae', 'deformation', default=None),
        load_original   = False,
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True)
    
    val_dataset = ConcatDataset([val_dataset] * validation_set_multiplier)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, drop_last=False)

    print('Dataloaders defined')

    if is_resumed:
        # Getting checkpoint from local log dir.
        checkpoint = torch.load(os.path.join(logdir, checkpoint_last), map_location=device)

        print(f'Resuming training at epoch {checkpoint["epoch"] + 1}.')
        
        continue_from_epoch = checkpoint['epoch'] + 1
        dae.load_state_dict(checkpoint['dae_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_validation_loss = checkpoint['best_validation_loss']

        del checkpoint

    else:
        print('Starting from scratch.')

        continue_from_epoch = 0
        best_validation_loss = np.inf

        save_checkpoint(
            path=os.path.join(logdir, checkpoint_last),
            epoch=-1,
            dae_state_dict=dae.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            best_validation_loss=np.inf,
        )

    if wandb_log:
        wandb.save(os.path.join(wandb_dir, checkpoint_last), base_path=wandb_dir)
        wandb.watch([dae], loss_func, log='all')

    training_losses = []
    validation_losses = []
    validation_scores = []

    # Training
    for epoch in range(continue_from_epoch, epochs):

        training_loss = 0
        n_samples_train = 0

        dae.train()

        print(f'Training for epoch {epoch}')
        for _, y, x, _, _ in train_dataloader:
            x = x.to(device).float()
            y = y.to(device)

            y_pred, logits = dae(x)

            loss = loss_func(y_pred, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                training_loss += loss.detach() * x.shape[0]
                n_samples_train += x.shape[0]
        
        training_loss /= n_samples_train
        training_losses.append(training_loss.item())

        if (epoch + 1) % validate_every != 0 and epoch != epochs - 1:
            continue

        # Evaluation
        dae.eval()

        validation_loss = 0
        validation_score = 0
        n_samples_val = 0

        print(f'Validating for epoch {epoch}')
        with torch.no_grad():
            for _, y, x, _, _ in val_dataloader:
                x = x.to(device).float()
                y = y.to(device)

                y_pred, logits = dae(x)

                loss = loss_func(y_pred, y)
                dice, dice_fg = dice_score(y_pred, y, soft=False, reduction='mean')

                validation_loss += loss * x.shape[0]
                validation_score += dice_fg * x.shape[0]
                n_samples_val += x.shape[0]

        validation_loss /= n_samples_val
        validation_score /= n_samples_val

        validation_losses.append(validation_loss.item())
        validation_scores.append(validation_score.item())

        save_checkpoint(
            path=os.path.join(logdir, checkpoint_last),
            epoch=epoch,
            dae_state_dict=dae.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            best_validation_loss=best_validation_loss,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                path=os.path.join(logdir, checkpoint_best),
                epoch=epoch,
                dae_state_dict=dae.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                best_validation_loss=best_validation_loss,
            )

        if wandb_log:
            wandb.log({
                'training_loss': training_loss,
                'validation_loss': validation_loss,
                'validation_score': validation_score,
            }, step=epoch)
            wandb.save(os.path.join(wandb_dir, checkpoint_last), base_path=wandb_dir)
            wandb.save(os.path.join(wandb_dir, checkpoint_best), base_path=wandb_dir)
    
    save_atlas(train_dataset, num_workers, logdir)

    write_to_csv(
        path=os.path.join(logdir, 'training_statistics.csv'),
        data=np.stack([training_losses], 1),
        header=['training_losses'],
    )
    write_to_csv(
        path=os.path.join(logdir, 'validation_statistics.csv'),
        data=np.stack([validation_losses, validation_scores], 1),
        header=['validation_losses', 'validation_scores'],
    )

    if wandb_log:
        wandb.finish()
