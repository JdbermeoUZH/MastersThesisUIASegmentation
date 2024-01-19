import argparse
import copy
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader, ConcatDataset, TensorDataset
import wandb

from dataset.dataset import get_datasets
from models import Normalization, UNet
from models.normalization import background_suppression
from tta. import test_volume
from utils.io import load_config, dump_config, print_config, save_checkpoint, write_to_csv, deep_get
from utils.utils import seed_everything, get_seed
from utils.loss import DiceLoss, class_to_onehot, dice_score, onehot_to_class



def tta_dae(
    volume_dataset,
    dataset,
    atlas,
    logdir,
    norm,
    seg,
    dae,
    device,
    batch_size,
    dataset_repetition,
    learning_rate,
    num_steps,
    update_dae_output_every,
    rescale_factor,
    alpha,
    beta,
    n_classes,
    num_workers,
    index,
    bg_suppression_opts,
    bg_suppression_opts_tta,
    save_checkpoints,
    calculate_dice_every,
    accumulate_over_volume,
    const_aug_per_volume,
):
    if rescale_factor is not None:
        assert (batch_size * rescale_factor[0]) % 1 == 0
        label_batch_size = int(batch_size * rescale_factor[0])
    else:
        label_batch_size = batch_size

    optimizer = torch.optim.Adam(
        norm.parameters(),
        lr=learning_rate
    )

    seg.requires_grad_(False)

    loss_func = DiceLoss()

    # Setting up metrics for model selection.
    tta_losses = []
    test_scores = []

    norm_dict = {'best_score': copy.deepcopy(norm.state_dict())}
    metrics_best = {'best_score': 0}

    dae_dataloader = DataLoader(
        volume_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    volume_dataloader = DataLoader(
        ConcatDataset([volume_dataset] * dataset_repetition),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    for step in range(num_steps):
        
        norm.eval()
        volume_dataset.dataset.set_augmentation(False)

        # Testing performance during adaptation.
        if step % calculate_dice_every == 0 and calculate_dice_every != -1:

            _, dices_fg = test_volume(
                volume_dataset=volume_dataset,
                dataset=dataset,
                logdir=logdir,
                norm=norm,
                seg=seg,
                device=device,
                batch_size=batch_size,
                n_classes=n_classes,
                num_workers=num_workers,
                index=index,
                iteration=step,
                bg_suppression_opts=bg_suppression_opts,
            )
            test_scores.append(dices_fg.mean().item())

        if step % update_dae_output_every == 0:
            

            with torch.no_grad():
                masks = []
                for x, _, _, _, bg_mask in dae_dataloader:
                    x = x.to(device).float()

                    bg_mask = bg_mask.to(device)
                    x_norm = norm(x)

                    x_norm = background_suppression(
                        x_norm, bg_mask, bg_suppression_opts_tta)

                    mask, _ = seg(x_norm)
                    masks.append(mask)

                masks = torch.cat(masks)
                masks = masks.permute(1,0,2,3).unsqueeze(0)

                if rescale_factor is not None:
                    masks = F.interpolate(masks, scale_factor=rescale_factor, mode='trilinear')

                dae_output, _ = dae(masks)

                dice_denoised, _ = dice_score(masks, dae_output, soft=True, reduction='mean')
                dice_atlas, _ = dice_score(masks, atlas, soft=True, reduction='mean')

                if dice_denoised / dice_atlas >= alpha and dice_atlas >= beta:
                    target_labels = dae_output
                    dice = dice_denoised
                else:
                    target_labels = atlas
                    dice = dice_atlas

                target_labels = target_labels.squeeze(0)
                target_labels = target_labels.permute(1,0,2,3)

            if metrics_best['best_score'] < dice:
                norm_dict['best_score'] = copy.deepcopy(norm.state_dict())
                metrics_best['best_score'] = dice

            label_dataloader = DataLoader(
                ConcatDataset([TensorDataset(target_labels.cpu())] * dataset_repetition),
                batch_size=label_batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
            )

        tta_loss = 0
        n_samples = 0

        norm.train()
        volume_dataset.dataset.set_augmentation(True)

        if accumulate_over_volume:
            optimizer.zero_grad()

        if const_aug_per_volume:
            volume_dataset.dataset.set_seed(get_seed())

        # Adapting to the target distribution.
        for (x,_,_,_, bg_mask), (y,) in zip(volume_dataloader, label_dataloader):

            if not accumulate_over_volume:
                optimizer.zero_grad()

            x = x.to(device).float()
            y = y.to(device)
            bg_mask = bg_mask.to(device)
            x_norm = norm(x)
            
            x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)

            mask, logits = seg(x_norm)

            if rescale_factor is not None:
                mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
                mask = F.interpolate(mask, scale_factor=rescale_factor, mode='trilinear')
                mask = mask.squeeze(0).permute(1, 0, 2, 3)

            loss = loss_func(mask, y)

            if accumulate_over_volume:
                loss /= len(volume_dataloader)

            loss.backward()

            if not accumulate_over_volume:
                optimizer.step()

            with torch.no_grad():
                tta_loss += loss.detach() * x.shape[0]
                n_samples += x.shape[0]

        if accumulate_over_volume:
            optimizer.step()

        tta_losses.append((tta_loss / n_samples).item())


    if save_checkpoints:
        os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)
        save_checkpoint(
            path=os.path.join(logdir, 'checkpoints',
                              f'checkpoint_tta_{dataset}_{index:02d}.pth'),
            norm_state_dict=norm_dict['best_score'],
            seg_state_dict=seg.state_dict(),
        )

    os.makedirs(os.path.join(logdir, 'metrics'), exist_ok=True)

    os.makedirs(os.path.join(logdir, 'tta_score'), exist_ok=True)
    write_to_csv(
        os.path.join(logdir, 'tta_score', f'{dataset}_{index:03d}.csv'),
        np.array([test_scores]).T,
        header=['tta_score'],
        mode='w',
    )

    return norm, norm_dict, metrics_best




def main():
    print(f'Running {__file__}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=False, type=int,
                        help='starting volume index to be used for testing')
    parser.add_argument('--stop', required=False, type=int,
                        help='stopping volume index to be used for testing (index not included)')

    testing_params = load_config('config/params.yaml')
    print('Loaded params')
    tta_mode = 'dae'
    logdir = deep_get(testing_params, 'testing', tta_mode, 'logdir', default='logs')
    dae_dir = deep_get(testing_params, 'testing', tta_mode, 'dae_dir')
    seg_dir = deep_get(testing_params, 'testing', tta_mode, 'seg_dir')
    
    params_path = os.path.join(logdir, 'params.yaml')
    params_dae = load_config(os.path.join(dae_dir, 'params.yaml'))
    params_seg = load_config(os.path.join(seg_dir, 'params.yaml'))
    params = params_seg
    params['testing'] = deep_get(testing_params, 'testing')
    params['datasets'] = deep_get(testing_params, 'datasets')
    params['training']['dae'] = deep_get(params_dae, 'training', 'dae')
    params['model']['dae'] = deep_get(params_dae, 'model', 'dae')

    os.makedirs(logdir, exist_ok=True)
    dump_config(params_path, params)
    print_config(params, keys=['testing', 'training', 'model', 'datasets'])

    checkpoint_best = deep_get(params, 'training', 'checkpoint_best', default='checkpoint_best.pth')

    device = deep_get(params, 'testing', 'device', default='cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print('No GPU available, using CPU instead')
    else:
        device = torch.device(device)

    wandb_log = deep_get(params, 'testing', 'wandb_log', default=False)
    if wandb_log:
        wandb.init(
            project=deep_get(params, 'testing', 'wandb_project'),
            config=params,
        )

    dataset                = deep_get(params, 'testing', 'dataset')
    num_workers            = deep_get(params, 'testing', 'num_workers', default=0)
    batch_size             = deep_get(params, 'testing', tta_mode, 'batch_size')
    dataset_repetition     = deep_get(params, 'testing', tta_mode, 'dataset_repetition', default=1)
    n_classes              = deep_get(params, 'datasets', dataset, 'n_classes')
    const_aug_per_volume   = deep_get(params, 'testing', tta_mode, 'const_aug_per_volume', default=False)
    accumulate_over_volume = deep_get(params, 'testing', tta_mode, 'accumulate_over_volume', default=True)
    calculate_dice_every   = deep_get(params, 'testing', tta_mode, 'calculate_dice_every', default=20)
    learning_rate          = deep_get(params, 'testing', tta_mode, 'learning_rate')
    num_steps              = deep_get(params, 'testing', tta_mode, 'num_steps')
    image_channels         = deep_get(params, 'model', 'normalization', 'image_channels', default=1)
    rescale_factor         = deep_get(params, 'training', 'dae', 'rescale_factor')
    alpha                  = deep_get(params, 'testing', tta_mode, 'alpha')
    beta                   = deep_get(params, 'testing', tta_mode, 'beta')

    # Model definition
    norm = Normalization(
        deep_get(params, 'model', 'normalization', 'n_layers'),
        image_channels,
        deep_get(params, 'model', 'normalization', 'channel_size'),
        deep_get(params, 'model', 'normalization', 'kernel_size'),
        deep_get(params, 'model', 'normalization', 'activation'),
        deep_get(params, 'model', 'normalization', 'batch_norm'),
        deep_get(params, 'model', 'normalization', 'residual'),
    ).to(device)

    seg = UNet(
        deep_get(params, 'model', 'segmentation', 'image_channels'),
        deep_get(params, 'model', 'segmentation', 'n_classes'),
        deep_get(params, 'model', 'segmentation', 'channel_size'),
        deep_get(params, 'model', 'segmentation', 'channels_bottleneck'),
        deep_get(params, 'model', 'segmentation', 'skips'),
        deep_get(params, 'model', 'segmentation', 'n_dimensions'),
    ).to(device)

    dae = UNet(
        n_classes,
        n_classes,
        deep_get(params, 'model', 'dae', 'channel_size'),
        deep_get(params, 'model', 'dae', 'channels_bottleneck'),
        deep_get(params, 'model', 'dae', 'skips'),
        deep_get(params, 'model', 'dae', 'n_dimensions'),
    ).to(device)


    print('Loading datasets')

    bg_suppression_opts = deep_get(params, 'testing', 'bg_suppression_opts')
    bg_suppression_opts_tta = deep_get(params, 'testing', tta_mode, 'bg_suppression_opts')

    test_dataset, = get_datasets(
        paths           = deep_get(params, 'datasets', dataset, 'paths_processed'),
        paths_original  = deep_get(params, 'datasets', dataset, 'paths_original'),
        splits          = ['test'],
        image_size      = deep_get(params, 'testing', 'image_size'),
        resolution_proc = deep_get(params, 'datasets', dataset, 'resolution_proc'),
        dim_proc        = deep_get(params, 'datasets', dataset, 'dim'),
        n_classes       = n_classes,
        aug_params      = deep_get(params, 'testing', tta_mode, 'augmentation', default=None),
        deformation     = None,
        load_original   = True,
        bg_suppression_opts=bg_suppression_opts,
    )

    indices_per_volume = test_dataset.get_volume_indices()

    args = parser.parse_args()
    start_idx = 0
    stop_idx = len(indices_per_volume)  # == number of volumess
    if args.start is not None:
        start_idx = args.start
    if args.stop is not None:
        stop_idx = args.stop

    print('Datasets loaded')

    save_checkpoints = deep_get(params, 'testing', tta_mode, 'save_checkpoints', default=False)
    update_dae_output_every = deep_get(params, 'testing', tta_mode, 'update_dae_output_every', default=3)

    # Getting checkpoint from local log dir.
    checkpoint = torch.load(os.path.join(seg_dir, checkpoint_best), map_location=device)
    norm.load_state_dict(checkpoint['norm_state_dict'])
    seg.load_state_dict(checkpoint['seg_state_dict'])
    norm_state_dict = checkpoint['norm_state_dict']

    checkpoint = torch.load(os.path.join(dae_dir, checkpoint_best), map_location=device)
    dae.load_state_dict(checkpoint['dae_state_dict'])

    checkpoint = torch.load(os.path.join(dae_dir, 'atlas.h5py'), map_location=device)
    atlas = checkpoint['atlas']

    del checkpoint


    seg.eval()
    dae.eval()

    if wandb_log:
        wandb.watch([norm], log='all', log_freq=1)

    dice_scores = torch.zeros((len(indices_per_volume), n_classes))

    for i in range(start_idx, stop_idx):
        
        seed = deep_get(params, 'seed', default=0)
        seed_everything(seed)

        indices = indices_per_volume[i]
        print(f'processing volume {i}')

        volume_dataset = Subset(test_dataset, indices)

        norm.load_state_dict(norm_state_dict)

        norm, norm_dict, metrics_best = tta_dae(
            volume_dataset=volume_dataset,
            dataset=dataset,
            atlas=atlas,
            logdir=logdir,
            norm=norm,
            seg=seg,
            dae=dae,
            device=device,
            batch_size=batch_size,
            dataset_repetition=dataset_repetition,
            learning_rate=learning_rate,
            num_steps=num_steps,
            update_dae_output_every=update_dae_output_every,
            rescale_factor=rescale_factor,
            alpha=alpha,
            beta=beta,
            n_classes=n_classes,
            num_workers=num_workers,
            index=i,
            bg_suppression_opts=bg_suppression_opts,
            bg_suppression_opts_tta=bg_suppression_opts_tta,
            save_checkpoints=save_checkpoints,
            calculate_dice_every=calculate_dice_every,
            accumulate_over_volume=accumulate_over_volume,
            const_aug_per_volume=const_aug_per_volume,
        )

        dice_scores[i, :], _ = test_volume(
            volume_dataset=volume_dataset,
            dataset=dataset,
            logdir=logdir,
            norm=norm,
            seg=seg,
            device=device,
            batch_size=batch_size,
            n_classes=n_classes,
            num_workers=num_workers,
            index=i,
            bg_suppression_opts=bg_suppression_opts,
        )

        write_to_csv(
            os.path.join(logdir, f'scores_{dataset}_last_iteration.csv'),
            np.hstack([[[f'volume_{i:02d}']], dice_scores[None, i, :].numpy()]),
            mode='a',
        )

        os.makedirs(os.path.join(logdir, 'optimal_metrics'), exist_ok=True)
        dump_config(
            os.path.join(logdir, 'optimal_metrics', f'{dataset}_{i:02d}.yaml'),
            metrics_best,
        )

        for key in norm_dict.keys():
            print(f'Model at minimum {key} = {metrics_best[key]}')

            norm.load_state_dict(norm_dict[key])
            scores, _ = test_volume(
                volume_dataset=volume_dataset,
                dataset=dataset,
                logdir=logdir,
                norm=norm,
                seg=seg,
                device=device,
                batch_size=batch_size,
                n_classes=n_classes,
                num_workers=num_workers,
                index=i,
                appendix=f'_min_{key}',
                bg_suppression_opts=bg_suppression_opts,
            )

            write_to_csv(
                os.path.join(logdir, f'scores_{dataset}_{key}.csv'),
                np.hstack([[[f'volume_{i:02d}']], scores.numpy()]),
                mode='a',
            )

    print(f'Overall mean dice (only foreground): {dice_scores[:, 1:].mean()}')

    if wandb_log:
        wandb.finish()


if __name__ == '__main__':
    main()
