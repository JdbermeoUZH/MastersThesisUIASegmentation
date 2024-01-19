import argparse
import copy
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader, TensorDataset, ConcatDataset
import wandb

from dataset.dataset import get_datasets, get_sectors_from_index, split_dataset
from models import Normalization, UNet, ViG, ProjectionHead, EmbeddingCNN
from models.normalization import background_suppression
from train_gnn import get_embeddings
from train_ph import train_ph
from utils.contrastive_loss import ContrastiveLoss
from utils.io import load_config, dump_config, print_config, save_checkpoint, write_to_csv, deep_get
from utils.loss import dice_score, DiceLoss
from utils.tta_loss import SliceSectorLoss, ContrastiveSliceSectorLoss, NearestNeighborLoss, KLDivergenceLoss
from utils.utils import assert_in, seed_everything, get_seed
from utils.visualization import export_images, multilabel_scatter


def test_volume(
    volume_dataset,
    dataset,
    logdir,
    norm,
    seg,
    device,
    batch_size,
    n_classes,
    num_workers,
    index,
    iteration=-1,
    appendix='',
    bg_suppression_opts=None,
):

    norm.eval()
    volume_dataset.dataset.set_augmentation(False)

    # Get original images
    x_original, y_original, bg = volume_dataset.dataset.get_original_images(index)
    _, C, D, H, W = y_original.shape  # xyz = HWD

    x_ = x_original.permute(0, 2, 3, 1).unsqueeze(0)  # NCHWD (= NCxyz)
    y_ = y_original.permute(0, 1, 3, 4, 2)  # NCHWD
    bg_ = torch.from_numpy(bg).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)  # NCHWD

    # Rescale x and y to the target resolution of the dataset
    original_pix_size = volume_dataset.dataset.pix_size_original[:, index]
    target_pix_size = volume_dataset.dataset.resolution_proc  # xyz
    scale_factor = original_pix_size / target_pix_size
    scale_factor[-1] = 1

    y_ = y_.float()
    bg_ = bg_.float()

    output_size = (y_.shape[2:] * scale_factor).round().astype(int).tolist()
    x_ = F.interpolate(x_, size=output_size, mode='trilinear')
    y_ = F.interpolate(y_, size=output_size, mode='trilinear')
    bg_ = F.interpolate(bg_, size=output_size, mode='trilinear')

    y_ = y_.round().byte()
    bg_ = bg_.round().bool()

    x_ = x_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
    y_ = y_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
    bg_ = bg_.squeeze(0).permute(3, 0, 1, 2)  # DCHW

    # Get segmentation
    volume_dataloader = DataLoader(
        TensorDataset(x_, y_, bg_),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    x_norm = []
    y_pred = []
    with torch.no_grad():
        for x, _, bg_mask in volume_dataloader:
            x_norm_part = norm(x.to(device))
            bg_mask = bg_mask.to(device)

            x_norm_part = background_suppression(x_norm_part, bg_mask, bg_suppression_opts)

            x_norm.append(x_norm_part.cpu())

            y_pred_part, _ = seg(x_norm_part)
            y_pred.append(y_pred_part.cpu())

    x_norm = torch.vstack(x_norm)
    y_pred = torch.vstack(y_pred)

    # Rescale x and y to the original resolution
    x_norm = x_norm.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)
    y_pred = y_pred.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)

    x_norm = F.interpolate(x_norm, size=(D, H, W), mode='trilinear')
    y_pred = F.interpolate(y_pred, size=(D, H, W), mode='trilinear')

    export_images(
        x_original,
        x_norm,
        y_original,
        y_pred,
        n_classes=n_classes,
        output_dir=os.path.join(logdir, 'segmentations'),
        image_name=f'{dataset}_test_{index:03}_{iteration:03}{appendix}.png'
    )

    dices, dices_fg = dice_score(y_pred, y_original, soft=False, reduction='none', epsilon=1e-5)
    print(f'Iteration {iteration} - dice score {dices_fg.mean().item()}')

    return dices.cpu(), dices_fg.cpu()


def tta_gnn(
    volume_dataset,
    dataset,
    logdir,
    norm,
    seg,
    gnn,
    projection_head,
    device,
    batch_size,
    dataset_repetition,
    learning_rate,
    num_steps,
    n_classes,
    num_workers,
    index,
    source_train_embeddings,
    source_sectors,
    source_slice_idxs,
    opt_type,
    opt_settings,
    bg_suppression_opts,
    bg_suppression_opts_tta,
    scheduler_settings,
    const_aug_per_volume,
    accumulate_over_volume,
    save_embeddings,
    save_checkpoints,
    calculate_dice_every,
    plot_pca_every,
    selection_criterion=['l2_distance', 'cosine_distance', 'tta_loss', 'tta_loss_no_aug']
):

    volume_dataloader = DataLoader(
        ConcatDataset([volume_dataset] * dataset_repetition),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    metric_dataloader = DataLoader(
        volume_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        norm.parameters(),
        lr=learning_rate
    )

    if deep_get(scheduler_settings, 'type') == 'MultiStepLR':
        milestones = deep_get(scheduler_settings, 'milestones', default=[])
        gamma = deep_get(scheduler_settings, 'gamma', default=1)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    else:
        lambda_lr = lambda epoch: 1
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda_lr)

    seg.requires_grad_(False)
    gnn.requires_grad_(False)
    projection_head.requires_grad_(False)

    slice_range = deep_get(opt_settings, 'slice_range', default=0.25)

    with torch.no_grad():
        source_train_embeddings = projection_head(
            source_train_embeddings.unsqueeze(-1).unsqueeze(-1)
        )
    
    if save_embeddings:
        os.makedirs(os.path.join(logdir, 'embeddings'), exist_ok=True)
        save_checkpoint(
            os.path.join(logdir, 'embeddings', f'source.pth'),
            embeddings=source_train_embeddings.cpu(),
            slice_sectors=source_sectors.cpu(),
        )

    if plot_pca_every > 0:
        os.makedirs(os.path.join(logdir, 'pca'), exist_ok=True)
        pca_vis = PCA(n_components=2)
        train_embeddings_pca = pca_vis.fit_transform(source_train_embeddings.cpu())


    # Setting up the loss functions.
    assert_in(
        opt_type, 'opt_type',
        ['opt_slice_sectors', 'opt_contrastive_slice_sectors',
            'opt_nearest_neighbor', 'opt_kl_divergence'],
    )
    if opt_type == 'opt_slice_sectors':
        
        loss_fn = SliceSectorLoss(
            source_train_embeddings,
            source_slice_idxs,
            source_sectors,
            device,
            slice_range,
            deep_get(opt_settings, 'pull_towards', default='cluster_center'),
            deep_get(opt_settings, 'distance_metric', default='l2_distance'),
            deep_get(opt_settings, 'random_nearby_samples_range', default=0.02),
        )

    elif opt_type == 'opt_contrastive_slice_sectors':
        
        loss_fn = ContrastiveSliceSectorLoss(
            source_train_embeddings,
            source_slice_idxs,
            device,
            deep_get(opt_settings, 'temperature', default=0.1),
            deep_get(opt_settings, 'source_batch_size', default=32),
            slice_range,
        )

    elif opt_type == 'opt_nearest_neighbor':

        loss_fn = NearestNeighborLoss(
            source_train_embeddings,
            device,
            deep_get(opt_settings, 'n_clusters', default=4),
        )

    elif opt_type == 'opt_kl_divergence':

        n_source, dim = source_train_embeddings.shape

        # TODO: do this for all losses
        # Suppressing embeddings which occur very often as these embeddings correspond to empty slices.
        embeddings_unique, counts = source_train_embeddings.unique(dim=0, return_counts=True)
        if counts.max() > 5:  # TODO parametrize this threshold
            empty_embedding = embeddings_unique[counts.argmax()]
            empty_mask = (empty_embedding.repeat(n_source, 1) == source_train_embeddings).all(1)
            source_train_embeddings = source_train_embeddings[~empty_mask]
            source_sectors = source_sectors[~empty_mask]
            source_slice_idxs = source_slice_idxs[~empty_mask]
            train_embeddings_pca = train_embeddings_pca[~empty_mask.cpu()]

        loss_fn = KLDivergenceLoss(
            source_train_embeddings,
            source_sectors,
            device,
            slice_range,
            deep_get(opt_settings, 'kl_mode', default='forward'),
        )

    # Setting up metrics for model selection.
    tta_losses = []
    test_scores = []

    norm_dict = {
        metric: copy.deepcopy(norm.state_dict()) for metric in selection_criterion
    }
    metrics = {metric: [] for metric in selection_criterion}
    metrics_best = {metric: torch.inf for metric in selection_criterion}

    n_clusters = len(source_sectors.unique())
    source_cluster_centers = torch.stack([
        source_train_embeddings[source_sectors == i].mean(0)
        for i in range(n_clusters)
    ])

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

        if accumulate_over_volume:
            optimizer.zero_grad()

        tta_loss = 0
        n_samples = 0

        norm.train()
        volume_dataset.dataset.set_augmentation(True)

        if const_aug_per_volume:
            volume_dataset.dataset.set_seed(get_seed())

        # Adapting to the target distribution.
        for x,_,_,rel_slice_idx, bg_mask in volume_dataloader:

            if not accumulate_over_volume:
                optimizer.zero_grad()

            x = x.to(device).float()
            bg_mask = bg_mask.to(device)
            x_norm = norm(x)
            
            x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)

            mask, logits = seg(x_norm)

            gnn_input = compress([x_norm, logits, mask], gnn.input_mask)
            graphs = gnn.get_graph(*gnn_input)
            z = gnn(graphs)
            embeddings = gnn.get_embedding(z)

            embeddings = projection_head(embeddings.unsqueeze(-1).unsqueeze(-1))

            if opt_type == 'opt_contrastive_slice_sectors':
                loss, n_defined = loss_fn(embeddings, rel_slice_idx)
                if n_defined == 0:
                    print('No positive examples in this batch. Batch skipped.')
                    continue
                loss /= n_defined

            else:
                # opt_type is in ['opt_slice_sectors', 'opt_nearest_neighbor', 'opt_kl_divergence']
                loss = loss_fn(embeddings, rel_slice_idx)

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

        scheduler.step()

        tta_losses.append((tta_loss / n_samples).item())

        norm.eval()
        volume_dataset.dataset.set_augmentation(False)

        # Getting embeddings for target slices.
        embeddings, test_slice_sectors, rel_slice_idx = get_embeddings(
            metric_dataloader, norm, seg, gnn, device,
            'pred', slice_range, bg_suppression_opts,
        )
        embeddings = projection_head(embeddings.unsqueeze(-1).unsqueeze(-1))

        if save_embeddings:
            save_checkpoint(
                os.path.join(logdir, 'embeddings', f'{dataset}_{index:02d}_{step:03d}.pth'),
                embeddings=embeddings.cpu(),
                slice_sectors=test_slice_sectors.cpu(),
            )

        # Creating PCA plot visualizing the embeddings of the source and target slices.
        if step % plot_pca_every == 0 and plot_pca_every != -1:

            test_embeddings_pca = pca_vis.transform(embeddings.cpu())

            train_labels = [f'SD compartment {i}' for i in np.unique(source_sectors.cpu())]
            test_labels = [f'TD compartment {i}' for i in np.unique(test_slice_sectors.cpu())]
            plt.figure(figsize=(8,7))
            multilabel_scatter(train_embeddings_pca[:, 0], train_embeddings_pca[:, 1], c=source_sectors.cpu()*2+1, marker='x', label=train_labels, cmap='Paired', vmin=0, vmax=12)
            multilabel_scatter(test_embeddings_pca[:,0], test_embeddings_pca[:,1], c=test_slice_sectors.cpu()*2, marker='+', label=test_labels, cmap='Paired', vmin=0, vmax=12)
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(logdir, 'pca', f'pca_{dataset}_{index:02d}_{step:03d}.jpg'), dpi=200)
            plt.close()

        # Calculating some metrics and selecting models.
        cc = torch.index_select(source_cluster_centers, 0, test_slice_sectors.to(device))
        for crit in selection_criterion:
            assert_in(
                crit, 'crit',
                ['l2_distance', 'cosine_distance', 'tta_loss', 'tta_loss_no_aug']
            )
            if crit == 'l2_distance':
                score = F.pairwise_distance(embeddings, cc).mean().item()

            elif crit == 'cosine_distance':
                score = 1 - F.cosine_similarity(embeddings, cc).mean().item()

            elif crit == 'tta_loss':
                score = tta_losses[-1]

            elif crit == 'tta_loss_no_aug':
                if opt_type == 'opt_contrastive_slice_sectors':
                    score, n_defined = loss_fn(embeddings, rel_slice_idx)
                    score /= n_defined

                else:
                    score = loss_fn(embeddings, rel_slice_idx)
                score = score.mean().item()

            metrics[crit].append(score)
            if score <= metrics_best[crit]:
                metrics_best[crit] = score
                norm_dict[crit] = copy.deepcopy(norm.state_dict())

    if save_checkpoints:
        os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)

        if 'tta_loss_no_aug' in selection_criterion:
            norm_state_dict = norm_dict['tta_loss_no_aug']
        else:
            norm_state_dict = norm.state_dict()

        save_checkpoint(
            path=os.path.join(logdir, 'checkpoints',
                              f'checkpoint_tta_{dataset}_{index:02d}.pth'),
            norm_state_dict=norm_state_dict,
            seg_state_dict=seg.state_dict(),
            gnn_state_dict=gnn.state_dict(),
            projection_head_state_dict=projection_head.state_dict(),
            train_embeddings=source_train_embeddings,
            train_cluster_assignment=source_sectors,
            train_rel_slice_idxs=source_slice_idxs,
        )

    os.makedirs(os.path.join(logdir, 'metrics'), exist_ok=True)
    write_to_csv(
        os.path.join(logdir, 'metrics', f'{dataset}_{index:03d}.csv'),
        np.stack([metrics[crit] for crit in selection_criterion], axis=1),
        header=selection_criterion,
        mode='w',
    )

    os.makedirs(os.path.join(logdir, 'tta_score'), exist_ok=True)
    write_to_csv(
        os.path.join(logdir, 'tta_score', f'{dataset}_{index:03d}.csv'),
        np.array([test_scores]).T,
        header=['tta_score'],
        mode='w',
    )

    return norm, norm_dict, metrics_best


def tta_gt(
    volume_dataset,
    dataset,
    logdir,
    norm,
    seg,
    device,
    batch_size,
    dataset_repetition,
    learning_rate,
    accumulate_over,
    num_steps,
    n_classes,
    num_workers,
    index,
    bg_suppression_opts,
    bg_suppression_opts_tta,
    const_aug_per_volume,
    accumulate_over_volume,
    save_checkpoints,
    calculate_dice_every,
):

    if not accumulate_over_volume:
        raise NotImplementedError()

    volume_dataloader = DataLoader(
        ConcatDataset([volume_dataset] * dataset_repetition),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        norm.parameters(),
        lr=learning_rate
    )

    seg.requires_grad_(False)

    loss_func = DiceLoss()

    tta_losses = []
    calculate_dice_every *= accumulate_over

    num_steps *= accumulate_over
    for step in range(num_steps):

        if step % calculate_dice_every == 0 and calculate_dice_every != -1:
            norm.eval()
            volume_dataset.dataset.set_augmentation(False)

            test_volume(
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

        if step % accumulate_over == 0:
            optimizer.zero_grad()
            tta_loss = 0

        norm.train()
        volume_dataset.dataset.set_augmentation(True)
        
        if const_aug_per_volume:
            volume_dataset.dataset.set_seed(get_seed())

        for x,y,_,_,bg_mask in volume_dataloader:

            x = x.to(device).float()
            y = y.to(device)
            bg_mask = bg_mask.to(device)

            x_norm = norm(x)

            x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)

            y_pred, logits = seg(x_norm)

            loss = loss_func(y_pred, y)

            loss = loss / (len(volume_dataloader) * accumulate_over)
            loss.backward()

            with torch.no_grad():
                tta_loss += loss.detach()

        if (step + 1) % accumulate_over == 0:
            tta_losses.append(tta_loss)
            optimizer.step()
    
    if save_checkpoints:
        os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)
        save_checkpoint(
            path=os.path.join(logdir, 'checkpoints',
                              f'checkpoint_tta_{dataset}_{index:02d}.pth'),
            norm_state_dict=norm.state_dict(),
            seg_state_dict=seg.state_dict(),
        )

    return norm, {}, {}


def main():

    print(f'Running {__file__}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=False, type=int,
                        help='starting volume index to be used for testing')
    parser.add_argument('--stop', required=False, type=int,
                        help='stopping volume index to be used for testing (index not included)')

    testing_params = load_config('config/params.yaml')
    print('Loaded params')
    tta_mode = deep_get(testing_params, 'testing', 'tta_mode', default='no_tta')
    logdir = deep_get(testing_params, 'testing', tta_mode, 'logdir', default='logs')
    
    params_path = os.path.join(logdir, 'params.yaml')
    params = load_config(params_path)
    params['testing'] = deep_get(testing_params, 'testing')
    params['datasets'] = deep_get(testing_params, 'datasets')
    params['training']['projection_head'] = deep_get(testing_params, 'training', 'projection_head')

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
    accumulate_over        = deep_get(params, 'testing', tta_mode, 'accumulate_over', default=1)
    const_aug_per_volume   = deep_get(params, 'testing', tta_mode, 'const_aug_per_volume', default=False)
    accumulate_over_volume = deep_get(params, 'testing', tta_mode, 'accumulate_over_volume', default=True)
    calculate_dice_every   = deep_get(params, 'testing', tta_mode, 'calculate_dice_every', default=20)
    plot_pca_every         = deep_get(params, 'testing', tta_mode, 'plot_pca_every', default=20)
    learning_rate          = deep_get(params, 'testing', tta_mode, 'learning_rate')
    num_steps              = deep_get(params, 'testing', tta_mode, 'num_steps')
    image_channels         = deep_get(params, 'model', 'normalization', 'image_channels', default=1)

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

    if tta_mode == 'gnn':
        embedding_model_type = deep_get(params, 'training', 'gnn', 'embedding_model_type', default='vig')
        assert_in(embedding_model_type, 'embedding_model_type', ['vig', 'cnn'])

        use_normalized_image = deep_get(params, 'model', 'vig', 'input_features', 'use_normalized_image', default=True)
        use_logits = deep_get(params, 'model', 'vig', 'input_features', 'use_logits', default=True)
        use_mask = deep_get(params, 'model', 'vig', 'input_features', 'use_mask', default=False)

        gnn_input_mask = [use_normalized_image, use_logits, use_mask]
        gnn_input_channels = use_normalized_image * image_channels + \
            use_logits * n_classes + use_mask * n_classes

        if embedding_model_type == 'vig':

            channels = deep_get(params, 'model', 'vig', 'channels', default=[48, 96, 240, 384])
            gnn_out_channels = channels[-1]

            gnn = ViG(
                deep_get(params, 'model', 'vig', 'k', default=9),
                deep_get(params, 'model', 'vig', 'act', default='relu'),
                deep_get(params, 'model', 'vig', 'norm', default='batch'),
                deep_get(params, 'model', 'vig', 'bias', default=True),
                deep_get(params, 'model', 'vig', 'epsilon', default=0),
                deep_get(params, 'model', 'vig', 'use_stochastic', default=False),
                deep_get(params, 'model', 'vig', 'conv', default='mr'),
                deep_get(params, 'model', 'vig', 'drop_path', default=0),
                deep_get(params, 'model', 'vig', 'blocks', default=[2, 2, 6, 2]),
                deep_get(params, 'model', 'vig', 'channels', default=[48, 96, 240, 384]),
                deep_get(params, 'model', 'vig', 'reduce_ratios', default=[4, 2, 1, 1]),
                deep_get(params, 'testing', 'image_size'),
                n_channels=gnn_input_channels,
            ).to(device)

        elif embedding_model_type == 'cnn':
            gnn = EmbeddingCNN(
                deep_get(params, 'model', 'embedding_cnn', 'backbone', default='resnet18'),
                n_channels=gnn_input_channels,
            ).to(device)

            gnn_out_channels = gnn.out_channels

        gnn.set_input_mask(gnn_input_mask)

        use_projection_head = deep_get(params, 'testing', 'gnn', 'use_projection_head', default=False)
        if use_projection_head:
            projection_head = ProjectionHead(gnn_out_channels).to(device)
        else:
            projection_head = torch.nn.Flatten()
        
        projection_head_training_settings = deep_get(params, 'training', 'projection_head')
        train_projection_head = deep_get(params, 'testing', 'gnn', 'train_projection_head', default=False)
        train_projection_head = (train_projection_head and use_projection_head and projection_head_training_settings is not None)


    print('Loading datasets')

    bg_suppression_opts = deep_get(params, 'testing', 'bg_suppression_opts')
    bg_suppression_opts_tta = deep_get(params, 'testing', tta_mode, 'bg_suppression_opts')

    # Dataset definition
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

    # Getting checkpoint from local log dir.
    checkpoint = torch.load(os.path.join(logdir, checkpoint_best), map_location=device)
    norm.load_state_dict(checkpoint['norm_state_dict'])
    seg.load_state_dict(checkpoint['seg_state_dict'])
    
    save_checkpoints = deep_get(params, 'testing', tta_mode, 'save_checkpoints', default=True)
    
    if tta_mode == 'gnn':
        opt_type = deep_get(params, 'testing', 'gnn', 'opt_type', default='opt_slice_sectors')
        opt_settings = deep_get(params, 'testing', 'gnn', opt_type)

        gnn.load_state_dict(checkpoint['gnn_state_dict'])
        norm_state_dict = checkpoint['norm_state_dict']
        source_train_embeddings = checkpoint['train_embeddings']
        source_sectors = checkpoint['train_cluster_assignment']
        source_train_rel_slice_idxs = checkpoint['train_rel_slice_idxs']
        source_val_embeddings = checkpoint['val_embeddings']
        source_val_rel_slice_idxs = checkpoint['val_rel_slice_idxs']
        
        if use_projection_head:
            projection_head_initial_checkpoint_name = deep_get(
                params, 'testing', 'gnn', 
                'projection_head_initial_checkpoint_name',
                default='contrastive_loss_state_dict'
            )

            projection_head_state_dict = checkpoint[projection_head_initial_checkpoint_name]
            projection_head.load_state_dict(projection_head_state_dict)

        scheduler_settings = deep_get(params, 'testing', 'gnn', 'scheduler_settings', default={'type': None})
        save_embeddings = deep_get(params, 'testing', 'gnn', 'save_embeddings', default=False)

    elif tta_mode == 'gt':
        norm_state_dict = checkpoint['norm_state_dict']

    del checkpoint

    seg.eval()
    if tta_mode == 'gnn':
        gnn.eval()

    if wandb_log and tta_mode in ['gt', 'gnn']:
        wandb.watch([norm], log='all', log_freq=1)

    dice_scores = torch.zeros((len(indices_per_volume), n_classes))

    for i in range(start_idx, stop_idx):
        
        seed = deep_get(params, 'seed', default=0)
        seed_everything(seed)

        indices = indices_per_volume[i]
        print(f'processing volume {i}')
        volume_dataset = Subset(test_dataset, indices)

        if tta_mode == 'gnn' and train_projection_head:
            target_train_dataset, target_val_dataset = split_dataset(volume_dataset, [0.5, 0.5])
            
            domain_contrast = deep_get(params, 'training', 'projection_head', 'domain_contrast', default=True)
            cluster_target_domain = deep_get(params, 'training', 'projection_head', 'cluster_target_domain', default=True)
            cluster_target_domain = cluster_target_domain and domain_contrast
            bg_suppression_opts_ph = deep_get(params, 'training', 'projection_head', 'bg_suppression_opts')
            
            contrastive_loss = ContrastiveLoss(
                gnn_out_channels,
                deep_get(params, 'training', 'projection_head', 'slice_position_range', default=0.25),
                deep_get(params, 'training', 'projection_head', 'temperature', default=0.1),
                deep_get(params, 'training', 'projection_head', 'negative_examples', default='relative_position'),
                domain_contrast,
                deep_get(params, 'training', 'projection_head', 'similarity_weighting', default='none'),
                use_projection_head=False,
            ).to(device)

            optimizer = torch.optim.Adam(
                projection_head.parameters(),
                lr=deep_get(params, 'training', 'projection_head', 'learning_rate')
            )

            source_train_dataset = TensorDataset(source_train_embeddings.cpu(), source_train_rel_slice_idxs.cpu())
            source_val_dataset = TensorDataset(source_val_embeddings.cpu(), source_val_rel_slice_idxs.cpu())
            projection_head.load_state_dict(projection_head_state_dict)

            projection_head.requires_grad_(True)

            projection_head_state_dict_trained, training_loss, validation_loss = train_ph(
                norm,
                seg,
                gnn,
                projection_head,
                contrastive_loss,
                optimizer,
                source_train_dataset,
                target_train_dataset,
                source_val_dataset,
                target_val_dataset,
                batch_size,
                cluster_target_domain,
                bg_suppression_opts_ph,
                num_workers,
                deep_get(params, 'training', 'projection_head', 'epochs'),
                device,
                wandb_log,
            )

            projection_head.load_state_dict(projection_head_state_dict_trained)

            output_dir = os.path.join(logdir, 'ttt_projection_head_training_stats')
            os.makedirs(output_dir, exist_ok=True)
            write_to_csv(
                path=os.path.join(output_dir, f'{i:03d}.csv'),
                data=np.stack([training_loss, validation_loss], 1),
                header=['training_losses', 'validation_losses'],
                mode='w',
            )

        if tta_mode == 'gnn':
            norm.load_state_dict(norm_state_dict)

            norm, norm_dict, metrics_best = tta_gnn(
                volume_dataset=volume_dataset,
                dataset=dataset,
                logdir=logdir,
                norm=norm,
                seg=seg,
                gnn=gnn,
                projection_head=projection_head,
                device=device,
                batch_size=batch_size,
                dataset_repetition=dataset_repetition,
                learning_rate=learning_rate,
                num_steps=num_steps,
                n_classes=n_classes,
                num_workers=num_workers,
                index=i,
                source_train_embeddings=source_train_embeddings,
                source_sectors=source_sectors,
                source_slice_idxs=source_train_rel_slice_idxs,
                opt_type=opt_type,
                opt_settings=opt_settings,
                bg_suppression_opts=bg_suppression_opts,
                bg_suppression_opts_tta=bg_suppression_opts_tta,
                scheduler_settings=scheduler_settings,
                const_aug_per_volume=const_aug_per_volume,
                accumulate_over_volume=accumulate_over_volume,
                save_embeddings=save_embeddings,
                save_checkpoints=save_checkpoints,
                calculate_dice_every=calculate_dice_every,
                plot_pca_every=plot_pca_every,
            )

        elif tta_mode == 'gt':
            norm.load_state_dict(norm_state_dict)

            norm, norm_dict, metrics_best = tta_gt(
                volume_dataset=volume_dataset,
                dataset=dataset,
                logdir=logdir,
                norm=norm,
                seg=seg,
                device=device,
                batch_size=batch_size,
                dataset_repetition=dataset_repetition,
                accumulate_over=accumulate_over,
                learning_rate=learning_rate,
                num_steps=num_steps,
                n_classes=n_classes,
                num_workers=num_workers,
                index=i,
                bg_suppression_opts=bg_suppression_opts,
                bg_suppression_opts_tta=bg_suppression_opts_tta,
                const_aug_per_volume=const_aug_per_volume,
                accumulate_over_volume=accumulate_over_volume,
                save_checkpoints=save_checkpoints,
                calculate_dice_every=calculate_dice_every,
            )

        else:
            norm_dict = metrics_best = {}

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
                appendix=f'_min_{key}',
                bg_suppression_opts=bg_suppression_opts,
            )

            write_to_csv(
                os.path.join(logdir, f'scores_{dataset}_{key}.csv'),
                np.hstack([[[f'volume_{i:02d}']], dice_scores[None, i, :].numpy()]),
                mode='a',
            )

    print(f'Overall mean dice (only foreground): {dice_scores[:, 1:].mean()}')

    if wandb_log:
        wandb.finish()


if __name__ == '__main__':
    main()
