from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import get_sectors_from_index
from utils.contrastive_loss import ContrastiveLoss
from utils.distributions import normal
from utils.utils import assert_in, random_select_from_tensor


class SliceSectorLoss(nn.Module):
    def __init__(
        self,
        source_embeddings,
        source_slice_idx,
        source_sectors,
        device,
        slice_range,
        pull_towards,
        distance_metric,
        nearby_samples_range,
    ):
        super().__init__()

        assert_in(
            pull_towards, 'pull_towards',
            ['cluster_center', 'random_center_sample',
             'random_compartment_samples', 'random_nearby_samples'],
        )
        assert_in(
            distance_metric, 'distance_metric',
            ['cosine_similarity', 'l2_distance']
        )

        self.source_embeddings = source_embeddings.clone()
        self.source_slice_idx = source_slice_idx.clone()
        self.source_sectors = source_sectors.clone()
        self.device = device
        self.slice_range = slice_range
        self.pull_towards = pull_towards
        self.distance_metric = distance_metric
        self.nearby_samples_range = nearby_samples_range

        self.n_clusters = len(self.source_embeddings.unique())
        self.cluster_centers = torch.stack([
            self.source_embeddings[self.source_sectors == i].mean(0)
            for i in range(self.n_clusters)
        ])


    def resample_cluster_centers(self, target_slice_idx):
        target_sectors = get_sectors_from_index(
            target_slice_idx,
            sector_size=self.slice_range,
        )
        target_sectors = target_sectors.to(self.device)

        if self.pull_towards == 'cluster_center':
            # Pulling all embeddings towards the compartment center of the source embeddings.
            cc = torch.index_select(self.cluster_centers, 0, target_sectors)

        elif self.pull_towards == 'random_center_sample':
            # Pulling all embeddings towards the same random sample of the source embeddings.
            cluster_centers = torch.stack([
                random_select_from_tensor(
                    self.source_embeddings[self.source_sectors == i],
                    dim=0) for i in range(self.n_clusters)
            ])

            cc = torch.index_select(cluster_centers, 0, target_sectors)

        elif self.pull_towards == 'random_compartment_samples':
            # Pulling each embeddings towards a (different) random sample of the source embeddings.
            cc = torch.stack([
                random_select_from_tensor(
                    self.source_embeddings[self.source_sectors == i],
                    dim=0) for i in target_sectors
            ])

        elif self.pull_towards == 'random_nearby_samples':
            # random_nearby_samples_range = deep_get(opt_settings, 'random_nearby_samples_range', default=0.02)
            cc = torch.stack([
                random_select_from_tensor(
                    self.source_embeddings[(self.source_slice_idx - i).abs() < self.nearby_samples_range],
                    dim=0) for i in target_slice_idx
            ])

        return cc
    
    
    def distance(self, target_embeddings, cc):
        if self.distance_metric == 'l2_distance':
            distances = F.pairwise_distance(target_embeddings, cc)
        elif self.distance_metric == 'cosine_similarity':
            distances = 1 - F.cosine_similarity(target_embeddings, cc)

        return distances.mean()


    def forward(self, target_embeddings, target_slice_idx):
        cc = self.resample_cluster_centers(target_slice_idx)
        distance = self.distance(target_embeddings, cc)

        return distance
    

class ContrastiveSliceSectorLoss(nn.Module):
    def __init__(
        self,
        source_embeddings,
        source_slice_idx,
        device,
        temperature,
        source_batch_size,
        slice_range,
        
    ):
        super().__init__()

        self.source_embeddings = source_embeddings.clone()
        self.source_slice_idx = source_slice_idx.clone()
        self.device = device
        self.source_batch_size = source_batch_size

        self.n_source, dim = self.source_embeddings.shape

        self.contrastive_loss = ContrastiveLoss(
            dim=dim,
            slice_idx_range=slice_range,
            temperature=temperature,
            domain_contrast=False,
            negative_examples='absolute_position',
            use_projection_head=False
        )

    def forward(self, target_embeddings, target_slice_idx):
        source_example_selection = torch.randint(
            0, self.n_source, (self.source_batch_size,))

        loss, n_defined = self.contrastive_loss(
            self.source_embeddings[source_example_selection, :],
            target_embeddings,
            self.source_slice_idx[source_example_selection],
            target_slice_idx.to(self.device),
        )

        return loss, n_defined


class NearestNeighborLoss(nn.Module):
    def __init__(
        self,
        source_embeddings,
        device,
        n_clusters,
    ):
        super().__init__()

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(source_embeddings.cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        self.cluster_centers = torch.from_numpy(cluster_centers).to(device)


    def forward(self, target_embeddings, target_slice_idx):
        distances = torch.cdist(target_embeddings, self.cluster_centers)
        distances = torch.min(distances, dim=1).values

        return distances.mean()


class KLDivergenceLoss(nn.Module):
    def __init__(
        self,
        source_embeddings,
        source_sectors,
        device,
        slice_range,
        kl_mode,
    ):
        super().__init__()

        assert_in(kl_mode, 'kl_mode', ['forward', 'reverse'])
        
        source_embeddings = source_embeddings.clone()

        self.device = device
        self.slice_range = slice_range
        self.kl_mode = kl_mode

        n_source, dim = source_embeddings.shape
        sectors_unique, sector_counts = source_sectors.unique(return_counts=True)
        n_sectors = len(sectors_unique)
        self.dim_pca = min(dim, *sector_counts.tolist())

        self.cluster_centers = torch.zeros((n_sectors, dim), device=device)
        self.cluster_rotations = torch.zeros((n_sectors, self.dim_pca, dim), device=device)
        self.cluster_scales = torch.zeros((n_sectors, self.dim_pca), device=device)

        source_embeddings_kl = torch.zeros((n_source, self.dim_pca), device=device)

        for i in range(n_sectors):
            # Centering each source cluster and storing the means.
            self.cluster_centers[i, :] = source_embeddings[source_sectors == i].mean(0)
            source_embeddings[source_sectors == i] -= self.cluster_centers[i, :]

            # Rotating each source cluster with PCA and storing the rotation matrices.
            pca = PCA(n_components=self.dim_pca)
            pca.fit(source_embeddings[source_sectors == i].cpu())
            self.cluster_rotations[i, :, :] = torch.from_numpy(pca.components_).to(device)
            source_embeddings_kl[source_sectors == i] = torch.matmul(
                source_embeddings[source_sectors == i],
                self.cluster_rotations[i, :, :].T
            )

            # Getting standard deviation for each dimension and rescaling embeddings.
            _, cov = normal.fit_distribution(
                source_embeddings_kl[source_sectors == i],
                diagonal_cov=True,
            )
            self.cluster_scales[i, :] = cov.diag() ** (-0.5)
            source_embeddings_kl[source_sectors == i] *= self.cluster_scales[i, :]

        # Defining distribution for transformed source embeddings.
        source_mean, source_cov = normal.fit_distribution(
            source_embeddings_kl,
            diagonal_cov=True,
        )
        self.source_distribution = MultivariateNormal(source_mean, source_cov, validate_args=True)


    def forward(self, target_embeddings, target_slice_idx):
        target_sectors = get_sectors_from_index(
            target_slice_idx,
            sector_size=self.slice_range
        )

        batch_size = target_embeddings.shape[0]
        embeddings_kl = torch.zeros((batch_size, self.dim_pca), device=self.device)

        for i in target_sectors.unique():
            target_embeddings[target_sectors == i] -= self.cluster_centers[i, :]

            embeddings_kl[target_sectors == i] = torch.matmul(
                target_embeddings[target_sectors == i],
                self.cluster_rotations[i, :, :].T
            )

            embeddings_kl[target_sectors == i] *= self.cluster_scales[i, :]

        target_mean, target_cov = normal.fit_distribution(
            embeddings_kl,
            diagonal_cov=True,
        )

        target_distribution = MultivariateNormal(target_mean, target_cov, validate_args=True)

        if self.kl_mode == 'forward':
            kl_div = kl_divergence(target_distribution, self.source_distribution)
        elif self.kl_mode == 'reverse':
            kl_div = kl_divergence(self.source_distribution, target_distribution)

        return kl_div
