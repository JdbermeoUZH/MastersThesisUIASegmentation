import torch
from torch.nn import Sequential
from dataset.dataset_in_memory import get_sectors_from_index
from utils.utils import assert_in


def cosine_similarity(x1, x2):
    sim = torch.cosine_similarity(x1, x2, dim=1)
    sim = sim.mean()
    return sim

class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        dim=2048,
        slice_idx_range=0,
        temperature=0.1,
        negative_examples='relative_position',
        domain_contrast=True,
        similarity_weighting='none',
        use_projection_head=True,
    ):
        super().__init__()

        if use_projection_head:
            self.global_mlp = Sequential(*[
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(dim, dim),
                torch.nn.BatchNorm1d(dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, dim),
            ])
        else:
            self.global_mlp = None

        self.temperature = temperature

        assert slice_idx_range > 0
        self.slice_idx_range = slice_idx_range
        assert_in(negative_examples, 'negative_examples', ['relative_position', 'absolute_position'])
        self.negative_examples = negative_examples
        self.domain_contrast = domain_contrast
        assert_in(similarity_weighting, 'similarity_weighting', ['none', 'exp_difference'])
        self.similarity_weighting = similarity_weighting


    def get_neg_pos_indices(self, b, B, rel_slice_idx):

        if self.negative_examples == 'relative_position':
            neg_idx = torch.where(
                (rel_slice_idx[b] - rel_slice_idx).abs() > self.slice_idx_range)[0]
            pos_idx = torch.where(
                (rel_slice_idx[b] - rel_slice_idx).abs() <= self.slice_idx_range)[0]

        elif self.negative_examples == 'absolute_position':
            sectors = get_sectors_from_index(
                rel_slice_idx, sector_size=self.slice_idx_range)
            neg_idx = torch.where(sectors != sectors[b])[0]
            pos_idx = torch.where(sectors == sectors[b])[0]

        else:
            raise NotImplementedError()

        pos_idx = pos_idx[pos_idx != b]

        if self.domain_contrast:
            neg_idx = torch.cat([neg_idx, torch.arange(B) + B])

        return neg_idx, pos_idx


    def get_similarity_weights(self, rel_slice_idx):
        if self.domain_contrast:
            rel_slice_idx = rel_slice_idx.repeat(2)
        
        size = rel_slice_idx.shape[0]
        sectors = get_sectors_from_index(rel_slice_idx, sector_size=self.slice_idx_range)

        if self.similarity_weighting == 'exp_difference':
            weights = 2 ** (sectors[None,:] - sectors[:,None]).abs()
            return weights
        
        return torch.ones((size, size))


    def forward(self, z0, z1, rel_slice_idx_0, rel_slice_idx_1):

        device = z0.device

        if self.domain_contrast:
            rel_slice_idx = rel_slice_idx_0
        else:
            rel_slice_idx = torch.cat([rel_slice_idx_0, rel_slice_idx_1])

        z = torch.cat([z0, z1], dim=0)
        if self.global_mlp:
            z = self.global_mlp(z)

        # Calculating similarities between all samples in the batch.
        B = z0.shape[0]
        similarities = torch.zeros((2*B, 2*B), device=device)
        for b in range(2*B):
            similarities[b, :] = torch.cosine_similarity(z[None, b], z)

        sim_weights = self.get_similarity_weights(rel_slice_idx).to(device)
        similarities = torch.exp(sim_weights * similarities / self.temperature)

        loss = 0
        n_defined = 0
        for b in range(B):
            neg_idx, pos_idx = self.get_neg_pos_indices(b, B, rel_slice_idx)

            if pos_idx.nelement() == 0:
                continue

            n_defined += 1
            loss += torch.mean(-torch.log(
                similarities[b, pos_idx] / 
                (similarities[b, pos_idx] + similarities[b, neg_idx].sum()))
            )

        return loss, n_defined
