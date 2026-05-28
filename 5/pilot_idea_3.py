"""
Pilot for Idea 2: Cross-Stage Edge Agreement Gating (CSEAG)

Simulates CSMGC with three stage graphs. One stage is pure noise.
Measures output variance and mean absolute difference from a clean baseline
with and without agreement gating.
"""

import torch
import torch.nn as nn
import numpy as np

N = 100
C = 64
k = 2

# Reproduce exact MGCA-Net knn/get_graph_feature from verify_pain_points.py
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

# Simulated CSMGC forward (vanilla)
class VanillaCSMGC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.annular_convolution = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 3), stride=(1, 3)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 2)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.MLP = nn.Sequential(
            nn.InstanceNorm2d(in_channels * 2, eps=1e-3),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
        )

    def forward(self, Stage_1, Stage_2, Stage_3):
        S1_graph = get_graph_feature(Stage_1, k=k)
        S2_graph = get_graph_feature(Stage_2, k=k)
        S3_graph = get_graph_feature(Stage_3, k=k)
        Combine = torch.cat([S1_graph, S2_graph, S3_graph], dim=-1)
        ANN_out = self.MLP(self.annular_convolution(Combine))
        out = ANN_out + Stage_3
        return out

# Simulated CSMGC with Edge Agreement Gating
class AgreementCSMGC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.annular_convolution = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 3), stride=(1, 3)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 2)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.MLP = nn.Sequential(
            nn.InstanceNorm2d(in_channels * 2, eps=1e-3),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
        )

    def forward(self, Stage_1, Stage_2, Stage_3):
        # Compute k-NN indices for each stage
        idx1 = knn(Stage_1.view(1, -1, N), k=k)
        idx2 = knn(Stage_2.view(1, -1, N), k=k)
        idx3 = knn(Stage_3.view(1, -1, N), k=k)

        # Build adjacency masks [B, N, N]
        def idx_to_mask(idx):
            B, Np, K = idx.shape
            mask = torch.zeros(B, Np, Np, device=idx.device)
            for b in range(B):
                for i in range(Np):
                    mask[b, i, idx[b, i]] = 1.0
            return mask

        m1 = idx_to_mask(idx1)
        m2 = idx_to_mask(idx2)
        m3 = idx_to_mask(idx3)
        agreement = (m1 + m2 + m3) / 3.0  # [B, N, N]

        S1_graph = get_graph_feature(Stage_1, k=k, idx=idx1)
        S2_graph = get_graph_feature(Stage_2, k=k, idx=idx2)
        S3_graph = get_graph_feature(Stage_3, k=k, idx=idx3)
        Combine = torch.cat([S1_graph, S2_graph, S3_graph], dim=-1)

        # Apply agreement gating: avg agreement per edge in the k-NN set
        # Combine shape [B, C, N, k*3]; we need a scalar weight per spatial location
        # Simplification: use mean agreement over the k neighbors for each stage
        B, Cc, Np, K3 = Combine.shape
        K = K3 // 3
        # Reshape to separate stages
        combine_5d = Combine.view(B, Cc, Np, 3, K)
        # agreement for each neighbor index
        agr_expanded = agreement.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, N]
        # We need to gather agreement scores for the actual neighbor indices
        # This is complex; for the pilot we use a simpler proxy:
        # weight each stage's graph by the fraction of edges that agree with the other two stages
        stage_agree = torch.stack([
            (m1 * m2).sum(dim=-1) + (m1 * m3).sum(dim=-1),
            (m2 * m1).sum(dim=-1) + (m2 * m3).sum(dim=-1),
            (m3 * m1).sum(dim=-1) + (m3 * m2).sum(dim=-1)
        ], dim=-1) / (2 * k)  # [B, N, 3]
        stage_agree = stage_agree / (stage_agree.sum(dim=-1, keepdim=True) + 1e-6)
        # Apply stage-wise weighting
        weights = stage_agree.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)  # [B, 1, 3, N, 1]
        weights = weights.permute(0, 1, 3, 2, 4)  # [B, 1, N, 3, 1]
        combine_5d = combine_5d * weights
        Combine = combine_5d.view(B, Cc, Np, K3)

        ANN_out = self.MLP(self.annular_convolution(Combine))
        out = ANN_out + Stage_3
        return out

np.random.seed(42)
torch.manual_seed(42)

vanilla = VanillaCSMGC(C, C)
agreement = AgreementCSMGC(C, C)
vanilla.eval()
agreement.eval()

# Clean stages
S1 = torch.randn(1, C, N, 1)
S2 = torch.randn(1, C, N, 1)
S3 = torch.randn(1, C, N, 1)

with torch.no_grad():
    clean_vanilla = vanilla(S1, S2, S3)
    clean_agree = agreement(S1, S2, S3)

# One noisy stage
S1_bad = torch.randn(1, C, N, 1) * 5.0

with torch.no_grad():
    bad_vanilla = vanilla(S1_bad, S2, S3)
    bad_agree = agreement(S1_bad, S2, S3)

var_vanilla = bad_vanilla.var().item()
var_agree = bad_agree.var().item()
diff_vanilla = (bad_vanilla - clean_vanilla).abs().mean().item()
diff_agree = (bad_agree - clean_agree).abs().mean().item()

print("=" * 70)
print("PILOT: Cross-Stage Edge Agreement Gating (Idea 2)")
print("=" * 70)
print(f"Vanilla CSMGC  - Variance: {var_vanilla:.4f}  MeanAbsDiff: {diff_vanilla:.4f}")
print(f"Agreement CSMGC - Variance: {var_agree:.4f}  MeanAbsDiff: {diff_agree:.4f}")
print(f"Variance Reduction: {(1 - var_agree/var_vanilla)*100:.1f}%")
print(f"Diff Reduction:     {(1 - diff_agree/diff_vanilla)*100:.1f}%")
print("=" * 70)
