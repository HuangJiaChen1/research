"""
Pilot for Idea 1: Residual-Weighted Global Pooling (RWGP)

Simulates MB-FFN global pooling with and without residual-based weighting.
Measures how much the global feature shifts from a clean baseline as outlier
ratio increases.
"""

import torch
import numpy as np

N = 200
C = 128
outlier_ratios = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]

def vanilla_pooling(feats):
    avg = feats.mean(dim=2, keepdim=True)
    max_p = feats.max(dim=2, keepdim=True)[0]
    return avg, max_p

def residual_weighted_pooling(feats, residual, tau=0.1):
    # residual: [B, 1, N, 1]  (higher = worse)
    w = torch.exp(-residual / tau)  # [B, 1, N, 1]
    w = w / (w.sum(dim=2, keepdim=True) + 1e-6)
    weighted_avg = (feats * w).sum(dim=2, keepdim=True)
    # For max pooling, we can weight and take max of weighted features
    weighted_max = (feats * w).max(dim=2, keepdim=True)[0]
    return weighted_avg, weighted_max

np.random.seed(42)
torch.manual_seed(42)

# Baseline clean features
base_feats = torch.randn(1, C, N, 1) * 0.3 + 1.0
base_avg, base_max = vanilla_pooling(base_feats)

print("=" * 70)
print("PILOT: Residual-Weighted Global Pooling (Idea 1)")
print("=" * 70)
print(f"{'Outlier':>8} | {'Vanilla Avg Shift':>18} | {'RWGP Avg Shift':>16} | {'Reduction':>10}")
print("-" * 70)

for ratio in outlier_ratios[1:]:
    n_outliers = int(N * ratio)
    n_inliers = N - n_outliers

    # Realistic mixture: inliers tight, outliers scattered but overlapping
    inlier_feats = torch.randn(1, C, n_inliers, 1) * 0.3 + 1.0
    outlier_feats = torch.randn(1, C, n_outliers, 1) * 3.0 - 1.0
    feats = torch.cat([inlier_feats, outlier_feats], dim=2)

    # Simulate residual: inliers have low residual (~0.01), outliers high (~0.5-1.0)
    inlier_res = torch.rand(1, 1, n_inliers, 1) * 0.02
    outlier_res = torch.rand(1, 1, n_outliers, 1) * 0.8 + 0.2
    residual = torch.cat([inlier_res, outlier_res], dim=2)

    # Random shuffle so outliers are not all at the end
    perm = torch.randperm(N)
    feats = feats[:, :, perm, :]
    residual = residual[:, :, perm, :]

    v_avg, v_max = vanilla_pooling(feats)
    r_avg, r_max = residual_weighted_pooling(feats, residual)

    v_shift = (v_avg - base_avg).abs().mean().item()
    r_shift = (r_avg - base_avg).abs().mean().item()
    reduction = (1 - r_shift / (v_shift + 1e-6)) * 100

    print(f"{ratio*100:7.0f}% | {v_shift:18.4f} | {r_shift:16.4f} | {reduction:9.1f}%")

print("=" * 70)
