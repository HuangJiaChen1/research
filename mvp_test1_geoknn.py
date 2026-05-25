#!/usr/bin/env python3
"""
MVP Test 1: GeoKNN 单独验证

目标：验证最简单的 geometric compatibility 是否能提升 >95% outlier bucket 的 F-score。

修改点：get_graph_feature 中，用 hybrid distance（feature + geometric）替代纯 feature distance。

成功标准：>95% bucket F-score 提升 > 1%
失败标准：>95% bucket F-score 提升 < 0.3%
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from collections import defaultdict
import time

sys.path.insert(0, '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/core')

import MGCA as MGCA_module

# Patch CPU-only
def cpu_batch_symeig(X):
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='L')
        bv[batch_idx, :, :] = v
    return bv

MGCA_module.batch_symeig = cpu_batch_symeig

from MGCA import MGCANet, get_graph_feature, knn, weighted_8points, batch_episym
from data import CorrespondencesDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = SimpleNamespace(
    net_depth=12,
    net_channels=128,
    clusters=500,
    iter_num=1,
    use_fundamental=False,
    share=False,
    use_ratio=0,
    use_mutual=0,
    ratio_test_th=0.8,
    obj_num_kp=2000,
    obj_geod_type="episym",
    obj_geod_th=1e-4,
)

# ---------------------------------------------------------------------------
# GeoKNN: get_graph_feature with geometric compatibility
# ---------------------------------------------------------------------------

def compute_geometric_compatibility(xs):
    """
    xs: (B, 1, N, 4) correspondences (x1, y1, x2, y2)
    Returns: (B, N, N) compatibility matrix where compat[i,j] = exp(-|d1-d2|/sigma)
    """
    B, _, N, _ = xs.shape
    x1 = xs[:, 0, :, :2]  # (B, N, 2)
    x2 = xs[:, 0, :, 2:4] # (B, N, 2)

    # Pairwise distances in image 1
    # d1[i,j] = ||x1[i] - x1[j]||
    diff1 = x1.unsqueeze(2) - x1.unsqueeze(1)  # (B, N, N, 2)
    d1 = torch.norm(diff1, dim=-1)  # (B, N, N)

    # Pairwise distances in image 2
    diff2 = x2.unsqueeze(2) - x2.unsqueeze(1)  # (B, N, N, 2)
    d2 = torch.norm(diff2, dim=-1)  # (B, N, N)

    # Geometric compatibility: closer distance difference -> higher compatibility
    sigma = 10.0  # pixel, hyperparameter
    compat = torch.exp(-torch.abs(d1 - d2) / sigma)

    return compat


def get_graph_feature_geo(x, xs=None, k=20, idx=None, alpha=0.1):
    """
    Modified get_graph_feature with geometric compatibility.

    x: (B, C, N, 1) features
    xs: (B, 1, N, 4) coordinates (optional, if None fallback to original)
    k: number of neighbors
    alpha: weight for geometric term (0 = pure feature, 1 = pure geometric)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x_view = x.view(batch_size, -1, num_points)

    if idx is None:
        # Compute feature distance
        xx = torch.sum(x_view ** 2, dim=1, keepdim=True)
        inner = -2 * torch.bmm(x_view.transpose(2, 1), x_view)
        feat_dist = xx.transpose(2, 1) + inner + xx  # (B, N, N)

        if xs is not None and alpha > 0:
            # Compute geometric compatibility
            compat = compute_geometric_compatibility(xs)  # (B, N, N)
            # Hybrid distance: lower is better (like distance)
            # feat_dist is negative squared distance (from original knn), so smaller = closer
            # We want compat to boost closer geometric neighbors
            # Original: pairwise_distance = -xx - inner - xx.transpose -> more negative = closer
            # So: hybrid = feat_dist - alpha * compat (more negative = closer)
            hybrid_dist = feat_dist - alpha * compat
        else:
            hybrid_dist = feat_dist

        idx = hybrid_dist.topk(k=k, dim=-1, largest=False)[1]  # (B, N, k)
    else:
        idx_out = idx
        return get_graph_feature_original(x, k=k, idx=idx_out)

    # Rest is same as original get_graph_feature
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_flat = idx + idx_base
    idx_flat = idx_flat.view(-1)
    _, num_dims, _ = x_view.size()
    x_t = x_view.transpose(2, 1).contiguous()
    feature = x_t.view(batch_size * num_points, -1)[idx_flat, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_rep = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x_rep, x_rep - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


# Keep original for fallback
def get_graph_feature_original(x, k=20, idx=None):
    """Original get_graph_feature."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


# ---------------------------------------------------------------------------
# Monkey-patch model to use GeoKNN
# ---------------------------------------------------------------------------

def patch_model_for_geoknn(model, alpha=0.1):
    """
    Patch model's GNN modules to use GeoKNN instead of standard KNN.
    We need to pass xs (coordinates) to get_graph_feature.
    """
    # Store original forward methods
    original_gnn_forwards = []

    # Patch subnetwork_init.GNN
    init_gnn = model.subnetwork_init.GNN
    original_init_forward = init_gnn.forward
    original_gnn_forwards.append(('init', init_gnn, original_init_forward))

    def init_gnn_forward(features):
        # features: (B, C, N, 1)
        # We need xs, but GNN.forward doesn't receive it
        # So we store xs in a global or closure variable
        # This is a hack - better approach: patch sub_MGCANet.forward
        return original_init_forward(features)

    # Actually, better to patch at sub_MGCANet level
    return original_gnn_forwards


# Better approach: patch sub_MGCANet.forward to pass xs to GNN
def create_patched_sub_forward(original_forward, alpha):
    """Create a patched forward that passes xs to GeoKNN."""
    def patched_forward(data, xs, i, x_last=None, x_last2=None, init_out=None):
        # Temporarily replace get_graph_feature with GeoKNN version
        # that captures xs from closure
        xs_closure = xs

        def geo_get_graph_feature(x, k=20, idx=None):
            return get_graph_feature_geo(x, xs=xs_closure, k=k, idx=idx, alpha=alpha)

        # Monkey patch
        old_get_graph_feature = MGCA_module.get_graph_feature
        MGCA_module.get_graph_feature = geo_get_graph_feature

        try:
            result = original_forward(data, xs, i, x_last, x_last2, init_out)
        finally:
            MGCA_module.get_graph_feature = old_get_graph_feature

        return result
    return patched_forward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_precision_recall_f(logits, ys, threshold=1e-4):
    gt_inlier = (ys < threshold).astype(bool)
    pred_inlier = (logits > 0.5).astype(bool)
    tp = np.sum(pred_inlier & gt_inlier)
    fp = np.sum(pred_inlier & ~gt_inlier)
    fn = np.sum(~pred_inlier & gt_inlier)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f_score = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f_score


def evaluate_model(model, dataset, max_samples=200, mode='baseline'):
    """Evaluate model on dataset subset."""
    bucket_stats = defaultdict(lambda: {
        'count': 0,
        'f_scores': [],
        'precisions': [],
        'recalls': [],
        'outlier_ratios': [],
    })

    model.eval()
    with torch.no_grad():
        for idx in range(min(len(dataset), max_samples)):
            if idx % 50 == 0:
                print(f"  {mode}: {idx}/{min(len(dataset), max_samples)}")

            sample = dataset[idx]
            xs = sample['xs']  # (1, N, 4)
            ys = sample['ys'][:, 0]  # (N,)
            outlier_ratio = float(np.mean(ys >= config.obj_geod_th))

            # Determine bucket
            if outlier_ratio >= 0.95:
                bucket = '>95%'
            elif outlier_ratio >= 0.90:
                bucket = '90-95%'
            elif outlier_ratio >= 0.80:
                bucket = '80-90%'
            elif outlier_ratio >= 0.70:
                bucket = '70-80%'
            else:
                bucket = '<70%'

            batch = {
                'xs': torch.from_numpy(xs).unsqueeze(0).float(),
                'ys': torch.from_numpy(sample['ys']).unsqueeze(0).float(),
                'Rs': torch.from_numpy(sample['R']).unsqueeze(0).float(),
                'ts': torch.from_numpy(sample['t']).unsqueeze(0).float(),
            }
            for key in batch:
                batch[key] = batch[key].to(torch.device('cpu'))

            try:
                res_logits, res_e_hat = model(batch)
                logits = res_logits[-1][0].cpu().numpy()  # Final stage
            except Exception as e:
                print(f"  Error on sample {idx}: {e}")
                continue

            prec, rec, f = compute_precision_recall_f(
                torch.sigmoid(torch.from_numpy(logits)).numpy(), ys)

            bucket_stats[bucket]['count'] += 1
            bucket_stats[bucket]['f_scores'].append(f)
            bucket_stats[bucket]['precisions'].append(prec)
            bucket_stats[bucket]['recalls'].append(rec)
            bucket_stats[bucket]['outlier_ratios'].append(outlier_ratio)

    return bucket_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MVP Test 1: GeoKNN")
    print("=" * 70)

    # Load model
    model = MGCANet(config)
    model_path = '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/weights/yfcc100m/model_best1.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load dataset
    data_path = '/Users/huangjiachen/Desktop/PROJECTS/research/data/yfcc-sift-2000-val.hdf5'
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return
    dataset = CorrespondencesDataset(data_path, config)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Store original forwards
    original_init_forward = model.subnetwork_init.forward
    original_sub_forwards = [sub.forward for sub in model.subnetwork]

    # --- Baseline ---
    print("\n--- Running BASELINE ---")
    baseline_stats = evaluate_model(model, dataset, max_samples=200, mode='baseline')

    # --- GeoKNN with alpha=0.1 ---
    print("\n--- Running GeoKNN (alpha=0.1) ---")
    model.subnetwork_init.forward = create_patched_sub_forward(original_init_forward, alpha=0.1)
    for i, sub in enumerate(model.subnetwork):
        sub.forward = create_patched_sub_forward(original_sub_forwards[i], alpha=0.1)

    geoknn_stats = evaluate_model(model, dataset, max_samples=200, mode='geoknn')

    # Restore original
    model.subnetwork_init.forward = original_init_forward
    for i, sub in enumerate(model.subnetwork):
        sub.forward = original_sub_forwards[i]

    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Bucket':<10} {'N':>5} {'Base-F1':>9} {'Geo-F1':>9} {'Δ':>7} {'Base-P':>8} {'Base-R':>8}")
    print("-" * 70)

    target_bucket = '>95%'
    for bucket in ['<70%', '70-80%', '80-90%', '90-95%', '>95%']:
        b = baseline_stats[bucket]
        g = geoknn_stats[bucket]
        if b['count'] == 0:
            continue

        base_f = np.mean(b['f_scores'])
        geo_f = np.mean(g['f_scores'])
        delta = geo_f - base_f
        base_p = np.mean(b['precisions'])
        base_r = np.mean(b['recalls'])

        marker = ""
        if bucket == target_bucket:
            if delta > 0.01:
                marker = " ✅ PASS"
            elif delta < 0.003:
                marker = " ❌ FAIL"
            else:
                marker = " ⚠️  UNCLEAR"

        print(f"{bucket:<10} {b['count']:>5} {base_f:>9.3f} {geo_f:>9.3f} {delta:>+7.4f} {base_p:>8.3f} {base_r:>8.3f}{marker}")

    # Final verdict
    if target_bucket in baseline_stats and baseline_stats[target_bucket]['count'] > 0:
        delta = np.mean(geoknn_stats[target_bucket]['f_scores']) - np.mean(baseline_stats[target_bucket]['f_scores'])
        print(f"\n{'='*70}")
        if delta > 0.01:
            print("VERDICT: ✅ GeoKNN shows promise. Continue refining.")
        elif delta < 0.003:
            print("VERDICT: ❌ GeoKNN signal too weak. Abandon this direction.")
        else:
            print("VERDICT: ⚠️  Inconclusive. Try different alpha/sigma or larger sample.")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
