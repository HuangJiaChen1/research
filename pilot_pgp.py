"""
Pilot: Progressive Graph Purification (PGP)

Test the core hypothesis: Adaptive graph rewiring based on stage confidence
improves performance, especially at high outlier ratios.

Three configurations tested:
1. Baseline: Original MGCA-Net with static KNN (k=6)
2. Hard-Prune: Low-confidence nodes excluded from neighbor search
3. Soft-Reweight: Feature distance modulated by confidence product

Metrics per outlier-ratio bucket:
- F-score (precision-recall)
- Mean Precision @ different thresholds
- Inlier-neighbor-purity (avg confidence of a node's k neighbors)
"""

import os
import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/core')

from MGCA import MGCANet, get_graph_feature, knn, batch_episym
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
    batch_size=1,
    knn_num=6,
)

# ---------------------------------------------------------------------------
# Patch CPU-only functions
# ---------------------------------------------------------------------------
import MGCA

def cpu_batch_symeig(X):
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='L')
        bv[batch_idx, :, :] = v
    return bv

MGCA.batch_symeig = cpu_batch_symeig

# ---------------------------------------------------------------------------
# Adaptive Graph Builders
# ---------------------------------------------------------------------------

class AdaptiveGraphState:
    """Global state to pass logits between forward passes."""
    def __init__(self):
        self.logits_init = None
        self.logits_0 = None
        self.logits_1 = None
        self.xs = None
        self.mode = 'baseline'  # 'baseline', 'hard', 'soft'
        self.stage_counter = 0

state = AdaptiveGraphState()


def adaptive_get_graph_feature(x, k=20, idx=None):
    """
    Drop-in replacement for get_graph_feature that supports adaptive rewiring.
    """
    if state.mode == 'baseline' or idx is not None:
        return get_graph_feature(x, k=k, idx=idx)

    batch_size = x.size(0)
    num_points = x.size(2)
    x_view = x.view(batch_size, -1, num_points)

    # Determine which logits to use based on call order / k value
    # k=3: CSMGC calls (skip for simplicity in pilot)
    # k=6 or 9: main GNN calls
    if k not in [6, 9]:
        return get_graph_feature(x, k=k, idx=idx)

    # Map stage_counter to logits
    # The forward order is: subnetwork_init GNN -> subnetwork[0] GNN -> subnetwork[1] GNN
    # CSMGC also uses get_graph_feature with k=3
    # We track by k value and call sequence
    if state.stage_counter == 0:
        logits = state.logits_init
    elif state.stage_counter == 1:
        logits = state.logits_0
    elif state.stage_counter == 2:
        logits = state.logits_1
    else:
        logits = None

    state.stage_counter += 1

    if logits is None:
        return get_graph_feature(x, k=k, idx=idx)

    confidences = torch.sigmoid(logits)  # (B, N)

    if state.mode == 'hard':
        # Hard prune: set low-confidence nodes' features to far away
        modified = x_view.transpose(1, 2).clone()  # (B, N, C)
        threshold = 0.3
        mask = confidences > threshold  # (B, N)
        far_val = modified.max() * 100
        for b in range(batch_size):
            modified[b, ~mask[b]] = far_val
        modified = modified.transpose(1, 2)  # (B, C, N)
        # Compute KNN on modified features
        dist = torch.cdist(modified.transpose(1, 2), modified.transpose(1, 2))
        _, idx_out = torch.topk(dist, k=k, largest=False, dim=-1)
        return get_graph_feature(x, k=k, idx=idx_out)

    elif state.mode == 'soft':
        # Soft reweight: modulate distance by confidence, NOT features
        # Keep features unchanged for GNN, but make low-confidence nodes
        # appear "farther away" in the distance metric
        dist = torch.cdist(x_view.transpose(1, 2), x_view.transpose(1, 2))  # (B, N, N)
        # Joint confidence: high-conf * high-conf = strong connection
        conf_gate = confidences.unsqueeze(2) * confidences.unsqueeze(1)  # (B, N, N)
        # Effective distance: original distance divided by confidence product
        # This makes high-confidence pairs closer without changing features
        effective_dist = dist / (conf_gate + 0.05)
        _, idx_out = torch.topk(effective_dist, k=k, largest=False, dim=-1)
        return get_graph_feature(x, k=k, idx=idx_out)

    return get_graph_feature(x, k=k, idx=idx)


# Monkey-patch
MGCA.get_graph_feature = adaptive_get_graph_feature


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_fscore(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_precision_recall(weights, labels, threshold=0.5):
    """
    weights: (N,) predicted weights (after sigmoid)
    labels: (N,) binary labels (0=outlier, 1=inlier)
    """
    pred = (weights > threshold).astype(float)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return precision, recall


def compute_neighbor_purity(nn_idx, confidences):
    """
    nn_idx: (N, k) neighbor indices
    confidences: (N,) confidence scores
    Returns: average confidence of neighbors for each node
    """
    N, k = nn_idx.shape
    neighbor_conf = confidences[nn_idx]  # (N, k)
    purity = neighbor_conf.mean(axis=1)
    return purity.mean()


# ---------------------------------------------------------------------------
# Run model in a mode
# ---------------------------------------------------------------------------

def run_model(model, xs, R, t, ys, mode='baseline'):
    """
    Run model and compute metrics.
    Returns dict with metrics.
    """
    state.mode = mode
    state.xs = xs
    state.stage_counter = 0

    data = {'xs': xs, 'Rs': R, 'ts': t}

    with torch.no_grad():
        res_weights, res_e_hat = model(data)

    # Final weights (last element in res_weights)
    final_logits = res_weights[-1]  # (B, N)
    final_weights = torch.sigmoid(final_logits).numpy().flatten()

    # Binary labels from epipolar distance
    labels = (ys < 1e-4).astype(float).flatten()

    # Outlier ratio
    outlier_ratio = 1.0 - labels.mean()

    # Metrics at multiple thresholds
    metrics = {
        'outlier_ratio': outlier_ratio,
        'num_inliers': int(labels.sum()),
        'num_outliers': int((1 - labels).sum()),
    }

    for thresh in [0.3, 0.5, 0.7]:
        p, r = compute_precision_recall(final_weights, labels, threshold=thresh)
        f = compute_fscore(p, r)
        metrics[f'precision@{thresh}'] = p
        metrics[f'recall@{thresh}'] = r
        metrics[f'fscore@{thresh}'] = f

    # Store intermediate logits for adaptive modes
    # Note: for baseline run, we save logits for later use
    if mode == 'baseline':
        state.logits_init = res_weights[0] if len(res_weights) > 0 else None
        # res_weights contains [init, stage0, stage1, final]
        # Actually res_weights is appended in forward:
        #   res_weights.append(logits)  -- subnetwork_init
        #   res_weights.append(logits)  -- subnetwork[0]
        #   res_weights.append(logits)  -- subnetwork[1]
        #   res_weights.append(logits)  -- final
        if len(res_weights) >= 4:
            state.logits_init = res_weights[0]
            state.logits_0 = res_weights[1]
            state.logits_1 = res_weights[2]
        else:
            state.logits_init = None
            state.logits_0 = None
            state.logits_1 = None

    return metrics


# ---------------------------------------------------------------------------
# Main pilot
# ---------------------------------------------------------------------------

def run_pilot(num_samples=100):
    print("=" * 70)
    print("PILOT: Progressive Graph Purification")
    print("=" * 70)

    # Load model
    print("\n[1] Loading model...")
    model = MGCANet(config)
    weight_path = '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/weights/yfcc100m/model_best1.pth'
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"    Loaded pretrained weights")
    else:
        print(f"    WARNING: No pretrained weights found")
        return
    model.eval()

    # Load data
    print("\n[2] Loading data...")
    data_path = '/Users/huangjiachen/Desktop/PROJECTS/research/data/yfcc-sift-2000-val.hdf5'
    if not os.path.exists(data_path):
        print(f"    Dataset not found: {data_path}")
        return

    # Collect results per outlier-ratio bucket
    buckets = {
        '60-70': [],
        '70-80': [],
        '80-90': [],
        '90-95': [],
        '95-100': [],
    }

    with h5py.File(data_path, 'r') as f:
        sample_keys = list(f['xs'].keys())[:num_samples]
        print(f"    Testing on {len(sample_keys)} samples")

        for key in tqdm(sample_keys, desc="Processing"):
            xs = torch.from_numpy(f[f'xs/{key}'][:]).float().unsqueeze(0)
            R = torch.from_numpy(f[f'Rs/{key}'][:]).float().unsqueeze(0)
            t = torch.from_numpy(f[f'ts/{key}'][:]).float().unsqueeze(0)
            ys = f[f'ys/{key}'][:]  # (N, 1)

            # Baseline
            baseline_metrics = run_model(model, xs, R, t, ys, mode='baseline')

            # Hard-prune (using baseline's logits)
            hard_metrics = run_model(model, xs, R, t, ys, mode='hard')

            # Soft-reweight (using baseline's logits)
            soft_metrics = run_model(model, xs, R, t, ys, mode='soft')

            # Bucket by outlier ratio
            or_ratio = baseline_metrics['outlier_ratio'] * 100
            if or_ratio < 70:
                bucket = '60-70'
            elif or_ratio < 80:
                bucket = '70-80'
            elif or_ratio < 90:
                bucket = '80-90'
            elif or_ratio < 95:
                bucket = '90-95'
            else:
                bucket = '95-100'

            buckets[bucket].append({
                'baseline': baseline_metrics,
                'hard': hard_metrics,
                'soft': soft_metrics,
            })

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n    {'Bucket':<10} {'N':>5} {'Base-F@0.5':>12} {'Hard-F@0.5':>12} {'Soft-F@0.5':>12} {'ΔSoft':>8}")
    print("    " + "-" * 65)

    for bucket_name in ['60-70', '70-80', '80-90', '90-95', '95-100']:
        samples = buckets[bucket_name]
        if len(samples) == 0:
            continue

        base_f = np.mean([s['baseline']['fscore@0.5'] for s in samples])
        hard_f = np.mean([s['hard']['fscore@0.5'] for s in samples])
        soft_f = np.mean([s['soft']['fscore@0.5'] for s in samples])
        delta = soft_f - base_f

        print(f"    {bucket_name:<10} {len(samples):>5} {base_f:>12.4f} {hard_f:>12.4f} {soft_f:>12.4f} {delta:>+8.4f}")

    # Statistical significance test for high-outlier bucket
    high_samples = buckets['90-95'] + buckets['95-100']
    if len(high_samples) > 5:
        base_scores = [s['baseline']['fscore@0.5'] for s in high_samples]
        soft_scores = [s['soft']['fscore@0.5'] for s in high_samples]
        from scipy.stats import ttest_rel
        tstat, pval = ttest_rel(soft_scores, base_scores)
        print(f"\n    High-outlier (>90%) paired t-test: t={tstat:.3f}, p={pval:.4f}")
        if pval < 0.05:
            print(f"    ✅ Significant improvement!")
        else:
            print(f"    ❌ Not significant (may need more samples)")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    If ΔSoft increases monotonically with outlier ratio:
        → Validates the "Paradox of High Outlier Ratios"
        → Strong evidence for Q1 paper

    If ΔSoft is uniform across buckets:
        → Rewiring helps, but not specifically for high outliers
        → Still publishable, but story is weaker

    If ΔSoft is negative for any bucket:
        → False-negative death spiral is real
        → Need residual-gate or exploration mechanism
    """)


if __name__ == '__main__':
    run_pilot(num_samples=200)
