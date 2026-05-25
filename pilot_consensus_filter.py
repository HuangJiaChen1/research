#!/usr/bin/env python3
"""
Pilot: Consensus Filtering for MGCA-Net

Zero-shot validation: Use pretrained model, modify only post-processing.

Hypothesis: If a correspondence is truly an inlier, it should have consistent
epipolar distances across all three stages. If it's an outlier, at least one
stage will show large epipolar distance.

Test on >90% outlier bucket (where precision collapses to 0.378).
"""

import os
import sys
import numpy as np
import torch
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

from MGCA import MGCANet
from data import CorrespondencesDataset
from utils import np_skew_symmetric

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
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
    obj_geod_th=1e-4,
    knn_num=6,
)

MAX_SAMPLES = 500
DEVICE = torch.device('cpu')

BUCKETS = [
    (0.0, 0.70, "<70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 0.95, "90-95%"),
    (0.95, 1.00, ">95%"),
]

# ------------------------------------------------------------------
# Geometric utilities
# ------------------------------------------------------------------

def compute_epipolar_distance(E, xs):
    """
    E: (B, 9) or (B, 3, 3)
    xs: (B, 1, N, 4) or (B, N, 4) — normalized coords (x1,y1,x2,y2)
    Returns: (B, N) epipolar distances
    """
    if E.dim() == 2:
        E = E.view(E.shape[0], 3, 3)

    if xs.dim() == 4:
        xs = xs[:, 0, :, :]  # (B, N, 4)

    B, N = xs.shape[0], xs.shape[1]
    x1 = xs[:, :, :2]  # (B, N, 2)
    x2 = xs[:, :, 2:4] # (B, N, 2)

    p1 = torch.cat([x1, torch.ones(B, N, 1, device=xs.device, dtype=xs.dtype)], dim=2)  # (B, N, 3)
    p2 = torch.cat([x2, torch.ones(B, N, 1, device=xs.device, dtype=xs.dtype)], dim=2)  # (B, N, 3)

    l2 = torch.bmm(E, p1.transpose(1, 2))  # (B, 3, N)
    d = torch.abs(torch.sum(p2.transpose(1, 2) * l2, dim=1))  # (B, N)
    norm = torch.sqrt(l2[:, 0, :]**2 + l2[:, 1, :]**2) + 1e-10
    return d / norm


def compute_precision_recall_f(weights, ys, threshold=1e-4, weight_threshold=0.5):
    """
    weights: (N,) predicted weights (after sigmoid)
    ys: (N,) ground truth epipolar distances
    """
    gt_inlier = (ys < threshold)
    pred_inlier = (weights > weight_threshold)

    tp = np.sum(pred_inlier & gt_inlier)
    fp = np.sum(pred_inlier & ~gt_inlier)
    fn = np.sum(~pred_inlier & gt_inlier)
    tn = np.sum(~pred_inlier & ~gt_inlier)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f_score = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f_score, tp, fp, fn, tn


def get_bucket(outlier_ratio):
    for low, high, label in BUCKETS:
        if low <= outlier_ratio < high:
            return label
    return BUCKETS[-1][2]


# ------------------------------------------------------------------
# Consensus metrics
# ------------------------------------------------------------------

def consensus_variance(dists):
    """
    dists: (S, B, N) epipolar distances from S stages
    Returns: (B, N) consensus score (high = consistent)
    """
    var = torch.var(dists, dim=0)  # (B, N)
    return 1.0 / (1.0 + var)


def consensus_min(dists):
    """
    dists: (S, B, N)
    Returns: (B, N) min distance (optimistic: if any stage says inlier, trust it)
    """
    return torch.min(dists, dim=0)[0]


def consensus_product(dists, semantic_conf):
    """
    dists: (S, B, N)
    semantic_conf: (B, N) model confidence
    Returns: (B, N) combined score
    """
    mean_dist = torch.mean(dists, dim=0)  # (B, N)
    geo_score = torch.exp(-mean_dist)  # (B, N)
    return semantic_conf * geo_score


def consensus_pairwise_agreement(dists):
    """
    dists: (S, B, N)
    Returns: (B, N) agreement score (how many stage pairs agree on inlier/outlier)
    """
    S = dists.shape[0]
    inlier_mask = (dists < 0.01).float()  # (S, B, N)

    # Count agreements: pairwise AND
    agreements = []
    for i in range(S):
        for j in range(i+1, S):
            agree = (inlier_mask[i] * inlier_mask[j])  # (B, N)
            agreements.append(agree)

    if len(agreements) == 0:
        return inlier_mask[0]

    return torch.stack(agreements).mean(dim=0)  # (B, N)


# ------------------------------------------------------------------
# Main validation
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Pilot: Consensus Filtering (Zero-Shot)")
    print("=" * 70)

    # Load model
    model_path = '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/weights/yfcc100m/model_best1.pth'
    print(f"\nLoading model from {model_path}")

    model = MGCANet(config)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded. Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load dataset
    data_path = '/Users/huangjiachen/Desktop/PROJECTS/research/data/yfcc-sift-2000-val.hdf5'
    dataset = CorrespondencesDataset(data_path, config)
    print(f"Dataset: {len(dataset)} samples")

    # Statistics containers
    methods = ['baseline', 'variance', 'min_dist', 'product', 'pairwise']

    bucket_stats = defaultdict(lambda: {
        m: {
            'precisions': [],
            'recalls': [],
            'f_scores': [],
            'tps': [], 'fps': [], 'fns': [], 'tns': [],
        } for m in methods
    })

    print("\nRunning validation...")
    start_time = time.time()

    with torch.no_grad():
        for idx in range(min(len(dataset), MAX_SAMPLES)):
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  [{idx}/{min(len(dataset), MAX_SAMPLES)}] {elapsed:.1f}s")

            # Get data
            sample = dataset[idx]
            xs = sample['xs']  # (1, N, 4)
            ys = sample['ys'][:, 0]  # (N,)
            R = sample['R']
            t = sample['t']

            # Compute outlier ratio
            outlier_ratio = float(np.mean(ys >= config.obj_geod_th))
            bucket = get_bucket(outlier_ratio)

            # Prepare batch
            batch = {
                'xs': torch.from_numpy(xs).unsqueeze(0).float(),  # (1, 1, N, 4)
                'ys': torch.from_numpy(sample['ys']).unsqueeze(0).float(),
                'Rs': torch.from_numpy(R).unsqueeze(0).float(),
                'ts': torch.from_numpy(t).unsqueeze(0).float(),
            }
            for key in batch:
                batch[key] = batch[key].to(DEVICE)

            # Forward pass
            try:
                res_logits, res_e_hat = model(batch)
            except Exception as e:
                print(f"  Error on sample {idx}: {e}")
                continue

            # Collect stage outputs
            # MGCA-Net outputs: [init, stage0, stage1, final]
            # We use first 3 stages for consensus, final for baseline
            num_stages = len(res_e_hat)

            if num_stages < 3:
                print(f"  Warning: only {num_stages} stages, skipping")
                continue

            # Get e_hats for all stages
            e_hats = []
            logits_list = []
            for s in range(num_stages):
                e_hats.append(res_e_hat[s][0])  # (9,)
                logits_list.append(res_logits[s][0])  # (N,)

            # Baseline: final stage only
            baseline_logits = logits_list[-1]
            baseline_weights = torch.sigmoid(baseline_logits).cpu().numpy()

            # Compute epipolar distances for all stages
            xs_tensor = batch['xs']  # (1, 1, N, 4)
            dists = []
            for s in range(num_stages):
                e_hat = e_hats[s].unsqueeze(0)  # (1, 9)
                d = compute_epipolar_distance(e_hat, xs_tensor)  # (1, N)
                dists.append(d[0])  # (N,)
            dists = torch.stack(dists)  # (S, N)

            # Compute consensus scores
            semantic_conf = torch.sigmoid(baseline_logits)  # (N,)

            scores = {
                'baseline': torch.ones_like(semantic_conf),  # no filtering
                'variance': consensus_variance(dists),  # (N,)
                'min_dist': torch.exp(-consensus_min(dists)),  # (N,)
                'product': consensus_product(dists, semantic_conf),  # (N,)
                'pairwise': consensus_pairwise_agreement(dists),  # (N,)
            }

            # Apply consensus and compute metrics
            ys_np = ys
            for method_name in methods:
                score = scores[method_name].cpu().numpy()

                # Refine weights: baseline * consensus_score
                if method_name == 'baseline':
                    refined_weights = baseline_weights
                else:
                    # Multiply baseline confidence with consensus
                    refined_weights = baseline_weights * score

                prec, rec, f, tp, fp, fn, tn = compute_precision_recall_f(
                    refined_weights, ys_np, threshold=config.obj_geod_th
                )

                bucket_stats[bucket][method_name]['precisions'].append(prec)
                bucket_stats[bucket][method_name]['recalls'].append(rec)
                bucket_stats[bucket][method_name]['f_scores'].append(f)
                bucket_stats[bucket][method_name]['tps'].append(tp)
                bucket_stats[bucket][method_name]['fps'].append(fp)
                bucket_stats[bucket][method_name]['fns'].append(fn)
                bucket_stats[bucket][method_name]['tns'].append(tn)

    elapsed = time.time() - start_time
    print(f"\nValidation complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RESULTS BY OUTLIER BUCKET")
    print("=" * 100)

    for bucket_label in [b[2] for b in BUCKETS]:
        stats = bucket_stats[bucket_label]
        n = len(stats['baseline']['f_scores'])
        if n == 0:
            continue

        print(f"\n{'='*100}")
        print(f"Bucket: {bucket_label} (N={n})")
        print(f"{'='*100}")
        print(f"{'Method':<15} {'Prec↑':>8} {'Rec↑':>8} {'F1↑':>8} {'ΔF1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 80)

        baseline_f = np.mean(stats['baseline']['f_scores'])

        for method_name in methods:
            prec = np.mean(stats[method_name]['precisions'])
            rec = np.mean(stats[method_name]['recalls'])
            f = np.mean(stats[method_name]['f_scores'])
            tp = np.mean(stats[method_name]['tps'])
            fp = np.mean(stats[method_name]['fps'])
            fn = np.mean(stats[method_name]['fns'])

            delta_f = f - baseline_f if method_name != 'baseline' else 0.0
            marker = " ✅" if delta_f > 0.01 else " ⚠️" if delta_f > 0 else " ❌"

            print(f"{method_name:<15} {prec:>8.3f} {rec:>8.3f} {f:>8.3f} {delta_f:>+8.3f} {tp:>6.1f} {fp:>6.1f} {fn:>6.1f}{marker}")

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    high_bucket = ">95%"
    if high_bucket in bucket_stats and len(bucket_stats[high_bucket]['baseline']['f_scores']) > 0:
        stats = bucket_stats[high_bucket]
        n = len(stats['baseline']['f_scores'])
        baseline_f = np.mean(stats['baseline']['f_scores'])

        print(f"\n>95% Outlier Bucket (N={n}):")
        print(f"  Baseline F-score: {baseline_f:.3f}")

        best_method = 'baseline'
        best_f = baseline_f
        for method_name in methods:
            if method_name == 'baseline':
                continue
            f = np.mean(stats[method_name]['f_scores'])
            if f > best_f:
                best_f = f
                best_method = method_name

        delta = best_f - baseline_f
        print(f"  Best method: {best_method} (F={best_f:.3f}, Δ={delta:+.3f})")

        if delta > 0.03:
            print(f"\n  ✅ SIGNIFICANT IMPROVEMENT: Consensus filtering works!")
            print(f"     → {best_method} improved F-score by {delta*100:.1f} percentage points")
            print(f"     → Next step: integrate into CSMGC and train end-to-end")
        elif delta > 0.01:
            print(f"\n  ⚠️  MARGINAL IMPROVEMENT: {delta*100:.1f}pp")
            print(f"     → Try different consensus metrics or hyperparameters")
        else:
            print(f"\n  ❌ NO IMPROVEMENT: {delta:+.3f}")
            print(f"     → Three-stage e_hat consistency is not a strong signal")
            print(f"     → Try alternative approaches (e.g., cross-sample consensus)")

    # Show precision improvement specifically
    print("\n" + "=" * 100)
    print("PRECISION ANALYSIS (The key metric for >95% bucket)")
    print("=" * 100)

    for bucket_label in ["90-95%", ">95%"]:
        if bucket_label not in bucket_stats:
            continue
        stats = bucket_stats[bucket_label]
        n = len(stats['baseline']['f_scores'])
        if n == 0:
            continue

        print(f"\n{bucket_label} (N={n}):")
        baseline_p = np.mean(stats['baseline']['precisions'])
        print(f"  Baseline Precision: {baseline_p:.3f}")

        for method_name in methods:
            if method_name == 'baseline':
                continue
            p = np.mean(stats[method_name]['precisions'])
            delta_p = p - baseline_p
            print(f"  {method_name:<15}: {p:.3f} (Δ={delta_p:+.3f})")


if __name__ == '__main__':
    main()
