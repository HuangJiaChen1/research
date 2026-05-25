#!/usr/bin/env python3
"""
验证核心假设：预训练 MGCA-Net 在不同 outlier bucket 下的 e_hat 质量

问题：在高 outlier (>90%) 场景下，e_hat 是否足够准确以支持极线约束注意力？

输出：按 outlier ratio 分 bucket 的统计
- e_hat 与 GT E 的 epipolar error
- baseline F-score / precision / recall
- confidence 分布
- 迭代过程中 e_hat 的改善（init / stage0 / stage1 / CSMGC）
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import h5py
from collections import defaultdict
import time

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MGCANET', 'core'))

import MGCA as MGCA_module

# Patch CPU-only: remove .cuda() from batch_symeig
_original_batch_symeig = MGCA_module.batch_symeig

def cpu_batch_symeig(X):
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='L')
        bv[batch_idx, :, :] = v
    return bv

MGCA_module.batch_symeig = cpu_batch_symeig

from MGCA import MGCANet
from data import CorrespondencesDataset, collate_fn
from config import get_config
from utils import np_skew_symmetric

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DEVICE = torch.device('cpu')
BUCKETS = [
    (0.0, 0.60, "<60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 0.95, "90-95%"),
    (0.95, 1.00, ">95%"),
]
MAX_SAMPLES = 500  # 快速验证用，设为 -1 处理全部

# ------------------------------------------------------------------
# Geometric utilities
# ------------------------------------------------------------------

def compute_epipolar_distance(E, x1, x2):
    """
    Compute symmetric epipolar distance for correspondences.
    E: (3, 3) essential matrix
    x1, x2: (N, 2) normalized coordinates
    Returns: (N,) epipolar distances
    """
    N = x1.shape[0]
    # Homogeneous coordinates
    p1 = np.concatenate([x1, np.ones((N, 1))], axis=1)  # (N, 3)
    p2 = np.concatenate([x2, np.ones((N, 1))], axis=1)  # (N, 3)

    # Epipolar line in image 2: l2 = E @ p1.T
    l2 = E @ p1.T  # (3, N)
    # Epipolar line in image 1: l1 = E.T @ p2.T
    l1 = E.T @ p2.T  # (3, N)

    # Distance from p2 to line l2
    d2 = np.abs(np.sum(p2.T * l2, axis=0)) / (np.sqrt(l2[0]**2 + l2[1]**2) + 1e-10)
    # Distance from p1 to line l1
    d1 = np.abs(np.sum(p1.T * l1, axis=0)) / (np.sqrt(l1[0]**2 + l1[1]**2) + 1e-10)

    # Symmetric distance
    return (d1 + d2) / 2.0


def compute_e_hat_error_metrics(e_hat, e_gt, xs):
    """
    Compute error metrics between predicted and GT essential matrix.
    Returns dict with various error metrics.
    """
    E_pred = e_hat.reshape(3, 3)
    E_gt = e_gt.reshape(3, 3)

    # 1. Frobenius norm difference (normalized)
    E_pred_norm = E_pred / (np.linalg.norm(E_pred) + 1e-10)
    E_gt_norm = E_gt / (np.linalg.norm(E_gt) + 1e-10)
    frob_error = np.linalg.norm(E_pred_norm - E_gt_norm)

    # 2. Angular difference (using SVD)
    U_pred, S_pred, Vt_pred = np.linalg.svd(E_pred)
    U_gt, S_gt, Vt_gt = np.linalg.svd(E_gt)
    # Essential matrix should have singular values [1, 1, 0]
    # Enforce constraint
    S_pred_corrected = np.diag([1.0, 1.0, 0.0])
    S_gt_corrected = np.diag([1.0, 1.0, 0.0])
    E_pred_enforced = U_pred @ S_pred_corrected @ Vt_pred
    E_gt_enforced = U_gt @ S_gt_corrected @ Vt_gt

    # Compute angle between the two essential matrices (as vectors)
    vec_pred = E_pred_enforced.flatten()
    vec_gt = E_gt_enforced.flatten()
    cos_angle = np.clip(np.dot(vec_pred, vec_gt) / (np.linalg.norm(vec_pred) * np.linalg.norm(vec_gt) + 1e-10), -1.0, 1.0)
    angle_error = np.arccos(np.abs(cos_angle)) * 180.0 / np.pi

    # 3. Epipolar distance using predicted E vs GT E
    x1 = xs[0, :, :2]
    x2 = xs[0, :, 2:4]
    epi_dist_pred = compute_epipolar_distance(E_pred, x1, x2)
    epi_dist_gt = compute_epipolar_distance(E_gt, x1, x2)

    # 4. Pose recovery error (decompose E and compare R, t)
    # This requires more computation, skip for quick validation

    return {
        'frob_error': frob_error,
        'angle_error_deg': angle_error,
        'epipolar_dist_pred_mean': np.mean(epi_dist_pred),
        'epipolar_dist_pred_median': np.median(epi_dist_pred),
        'epipolar_dist_gt_mean': np.mean(epi_dist_gt),
        'epipolar_dist_gt_median': np.median(epi_dist_gt),
    }


def compute_precision_recall_f(logits, ys, threshold=1e-4):
    """
    Compute precision, recall, F-score.
    logits: (N,) predicted weights (after sigmoid)
    ys: (N,) ground truth epipolar distances
    threshold: threshold to determine inlier/outlier
    """
    gt_inlier = (ys < threshold).astype(bool)
    pred_inlier = (logits > 0.5).astype(bool)  # sigmoid threshold

    tp = np.sum(pred_inlier & gt_inlier)
    fp = np.sum(pred_inlier & ~gt_inlier)
    fn = np.sum(~pred_inlier & gt_inlier)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f_score = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f_score


def get_bucket(outlier_ratio):
    """Return bucket label for outlier ratio."""
    for low, high, label in BUCKETS:
        if low <= outlier_ratio < high:
            return label
    return BUCKETS[-1][2]


# ------------------------------------------------------------------
# Main validation
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("验证核心假设：e_hat 质量 vs Outlier Ratio")
    print("=" * 70)

    # Setup config
    config, _ = get_config()
    config.data_va = '../data_dump/yfcc-sift-2000-val.hdf5'
    config.use_fundamental = False
    config.gpu_id = '-1'  # CPU only

    # Load model
    model = MGCANet(config)
    model_path = os.path.join(os.path.dirname(__file__), 'MGCANET', 'weights', 'yfcc100m', 'model_best1.pth')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Available weights:")
        weights_dir = os.path.join(os.path.dirname(__file__), 'MGCANET', 'weights')
        if os.path.exists(weights_dir):
            for d in os.listdir(weights_dir):
                print(f"  - {d}")
        return

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded. Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'yfcc-sift-2000-val.hdf5')
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    dataset = CorrespondencesDataset(data_path, config)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Statistics containers
    bucket_stats = defaultdict(lambda: {
        'count': 0,
        'frob_errors': [],
        'angle_errors': [],
        'epi_pred_means': [],
        'epi_gt_means': [],
        'precisions': [],
        'recalls': [],
        'f_scores': [],
        'confidences': [],
        'outlier_ratios': [],
        'stage_errors': defaultdict(list),  # init, stage0, stage1, final
    })

    print("\nRunning validation...")
    start_time = time.time()

    with torch.no_grad():
        for idx in range(min(len(dataset), MAX_SAMPLES if MAX_SAMPLES > 0 else len(dataset))):
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {idx}/{min(len(dataset), MAX_SAMPLES if MAX_SAMPLES > 0 else len(dataset))} samples ({elapsed:.1f}s)")

            # Get data
            sample = dataset[idx]
            xs = sample['xs']  # (1, N, 4)
            ys = sample['ys'][:, 0]  # (N,)
            R = sample['R']
            t = sample['t']

            # Compute outlier ratio
            outlier_ratio = float(np.mean(ys >= config.obj_geod_th))
            bucket = get_bucket(outlier_ratio)

            # Ground truth essential matrix
            e_gt_unnorm = np.reshape(
                np.matmul(
                    np.reshape(np_skew_symmetric(t.astype('float64').reshape(1, 3)), (3, 3)),
                    np.reshape(R.astype('float64'), (3, 3))
                ), (3, 3)
            )
            e_gt = e_gt_unnorm / (np.linalg.norm(e_gt_unnorm) + 1e-10)

            # Prepare batch
            batch = {
                'xs': torch.from_numpy(xs).unsqueeze(0).float(),  # (1, 1, N, 4)
                'ys': torch.from_numpy(sample['ys']).unsqueeze(0).float(),  # (1, N, 1)
                'Rs': torch.from_numpy(R).unsqueeze(0).float(),
                'ts': torch.from_numpy(t).unsqueeze(0).float(),
            }

            # Move to device
            for key in batch:
                batch[key] = batch[key].to(DEVICE)

            # Forward pass
            try:
                res_logits, res_e_hat = model(batch)
            except Exception as e:
                print(f"  Error on sample {idx}: {e}")
                continue

            # Process each stage's output
            stages = ['init', 'stage0', 'stage1', 'final']
            for stage_idx, stage_name in enumerate(stages):
                if stage_idx < len(res_logits):
                    logits = res_logits[stage_idx]
                    e_hat = res_e_hat[stage_idx]

                    # Convert to numpy
                    logits_np = logits[0].cpu().numpy()  # (N,)
                    e_hat_np = e_hat[0].cpu().numpy()  # (9,)
                    ys_np = ys

                    # Compute metrics
                    try:
                        error_metrics = compute_e_hat_error_metrics(e_hat_np, e_gt, xs)
                        prec, rec, f = compute_precision_recall_f(torch.sigmoid(torch.from_numpy(logits_np)).numpy(), ys_np)

                        bucket_stats[bucket]['count'] += 1
                        bucket_stats[bucket]['frob_errors'].append(error_metrics['frob_error'])
                        bucket_stats[bucket]['angle_errors'].append(error_metrics['angle_error_deg'])
                        bucket_stats[bucket]['epi_pred_means'].append(error_metrics['epipolar_dist_pred_mean'])
                        bucket_stats[bucket]['epi_gt_means'].append(error_metrics['epipolar_dist_gt_mean'])
                        bucket_stats[bucket]['precisions'].append(prec)
                        bucket_stats[bucket]['recalls'].append(rec)
                        bucket_stats[bucket]['f_scores'].append(f)
                        bucket_stats[bucket]['confidences'].append(float(np.mean(torch.sigmoid(torch.from_numpy(logits_np)).numpy())))
                        bucket_stats[bucket]['outlier_ratios'].append(outlier_ratio)
                        bucket_stats[bucket]['stage_errors'][stage_name].append(error_metrics['angle_error_deg'])
                    except Exception as e:
                        print(f"  Metric error on sample {idx}, stage {stage_name}: {e}")
                        continue

    elapsed = time.time() - start_time
    print(f"\nValidation complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS BY OUTLIER BUCKET")
    print("=" * 70)

    # Header
    print(f"\n{'Bucket':<10} {'N':>5} {'Frob↓':>8} {'Angle↓':>8} {'EpiPred↓':>10} {'EpiGT↓':>8} {'Prec↑':>7} {'Rec↑':>7} {'F1↑':>7} {'Conf':>7}")
    print("-" * 85)

    for low, high, label in BUCKETS:
        stats = bucket_stats[label]
        if stats['count'] == 0:
            print(f"{label:<10} {'N/A':>5} {'-':>8} {'-':>8} {'-':>10} {'-':>8} {'-':>7} {'-':>7} {'-':>7} {'-':>7}")
            continue

        n = stats['count'] // 4  # Divide by 4 because we have 4 stages per sample
        frob_mean = np.mean(stats['frob_errors'])
        angle_mean = np.mean(stats['angle_errors'])
        epi_pred_mean = np.mean(stats['epi_pred_means'])
        epi_gt_mean = np.mean(stats['epi_gt_means'])
        prec_mean = np.mean(stats['precisions'])
        rec_mean = np.mean(stats['recalls'])
        f_mean = np.mean(stats['f_scores'])
        conf_mean = np.mean(stats['confidences'])
        outlier_mean = np.mean(stats['outlier_ratios'])

        print(f"{label:<10} {n:>5} {frob_mean:>8.3f} {angle_mean:>8.2f} {epi_pred_mean:>10.4f} {epi_gt_mean:>8.4f} {prec_mean:>7.3f} {rec_mean:>7.3f} {f_mean:>7.3f} {conf_mean:>7.3f}")

    # ------------------------------------------------------------------
    # Stage-wise error improvement
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE-WISE E_HAT ERROR IMPROVEMENT (Angle Error in degrees)")
    print("=" * 70)
    print(f"\n{'Bucket':<10} {'N':>5} {'Init↓':>8} {'Stage0↓':>8} {'Stage1↓':>8} {'Final↓':>8} {'Δ(Fin-Init)':>12}")
    print("-" * 75)

    for low, high, label in BUCKETS:
        stats = bucket_stats[label]
        if stats['count'] == 0:
            continue

        n = stats['count'] // 4
        init_err = np.mean(stats['stage_errors']['init']) if stats['stage_errors']['init'] else 0
        s0_err = np.mean(stats['stage_errors']['stage0']) if stats['stage_errors']['stage0'] else 0
        s1_err = np.mean(stats['stage_errors']['stage1']) if stats['stage_errors']['stage1'] else 0
        final_err = np.mean(stats['stage_errors']['final']) if stats['stage_errors']['final'] else 0
        delta = final_err - init_err

        print(f"{label:<10} {n:>5} {init_err:>8.2f} {s0_err:>8.2f} {s1_err:>8.2f} {final_err:>8.2f} {delta:>+12.2f}")

    # ------------------------------------------------------------------
    # Key conclusions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find >90% bucket
    high_outlier_stats = None
    for low, high, label in BUCKETS:
        if label in [">95%", "90-95%"]:
            if bucket_stats[label]['count'] > 0:
                high_outlier_stats = bucket_stats[label]
                break

    if high_outlier_stats:
        init_err = np.mean(high_outlier_stats['stage_errors']['init'])
        final_err = np.mean(high_outlier_stats['stage_errors']['final'])
        f_score = np.mean(high_outlier_stats['f_scores'])
        print(f"\n>90% Outlier bucket:")
        print(f"  - Init stage e_hat angle error: {init_err:.2f}°")
        print(f"  - Final stage e_hat angle error: {final_err:.2f}°")
        print(f"  - F-score: {f_score:.3f}")
        print(f"  - Improvement (Init→Final): {init_err - final_err:.2f}°")

        if final_err < 15.0:
            print(f"\n  ✅ CONCLUSION: e_hat is RELATIVELY ACCURATE in >90% outlier.")
            print(f"     → 极线约束注意力（方案A）在 Final stage 可用")
            print(f"     → 但 Init stage 误差 {init_err:.1f}°，方案B（不依赖E）更稳健")
        elif final_err < 30.0:
            print(f"\n  ⚠️  CONCLUSION: e_hat is MODERATELY ACCURATE in >90% outlier.")
            print(f"     → 极线约束可能有噪声，需要 robust E 估计（top-K confidence）")
        else:
            print(f"\n  ❌ CONCLUSION: e_hat is VERY INACCURATE in >90% outlier.")
            print(f"     → 方案A（依赖E）风险极高")
            print(f"     → 优先做方案B（局部几何兼容性，不依赖E）")
    else:
        print("\n  No >90% outlier samples found in validation set.")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
基于以上结果，下一步行动：

1. 如果 Final stage e_hat error < 15° 且样本数 > 20:
   → 方案A可行，但建议用 top-K confidence 鲁棒估计 E

2. 如果 Final stage e_hat error > 30°:
   → 放弃方案A，只做方案B（geometric_compatibility KNN）

3. 如果 Init stage error >> Final stage error:
   → 说明迭代结构有效，极线约束应放在后期 stage 使用
   → 不要放在 Init stage（信号太弱）
""")


if __name__ == '__main__':
    main()
