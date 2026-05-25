#!/usr/bin/env python3
"""
MVP Test 2: EpiAttn 核心假设验证

不修改模型内部，用最简单的后处理验证极线约束是否有价值：
  modulated_logits = logits - d_epi * scale

如果后处理有效 → 极线约束有价值 → 继续做真正的 EpiAttn
如果后处理无效 → 即使 E 准，极线约束也没用 → 放弃

成功标准：>95% bucket F-score 提升 > 1%
失败标准：>95% bucket F-score 提升 < 0.3%
"""

import os
import sys
import numpy as np
import torch
from types import SimpleNamespace
from collections import defaultdict

sys.path.insert(0, '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/core')

import MGCA as MGCA_module

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

config = SimpleNamespace(
    net_depth=12, net_channels=128, clusters=500, iter_num=1,
    use_fundamental=False, share=False, use_ratio=0, use_mutual=0,
    ratio_test_th=0.8, obj_num_kp=2000, obj_geod_type="episym", obj_geod_th=1e-4,
)

# ---------------------------------------------------------------------------
# Epipolar utilities
# ---------------------------------------------------------------------------

def compute_epipolar_distance(E, xs):
    """
    E: (B, 9) essential matrix
    xs: (B, 1, N, 4) correspondences
    Returns: (B, N) epipolar distances
    """
    B = E.shape[0]
    N = xs.shape[2]
    E_mat = E.view(B, 3, 3)

    x1 = xs[:, 0, :, :2]
    x2 = xs[:, 0, :, 2:4]

    p1 = torch.cat([x1, torch.ones(B, N, 1)], dim=2)
    p2 = torch.cat([x2, torch.ones(B, N, 1)], dim=2)

    l2 = torch.bmm(E_mat, p1.transpose(1, 2))
    d = torch.abs(torch.sum(p2.transpose(1, 2) * l2, dim=1))
    norm = torch.sqrt(l2[:, 0, :]**2 + l2[:, 1, :]**2) + 1e-10
    return d / norm


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


def evaluate_with_epipolar_modulation(model, dataset, max_samples=200, scale=1.0):
    """
    Evaluate with epipolar distance modulation of logits.
    modulated_logits = sigmoid^{-1}(confidence) - d_epi * scale
    """
    bucket_stats = defaultdict(lambda: {'count': 0, 'f_scores': [], 'precisions': [], 'recalls': []})

    model.eval()
    with torch.no_grad():
        for idx in range(min(len(dataset), max_samples)):
            if idx % 50 == 0:
                print(f"  scale={scale}: {idx}/{min(len(dataset), max_samples)}")

            sample = dataset[idx]
            xs = sample['xs']
            ys = sample['ys'][:, 0]
            outlier_ratio = float(np.mean(ys >= config.obj_geod_th))

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
                batch[key] = batch[key].cpu()

            try:
                res_logits, res_e_hat = model(batch)
            except Exception as e:
                print(f"  Error sample {idx}: {e}")
                continue

            # Use final stage logits and e_hat
            logits = res_logits[-1][0].cpu().numpy()
            e_hat = res_e_hat[-1][0:1].cpu()  # (1, 9)
            xs_t = batch['xs'].cpu()

            # Compute epipolar distance
            d_epi = compute_epipolar_distance(e_hat, xs_t)[0].cpu().numpy()

            # Convert logits to confidence and back to log-odds for modulation
            confidence = torch.sigmoid(torch.from_numpy(logits)).numpy()
            # Avoid log(0)
            confidence = np.clip(confidence, 1e-5, 1 - 1e-5)
            log_odds = np.log(confidence / (1 - confidence))

            # Modulate: penalize high epipolar distance
            modulated_log_odds = log_odds - d_epi * scale
            modulated_confidence = 1.0 / (1.0 + np.exp(-modulated_log_odds))
            modulated_logits = np.log(modulated_confidence / (1 - modulated_confidence + 1e-10))

            # Evaluate both
            prec_base, rec_base, f_base = compute_precision_recall_f(logits, ys)
            prec_mod, rec_mod, f_mod = compute_precision_recall_f(modulated_logits, ys)

            # Store delta
            bucket_stats[bucket]['count'] += 1
            bucket_stats[bucket]['f_scores'].append(f_mod - f_base)
            bucket_stats[bucket]['precisions'].append(prec_mod - prec_base)
            bucket_stats[bucket]['recalls'].append(rec_mod - rec_base)

    return bucket_stats


def main():
    print("=" * 70)
    print("MVP Test 2: EpiAttn (Post-hoc modulation)")
    print("=" * 70)

    model = MGCANet(config)
    model_path = '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/weights/yfcc100m/model_best1.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Model loaded")

    data_path = '/Users/huangjiachen/Desktop/PROJECTS/research/data/yfcc-sift-2000-val.hdf5'
    dataset = CorrespondencesDataset(data_path, config)
    print(f"Dataset: {len(dataset)} samples")

    # Try multiple scales
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    all_results = {}

    for scale in scales:
        print(f"\n--- Testing scale={scale} ---")
        stats = evaluate_with_epipolar_modulation(model, dataset, max_samples=200, scale=scale)
        all_results[scale] = stats

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS: ΔF-score (Modulated - Baseline) by scale and bucket")
    print("=" * 70)
    print(f"\n{'Scale':>8} {'<70%':>10} {'70-80%':>10} {'80-90%':>10} {'90-95%':>10} {'>95%':>10} {'Verdict':>10}")
    print("-" * 90)

    target_bucket = '>95%'
    for scale in scales:
        stats = all_results[scale]
        vals = []
        for bucket in ['<70%', '70-80%', '80-90%', '90-95%', '>95%']:
            if stats[bucket]['count'] > 0:
                vals.append(np.mean(stats[bucket]['f_scores']))
            else:
                vals.append(0)

        delta_target = vals[4]
        if delta_target > 0.01:
            verdict = "✅ PASS"
        elif delta_target < -0.005:
            verdict = "❌ HARM"
        elif delta_target < 0.003:
            verdict = "❌ FAIL"
        else:
            verdict = "⚠️  UNCLEAR"

        print(f"{scale:>8.1f} {vals[0]:>+10.4f} {vals[1]:>+10.4f} {vals[2]:>+10.4f} {vals[3]:>+10.4f} {vals[4]:>+10.4f} {verdict:>10}")

    # Best scale analysis
    print("\n" + "=" * 70)
    print("BEST SCALE ANALYSIS")
    print("=" * 70)

    best_scale = max(scales, key=lambda s: np.mean(all_results[s][target_bucket]['f_scores']))
    best_stats = all_results[best_scale]
    if best_stats[target_bucket]['count'] > 0:
        delta = np.mean(best_stats[target_bucket]['f_scores'])
        print(f"Best scale: {best_scale}")
        print(f"Delta F-score @ >95%: {delta:+.4f}")

        if delta > 0.01:
            print("\nVERDICT: ✅ Epipolar modulation shows clear signal.")
            print("         Proceed to implement true EpiAttn inside attention.")
        elif delta > 0.003:
            print("\nVERDICT: ⚠️  Weak but positive signal.")
            print("         Try more sophisticated modulation or larger sample.")
        else:
            print("\nVERDICT: ❌ No meaningful signal from epipolar constraint.")
            print("         Even with accurate E, epipolar distance doesn't help.")
            print("         This suggests the bottleneck is elsewhere.")


if __name__ == '__main__':
    main()
