#!/usr/bin/env python3
"""
Analyze E-matrix estimation quality across stages and outlier buckets.

Key question: MGCA-Net uses top-50% correspondences (by logits) to estimate E-matrix.
Does this effectively filter outliers and produce reliable E-matrices even when
overall outlier ratio >95%?

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_e_matrix_quality.py \
        --pretrained ../weights/yfcc100m/model_best1.pth \
        --data_va ../data_dump/yfcc-sift-2000-val.hdf5 \
        --max_samples 1000

Output: Table showing per-stage, per-bucket statistics:
    - Outlier ratio in top-50% subset
    - E-matrix angular error vs ground truth
    - Epipolar distance statistics for inliers/outliers
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from MGCA import MGCANet
from data import CorrespondencesDataset, collate_fn
from utils import tocuda


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_va", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max samples to analyze (set -1 for all)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--out_csv", type=str, default="e_matrix_quality.csv")
    return parser.parse_args()


class SimpleConfig:
    net_depth = 12
    net_channels = 128
    iter_num = 2
    clusters = 500
    use_fundamental = False
    obj_geod_th = 1e-4
    use_ratio = 0
    use_mutual = 0
    ratio_test_th = 0.8


def compute_e_matrix_error(e_pred, e_gt):
    """
    Compute angular error between predicted and GT essential matrices.
    e_pred, e_gt: [B, 9] or [9], normalized.
    Returns: [B] angular errors in degrees.
    """
    if e_pred.dim() == 1:
        e_pred = e_pred.unsqueeze(0)
    if e_gt.dim() == 1:
        e_gt = e_gt.unsqueeze(0)

    # Handle sign ambiguity: E and -E represent the same epipolar geometry
    dot = (e_pred * e_gt).sum(dim=1)
    dot_abs = torch.abs(dot)
    # Clamp to avoid numerical issues
    dot_abs = torch.clamp(dot_abs, -1.0, 1.0)
    # Angular error: arccos(|dot|) in radians -> degrees
    angle_rad = torch.acos(dot_abs)
    angle_deg = angle_rad * 180.0 / np.pi
    return angle_deg


@torch.no_grad()
def analyze_sample(model, data, config):
    """
    For a single batch, compute per-stage statistics.
    Returns list of dicts, one per sample in batch.
    """
    device = next(model.parameters()).device
    data = tocuda(data)

    # Forward
    res_logits, res_e_hat = model(data)
    xs = data['xs']  # [B, 1, N, 4]
    ys = data['ys'][:, :, 0]  # [B, N] ground truth geodesic distance
    is_pos = (ys < config.obj_geod_th).float()  # [B, N]

    # Ground truth E-matrix
    R_in, t_in = data['Rs'], data['ts']
    e_gt_unnorm = torch.matmul(
        torch_skew_symmetric_batch(t_in),
        R_in
    )
    e_gt = e_gt_unnorm / (torch.norm(e_gt_unnorm, dim=1, keepdim=True) + 1e-10)

    B, N = ys.shape
    num_stages = len(res_e_hat)  # Typically 4 (init + 2 iter + final)

    results = []
    for b in range(B):
        sample = {
            'overall_outlier_ratio': (1.0 - is_pos[b].mean()).item(),
            'num_inliers': int(is_pos[b].sum().item()),
            'num_total': N,
        }

        for s in range(num_stages):
            logits_s = res_logits[s][b]  # [N]
            e_hat_s = res_e_hat[s][b]     # [9]

            # Top-50% indices (same as down_sampling in MGCA-Net)
            K = N // 2
            top_indices = torch.argsort(logits_s, descending=True)[:K]

            # Outlier ratio in top-50% subset
            inliers_in_top = is_pos[b][top_indices].sum().item()
            outliers_in_top = K - inliers_in_top
            top_outlier_ratio = outliers_in_top / K

            # E-matrix angular error
            e_err = compute_e_matrix_error(e_hat_s, e_gt[b]).item()

            sample[f'stage{s}_top50_outlier_ratio'] = top_outlier_ratio
            sample[f'stage{s}_e_error_deg'] = e_err
            sample[f'stage{s}_inliers_in_top50'] = int(inliers_in_top)

        results.append(sample)

    return results


def torch_skew_symmetric_batch(t):
    """Batch skew-symmetric matrix from translation vector t [B, 3]."""
    B = t.shape[0]
    zero = torch.zeros(B, device=t.device, dtype=t.dtype)
    return torch.stack([
        torch.stack([zero, -t[:, 2], t[:, 1]], dim=1),
        torch.stack([t[:, 2], zero, -t[:, 0]], dim=1),
        torch.stack([-t[:, 1], t[:, 0], zero], dim=1),
    ], dim=1)  # [B, 3, 3]


def bucket_name(outlier_ratio):
    if outlier_ratio < 0.50:
        return '0-50'
    elif outlier_ratio < 0.75:
        return '50-75'
    elif outlier_ratio < 0.90:
        return '75-90'
    elif outlier_ratio < 0.95:
        return '90-95'
    else:
        return '>95'


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    config = SimpleConfig()
    model = MGCANet(config).to(device)

    print(f"Loading pretrained: {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    # Data
    dataset = CorrespondencesDataset(args.data_va, config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    all_results = []
    total_samples = 0
    for batch_idx, data in enumerate(loader):
        if args.max_samples > 0 and total_samples >= args.max_samples:
            break
        batch_results = analyze_sample(model, data, config)
        all_results.extend(batch_results)
        total_samples += len(batch_results)
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {total_samples} samples...")

    print(f"\nTotal samples analyzed: {len(all_results)}")

    # ---- Aggregate by bucket ----
    buckets = {
        '0-50': [], '50-75': [], '75-90': [], '90-95': [], '>95': []
    }
    for r in all_results:
        bname = bucket_name(r['overall_outlier_ratio'])
        buckets[bname].append(r)

    num_stages = max(
        sum(1 for k in r.keys() if k.startswith('stage') and k.endswith('_e_error_deg'))
        for r in all_results
    )

    # Print table
    print("\n" + "=" * 100)
    print("E-Matrix Quality Analysis by Bucket and Stage")
    print("=" * 100)
    print(f"{'Bucket':>8s}  {'N':>6s}  {'Stage':>6s}  "
          f"{'Top50_Out%':>11s}  {'Inliers_in_T50':>15s}  {'E_Err(deg)':>12s}")
    print("-" * 100)

    csv_lines = ["bucket,n,stage,top50_outlier_ratio_mean,top50_outlier_ratio_std,"
                 "inliers_in_top50_mean,inliers_in_top50_std,"
                 "e_error_deg_mean,e_error_deg_std"]

    for bname in ['0-50', '50-75', '75-90', '90-95', '>95']:
        samples = buckets[bname]
        if len(samples) == 0:
            continue

        for s in range(num_stages):
            top50_out = [r[f'stage{s}_top50_outlier_ratio'] for r in samples]
            inliers_top = [r[f'stage{s}_inliers_in_top50'] for r in samples]
            e_errs = [r[f'stage{s}_e_error_deg'] for r in samples]

            print(f"{bname:>8s}  {len(samples):>6d}  stage{s}:>5d  "
                  f"{np.mean(top50_out)*100:>10.1f}%  "
                  f"{np.mean(inliers_top):>14.1f}   "
                  f"{np.mean(e_errs):>10.2f} deg")

            csv_lines.append(
                f"{bname},{len(samples)},stage{s},"
                f"{np.mean(top50_out):.4f},{np.std(top50_out):.4f},"
                f"{np.mean(inliers_top):.2f},{np.std(inliers_top):.2f},"
                f"{np.mean(e_errs):.4f},{np.std(e_errs):.4f}"
            )

    print("=" * 100)

    # Save CSV
    with open(args.out_csv, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"\nSaved detailed results to: {args.out_csv}")

    # ---- Key insight: Does top-50% help? ----
    print("\n" + "=" * 100)
    print("KEY QUESTION: Does top-50% selection effectively filter outliers?")
    print("=" * 100)

    for bname in ['90-95', '>95']:
        samples = buckets[bname]
        if len(samples) == 0:
            continue

        overall_out = np.mean([r['overall_outlier_ratio'] for r in samples]) * 100
        # Final stage (usually stage 3 or 4)
        final_stage = num_stages - 1
        top50_out = np.mean([r[f'stage{final_stage}_top50_outlier_ratio'] for r in samples]) * 100
        inliers_top = np.mean([r[f'stage{final_stage}_inliers_in_top50'] for r in samples])

        print(f"\nBucket {bname} (N={len(samples)}):")
        print(f"  Overall outlier ratio:     {overall_out:.1f}%")
        print(f"  Top-50% outlier ratio:     {top50_out:.1f}%")
        print(f"  Inliers captured in top50: {inliers_top:.1f} / {np.mean([r['num_inliers'] for r in samples]):.1f}")
        print(f"  E-matrix angular error:    {np.mean([r[f'stage{final_stage}_e_error_deg'] for r in samples]):.2f} deg")

        if top50_out < overall_out:
            reduction = overall_out - top50_out
            print(f"  => Top-50% REDUCES outlier ratio by {reduction:.1f} percentage points")
        else:
            print(f"  => WARNING: Top-50% does NOT reduce outlier ratio!")

    print("=" * 100)


if __name__ == '__main__':
    main()
