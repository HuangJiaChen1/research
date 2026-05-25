#!/usr/bin/env python3
"""
Test script for LearnableConsensus on MGCA-Net.

Loads a trained checkpoint and evaluates on test set.
Outputs global metrics + bucket-wise P/R/F1.
Supports cross-dataset evaluation (e.g. YFCC -> SUN3D).

Usage:
    # Evaluate on YFCC100M test set
    CUDA_VISIBLE_DEVICES=0 python test_consensus.py \
        --checkpoint ./log_consensus/learned/model_best.pth \
        --data_te ../data_dump/yfcc-sift-2000-test.hdf5 \
        --out_json ./results_yfcc.json

    # Cross-dataset: YFCC-trained model on SUN3D
    CUDA_VISIBLE_DEVICES=0 python test_consensus.py \
        --checkpoint ./log_consensus/learned/model_best.pth \
        --data_te ../data_dump/sun3d-test.hdf5 \
        --out_json ./results_sun3d.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from MGCA import MGCANet
from data import CorrespondencesDataset, collate_fn
from loss import MatchLoss
from test import test_process
from train_consensus import Config, evaluate_by_bucket
from utils import tocuda


def get_args():
    parser = argparse.ArgumentParser(description="Test LearnableConsensus")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_best.pth or checkpoint.pth")
    parser.add_argument("--data_te", type=str, required=True,
                        help="Test HDF5 dataset")
    parser.add_argument("--out_json", type=str, default="",
                        help="Output JSON path for results")
    parser.add_argument("--out_txt", type=str, default="",
                        help="Output TXT path for results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    # Config overrides (must match training config)
    parser.add_argument("--obj_geod_th", type=float, default=1e-4)
    parser.add_argument("--geo_loss_margin", type=float, default=0.1)
    parser.add_argument("--loss_essential", type=float, default=0.5)
    parser.add_argument("--loss_classif", type=float, default=1.0)
    parser.add_argument("--net_depth", type=int, default=12)
    parser.add_argument("--net_channels", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=500)
    parser.add_argument("--use_fundamental", action="store_true")
    return parser.parse_args()


def print_results(results_dict, title="Results"):
    """Pretty-print results to console."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("-" * 70)

    # Global
    g = results_dict['global']
    print(f"  GLOBAL    mAP={g['mAP']:.4f}  "
          f"P={g['precision']:.4f}  R={g['recall']:.4f}  F1={g['f1']:.4f}  "
          f"(n={g['n']})")
    print("-" * 70)

    # Buckets
    for name in ['0-50', '50-75', '75-90', '90-95', '>95']:
        b = results_dict['buckets'][name]
        if b['n'] > 0:
            print(f"  {name:8s}  P={b['precision']:.4f}  R={b['recall']:.4f}  "
                  f"F1={b['f1']:.4f}  (n={b['n']})")
        else:
            print(f"  {name:8s}  -- no samples --")
    print("=" * 70)


def save_txt(results_dict, path, checkpoint_name, dataset_name):
    """Save human-readable results to TXT."""
    with open(path, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_name}\n")
        f.write(f"Dataset:    {dataset_name}\n")
        f.write(f"Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 60 + "\n")

        g = results_dict['global']
        f.write(f"GLOBAL    mAP={g['mAP']:.4f}  "
                f"P={g['precision']:.4f}  R={g['recall']:.4f}  F1={g['f1']:.4f}\n")
        f.write("-" * 60 + "\n")

        for name in ['0-50', '50-75', '75-90', '90-95', '>95']:
            b = results_dict['buckets'][name]
            if b['n'] > 0:
                f.write(f"{name:8s}  P={b['precision']:.4f}  "
                        f"R={b['recall']:.4f}  F1={b['f1']:.4f}  n={b['n']}\n")

        f.write("-" * 60 + "\n")
        # Latex-ready table row
        buckets = results_dict['buckets']
        f.write("\nLaTeX table row:\n")
        f.write(f"  & {g['precision']:.3f} & {g['recall']:.3f} & {g['f1']:.3f}")
        for name in ['0-50', '50-75', '75-90', '90-95', '>95']:
            b = buckets[name]
            f.write(f" & {b['f1']:.3f}")
        f.write(" \\\\\n")


def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config
    config = Config(args)
    config.log_path = ""
    config.res_path = ""

    # Model
    model = MGCANet(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"  Loaded at step {ckpt.get('epoch', 'unknown')}, "
          f"ablation={ckpt.get('ablation', 'unknown')}")

    model = model.to(device)
    model.eval()

    # Data
    print(f"Loading test set: {args.data_te}")
    test_dataset = CorrespondencesDataset(args.data_te, config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    print(f"  Test batches: {len(test_loader)}")

    # ---- Global metrics (mAP + P/R/F1) ----
    print("\nRunning global evaluation ...")
    va_res = test_process("test", model, 0, test_loader, config)
    mAP, geo_loss, cla_loss, l2_loss, precision, recall, fscore = va_res

    # Count total samples
    total_n = 0
    for data in test_loader:
        total_n += data['xs'].shape[0]

    # ---- Bucket-wise metrics ----
    print("Running bucket-wise evaluation ...")
    bucket_res = evaluate_by_bucket(test_loader, model, config)

    # Assemble results
    results = {
        'checkpoint': args.checkpoint,
        'dataset': args.data_te,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'global': {
            'n': total_n,
            'mAP': float(mAP),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(fscore),
            'geo_loss': float(geo_loss),
            'cla_loss': float(cla_loss),
            'l2_loss': float(l2_loss),
        },
        'buckets': {
            name: {
                'n': stats['n'],
                'precision': float(stats['precision']),
                'recall': float(stats['recall']),
                'f1': float(stats['f1']),
            }
            for name, stats in bucket_res.items()
        },
    }

    # Print
    ablation_name = ckpt.get('ablation', 'unknown')
    print_results(results, title=f"Test Results  [{ablation_name}]")

    # Save JSON
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")

    # Save TXT
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or '.', exist_ok=True)
        save_txt(results, args.out_txt, args.checkpoint, args.data_te)
        print(f"Results saved to: {args.out_txt}")

    # Also print to console as JSON for easy parsing
    print("\n--- JSON (for copy-paste) ---")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
