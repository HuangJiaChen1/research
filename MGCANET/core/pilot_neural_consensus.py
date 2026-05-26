#!/usr/bin/env python3
"""
Pilot: NeuralConsensus vs Hand-crafted Product Consensus on MGCA-Net.

Trains only the consensus_module + CSMGC with frozen pretrained backbone.
Compares:
  - fixed_product:  Hand-crafted consensus = sem * exp(-geo_dist) [zero-shot MVP]
  - mlp:            NeuralConsensus (MLP learns fusion from 8-dim features)

Usage:
    # Hand-crafted product baseline (zero-shot, no training needed for consensus)
    CUDA_VISIBLE_DEVICES=0 python pilot_neural_consensus.py \
        --pretrained ../weights/yfcc100m/model_best1.pth \
        --data_tr ../data_dump/yfcc-sift-2000-train.hdf5 \
        --data_va ../data_dump/yfcc-sift-2000-val.hdf5 \
        --log_dir ./log_pilot \
        --consensus_mode fixed_product

    # NeuralConsensus (MLP-based)
    CUDA_VISIBLE_DEVICES=0 python pilot_neural_consensus.py \
        --pretrained ../weights/yfcc100m/model_best1.pth \
        --data_tr ../data_dump/yfcc-sift-2000-train.hdf5 \
        --data_va ../data_dump/yfcc-sift-2000-val.hdf5 \
        --log_dir ./log_pilot \
        --consensus_mode mlp \
        --lr_consensus 1e-4 --lr_csmgc 1e-5
"""

import argparse
import os
import sys
import json
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))
from MGCA import MGCANet
from data import CorrespondencesDataset, collate_fn
from loss import MatchLoss
from test import valid
from utils import tocuda
from logger import Logger

# Reduce CUDA memory fragmentation (helpful for 11GB GPUs)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Pilot: NeuralConsensus vs Baseline")
    parser.add_argument("--data_tr", type=str,
                        default="../data_dump/yfcc-sift-2000-train.hdf5",
                        help="Training HDF5 dataset")
    parser.add_argument("--data_va", type=str,
                        default="../data_dump/yfcc-sift-2000-val.hdf5",
                        help="Validation HDF5 dataset")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pretrained model_best1.pth")
    parser.add_argument("--log_dir", type=str, default="./log_pilot",
                        help="Base log directory")
    parser.add_argument("--consensus_mode", type=str, default="mlp",
                        choices=["fixed_product", "mlp"],
                        help="Consensus mode: fixed_product (hand-crafted) or mlp (NeuralConsensus)")
    parser.add_argument("--train_iter", type=int, default=30000,
                        help="Training iterations (pilot: 30K)")
    parser.add_argument("--val_intv", type=int, default=5000,
                        help="Validation interval (steps)")
    parser.add_argument("--save_intv", type=int, default=1000,
                        help="Checkpoint save interval")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default 16 for 11GB GPU)")
    parser.add_argument("--lr_consensus", type=float, default=1e-4,
                        help="LR for consensus module")
    parser.add_argument("--lr_csmgc", type=float, default=1e-5,
                        help="LR for CSMGC")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="CUDA device ID(s)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (reduce if OOM)")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable pin_memory (may cause OOM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint path")
    # Dataset-specific overrides (must match pretrained model)
    parser.add_argument("--obj_geod_th", type=float, default=1e-4)
    parser.add_argument("--geo_loss_margin", type=float, default=0.1)
    parser.add_argument("--loss_essential", type=float, default=0.5)
    parser.add_argument("--loss_classif", type=float, default=1.0)
    parser.add_argument("--net_depth", type=int, default=12)
    parser.add_argument("--net_channels", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=500)
    parser.add_argument("--use_fundamental", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
class Config:
    """Minimal config object matching the original config.py interface."""
    def __init__(self, args):
        self.net_depth = args.net_depth
        self.net_channels = args.net_channels
        self.iter_num = 2
        self.clusters = args.clusters
        self.use_fundamental = args.use_fundamental
        self.obj_geod_th = args.obj_geod_th
        self.geo_loss_margin = args.geo_loss_margin
        self.loss_essential = args.loss_essential
        self.loss_classif = args.loss_classif
        self.loss_essential_init_iter = 0
        self.use_ransac = False
        self.use_ratio = 0
        self.use_mutual = 0
        self.ratio_test_th = 0.8
        self.obj_top_k = -1
        self.log_path = ""
        self.res_path = ""
        self.consensus_mode = args.consensus_mode


# ---------------------------------------------------------------------------
# Backbone freezing
# ---------------------------------------------------------------------------
def freeze_backbone(model):
    """Freeze subnetwork_init and subnetwork."""
    for param in model.subnetwork_init.parameters():
        param.requires_grad = False
    for param in model.subnetwork.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_step(step, optimizer, model, match_loss, data):
    model.train()
    res_logits, res_e_hat = model(data)

    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, prec, rec = \
            match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss, prec, rec]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val, loss.item()


# ---------------------------------------------------------------------------
# Bucket-wise evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_by_bucket(data_loader, model, config):
    """
    Compute precision / recall / F1 per outlier-ratio bucket.
    """
    model.eval()
    match_loss = MatchLoss(config)

    buckets = {
        '0-50':  {'precisions': [], 'recalls': [], 'f_scores': []},
        '50-75': {'precisions': [], 'recalls': [], 'f_scores': []},
        '75-90': {'precisions': [], 'recalls': [], 'f_scores': []},
        '90-95': {'precisions': [], 'recalls': [], 'f_scores': []},
        '>95':   {'precisions': [], 'recalls': [], 'f_scores': []},
    }

    for data in data_loader:
        data = tocuda(data)
        res_logits, res_e_hat = model(data)
        y_hat = res_logits[-1]

        gt_geod = data['ys'][:, :, 0]
        is_pos = (gt_geod < config.obj_geod_th).float()
        outlier_ratio = 1.0 - is_pos.mean(dim=1)

        pred_pos = (y_hat > 2).float()
        true_pos = pred_pos * is_pos
        precision = true_pos.sum(dim=1) / (pred_pos.sum(dim=1) + 1e-10)
        recall = true_pos.sum(dim=1) / (is_pos.sum(dim=1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        for b in range(y_hat.shape[0]):
            oratio = outlier_ratio[b].item()
            if   oratio < 0.50: bucket = '0-50'
            elif oratio < 0.75: bucket = '50-75'
            elif oratio < 0.90: bucket = '75-90'
            elif oratio < 0.95: bucket = '90-95'
            else:               bucket = '>95'

            buckets[bucket]['precisions'].append(precision[b].item())
            buckets[bucket]['recalls'].append(recall[b].item())
            buckets[bucket]['f_scores'].append(f1[b].item())

    results = {}
    for name, vals in buckets.items():
        n = len(vals['precisions'])
        if n > 0:
            results[name] = {
                'n': n,
                'precision': float(np.mean(vals['precisions'])),
                'recall':    float(np.mean(vals['recalls'])),
                'f1':        float(np.mean(vals['f_scores'])),
            }
        else:
            results[name] = {'n': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # ---- CUDA ----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Pilot: consensus_mode={args.consensus_mode}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ---- Dirs ----
    log_dir = os.path.join(args.log_dir, args.consensus_mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'valid'), exist_ok=True)

    # ---- Config ----
    config = Config(args)
    config.log_path = os.path.join(log_dir, 'train')

    # ---- Model ----
    model = MGCANet(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load pretrained backbone
    print(f"Loading pretrained: {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print("Pretrained weights loaded.")

    # Freeze backbone
    freeze_backbone(model)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Print consensus module details
    if args.consensus_mode == 'mlp':
        consensus_params = sum(p.numel() for p in model.consensus_module.parameters())
        print(f"  - NeuralConsensus params: {consensus_params:,}")
    else:
        print(f"  - FixedProductConsensus (hand-crafted, no params)")

    # Optimizer
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'consensus_module' in name:
            param_groups.append({'params': param, 'lr': args.lr_consensus,
                                 'weight_decay': args.weight_decay})
        elif 'CSMGC' in name:
            param_groups.append({'params': param, 'lr': args.lr_csmgc,
                                 'weight_decay': args.weight_decay})
        else:
            param_groups.append({'params': param, 'lr': args.lr_csmgc,
                                 'weight_decay': args.weight_decay})

    optimizer = optim.Adam(param_groups)
    model = model.to(device)

    # ---- Data ----
    print("\nLoading datasets ...")
    train_dataset = CorrespondencesDataset(args.data_tr, config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)

    valid_dataset = CorrespondencesDataset(args.data_va, config)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_memory, collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}\n")

    match_loss = MatchLoss(config)
    writer = SummaryWriter(log_dir)

    logger_train = Logger(os.path.join(log_dir, 'log_train.txt'),
                          title='pilot')
    logger_train.set_names(
        ['LR', 'GeoLoss', 'ClaLoss', 'L2Loss', 'Precision', 'Recall', 'F1'])

    logger_valid = Logger(os.path.join(log_dir, 'log_valid.txt'),
                          title='pilot')
    logger_valid.set_names(
        ['mAP', 'GeoLoss', 'ClaLoss', 'L2Loss', 'Precision', 'Recall', 'F1'])

    # ---- Resume ----
    start_step = 0
    best_f1 = -1.0
    checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from: {args.resume}")
        rckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(rckpt['state_dict'])
        optimizer.load_state_dict(rckpt['optimizer'])
        start_step = rckpt['epoch']
        best_f1 = rckpt.get('best_f1', -1.0)

    # ---- Training loop ----
    train_loader_iter = iter(train_loader)
    for step in trange(start_step, args.train_iter, ncols=79,
                       desc=f"Train [{args.consensus_mode}]"):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)

        train_data = tocuda(train_data)
        loss_vals, total_loss = train_step(step, optimizer, model,
                                           match_loss, train_data)

        # Logging: extract final-stage metrics
        n_metrics = 5
        n_outputs = len(loss_vals) // n_metrics
        off = (n_outputs - 1) * n_metrics
        final_geo  = loss_vals[off + 0]
        final_cla  = loss_vals[off + 1]
        final_l2   = loss_vals[off + 2]
        final_prec = loss_vals[off + 3]
        final_rec  = loss_vals[off + 4]
        final_f1   = 2 * final_prec * final_rec / (final_prec + final_rec + 1e-10)

        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/TotalLoss', total_loss, step)
        writer.add_scalar('Train/LR', cur_lr, step)
        writer.add_scalar('Train/Precision', final_prec, step)
        writer.add_scalar('Train/Recall', final_rec, step)
        writer.add_scalar('Train/F1', final_f1, step)
        logger_train.append([cur_lr, final_geo, final_cla, final_l2,
                             final_prec, final_rec, final_f1])

        # ---- Validation ----
        b_validate = ((step + 1) % args.val_intv) == 0
        b_save = ((step + 1) % args.save_intv) == 0

        if b_validate:
            va_res = valid(valid_loader, model, step, config)
            mAP, geo_loss, cla_loss, l2_loss, precision, recall, fscore = va_res

            writer.add_scalar('Val/mAP', mAP, step)
            writer.add_scalar('Val/Precision', precision, step)
            writer.add_scalar('Val/Recall', recall, step)
            writer.add_scalar('Val/F1', fscore, step)
            logger_valid.append([mAP, geo_loss, cla_loss, l2_loss,
                                 precision, recall, fscore])

            # Bucket-wise metrics
            print("\n" + "=" * 70)
            print(f"  Step {step+1} | GLOBAL  P={precision:.4f}  R={recall:.4f}  "
                  f"F1={fscore:.4f}  mAP={mAP:.4f}")
            print("-" * 70)
            bucket_res = evaluate_by_bucket(valid_loader, model, config)
            for name, stats in bucket_res.items():
                if stats['n'] > 0:
                    print(f"  {name:8s}  n={stats['n']:5d}  "
                          f"P={stats['precision']:.4f}  "
                          f"R={stats['recall']:.4f}  "
                          f"F1={stats['f1']:.4f}")
                    writer.add_scalar(f'Bucket/{name}/Precision',
                                      stats['precision'], step)
                    writer.add_scalar(f'Bucket/{name}/Recall',
                                      stats['recall'], step)
                    writer.add_scalar(f'Bucket/{name}/F1',
                                      stats['f1'], step)
            print("=" * 70)

            # Save best by >95% bucket F1
            if bucket_res['>95']['f1'] > best_f1:
                best_f1 = bucket_res['>95']['f1']
                print(f"[BEST] >95% F1 = {best_f1:.4f}  -- saving model")
                torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                    'consensus_mode': args.consensus_mode,
                    'args': vars(args),
                }, os.path.join(log_dir, 'model_best.pth'))

                # Save bucket results for easy comparison
                with open(os.path.join(log_dir, 'best_bucket_results.json'), 'w') as f:
                    json.dump({
                        'step': step + 1,
                        'global': {
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1': float(fscore),
                            'mAP': float(mAP),
                        },
                        'buckets': bucket_res,
                    }, f, indent=2)

        if b_save:
            torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
                'consensus_mode': args.consensus_mode,
                'args': vars(args),
            }, checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Training complete. consensus_mode={args.consensus_mode}")
    print(f"Best >95% bucket F1: {best_f1:.4f}")
    print(f"{'='*60}\n")
    writer.close()


if __name__ == '__main__':
    main()
