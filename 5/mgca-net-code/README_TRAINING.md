# RWGP Training & Evaluation Guide

## What Changed

We modified **MGCA-Net** to add **Residual-Weighted Global Pooling (RWGP)** in the MB-FFN module.

### Code Changes (6 files)

| File | Change |
|------|--------|
| `core/MGCA.py` | Added `use_rwgp` and `tau` to `MBFFN`, `CGA_Module`, `sub_MGCANet`, `MGCANet`. Residual from previous stage is extracted and used as soft weighting for global avg/max pooling. |
| `core/config.py` | Added `--use_rwgp` (bool) and `--rwgp_tau` (float) CLI args. |
| `core/main.py` | Passes `use_rwgp` and `tau` to model constructor. |
| `core/test.py` | Saves Precision/Recall/F1 to `precision_recall_f1.txt` in test output dir. |
| `scripts/*.sh` | 6 training scripts + 1 test script + 1 parallel launcher + 1 result collector. |

### RWGP Mechanism

```
Inside MBFFN.forward:
  if use_rwgp and residual is not None:
      w = exp(-residual / tau)          # lower residual = higher weight
      w = w / (w.sum(dim=2) + eps)      # normalize
      weighted_avg = (features * w).sum(dim=2, keepdim=True)
      weighted_max = (features * w).max(dim=2, keepdim=True)[0]
  else:
      # original uniform pooling
```

**Key property**: Zero new parameters. Reuses the `batch_episym` residual already computed in every stage.

**Stage 1 behavior**: No prior residual available -> falls back to uniform pooling (same as baseline).
**Stage 2+ behavior**: Uses residual from previous stage as weighting signal.

---

## Environment Setup

### 1. Install Dependencies

```bash
cd mgca-net-code
conda create -n rwgp python=3.12 -y
conda activate rwgp
pip install -r requirements.txt
```

Required: PyTorch 2.6.0 with CUDA support, h5py, opencv, tensorboardX.

### 2. Data Preparation

Download YFCC100M / SUN3D preprocessed HDF5 files and place them:

```
mgca-net-code/
  data_dump/
    yfcc-sift-2000-train.hdf5
    yfcc-sift-2000-val.hdf5
    yfcc-sift-2000-test.hdf5
```

If your data is elsewhere, set the environment variable:

```bash
export DATA_DIR=/path/to/your/data
```

Or edit the paths directly in `scripts/*.sh`.

### 3. Make Scripts Executable

```bash
chmod +x scripts/*.sh
```

---

## Training

### Single Experiment

```bash
# Baseline (original MGCA-Net, no RWGP)
bash scripts/train_baseline.sh 0

# RWGP with tau=1.0 (recommended default)
bash scripts/train_rwgp_tau1.0.sh 0
```

GPU ID is the first argument (default: 0).

### Tau Ablation (Recommended)

Run all 5 experiments to find optimal tau:

```bash
# If you have 5 GPUs, run everything in parallel:
bash scripts/run_all_ablations.sh

# If you have 1 GPU, run sequentially:
bash scripts/train_baseline.sh 0
bash scripts/train_rwgp_tau0.5.sh 0
bash scripts/train_rwgp_tau1.0.sh 0
bash scripts/train_rwgp_tau2.0.sh 0
bash scripts/train_rwgp_tau5.0.sh 0
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Learning rate | 3e-4 |
| Training iterations | 500,000 |
| Validation interval | 10,000 steps |
| Save interval | 1,000 steps |
| LR schedule | Warmup (100k steps) + decay at 200k, 400k |

### Expected Training Time

| Hardware | Approx. Time |
|----------|--------------|
| 1x A100 80GB | ~5-7 days |
| 1x RTX 4090 | ~7-10 days |
| 1x V100 32GB | ~10-14 days |

---

## Testing

After training completes, test the best model:

```bash
bash scripts/test_model.sh ./log/rwgp_tau1.0_yfcc/train/model_best1.pth 0
```

Results are saved to `./log/rwgp_tau1.0_yfcc/test/`:

| File | Content |
|------|---------|
| `precision_recall_f1.txt` | Precision, Recall, F1-score (Table 1 metrics) |
| `map.txt` | mAP@5, mAP@10, mAP@15, mAP@20 (Table 2 metrics) |
| `auc.txt` | AUC@5, AUC@10, AUC@20 (Table 2 metrics) |
| `median_err_q.txt` | Median rotation error (deg) |
| `median_err_t.txt` | Median translation error (deg) |
| `acc_qt_auc20_ours.txt` | AUC@20 (qt) used for model selection |

### Collect Results from All Ablations

```bash
bash scripts/collect_results.sh
```

Outputs a CSV: `./log/ablation_summary.csv`

Example output:
```
experiment,tau,precision,recall,f1,mAP5,mAP10,mAP15,mAP20,AUC5,AUC10,AUC20
baseline_yfcc,None,0.8484,0.8413,0.8383,65.18,77.50,80.20,81.40,35.79,55.30,71.35
rwgp_tau1.0_yfcc,1.0,0.8550,0.8450,0.8420,66.50,78.20,81.10,82.60,36.50,56.20,72.80
```

---

## Paper Metrics Reference

### MGCA-Net (Paper Numbers)

**Table 1: Outlier Rejection (YFCC100M)**

| Scene | P (%) | R (%) | F (%) |
|-------|-------|-------|-------|
| Known | 84.84 | 84.13 | 83.83 |
| Unknown | 83.62 | 81.07 | 81.82 |

**Table 2: Relative Pose Estimation (YFCC100M)**

| Scene | mAP@5 | AUC@5 | mAP@20 | AUC@20 |
|-------|-------|-------|--------|--------|
| Known | 65.18 | 35.79 | 81.40 | 71.35 |
| Unknown | 77.10 | 44.62 | 89.23 | 79.36 |

**Table 3: Ablation (YFCC Unknown)**

| Model | mAP@5 | mAP@20 |
|-------|-------|--------|
| Baseline | 56.70 | 79.33 |
| +Iter | 58.20 | 80.46 |
| +CGA | 66.57 | 84.58 |
| +CSMGC | 73.70 | 88.26 |
| Full | 77.10 | 89.23 |

### What "Success" Looks Like for RWGP

| Metric | Target |
|--------|--------|
| **F1-score** | Match or exceed baseline F1 (83.83 known / 81.82 unknown) |
| **mAP@5** | Match or exceed baseline mAP@5 (65.18 known / 77.10 unknown) |
| **AUC@20** | Match or exceed baseline AUC@20 (71.35 known / 79.36 unknown) |

**Minimum bar**: RWGP (best tau) matches baseline within 0.5% on all metrics.
**Good result**: RWGP exceeds baseline by >= 1% on F1 or mAP@5.
**Strong result**: RWGP exceeds baseline by >= 2% on F1 AND shows faster convergence.

### Red Flags During Training

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Val mAP stuck at ~0.3 | tau too small (0.5), suppressing too many points | Try tau=2.0 or 5.0 |
| Geo loss diverges / NaN | Learning rate too high for RWGP-modified features | Reduce lr to 1e-4 |
| No improvement over baseline after 200k steps | RWGP not impactful enough | Check if residual values are reasonable; verify MBFFN is actually receiving residual |

---

## Known Limitations

### Known / Unknown Scene Split

The paper reports separate metrics for "Known Scenes" (68 sequences seen during training) and "Unknown Scenes" (4 held-out sequences for generalization testing).

**Current limitation**: Our test script computes aggregate metrics over the ENTIRE test set. It does NOT separate known from unknown scenes.

**Why**: The test HDF5 file does not expose scene labels in the current data loader. To separate known/unknown, we would need:
1. Scene labels for each test sample, OR
2. A pre-defined list of which samples belong to known vs unknown scenes.

**Impact**: For Strategy A, aggregate metrics are sufficient to validate RWGP. For the final paper, you may need to:
- Contact MGCA-Net authors for the exact scene split
- Or use the official test code from their repository (if it handles the split)

### SUN3D Training

To train on SUN3D (optional secondary benchmark):

1. Prepare SUN3D HDF5 files in `data_dump/`.
2. Create a new script by copying `scripts/train_rwgp_tau1.0.sh` and changing data paths:

```bash
--data_tr ${DATA_DIR}/sun3d-sift-2000-train.hdf5 \
--data_va ${DATA_DIR}/sun3d-sift-2000-val.hdf5 \
--data_te ${DATA_DIR}/sun3d-sift-2000-test.hdf5 \
--log_suffix rwgp_tau1.0_sun3d \
```

---

## File Reference

```
mgca-net-code/
  core/
    MGCA.py           # Model definition (MODIFIED with RWGP)
    config.py         # CLI args (MODIFIED with --use_rwgp, --rwgp_tau)
    main.py           # Entry point (MODIFIED)
    test.py           # Evaluation (MODIFIED: saves P/R/F1)
    train.py          # Training loop (UNCHANGED)
    loss.py           # Loss functions (UNCHANGED)
  scripts/
    train_baseline.sh       # Baseline (no RWGP)
    train_rwgp_tau0.5.sh    # RWGP tau=0.5
    train_rwgp_tau1.0.sh    # RWGP tau=1.0
    train_rwgp_tau2.0.sh    # RWGP tau=2.0
    train_rwgp_tau5.0.sh    # RWGP tau=5.0
    run_all_ablations.sh    # Launch all 5 in parallel
    test_model.sh           # Test a trained checkpoint
    collect_results.sh      # Gather results into CSV
  README_TRAINING.md  # This file
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'loss'"

Run training from the `core/` directory (scripts do this automatically with `cd core`).

### CUDA Out of Memory

Reduce batch size in scripts (default: 32). Try 24 or 16. Note: this may require adjusting learning rate proportionally.

### Cannot Load Pretrained Weights

The modified `MGCANet` class adds `use_rwgp` and `tau` parameters. Original pretrained weights (from MGCA-Net authors) do not include these flags.

- To load original weights into the **baseline** model: use `--use_rwgp False` -- the architecture is identical to original when `use_rwgp=False`.
- To load original weights into the **RWGP** model: not directly supported. You would need to write a weight-loading script that maps original keys to new model. **Recommended**: train RWGP from scratch.

### Residual Values Are All Zero in Stage 2+

Add debug print in `MBFFN.forward`:

```python
if residual is not None:
    print(f"  residual range: [{residual.min():.4f}, {residual.max():.4f}] mean={residual.mean():.4f}")
```

Normal range: residual should be > 0, typically 0.0 to ~5.0. If all zeros, check that `sub_MGCANet` is correctly extracting `data[:, 4:5, :, :]`.

---

## Next Steps After Training

1. Compare `ablation_summary.csv` across tau values.
2. Pick best tau (highest F1 and mAP@5).
3. Run full test on best model: `bash scripts/test_model.sh <best_model_path> 0`
4. Report results back for analysis and decision on Strategy B (cross-network validation).
