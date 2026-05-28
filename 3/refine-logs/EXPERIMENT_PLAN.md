# Experiment Plan: IPG-SAD

## Overview

Validate Inlier-Prior Guided MoE with Scale-Aware Decomposition on standard benchmarks.
Base code: GeoMoE (https://github.com/JiajunLe/GeoMoE)

---

## Phase 1: Pilot (2-3 hours)

### Pilot 1: Individual Fix Verification
**Goal**: Verify each of the 3 changes works independently

**Setup**:
- Dataset: YFCC100M validation subset (first 1000 pairs)
- Descriptor: SIFT-2000
- Metric: mAP@5°, Precision, Recall, F1

**Runs**:
1. Baseline GeoMoE (original code, no changes)
2. GeoMoE + Fix 1 only (inlier-weighted load balance)
3. GeoMoE + Fix 2 only (sigmoid + size normalization)
4. GeoMoE + Fix 3 only (dynamic expert capacity)

**Expected outcome**: Each fix shows marginal improvement; combined shows synergy

### Pilot 2: Failure Case Analysis
**Goal**: Identify when IPG-SAD still fails

**Setup**:
- Synthetic data: varying outlier ratios (50%, 70%, 90%, 95%, 99%)
- Measure: inlier recovery rate, expert usage distribution, sub-field size accuracy

**Expected outcome**: Performance degrades gracefully; clear failure mode at extreme ratios

---

## Phase 2: Full Benchmarks (8-12 hours)

### Experiment 1: Standard Benchmarks
**Datasets**:
- YFCC100M (outdoor, SIFT/SuperPoint)
- SUN3D (indoor, SIFT/SuperPoint)
- HPatches (homography)

**Metrics**:
- mAP@5° / mAP@10° / mAP@20°
- AUC (pose estimation)
- Precision / Recall / F1
- Runtime (ms per pair)

**Baselines**:
- GeoMoE (original)
- CorrMoE
- DeMo
- DeMatch
- NACNet
- CLNet
- OANet

### Experiment 2: Cross-Descriptor Generalization
**Setup**:
- Train on SIFT, test on SuperPoint, ORB, RootSIFT
- Measure generalization gap

### Experiment 3: Cross-Scene Generalization
**Setup**:
- Train on YFCC100M (outdoor), test on SUN3D (indoor)
- Measure domain gap

---

## Phase 3: Ablations (4-6 hours)

### Ablation 1: Component Analysis
- Full IPG-SAD
- Without Fix 1 (standard load balance)
- Without Fix 2 (softmax decomposition)
- Without Fix 3 (fixed hidden_dim=16)
- Without Fix 1+2
- Without Fix 1+3
- Without Fix 2+3

### Ablation 2: Hyperparameter Sensitivity
- num_experts: {2, 4, 8, 16}
- hidden_dim base: {8, 16, 32, 64}
- sigmoid temperature: {0.5, 1.0, 2.0}
- load balance strength: {0.01, 0.1, 1.0}

### Ablation 3: Alternative Designs
- Softmax + learned temperature vs Sigmoid + normalization
- Fixed capacity vs linear scaling vs logarithmic scaling
- Hard routing vs soft routing (weighted combination)

---

## Phase 4: Analysis (2-3 hours)

### Analysis 1: Expert Specialization
- Visualize which experts handle which motion patterns
- Measure expert usage distribution across test scenes
- Compare with GeoMoE's uniform usage

### Analysis 2: Attention Patterns
- Visualize attention maps for inliers vs outliers
- Measure attention entropy before/after fixes
- Compare with NACNet's Deep Sets approach

### Analysis 3: Sub-Field Decomposition Quality
- Measure correlation between predicted sub-fields and ground truth motion patterns
- Evaluate size preservation accuracy
- Compare with DeMatch's Fourier decomposition

---

## Compute Budget

| Phase | GPU Hours | Priority |
|-------|-----------|----------|
| Pilot | 3 | MUST |
| Full Benchmarks | 10 | MUST |
| Ablations | 6 | HIGH |
| Analysis | 3 | MEDIUM |
| **Total** | **22** | |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Training instability with dynamic capacity | Medium | Gradient clipping, warmup |
| Sigmoid causes gradient vanishing | Low | Use GELU-sigmoid hybrid |
| No improvement on clean scenes | High | Focus on high-outlier subsets |
| CorrMoE already published similar | Low | Distinct problem and approach |
| GeoMoE code incompatible | Low | Minimal changes, easy integration |
