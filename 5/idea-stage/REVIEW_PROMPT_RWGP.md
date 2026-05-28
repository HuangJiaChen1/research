# External Review Prompt: Residual-Weighted Global Pooling (RWGP)

## Role
You are a senior reviewer for NeurIPS/ICML/CVPR. Your job is to provide brutally honest, actionable feedback on a proposed research contribution.

---

## Project Context

### Domain
Two-view correspondence learning / outlier rejection. Given a set of putative correspondences between two images (many of which are outliers), the goal is to identify inliers and estimate the epipolar geometry (essential/fundamental matrix).

### Target Paper
MGCA-Net (Multi-Graph Contextual Attention Network), accepted at IJCAI 2025. It is currently a SOTA method on YFCC100M and SUN3D benchmarks.

### What We Found (Code-Level Audit + Real Data Verification)
We audited the official MGCA-Net source code and ran the pretrained model on real YFCC100M test data (N=200 samples, outlier ratios 11%–98%, mean ~79%). We discovered four structural pain points:

1. **Fixed k-NN graph construction** in feature space: At 95–100% outliers, 36.96% of an inlier's k-NN neighbors are actually outliers.
2. **MB-FFN global pooling contamination**: `AdaptiveAvgPool2d(1)` and `AdaptiveMaxPool2d(1)` pool over ALL correspondences without outlier weighting. At 95–100% outliers, avg-pool shifts by 1.18 vs a residual-weighted baseline.
3. **CSMGC lacks real consensus**: Three stage graphs are naively concatenated. One noisy stage increases fused variance by 808% (synthetic).
4. **CPT positional attention pollution**: Raw outlier coordinates are injected into attention without geometric validation.

### Proposed Contribution: Residual-Weighted Global Pooling (RWGP)

**Target**: Pain Point #2 (MB-FFN global pooling contamination).

**Method**: MGCA-Net already computes per-correspondence epipolar residuals (`batch_episym`) in every stage. We propose using these residuals as soft weights for global pooling in the MB-FFN block:

```python
# Inside MBFFN.forward
w = torch.exp(-residual / tau)  # [B, 1, N, 1]
w = w / (w.sum(dim=2, keepdim=True) + eps)
weighted_avg = (features * w).sum(dim=2, keepdim=True)
# Replace AdaptiveAvgPool2d with weighted_avg
```

**Key Properties**:
- **Zero new parameters**: reuses existing residual tensor.
- **Minimal change**: ~5 lines of code in one module.
- **Domain-specific**: weights are derived from geometric plausibility (epipolar distance), not learned attention.

**Novelty Assessment**:
- Closest prior: **ACNe** (CVPR 2020) uses *learned* attentive weights for **Context Normalization** (feature normalization, NOT pooling). ACNe's weights are feature-driven MLP outputs.
- **Delta**: RWGP targets **pooling** (not normalization) and uses **geometry-derived** weights (not learned attention). We believe this combination is novel.
- No direct concurrent work found in 2025–2026 arXiv.

**Pilot Results**:
- Synthetic: 70–95% reduction in global-context shift across outlier ratios 30%–95%.
- Real data: Uniform vs residual-weighted pooling diverges monotonically with outlier ratio (shift = 0.86 at 50–70% → 1.18 at 95–100%).

**Current Status**:
- Idea validated on synthetic data and real intermediate activations.
- **Not yet implemented** in the training pipeline (no retraining done).

---

## Questions for Reviewer

Please act as a senior ML reviewer and answer the following:

### 1. Contribution Sufficiency
Is a single-module fix (RWGP) to an existing SOTA network sufficient for a top-tier venue (e.g., ICCV, CVPR, ICLR)? Or is it too incremental? What would make it "sufficient"?

### 2. Novelty & Prior Art
How strong is the novelty delta relative to ACNe? Is "applying attentive normalization's insight to pooling" a valid contribution, or would reviewers dismiss it as obvious?

### 3. Experimental Bar
What is the **minimum experiment package** needed to make this paper convincing? Specifically:
   - Do we need to retrain MGCA-Net from scratch with RWGP, or is a fine-tuning / ablation study enough?
   - What benchmarks are mandatory (YFCC100M, SUN3D, others)?
   - What baselines must we compare against?
   - Should we also implement RWGP in other networks (OANet, CLNet) to show generalization?

### 4. Claims & Positioning
How should we frame the contribution to maximize acceptance probability? Options:
   - A) "We identify a critical flaw in MGCA-Net's MB-FFN and fix it with RWGP."
   - B) "We propose a general principle: geometric residuals should guide global pooling in correspondence networks. We validate it by improving MGCA-Net."
   - C) "We present a unified analysis of structural weaknesses in modern correspondence networks and propose RWGP as a minimal, principled fix."

### 5. Weaknesses & Risks
What is the strongest criticism a reviewer could raise? How do we preempt it?

### 6. Mock Review
Please write a mock NeurIPS/CVPR review with: Summary, Strengths, Weaknesses, Questions for Authors, Score (1–10), Confidence (1–5), and "What Would Move Toward Accept."

---

## Background Papers for Reference
- MGCA-Net (Lin et al., IJCAI 2025): SOTA two-view correspondence network with CGA + CSMGC modules.
- ACNe (Sun et al., CVPR 2020): Attentive Context Normalization for robust permutation-equivariant learning.
- Context Normalization (Yi et al., ECCV 2018): PointCN uses global mean/variance normalization over correspondences.
- CLNet (Zhao et al., ICCV 2021): Progressive correspondence pruning by consensus learning.
- NCMNet (Liu et al., CVPR 2023): Progressive neighbor consistency mining.
