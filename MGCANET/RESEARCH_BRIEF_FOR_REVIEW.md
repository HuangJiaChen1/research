# Research Brief: LearnableConsensus for Two-View Correspondence Learning

**For external review. Please paste this into GPT-4o / Claude / Gemini for critical feedback.**

---

## 1. Project Context

**Base architecture:** MGCA-Net (Lin et al., IJCAI 2025) — open-source, we are NOT the authors.
**Our contribution:** A neural plug-in module called **NeuralConsensus** (revised from an earlier scalar-parameter design that was abandoned as "hyperparameter tuning, not learning").
**Problem:** MGCA-Net achieves strong aggregate F1 (~0.80) on YFCC100M, but suffers **precision collapse** at extreme outlier ratios (>95%).

| Bucket | Baseline Precision | Baseline Recall | Baseline F1 |
|--------|-------------------|-----------------|-------------|
| 90-95% | 0.719 | 0.928 | 0.813 |
| **>95%** | **0.424** | **0.838** | **0.556** |

The model finds inliers well (high recall) but admits too many false positives (low precision) at >95% outlier.

---

## 2. Proposed Method: NeuralConsensus

### 2.1 Core Idea

Multi-stage correspondence networks (like MGCA-Net) predict increasingly accurate essential matrices across stages. The **epipolar distances** from these predictions contain valuable geometric consistency information that is **not exploited** by existing architectures. We propose to:

1. Compute epipolar distances from each stage's predicted E-matrix
2. Extract an 8-dimensional feature vector per correspondence (semantic confidence, geometric statistics, stage trends)
3. Learn a **non-linear fusion** via a small MLP — not a handcrafted formula
4. Use the consensus score to refine features before cross-stage aggregation

**Design evolution**: We initially explored a scalar-parameter version (`alpha`, `sigma`, `stage_weights`) but abandoned it because it was merely tuning hyperparameters within a fixed function family. The current MLP-based version learns the fusion function from data, which is the correct formulation for a "learnable" module.

### 2.2 Module Design

```python
class NeuralConsensus(nn.Module):
    def __init__(self, num_stages=3, hidden_dim=64):
        super().__init__()
        self.num_stages = num_stages

        # MLP: 8-dim input -> hidden -> 1 output
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, stage_logits, stage_e_hats, xs):
        S = len(stage_e_hats)
        # 1. Compute epipolar distances per stage
        dists = [compute_epipolar_distance(e, xs) for e in stage_e_hats]
        dists = torch.stack(dists, dim=2)  # [B, N, S]

        # 2. Build 8-dimensional per-point features
        sem_conf = torch.sigmoid(stage_logits[-1])           # [B, N]
        geo_mean = dists.mean(dim=2)                          # [B, N]
        geo_var = dists.var(dim=2) if S > 1 else torch.zeros_like(geo_mean)
        global_mean = geo_mean.mean(dim=1, keepdim=True)
        global_std = geo_mean.std(dim=1, keepdim=True) + 1e-6
        rel_pos = (geo_mean - global_mean) / global_std       # [B, N]
        dist_init, dist_final = dists[:, :, 0], dists[:, :, -1]
        trend = dist_final - dist_init                        # [B, N]

        features = torch.stack([
            sem_conf, torch.exp(-geo_mean), geo_mean, geo_var,
            rel_pos, dist_init, dist_final, trend
        ], dim=2)  # [B, N, 8]

        # 3. MLP -> consensus weights
        consensus = self.mlp(features).squeeze(-1)  # [B, N]
        return consensus
```

**Total learnable params:** ~2.7K (MLP) + CSMGC (which we also fine-tune).

### 2.3 Integration into MGCA-Net

Inserted **before CSMGC** in `MGCANet.forward()`:

```python
# Before: sub_l_input = self.CSMGC(stage_out[0], stage_out[1], stage_out[2])
# After:
consensus = self.consensus_module(res_weights, res_e_hat, data['xs'])
refined_stage2 = stage_out[2] * consensus.unsqueeze(1).unsqueeze(-1)
sub_l_input = self.CSMGC(stage_out[0], stage_out[1], refined_stage2)
```

**Note**: `unsqueeze(1).unsqueeze(-1)` produces `[B, 1, N, 1]` which broadcasts correctly with `stage_out[2]` `[B, 128, N, 1]`. The old `unsqueeze(1).unsqueeze(1)` would create `[B, 1, 1, N]` causing a 30GB OOM via incorrect broadcasting.

### 2.4 Training Strategy

- Load pretrained MGCA-Net weights
- **Freeze backbone** (subnetwork_init + subnetwork)
- Train only: `consensus_module` + `CSMGC`
- Optimizer: Adam with lr=1e-4 (consensus), lr=1e-5 (CSMGC)
- 50K iterations, batch_size=32

---

## 3. Existing Experimental Evidence

### 3.1 Zero-Shot MVP (Completed)

We first validated the approach without any learning — using fixed `alpha=0.5`, uniform stage weights, fixed `sigma=1.0`.

| Bucket | Base F1 | +Consensus (fixed) | Delta | Base Prec | +Cons Prec | Delta |
|--------|---------|-------------------|-------|-----------|------------|-------|
| 90-95% | 0.813 | **0.843** | **+3.1pp** | 0.719 | **0.787** | **+6.8pp** |
| **>95%** | **0.556** | **0.624** | **+6.8pp** | **0.424** | **0.516** | **+9.2pp** |

**Key finding:** Product fusion (`sem * geo`) is the only effective formulation. Simple variance (+0.2pp), min_dist (+0.4pp), and pairwise agreement (+1.6pp) barely work.

### 3.2 Best Consensus Metric

| Metric | >95% F1 Delta | Note |
|--------|--------------|------|
| Variance | +0.2pp | Too noisy |
| min_dist | +0.4pp | Insensitive |
| Pairwise agreement | +1.6pp | Too conservative (P=0.732, R=0.512) |
| **Product** | **+6.8pp** | **Best — preserves AND-gate** |

---

## 4. Novelty Assessment (from literature search)

| Claim | Novelty | Evidence |
|-------|---------|----------|
| Multi-stage epipolar distance weighting | **HIGH** | No prior work learns to weight multi-stage epipolar distances |
| Log-domain product fusion (sem + geo) | **HIGH** | No prior work uses learnable product fusion in correspondence learning |
| >95% outlier optimization | **MEDIUM** | Problem known but under-reported; solution is new |
| Frozen-backbone fine-tuning | **LOW** | Standard practice |

**Closest prior work:**
- **MGCA-Net CSMGC (IJCAI 2025):** Static KNN graph + Annular Convolution. No epipolar distance, no learnable weights.
- **MSGSA GTC (TIP 2024):** Passes intermediate features between stages. No epipolar distance, no product fusion.
- **ACNe (CVPR 2020):** Single-stage attention weights for context normalization. No cross-stage, no geometric consistency scoring.
- **DeMo (AAAI 2025):** Motion field consensus via RKHS kernels. Not epipolar-based.

**Overall novelty score: 7.5/10.**

---

## 5. Planned Experiments

### 5.1 Ablation Study (Mandatory)

| Variant | How | Expected Result |
|---------|-----|----------------|
| Baseline | Pretrained MGCA-Net, no consensus | F1=0.556 @ >95% |
| Fixed product | Handcrafted: `sem * exp(-geo_mean)` | F1≈0.624 (zero-shot upper bound) |
| **Neural (MLP)** | **Learned 8-dim feature fusion** | **F1 > 0.635 (target: +8-10pp total)** |

**Success criterion:** Learned version achieves F1 > 0.635 @ >95% bucket (i.e., >+1pp over zero-shot).

### 5.2 Cross-Dataset Validation

- Train on YFCC100M
- Test on SUN3D (zero-shot transfer)
- Expected: learned consensus should generalize better than baseline due to explicit geometric constraints

### 5.3 Optional: Cross-Architecture Validation

- Plug LearnableConsensus into OANet or ACNe
- If it also improves, contribution becomes "general framework" rather than "MGCA-Net improvement"

---

## 6. Known Weaknesses & Risks

**Please be brutal in reviewing these:**

1. **Improvement magnitude:** We are improving an existing open-source method. Is +6.8pp (zero-shot) / +8-10pp (learned) at >95% bucket enough for a top venue? Or do we need aggregate SOTA?

2. **Generalizability:** Currently only validated on MGCA-Net. If we can't show it works on other architectures, is the contribution too narrow?

3. **Design iteration:** This is the second version in one week (scalar parameters → MLP). How do we convincingly frame this as a principled evolution rather than "moving goalposts"?

4. **Parametric cost:** The MLP has ~2.7K parameters — more defensible than ~5 scalars, but some reviewers may still argue it's "just fitting a heuristic." Cross-architecture validation is the strongest counter.

5. **Fair comparison:** We are modifying someone else's architecture. Are we sure the comparison protocol (training data, batch size, evaluation) is identical to MGCA-Net's reported numbers?

6. **Field maturity:** Correspondence learning is crowded (21+ methods in MGCA-Net comparison). Do we need to beat all of them on aggregate metrics, or is bucket-specific improvement enough?

7. **Message clarity:** Is the story "we improve MGCA-Net" or "we propose a general consensus filtering principle"? The latter is stronger but requires cross-architecture evidence.

---

## 7. Questions for the Reviewer

Please address the following:

1. **Is the contribution sufficient for CVPR/ICCV/ECCV?** What minimum additional experiments would make it competitive?

2. **Should we frame this as:**
   - (A) "An improvement to MGCA-Net" — straightforward but narrow
   - (B) "A general learnable consensus framework for multi-stage correspondence networks" — stronger but requires cross-architecture validation

3. **What is the weakest claim?** Which claim would a reviewer most likely attack, and how should we preempt it?

4. **Is the frozen-backbone strategy a weakness or a strength?** On one hand, it shows the module is plug-and-play. On the other, "only training consensus + CSMGC" might look insufficient if the improvement is small.

5. **How do we frame the design evolution?** We abandoned a scalar-parameter version in favor of an MLP. How do we present this as a principled design choice rather than ad-hoc iteration?

5. **What additional baselines should we compare against?** (e.g., using epipolar loss directly in training? Using RANSAC post-processing?)

6. **Paper title suggestions:** Which framing maximizes perceived novelty?

7. **Mock review:** Please write a mock NeurIPS/CVPR review with Summary, Strengths, Weaknesses, Questions, Score, and Confidence.

---

## 8. Technical Appendix

### 8.1 Epipolar Distance Computation

```python
def compute_epipolar_distance(E, xs):
    E = E.view(B, 3, 3)
    xs = xs[:, 0, :, :]  # [B, N, 4]
    x1, x2 = xs[:, :, :2], xs[:, :, 2:4]
    p1 = torch.cat([x1, torch.ones(B, N, 1)], dim=2)
    p2 = torch.cat([x2, torch.ones(B, N, 1)], dim=2)
    l2 = torch.bmm(E, p1.transpose(1, 2))  # [B, 3, N]
    d = torch.abs(torch.sum(p2.transpose(1, 2) * l2, dim=1))
    norm = torch.sqrt(l2[:, 0, :]**2 + l2[:, 1, :]**2) + 1e-10
    return d / norm
```

### 8.2 Why Log-Domain Product?

The zero-shot MVP proved that `product` (`sem * geo`) works, but linear combination (`alpha*sem + (1-alpha)*geo`) does not. The log-domain formulation preserves the product behavior while making `alpha` differentiable:

```
product:      consensus = sem * geo
log-domain:   log(consensus) = log(sem) + log(geo)
learnable:    log(consensus) = alpha * log(sem) + (1-alpha) * log(geo)
```

When alpha → 1: semantic only. When alpha → 0: geometric only. When alpha = 0.5: equal contribution, equivalent to product.

### 8.3 Why Frozen Backbone?

The backbone (subnetwork_init + subnetwork) is already well-trained. Unfreezing it would:
- Require 500K iterations (vs 50K frozen)
- Risk catastrophic forgetting
- Make ablation interpretation harder

Frozen training isolates the effect of the consensus module.

---

## 9. References

- MGCA-Net (IJCAI 2025): https://arxiv.org/abs/2512.23369
- MSGSA (TIP 2024): https://ieeexplore.ieee.org/document/10508303/
- ACNe (CVPR 2020): Attentive Context Normalization
- DeMo (AAAI 2025): Deep Motion Field Consensus
- OANet (ICCV 2019): Order-Aware Network
- NCMNet (CVPR 2023): Progressive Neighbor Consistency Mining

---

**End of brief. Please provide critical, actionable feedback.**
