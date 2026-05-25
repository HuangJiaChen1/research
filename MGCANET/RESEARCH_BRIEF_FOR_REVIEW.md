# Research Brief: LearnableConsensus for Two-View Correspondence Learning

**For external review. Please paste this into GPT-4o / Claude / Gemini for critical feedback.**

---

## 1. Project Context

**Base architecture:** MGCA-Net (Lin et al., IJCAI 2025) — open-source, we are NOT the authors.
**Our contribution:** A lightweight plug-in module called **LearnableConsensus**.
**Problem:** MGCA-Net achieves strong aggregate F1 (~0.80) on YFCC100M, but suffers **precision collapse** at extreme outlier ratios (>95%).

| Bucket | Baseline Precision | Baseline Recall | Baseline F1 |
|--------|-------------------|-----------------|-------------|
| 90-95% | 0.719 | 0.928 | 0.813 |
| **>95%** | **0.424** | **0.838** | **0.556** |

The model finds inliers well (high recall) but admits too many false positives (low precision) at >95% outlier.

---

## 2. Proposed Method: LearnableConsensus

### 2.1 Core Idea

Multi-stage correspondence networks (like MGCA-Net) predict increasingly accurate essential matrices across stages. The **epipolar distances** from these predictions contain valuable geometric consistency information that is **not exploited** by existing architectures. We propose to:

1. Compute epipolar distances from each stage's predicted E-matrix
2. Learnably weight them (not all stages are equally reliable)
3. Fuse with semantic confidence (final-stage logits) via **log-domain product fusion**
4. Use the consensus score to refine features before cross-stage aggregation

### 2.2 Module Design

```python
class LearnableConsensus(nn.Module):
    def __init__(self, num_stages=3):
        self.stage_weights = nn.Parameter(torch.ones(num_stages))  # stage importance
        self.sigma = nn.Parameter(torch.tensor(1.0))                # geo sensitivity
        self.alpha = nn.Parameter(torch.tensor(0.5))                # sem-geo balance

    def forward(self, stage_logits, stage_e_hats, xs):
        # 1. Epipolar distances per stage
        dists = [compute_epipolar_distance(e, xs) for e in stage_e_hats]
        dists = torch.stack(dists)  # [S, B, N]

        # 2. Weighted geometric consistency
        w = F.softmax(self.stage_weights, dim=0)
        mean_dist = (dists * w.view(S, 1, 1)).sum(dim=0)
        geo_score = torch.exp(-mean_dist / F.softplus(self.sigma))

        # 3. Semantic confidence
        sem_conf = torch.sigmoid(stage_logits[-1])

        # 4. Log-domain product fusion (AND-gate)
        alpha = torch.sigmoid(self.alpha)
        log_consensus = alpha * torch.log(sem_conf + 1e-6) + \
                       (1 - alpha) * torch.log(geo_score + 1e-6)
        consensus = torch.exp(log_consensus)
        return consensus
```

**Total learnable params:** ~5 scalars + CSMGC (which we also fine-tune).

### 2.3 Integration into MGCA-Net

Inserted **before CSMGC** in `MGCANet.forward()`:

```python
# Before: sub_l_input = self.CSMGC(stage_out[0], stage_out[1], stage_out[2])
# After:
consensus = self.consensus_module(res_weights, res_e_hat, data['xs'])
refined_stage2 = stage_out[2] * consensus.unsqueeze(1).unsqueeze(1)
sub_l_input = self.CSMGC(stage_out[0], stage_out[1], refined_stage2)
```

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
| Fixed product | alpha=0.5 frozen, uniform weights, sigma fixed | F1≈0.624 (zero-shot upper bound) |
| Semantic only | alpha=1.0 frozen | F1≈0.556 (back to baseline) |
| Geo only | alpha=0.0 frozen | F1≈0.572 (like pairwise) |
| **Learned full** | **All params learnable** | **F1 > 0.624 (target: +8-10pp total)** |

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

3. **Theoretical grounding:** The product fusion was empirically discovered (zero-shot MVP). Is there a stronger theoretical justification for why product works and linear combination fails?

4. **Parametric cost:** The module has only ~5 scalar parameters. Reviewers might say: "This is just a clever hand-designed feature, not deep learning."

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

4. **Is the frozen-backbone strategy a weakness or a strength?** On one hand, it shows the module is plug-and-play. On the other, "only training 5 scalars" might look insufficient.

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
