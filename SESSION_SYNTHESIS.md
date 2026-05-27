# Session Synthesis: MGCA-Net Architecture Audit & Idea Validation

**Session Date**: 2026-05-23 ~ 2026-05-25
**Codebase**: `/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/`
**Dataset**: YFCC100M + SUN3D (SIFT-2000)

---

## Part 1: Critical Architecture Findings (Still Valid)

### Finding 1.1: No Multi-Head Attention Exists

**Location**: `core/MGCA.py` lines 239-283 (`CPT` class)

- **Attention shape**: `(batch, 128, 128)` — channel-wise, not node-wise
- **No `num_heads` parameter** anywhere
- **Geometric info enters via bias term**: `graph_context_position = matmul(q, graph_context.T)`

**Impact**: Any idea assuming "multi-head attention" or "attention over nodes" is **fundamentally mismatched**.

### Finding 1.2: Graph Propagation is Decoupled from Attention

| Module | Function | Graph Type |
|--------|----------|------------|
| `GNN` | `get_graph_feature(x, k=6)` | Static KNN on **feature space** |
| `CGA` (CPT) | Channel-wise attention + coordinate bias | No graph structure |
| `CSMGC` | `get_graph_feature(x, k=3)` | Static KNN on **feature space** |

### Finding 1.3: Three-Stage Iterative Architecture

**Location**: `core/MGCA.py` lines 625-675 (`MGCANet`)

```python
self.iter_num = 2  # Hardcoded
# Creates: subnetwork_init + subnetwork[0] + subnetwork[1]
```

Each stage outputs: `logits` (B,N), `e_hat` (B,9), `residual` (B,1,N,1).

### Finding 1.4: Output is Soft Weights, Not Classification

- Weights feed into `weighted_8points()` for essential matrix estimation
- Loss = classification loss + `batch_episym`
- **No hard inlier/outlier labels** during inference

### Finding 1.5: e_hat Quality by Stage (New)

**Source**: `verify_epipolar_assumption.py` (500 samples)

| Stage | >95% Outlier E Error | 90-95% Outlier E Error | Verdict |
|-------|---------------------|------------------------|---------|
| Init | **34.2°** | 16.2° | ❌ Unusable for epipolar constraints |
| Stage0 | 16.7° | 7.4° | ⚠️ Marginal |
| Stage1 | 12.0° | 5.0° | ✅ Reliable |
| Final | 12.0° | 5.1° | ✅ Reliable |

**Key insight**: E estimation improves dramatically across stages. Any method relying on E must **not** be applied in Init stage.

---

## Part 2: Updated Mental Model — What We Learned

### The Real Bottleneck is Precision, Not Recall

**Source**: `verify_epipolar_assumption.py`

| Bucket | Precision | Recall | F-score |
|-------|----------|--------|---------|
| 80-90% | 0.853 | 0.947 | 0.896 |
| 90-95% | 0.671 | 0.923 | 0.772 |
| **>95%** | **0.378** | **0.838** | **0.506** |

**Interpretation**: At >95% outlier, model retains 83.8% of inliers (recall is fine) but **only 37.8% of predicted inliers are true inliers** (precision collapsed). The problem is **false positives**, not false negatives.

**Implication**: Methods targeting "helping find inliers" (recall) are attacking the wrong problem. We need methods that **reduce false positives** (improve precision).

---

## Part 3: MVP Validation Results

### Test 1: GeoKNN — ❌ FAILED

**Script**: `mvp_test1_geoknn.py` (200 samples)

**Hypothesis**: Mixing geometric compatibility (`|d1-d2|`) into KNN distance improves graph quality.

**Result**:

| Bucket | Base F1 | GeoKNN F1 | Δ |
|--------|---------|-----------|---|
| >95% | 0.789 | 0.787 | **-0.0019** |

**Why it failed**: The `|d1-d2|` compatibility metric assumes distance preservation between views. This holds for pure rotation but **fails under projective distortion** (YFCC100M wide baseline general scenes). GeoKNN penalizes genuine inliers with large parallax.

**Verdict**: Geometric compatibility based on local distance metrics is **not viable** for general two-view correspondence.

### Test 2: EpiAttn (Post-hoc Modulation) — ❌ FAILED

**Script**: `mvp_test2_epiattn.py` (200 samples, 5 scales tested)

**Hypothesis**: Epipolar distance modulation of logits improves performance in high-outlier regimes.

**Result** (best scale=5.0):

| Bucket | Δ F-score |
|--------|-----------|
| >95% | **+0.0024** (+0.24%) |

**Why it failed**: Even with accurate E (Stage1/Final error 5-12°), explicit epipolar modulation provides **negligible improvement**. The model's implicit learning from epipolar loss already captures similar geometric patterns. Explicit bias is redundant.

**Verdict**: Explicit epipolar constraints as attention/graph bias are **not a productive direction** for MGCA-Net enhancement.

### Test 3: Consensus Filtering — ✅ SUCCESS

**Script**: `pilot_consensus_filter.py` (500 samples, zero-shot)

**Hypothesis**: A correspondence that is truly an inlier should have consistent epipolar distances across all three stages. If it's an outlier, at least one stage will show large epipolar distance. Using this cross-stage geometric consistency as a confidence discount factor should reduce false positives (improve precision) in high-outlier regimes.

**Result** (best method: `product` = semantic_confidence × geometric_consistency):

| Bucket | Base F1 | Consensus F1 | Δ | Base Prec | Consensus Prec | Δ |
|--------|---------|--------------|---|-----------|----------------|---|
| 90-95% | 0.813 | **0.843** | **+3.1pp** | 0.719 | **0.787** | **+6.8pp** |
| **>95%** | **0.556** | **0.624** | **+6.8pp** | **0.424** | **0.516** | **+9.2pp** |

**Why it works**:
- Product method combines **semantic confidence** (model's belief) with **geometric consistency** (cross-stage epipolar distance variance). Both signals are necessary — simple variance (−0.2pp) or min_dist (+0.4pp) alone are insufficient.
- Recall cost is modest (−3~5pp), making this a favorable precision-recall trade-off.
- Zero-shot validation (no retraining) already shows strong signal.

**Key insight**: The `pairwise` method achieves precision 0.732 (+30.8pp) but recall collapses to 0.512 (−36pp). This confirms that **hard agreement thresholds are too conservative**. Soft weighted fusion (product) is the right balance.

**Verdict**: ✅ **Validated as primary research direction**. Next step: integrate as learnable module into CSMGC and fine-tune.

### Combined Implication

GeoKNN and EpiAttn target "helping the model find inliers" — but **the model already finds inliers well**. The real problem (precision collapse) requires a different class of solutions. **Consensus Filtering directly targets precision** by cross-stage geometric verification, and the MVP confirms it works.

| Direction | Status | Why Not |
|-----------|--------|---------|
| **GeoKNN** (geometric compatibility KNN) | ❌ Eliminated (MVP) | `\|d1-d2\|` assumption fails under projective distortion; MVP shows −0.2% F-score. |
| **EpiAttn** (epipolar attention bias) | ❌ Eliminated (MVP) | Even with accurate E, post-hoc modulation gives +0.24% F-score; implicit epipolar loss already sufficient. |
| **Consensus Filtering** | ✅ **Primary Direction** | Zero-shot MVP shows +6.8pp F-score at >95% outlier; targets precision collapse directly. |
| **PGE** (Progressive Geometric Enhancement) | ❌ Eliminated | Combines two failed components. |
| **PGP** (confidence-based graph rewiring) | ❌ Eliminated (pilot) | Soft-reweight Δ≈0; hard-prune causes death spiral; feature-space KNN already near-optimal. |
| **CounterMatch** | ❌ Abandoned by user | Scope too broad for a first paper ("太大了，不敢做"). |
| **Attn2Geo** (Idea 5) | ❌ Eliminated | MGCA-Net has no multi-head attention; core hypothesis untestable. |
| **ConsistentAttn** | ⚠️ On hold | Simple loss modification, but targets wrong problem (recall, not precision). |
| **RobustGNN-Corr** (evidential learning) | ⚠️ On hold | Requires retraining + modifying output head; evidential learning unstable; effort high. |
| **StageAttn** (stage-wise attention analysis) | ⚠️ On hold | Pure analysis, no method contribution; novelty 6/10. |
| **Pure XGraph-Corr** | ❌ Eliminated | Too similar to PMA-Net ADGC; needs stronger differentiation. |
| **MetaGraph-Corr** | ⚠️ On hold | High effort (5–7 days), no evidence graph construction is the bottleneck. |
| **AdaGraph-Corr** | ❌ Eliminated | Too similar to PMA-Net. |
| **DHG-MGCA** | ❌ Eliminated | EGH-Net already applies hypergraphs. |

---

## Part 5: Active Direction — Learnable Consensus Filtering

**Status**: ✅ MVP validated. Proceeding to implementation.

### Core Idea

Use cross-stage geometric consistency as a learnable confidence refinement mechanism. A correspondence is reliable only if:
1. The model assigns it high semantic confidence (high logit)
2. Its epipolar distance is consistently low across all three stages

### Architecture

```
Stage Init  → logits_0, e_hat_0 ──┐
Stage 0     → logits_1, e_hat_1 ──┼→ LearnableConsensus → refined_logits → CSMGC
Stage 1     → logits_2, e_hat_2 ──┘
```

**LearnableConsensus module** (CORRECTED DESIGN):

```python
class LearnableConsensus(nn.Module):
    def __init__(self, num_stages=3):
        super().__init__()
        self.stage_weights = nn.Parameter(torch.ones(num_stages))
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, stage_logits, stage_e_hats, xs):
        S = len(stage_e_hats)
        
        # 1. Epipolar distances per stage
        dists = torch.stack([
            compute_epipolar_distance(e, xs) for e in stage_e_hats
        ])  # [S, B, N]
        
        # 2. Weighted geometric consistency (learnable stage weights)
        w = F.softmax(self.stage_weights, dim=0)  # [S]
        mean_dist = (dists * w.view(S, 1, 1)).sum(0)  # [B, N]
        geo_score = torch.exp(-mean_dist / F.softplus(self.sigma))  # [B, N]
        
        # 3. Semantic confidence from final stage
        sem_conf = torch.sigmoid(stage_logits[-1])  # [B, N]
        
        # 4. Product fusion in log domain (NOT linear combination!)
        # Zero-shot MVP proved product >> linear. Keep product structure.
        alpha = torch.sigmoid(self.alpha)  # scalar
        log_consensus = alpha * torch.log(sem_conf + 1e-6) + \
                       (1 - alpha) * torch.log(geo_score + 1e-6)
        consensus = torch.exp(log_consensus)  # = sem^alpha * geo^(1-alpha)
        
        return consensus  # [B, N]
```

**CRITICAL**: The zero-shot MVP proved `product` (semantic * geometric) works (+6.8pp). `Linear combination` (α*sem + (1-α)*geo) does NOT recover product behavior. Must use **log-domain product fusion**.

**Module placement**: Insert in `MGCANet.forward` BEFORE CSMGC. Use `refined_logits` to weight/modulate the final stage output before CSMGC fusion.

### MVP Results (Zero-Shot)

| Bucket | Base F1 | +Consensus | Δ | Base Prec | +Consensus Prec | Δ |
|--------|---------|------------|---|-----------|-----------------|---|
| 90-95% | 0.813 | 0.843 | +3.1pp | 0.719 | 0.787 | +6.8pp |
| >95% | 0.556 | **0.624** | **+6.8pp** | 0.424 | **0.516** | **+9.2pp** |

**Key**: Simple variance/min_dist consensus barely works (+0.2~0.4pp). The `product` of semantic confidence and geometric consistency is the only effective formulation.

### Implementation Plan

| Phase | Task | Est. Time | Risk | GPU? |
|-------|------|-----------|------|------|
| 1 | Write `LearnableConsensus` module | 4-6 hours | Low | No |
| 2 | Frozen-backbone fine-tune (consensus+CSMGC only) | 2-4 hours | Low | Yes |
| 3 | Ablation: semantic-only, geo-only, fixed-product, learned | 4-6 hours | Low | Yes |
| 4 | SUN3D validation | 2-3 hours | Medium | Yes |
| 5 | Cross-model migration (OANet plug-in) | 4-8 hours | Medium | Yes |

### Training Config (Phase 2)

```python
# Freeze all backbone parameters
for param in [model.subnetwork_init, model.subnetwork]:
    param.requires_grad = False

# Only train consensus + final fusion
optimizer = torch.optim.Adam([
    {'params': model.consensus_module.parameters(), 'lr': 1e-4},
    {'params': model.CSMGC.parameters(), 'lr': 1e-5},
], weight_decay=0)

# 50K steps expected, batch_size=32
# Expected time: single V100 ~ 2-4 hours
```

### Ablation Checklist

| Ablation | Config | Expected Result |
|----------|--------|-----------------|
| Baseline (frozen) | No consensus, use pretrained as-is | F1 = 0.556 @ >95% |
| Fixed product | α=0.5, uniform stage weights, σ fixed | F1 ≈ 0.624 @ >95% (zero-shot) |
| Semantic only | α=1.0, ignore geometric | F1 ≈ 0.556 (back to baseline) |
| Geo only | α=0.0, ignore semantic | F1 ≈ 0.572 (like pairwise) |
| Learned full | Learned α, stage weights, σ | F1 > 0.624 @ >95% (target: +8-10pp total) |

### Alternative Directions (On Hold)

| Direction | Status | Why |
|-----------|--------|-----|
| Top-K Adaptive E Estimation | ⏸️ On hold | May be combined with consensus later |
| Hard Negative Mining Loss | ⏸️ On hold | Requires full retraining; try if consensus plateaus |
| Cross-Sample Consensus | ⏸️ Future work | Use batch-level statistics for K estimation

---

## Part 6: File Inventory

### Generated Files

| File | Purpose | Status |
|------|---------|--------|
| `verify_idea5_attn2geo.py` | Attention stats extraction | ✅ Complete |
| `pilot_pgp.py` | PGP pilot (confidence rewiring) | ✅ Complete |
| `verify_epipolar_assumption.py` | e_hat quality by outlier bucket | ✅ Complete |
| `mvp_test1_geoknn.py` | GeoKNN MVP (200 samples) | ✅ Complete — FAILED |
| `mvp_test2_epiattn.py` | EpiAttn MVP (200 samples, 5 scales) | ✅ Complete — FAILED |
| `pilot_consensus_filter.py` | Consensus Filtering MVP (500 samples, zero-shot) | ✅ Complete — SUCCESS (+6.8pp @ >95%) |
| `IDEA_REPORT_REVISED.md` | Pre-MVP idea ranking | ⚠️ Outdated (GeoKNN/EpiAttn assumed feasible) |
| `SESSION_SYNTHESIS.md` | This file | 🔄 Updated |

---

## Part 7: Open Questions

1. **✅ Answered: Cross-stage consistency diverges more on false positives?** Yes. MVP confirms that `product` consensus (semantic × geometric) reduces false positives by 9.2pp at >95% outlier. False positives have inconsistent epipolar distances across stages.
2. **How much improvement is achievable with learnable fusion?** Zero-shot gave +6.8pp F-score. Can a learnable module (with stage weights, learned σ, learned α) push this to +10pp?
3. **Where to integrate consensus?** CSMGC fusion (post-stage) vs. per-stage refinement? Per-stage might provide iterative benefit but increases complexity.
4. **SUN3D generalization**: Does consensus filtering generalize to indoor scenes (SUN3D) where epipolar geometry differs?
5. **Training stability**: Frozen-backbone fine-tune vs. end-to-end? If frozen works, we can leverage pretrained weights without full retraining.

---

## Next Steps

- [x] ✅ MVP validation: Consensus Filtering (+6.8pp @ >95% outlier)
- [x] ✅ Design corrected: log-domain product fusion (not linear combination)
- [x] ✅ Write `NeuralConsensus` module + integrate into MGCA-Net (COMPLETED 2026-05-26)
- [x] ✅ Frozen-backbone fine-tune on YFCC100M (COMPLETED 2026-05-27)
- [x] ✅ Ablation: fixed-product vs neural (COMPLETED 2026-05-27)
- [ ] Identity ablation (CSMGC retraining vs CF effect) — NOT STARTED
- [ ] SUN3D validation (2-3 hours GPU)
- [ ] Cross-model plug-in (OANet) (4-8 hours GPU, optional)
- [ ] Paper writing if all experiments succeed

---

## Part 8: Major Redesign — NeuralConsensus (2026-05-26 Session)

### 8.1 Why LearnableConsensus Was Abandoned

User raised a fundamental critique: the original `LearnableConsensus` (~5 scalar parameters: `alpha`, `sigma`, `stage_weights`) was **not true learning** — it was merely tuning hyperparameters within a predefined function family:

```
f_θ(s,g) = s^α · g^(1−α) · exp(−d/σ)
```

This is insufficient for a paper-worthy contribution. The user explicitly requested: "Can we design the consensus filter as a genuine neural module, not a handcrafted method?"

### 8.2 New Design: NeuralConsensus

**Architecture**: 3-layer MLP (`8 → 64 → 32 → 1 + Sigmoid`), ~2.7K parameters

**Input features** (8-dim per correspondence):
| Feature | Description |
|---------|-------------|
| `sem_conf` | Semantic confidence (sigmoid of final logits) |
| `exp(-geo_mean)` | Geometric score (exponential decay) |
| `geo_mean` | Cross-stage average epipolar distance |
| `geo_var` | Cross-stage variance (stability measure) |
| `rel_pos` | Relative position in global distribution (z-score) |
| `dist_init` | Initial stage epipolar distance |
| `dist_final` | Final stage epipolar distance |
| `trend` | Distance evolution trend (final − init) |

**Key advantage**: The MLP learns a **non-linear fusion function** from data, not constrained to any predefined parametric family. This is true learning.

### 8.3 Baseline: FixedProductConsensus

Handcrafted zero-parameter baseline implementing the zero-shot MVP formula:
```python
consensus = sigmoid(logits_final) * exp(-mean_epipolar_distance)
```

**Purpose**: Direct comparison between handcrafted heuristic and learned neural fusion.

### 8.4 Critical Bug Fixes

| Bug | Impact | Fix |
|-----|--------|-----|
| Broadcast dimension | OOM (30GB allocation) | `unsqueeze(1).unsqueeze(1)` → `unsqueeze(1).unsqueeze(-1)` |
| AMP + `linalg.eigh` | `NotImplementedError` for Half + NaN/Inf | **Removed AMP entirely** from pilot script |

### 8.5 Training Infrastructure

- **Script**: `pilot_neural_consensus.py` (replaces `train_consensus.py` for pilots)
- **Modes**: `fixed_product` vs `mlp`
- **Default**: 30K iters, batch_size=16, val every 5K
- **Best model criterion**: >95% bucket F1

### 8.6 Narrative Direction

- **Module name**: NeuralConsensus (working name)
- **Paper framing**: Plug-in module (EGCG) with cross-model validation
- **Key argument**: CSMGC operates in 128-dim feature space without geometric coordinates — it cannot learn epipolar consistency. NeuralConsensus injects explicit geometric supervision.
- **Cross-architecture**: If pilots succeed, validate on OANet/ACNe to show generalizability

### 8.7 Files Changed in This Session

| File | Action | Notes |
|------|--------|-------|
| `core/MGCA.py` | Modified | Deleted `LearnableConsensus`, added `FixedProductConsensus` + `NeuralConsensus`, fixed broadcast bug |
| `core/config.py` | Modified | Added `consensus_mode` argument |
| `core/pilot_neural_consensus.py` | Created | Frozen-backbone training comparing fixed_product vs mlp |
| `CONSENSUS_TRAINING_GUIDE.md` | Updated | Reflects new design |
| `NOVELTY_CHECK_REPORT.md` | Updated | Reflects new design |
| `RESEARCH_BRIEF_FOR_REVIEW.md` | Updated | Reflects new design |

### 8.8 Open Questions (Pre-Experiment)

1. Will NeuralConsensus (MLP) outperform FixedProductConsensus (handcrafted) at >95% outlier?
2. How much of the MLP's improvement comes from non-linearity vs. additional features (variance, trend, rel_pos)?
3. Should we add cross-architecture validation (OANet) before or after MGCA-Net experiments?
4. How to frame the design evolution (scalar → MLP) in the paper without looking like "moving goalposts"?

---

## Part 9: Pilot Experiment Results (2026-05-27)

### 9.1 Experiment Setup

- **Script**: `core/pilot_neural_consensus.py`
- **Backbone**: Frozen (subnetwork_init + subnetwork)
- **Trainable**: consensus_module + CSMGC
- **Dataset**: YFCC100M SIFT-2000 (train/val)
- **Training**: 30K steps, batch_size=16, val every 5K
- **Best model**: Selected by >95% bucket F1

### 9.2 FixedProductConsensus Results (Handcrafted Baseline)

**Best checkpoint**: Step 10000 (saturated, no further improvement)

| Metric | Baseline (No CF) | FixedProduct (Trained) | Δ |
|--------|-----------------|------------------------|---|
| **Global F1** | ~0.80 | **0.8549** | +5.5pp |
| **Global mAP** | — | **0.8263** | — |
| **>95% Precision** | 0.424 | **0.6072** | **+18.3pp** |
| **>95% Recall** | 0.838 | 0.7097 | -12.8pp |
| **>95% F1** | **0.556** | **0.6415** | **+8.5pp** |

**Breakdown by bucket**:

| Bucket | Baseline F1 | FixedProduct F1 | Δ |
|--------|-------------|-----------------|---|
| 0-50% | — | 0.9713 | — |
| 50-75% | — | 0.9501 | — |
| 75-90% | — | 0.9187 | — |
| 90-95% | 0.813 | 0.8483 | +3.5pp |
| **>95%** | **0.556** | **0.6415** | **+8.5pp** |

### 9.3 NeuralConsensus Results (MLP)

**Best checkpoint**: Step 15000

| Metric | FixedProduct | NeuralConsensus (MLP) | Δ |
|--------|-------------|----------------------|---|
| **Global F1** | 0.8549 | **0.8570** | +0.2pp |
| **Global mAP** | 0.8263 | **0.8289** | +0.3pp |
| **>95% Precision** | 0.6072 | **0.6020** | -0.5pp |
| **>95% Recall** | 0.7097 | 0.7124 | +0.3pp |
| **>95% F1** | **0.6415** | **0.6393** | **-0.2pp** |

### 9.4 Key Findings

1. **MLP did NOT outperform handcrafted**: >95% F1 gap is only 0.2pp (within noise). MLP converged to essentially the same performance.
2. **Handcrafted formula captures the dominant signal**: `sem * exp(-geo_mean)` is near-optimal for this feature space.
3. **Additional features (variance, trend, rel_pos) provide no independent signal**: They are either redundant with `geo_mean` or swamped by noise at >95% outlier.
4. **Precision-recall tradeoff is real and intentional**: Baseline had recall=0.838 ("accept everything"), FixedProduct has precision=0.607 ("filter false positives"). F1 improves because precision gain outweighs recall loss.
5. **Saturation at ~10K steps**: >95% F1 does not improve beyond step 10000. Global F1 continues to rise due to easier buckets.

### 9.5 Why MLP Failed

| Hypothesis | Evidence | Verdict |
|-----------|----------|---------|
| Feature redundancy | 6 extra features add no value | ✅ Confirmed |
| MLP too shallow | Deeper might help, but unlikely | ⚠️ Possible but low ROI |
| Frozen backbone limits expressivity | MLP can only reweight stage_out[2] | ✅ Major factor |
| Task ceiling at ~0.64 F1 | E-matrix noisy at >95% outlier | ✅ Likely |
| Need per-stage integration | Current design only filters final stage | ⚠️ Unexplored |

### 9.6 Critical Gap: Identity Ablation NOT Done

**We cannot yet prove that the +8.5pp improvement comes from CF itself vs. CSMGC retraining.**

Required experiment:
- **Identity/bypass ablation**: consensus = 1.0 (no filtering), but CSMGC is retrained
- If identity ≈ baseline (0.556): CF is the main driver ✅
- If identity ≈ FixedProduct (0.641): CSMGC retraining is the main driver ⚠️

**This ablation is BLOCKING for paper claims.**

### 9.7 Implications for Paper Direction

**Original hypothesis (discredited)**: "Neural learning > handcrafted heuristic"

**Alternative narratives**:

| Narrative | Strength | Risk |
|-----------|----------|------|
| A: "Handcrafted insight identifies optimal fusion principle" | Honest, strong zero-shot result | Workload too small for top venue |
| B: "Explicit geometric consistency as plug-in module" | Generalizable beyond MGCA-Net | Needs cross-architecture validation |
| C: "CSMGC retraining reveals untapped capacity" | Interesting but credits CSMGC, not CF | Conflicts with CF narrative |

**User's concern**: Does not want a paper that just says "sem+geo fusion improves F1." Wants something bigger and more substantial.

### 9.8 Open Questions (Post-Experiment)

1. **Is the +8.5pp real?** Identity ablation required to disentangle CF vs. CSMGC retraining.
2. **Can we beat the ~0.64 ceiling?** Current designs hit a wall. Need new ideas (per-stage integration? attention? evidential?)
3. **Cross-architecture generalization**: Does CF improve OANet/ACNe? If yes, contribution becomes "general principle."
4. **SUN3D transfer**: Does CF generalize to indoor scenes?
5. **How to scale up the idea**: User explicitly wants to "做大做好" — not just a simple fusion trick.
6. **Theoretical grounding**: Why does product fusion work? Independent noise assumption? MAP estimation? Needs formal justification.
