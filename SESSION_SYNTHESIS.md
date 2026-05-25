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
- [ ] Write `LearnableConsensus` module + integrate into MGCA-Net (4-6 hours)
- [ ] Frozen-backbone fine-tune on YFCC100M (2-4 hours GPU)
- [ ] Ablation: semantic-only, geo-only, fixed-product, learned (4-6 hours GPU)
- [ ] SUN3D validation (2-3 hours GPU)
- [ ] Cross-model plug-in (OANet) (4-8 hours GPU, optional)
- [ ] Paper writing if all experiments succeed
