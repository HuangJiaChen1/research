# Research Idea Report — REVISED

**Direction**: Two-view correspondence learning with outlier removal, building on MGCA-Net (IJCAI 2025). Focus: dynamic graph construction and interpretability.
**Generated**: 2026-05-16
**Revised**: 2026-05-16 (after codebase verification)

---

## Critical Findings from Code Verification

Before revising the ideas, here is what the actual MGCA-Net codebase reveals:

### 1. Attention is NOT Multi-Head and NOT Node-Wise
- The `CPT` module (Context Position Attention) computes **single-head, channel-wise attention**.
- Attention matrix shape: `(batch, 128, 128)` — it mixes information **across feature channels**, not across correspondence nodes.
- There is **no `num_heads` parameter anywhere** in the codebase.
- **Implication**: Any idea assuming "attention heads" or "attention over nodes" is fundamentally mismatched with the architecture.

### 2. Graph Propagation is Decoupled from Attention
- The `GNN` module (`knn_num=6`) builds a static k-NN graph on feature space and performs graph convolution.
- The `CGA` module (attention) operates on the per-node features **after** graph propagation, not on the graph structure itself.
- **Implication**: Ideas conflating "attention" with "graph structure" need to be split into two separate interventions.

### 3. Geometric Information Enters via a Bias Term
- `graph_context_position = matmul(q, graph_context.transpose)` where `graph_context` is derived from 2D coordinates.
- This is added as a **bias to attention scores**, not as a separate attention mechanism.
- **Implication**: "Epipolar-constrained attention" should target this bias term, not attention heads.

### 4. Model Has 3 Stages (Not 2)
- `subnetwork_init`: initial feature extraction (CGA1 → CGA2)
- `subnetwork[0]`: first refinement stage (CGA1 → CGA2)
- `subnetwork[1]`: second refinement stage (CGA1 → CGA2)
- Each stage outputs `logits` (per-correspondence weights) and `e_hat` (essential matrix estimate).
- Final fusion: `CSMGC` merges all three stage outputs.
- **Implication**: Multi-stage analysis is possible, but ideas should clarify which stage they modify.

### 5. Output is a Weighted 8-Point Algorithm, Not Direct Classification
- The network outputs **soft weights** (logits → sigmoid) for each correspondence.
- These weights feed into `weighted_8points()` to estimate the essential matrix.
- The loss is a combination of:
  - Classification-style loss on the weights
  - Epipolar symmetric distance (`batch_episym`)
- **Implication**: "Outlier removal" is implicit via down-weighting, not explicit binary classification. Counterfactual ideas must perturb features to change weights, not binary labels.

---

## Revised Ideas (ranked by feasibility × novelty)

### Idea 1 (REVISED): XGraph-Corr — Explainable Dynamic Graphs for Correspondence Pruning
**Original ranking**: #3 | **Revised ranking**: #1

- **Why promoted**: This is the **most compatible** with actual MGCA-Net architecture.
- **What changes**: Replace the static `knn` in `get_graph_feature()` with a **learned, interpretable edge scorer**.
- **Implementation**: Add an edge scoring network that takes two correspondences and outputs:
  - A scalar edge weight
  - A 3D explanation vector: `[epipolar_consistency, angle_consistency, scale_consistency]`
  - The final edge weight is a weighted sum of these terms (interpretable by design).
- **Where to modify**: `core/MGCA.py` lines 119-137 (`get_graph_feature`), `GNN` class (lines 170-209).
- **Feasibility**: Very High. Pure architecture change, no retraining of existing weights needed (can fine-tune).
- **Novelty**: 8/10 — PMA-Net has adaptive graphs but no interpretability. This combines both.
- **Risk**: Low-Medium.

---

### Idea 2 (REVISED): ConsistentAttn — Geometrically Constrained Position Bias for Attention
**Original ranking**: #4 | **Revised ranking**: #2

- **Why reframed**: Original assumed multi-head attention. Now targets the `graph_context_position` bias term.
- **Hypothesis**: Adding an epipolar consistency regularization to the `graph_context_position` term improves generalization and makes the geometric bias directly interpretable.
- **Implementation**:
  - Compute per-correspondence epipolar distance.
  - Add a loss term: `L_epi = mean(epi_distance * attention_weight)` — penalize high attention to epipolar-violating correspondences.
  - Alternatively, directly modulate `graph_context_position` by epipolar distance: `graph_context_position = graph_context_position * f(epi_distance)`.
- **Where to modify**: `core/MGCA.py` `CPT.forward` (lines 264-283) and `core/loss.py`.
- **Feasibility**: High. Simple loss modification or attention bias modulation.
- **Novelty**: 7/10 — Epipolar-constrained attention exists for reconstruction but not for correspondence pruning via position bias.
- **Risk**: Low.

---

### Idea 3 (REVISED): CounterMatch — Counterfactual Explanation for Correspondence Weighting
**Original ranking**: #1 | **Revised ranking**: #3

- **Why demoted**: Still feasible, but the original framing assumed flipping "inlier/outlier predictions." MGCA-Net outputs **soft weights**, not hard labels.
- **Reframed hypothesis**: Generating minimal perturbations to correspondence features that flip the model's weight ranking (i.e., an outlier-weighted correspondence becomes inlier-weighted) reveals which geometric constraints drive the weighting decisions.
- **Implementation**:
  - Load pretrained MGCA-Net.
  - Select correspondences with low weights (predicted outliers).
  - Optimize minimal perturbation to `xs` (coordinates) that increases the weight above a threshold.
  - Analyze which geometric quantities (epipolar distance, local angle, scale) correlate with required perturbation magnitude.
- **Where to modify**: New script, no model changes needed.
- **Feasibility**: High. No retraining needed.
- **Novelty**: 9/10 — No existing work on counterfactuals for correspondence learning.
- **Risk**: Medium. Optimization may be unstable; soft weights have no natural "flip boundary."

---

### Idea 4 (REVISED): StageAttn — Stage-Wise Attention Evolution Analysis
**Original ranking**: N/A (replaces Idea 5) | **Revised ranking**: #4

- **Replaces**: Idea 5 (Attn2Geo), which is infeasible due to lack of multi-head attention.
- **Hypothesis**: Different stages (init / stage 0 / stage 1) and different CGAs within a stage (CGA1 vs CGA2) exhibit different attention behaviors (entropy, geometric bias ratio, sharpness), revealing a coarse functional specialization.
- **What we already found**:
  | Module | Entropy | Geo Ratio | Max Weight |
  |---|---|---|---|
  | init.CGA1 | 0.018 | 0.95 | 0.992 |
  | init.CGA2 | 0.325 | 1.57 | 0.880 |
  | stage0.CGA1 | 0.044 | 2.03 | 0.982 |
  | stage0.CGA2 | 0.480 | 1.16 | 0.835 |
  | stage1.CGA1 | 0.020 | 1.52 | 0.992 |
  | stage1.CGA2 | 0.377 | 0.96 | 0.878 |
- **Interpretation**: CGA1 modules are "selectors" (low entropy, sharp focus), CGA2 modules are "redistributors" (high entropy, diffuse). Geometric bias peaks at stage0.CGA1.
- **Where to modify**: Analysis script only (`verify_idea5_attn2geo.py` extended).
- **Feasibility**: Very High. Pure analysis.
- **Novelty**: 6/10 — Attention analysis is common; stage-wise analysis in geometric vision is rarer.
- **Risk**: Low.

---

### Idea 5 (REVISED): RobustGNN-Corr — Uncertainty-Aware Dynamic Graphs
**Original ranking**: #2 | **Revised ranking**: #5

- **Why demoted**: Still valid, but modifying the output layer + implementing uncertainty-based graph masking is more invasive than Ideas 1-2.
- **Hypothesis unchanged**: Modeling epistemic uncertainty enables robust performance at >95% outlier ratios by abstaining from uncertain predictions.
- **Implementation**:
  - Replace the final `nn.Conv2d(128, 1)` with an evidential layer (Dirichlet output).
  - Add uncertainty-based masking in the `GNN` module: mask edges where either endpoint has high predictive uncertainty.
- **Where to modify**: `core/MGCA.py` (output layers, GNN), `core/loss.py` (evidential loss).
- **Feasibility**: Medium. Requires modifying the output head and retraining.
- **Novelty**: 8/10.
- **Risk**: Medium. Evidential learning can be unstable.

---

### Idea 6 (REVISED): MetaGraph-Corr — Learning to Construct Correspondence Graphs
**Original ranking**: #6 | **Revised ranking**: #6

- **Unchanged**. Still valid but highest effort.
- **Implementation**: Meta-network selects graph construction parameters (graph type, k, distance threshold) per scene.
- **Feasibility**: Medium. Requires implementing multiple graph types + meta-training loop.
- **Risk**: Medium-High.

---

## Eliminated Ideas (Updated)

| Idea | Original Status | New Status | Reason |
|------|-----------------|------------|--------|
| **Attn2Geo** (Idea 5) | Recommended | **ELIMINATED** | MGCA-Net has no multi-head attention. Core hypothesis untestable. Replaced by StageAttn (Idea 4). |
| AdaGraph-Corr | Eliminated | Eliminated | Too similar to PMA-Net. |
| DHG-MGCA | Eliminated | Eliminated | EGH-Net already applies hypergraphs. |

---

## Revised Pilot Experiment Plan

| Priority | Idea | GPU | Est. Time | Key Metric | Success Threshold |
|----------|------|-----|-----------|------------|-------------------|
| 1 | **XGraph-Corr** | GPU 0 | 2-3 hr | F-score vs MGCA-Net | +1% F-score on YFCC100M subset |
| 2 | **ConsistentAttn** | GPU 1 | 1-2 hr | Cross-descriptor mAP | +2% mAP@5deg on SuperPoint |
| 3 | **CounterMatch** | GPU 2 | 1-2 hr | Perturbation interpretability | >70% cluster to single constraint |
| 4 | **StageAttn** | CPU | 2-4 hr | Stage-wise pattern divergence | Significant difference in entropy/geo_ratio across stages |

**Total estimated GPU time**: ~4-7 hours

---

## Suggested Execution Order

1. **Start with StageAttn** (CPU, no training, validates understanding of attention mechanism)
2. **Run XGraph-Corr pilot** (main method contribution, highest feasibility)
3. **Run ConsistentAttn pilot** in parallel (simple loss modification)
4. **Run CounterMatch pilot** if StageAttn shows interesting patterns

---

## Next Steps

- [ ] Update `IDEA_REPORT.md` to reflect these revisions
- [ ] Run StageAttn analysis on full val set (no GPU needed)
- [ ] Implement XGraph-Corr edge scorer prototype
- [ ] Implement ConsistentAttn loss modification
- [ ] If pilots succeed, invoke `/novelty-check` for deep validation
