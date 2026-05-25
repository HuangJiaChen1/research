# Research Idea Report

**Direction**: Two-view correspondence learning with outlier removal, building on MGCA-Net (IJCAI 2025). Focus: dynamic graph construction and interpretability.
**Generated**: 2026-05-16
**Ideas evaluated**: 12 generated → 6 survived filtering → 3 recommended for pilots

---

## Landscape Summary

The field of two-view correspondence learning has rapidly evolved from simple GNN+MLP architectures (2021-2023) to sophisticated attention-based and hypergraph-based methods (2024-2025). Three major trends dominate:

**1. Attention Mechanism Evolution** — MGCA-Net and SC-Net represent the current pinnacle. MGCA-Net's CGA module uses contextual geometric attention; its CSMGC module enforces cross-stage geometric consensus via sparse graph neural networks (k=3 neighbors). SC-Net extends this with bilateral field adjustment. TrGa takes a different path, replacing MLPs entirely with Transformers and decoupling attribute features from global structure.

**2. Higher-Order Modeling** — EGH-Net pushes beyond pairwise graphs to hypergraphs for correspondence learning, using energy-guided optimization. HyperGCT extends dynamic hypergraphs to 3D registration (ICCV 2025).

**3. Paradigm Shifts** — Regor challenges the fundamental "outlier removal" assumption by progressively regenerating correspondences, yielding 10x more inliers.

**Critical gaps identified:**
- All 2D-2D methods (including MGCA-Net, SC-Net, TrGa) use **static KNN graphs** — no dynamic topology adaptation during inference
- Attention maps show *where* the model focuses but not *which geometric constraints* drive decisions
- **PMA-Net** (recent) proposes adaptive dynamic graph construction (ADGC) but uses affinity-based filtering rather than true geometric confidence
- **Evidential GNNs** (GEL, KDD 2025) exist for general anomaly detection but have not been applied to correspondence outlier rejection
- **Epipolar-constrained attention** exists for reconstruction (EpiS, GPS-Gaussian+) but not for correspondence pruning
- Counterfactual explanations have not been explored for correspondence classification

---

## Recommended Ideas (ranked)

### Idea 1: CounterMatch — Counterfactual Explanation for Correspondence Classification
- **Hypothesis**: Generating minimal perturbations that flip inlier/outlier predictions reveals that outlier decisions are driven by violation of specific geometric constraints, providing actionable interpretability.
- **Minimum experiment**: Implement counterfactual optimization on top of pretrained MGCA-Net. For a sample of predicted outliers, optimize the minimal feature perturbation that flips the prediction to inlier. Analyze which geometric quantities (epipolar distance, local angle consistency, scale ratio) correlate with required perturbation magnitude.
- **Expected outcome**: Success = counterfactuals cluster by geometric constraint violation type (e.g., all flipped by adjusting epipolar distance). Failure = perturbations are diffuse and uninterpretable.
- **Novelty**: 9/10 — No existing work applies counterfactual explanations to correspondence learning. Closest work is general XAI for vision; correspondence-specific counterfactuals are unexplored.
- **Feasibility**: High. Requires only a pretrained MGCA-Net model + optimization loop. No retraining needed. ~1 day implementation on 1 GPU.
- **Risk**: MEDIUM. Counterfactual optimization may be unstable for discrete classification boundaries.
- **Contribution type**: Empirical finding + diagnostic tool
- **Pilot result**: SKIPPED — needs GPU (estimated 30 min - 1 hour)
- **Reviewer's likely objection**: "Counterfactual explanations are well-studied in general XAI; what's new about applying them to correspondence?" Response: Correspondences have unique geometric structure that makes counterfactuals interpretable as geometric constraint violations — this is not true for general image classification.
- **Why we should do this**: Provides a principled way to debug *why* MGCA-Net rejects specific correspondences, which is currently impossible. Could become a standard diagnostic tool for the field.

---

### Idea 2: RobustGNN-Corr — Uncertainty-Aware Dynamic Graphs for Extreme Outlier Regimes
- **Hypothesis**: Modeling epistemic uncertainty in both node features and graph topology enables robust performance at >95% outlier ratios by abstaining from uncertain predictions and focusing on high-confidence subgraphs.
- **Minimum experiment**: Add evidential deep learning layers (Dirichlet output) to MGCA-Net's node classification head. Implement uncertainty-based graph masking: during CSMGC, mask edges where either endpoint has high predictive uncertainty. Test on YFCC100M with artificially increased outlier ratios (90%, 95%, 99%).
- **Expected outcome**: Success = maintained or improved F-score at 95% outliers vs baseline MGCA-Net; uncertainty scores correlate with true outlier probability. Failure = uncertainty estimates are uninformative (no better than confidence scores).
- **Novelty**: 8/10 — GEL (KDD 2025) applies evidential learning to general graph anomaly detection but not to correspondence outlier rejection. The combination with dynamic graph masking is novel.
- **Feasibility**: Medium. Requires modifying MGCA-Net's output layer + implementing uncertainty-based masking. ~3-5 days implementation, ~6-12 hours training on 1 GPU.
- **Risk**: MEDIUM. Evidential learning can be unstable; uncertainty estimates may not be well-calibrated for this task.
- **Contribution type**: New method
- **Pilot result**: SKIPPED — needs GPU (estimated 2-3 hours training + eval)
- **Reviewer's likely objection**: "Does adding uncertainty quantification actually improve accuracy, or just provide calibrated confidence?" Response: The key claim is that uncertainty-aware graph masking improves accuracy at extreme outlier ratios by selectively propagating information through reliable subgraphs.
- **Why we should do this**: Addresses a real gap — current methods degrade significantly at >90% outliers. Could be the first correspondence method with explicit uncertainty quantification.

---

### Idea 3: XGraph-Corr — Explainable Dynamic Graphs for Correspondence Pruning
- **Hypothesis**: Combining dynamic graph construction with built-in geometric rationale (each edge has an associated justifying constraint) makes graph dynamics interpretable by design, enabling debugging of failure cases.
- **Minimum experiment**: Replace MGCA-Net's static KNN graph in CSMGC with an edge scoring function that produces both a weight and an explanation vector (e.g., [epipolar_score, angle_consistency_score, scale_score]). Use the explanation vector to visualize which geometric constraints justify each edge. Compare pruning decisions with and without dynamic rewiring.
- **Expected outcome**: Success = edges cluster by dominant geometric constraint; dynamic rewiring improves F-score; explanations align with human intuition. Failure = explanations are noisy or rewiring hurts performance.
- **Novelty**: 8/10 — PMA-Net has adaptive graph construction but no interpretability. EGH-Net has hypergraphs but static construction. This combines dynamic graphs with explicit geometric explanations.
- **Feasibility**: High. Builds directly on MGCA-Net architecture. ~2-3 days implementation.
- **Risk**: LOW-MEDIUM. Edge scoring with explanations is a straightforward extension; main risk is that explanations may not be informative.
- **Contribution type**: New method + diagnostic tool
- **Pilot result**: SKIPPED — needs GPU (estimated 1-2 hours)
- **Reviewer's likely objection**: "Are the explanations post-hoc rationalizations or genuine causal factors?" Response: The explanations are produced by construction (each edge score is a weighted sum of explicit geometric terms), not post-hoc attribution.
- **Why we should do this**: Directly addresses both focus areas (dynamic graphs + interpretability) in a unified framework. Most likely to produce a complete paper with both quantitative and qualitative results.

---

### Idea 4: ConsistentAttn — Geometrically Constrained Attention for Interpretable Correspondence Learning
- **Hypothesis**: Replacing free-form attention in MGCA-Net's CGA with epipolar-consistency-regularized attention improves generalization while making attention patterns directly interpretable as geometric consistency scores.
- **Minimum experiment**: Add an epipolar consistency regularization term to the attention weights in MGCA-Net's CPA module. The regularization penalizes attention between correspondences that violate epipolar geometry. Evaluate on cross-descriptor generalization (train SIFT, test SuperPoint/RootSIFT).
- **Expected outcome**: Success = improved cross-descriptor mAP; attention weights correlate with epipolar distance. Failure = regularization harms flexibility, reducing in-distribution performance.
- **Novelty**: 7/10 — Epipolar-constrained attention exists for reconstruction (EpiS, GPS-Gaussian+) but not for correspondence pruning. The cross-descriptor generalization angle is novel.
- **Feasibility**: High. Simple modification to loss function. ~1-2 days implementation.
- **Risk**: LOW. Even if performance doesn't improve, the interpretability analysis is valuable.
- **Contribution type**: Empirical finding + method modification
- **Pilot result**: SKIPPED — needs GPU (estimated 1-2 hours)
- **Reviewer's likely objection**: "How is this different from just using epipolar distance as a feature?" Response: The constraint acts on attention weights, not node features, biasing the model to attend to geometrically consistent patterns globally rather than locally.

---

### Idea 5: Attn2Geo — Attention-to-Geometry Mapping for Debugging Correspondence Networks
- **Hypothesis**: Different attention heads in MGCA-Net's CGA module specialize in different geometric aspects (global epipolar consistency vs local neighborhood similarity), and this specialization can be extracted post-hoc.
- **Minimum experiment**: Apply attention head analysis to MGCA-Net's CPA module. For each head, compute correlation between attention weights and various geometric quantities (epipolar residual, local angle consistency, neighborhood density, descriptor similarity). Identify "specialist" heads.
- **Expected outcome**: Success = clear specialization emerges (e.g., Head 3 correlates with epipolar distance, Head 7 with local angle). Failure = all heads correlate equally with all metrics (no specialization).
- **Novelty**: 7/10 — Attention head analysis is common in NLP (BERTology) but rare in geometric vision. VGGT shows correspondence matching in attention but doesn't analyze head specialization for pruning.
- **Feasibility**: Very high. Pure analysis of pretrained model. ~1 day implementation, no training.
- **Risk**: LOW. Analysis is inherently low-risk; may find null result (no specialization), which is still publishable.
- **Contribution type**: Diagnostic / empirical finding
- **Pilot result**: SKIPPED — no GPU needed, can run on CPU (estimated 2-4 hours)
- **Reviewer's likely objection**: "So what if heads specialize? Does this lead to any practical improvement?" Response: Specialization suggests architectural priors that could guide future network design (e.g., replacing generic attention with geometry-specific modules).

---

### Idea 6: MetaGraph-Corr — Learning to Construct Correspondence Graphs
- **Hypothesis**: The optimal graph structure (KNN vs hypergraph vs fully-connected, with what k and thresholds) varies by scene type, and a meta-learned selector outperforms fixed heuristics across diverse scenes.
- **Minimum experiment**: Train a lightweight meta-network (e.g., small MLP or GNN) that takes initial correspondences as input and outputs graph construction parameters (graph type, k, distance threshold). The meta-network is trained to maximize downstream pruning accuracy. Evaluate on mixed indoor/outdoor scenes.
- **Expected outcome**: Success = meta-selected graphs outperform fixed KNN(k=3) on mixed benchmarks. Failure = meta-network learns trivial policy (always uses same graph type).
- **Novelty**: 7/10 — Meta-learning for graph construction exists in general NAS but not for correspondence tasks. The scene-adaptive aspect is novel.
- **Feasibility**: Medium. Requires implementing multiple graph types + meta-training loop. ~5-7 days implementation.
- **Risk**: MEDIUM-HIGH. Meta-training can be unstable; correspondence datasets may not be diverse enough for meta-learning.
- **Contribution type**: New method
- **Pilot result**: SKIPPED — needs GPU (estimated 2-3 hours)
- **Reviewer's likely objection**: "Does the overhead of meta-selection justify the gains?" Response: The meta-network is lightweight (~1K parameters) and runs once per scene, not per correspondence.

---

## Eliminated Ideas

| Idea | Reason Eliminated |
|------|-------------------|
| AdaGraph-Corr (adaptive graph rewiring) | Too similar to PMA-Net's ADGC module, which already implements affinity-based dynamic graph construction for correspondence learning |
| DHG-MGCA (dynamic hypergraph consensus) | EGH-Net already applies hypergraphs to correspondence learning; incremental to add dynamic construction without clear differentiation |
| TempGraph-Corr (temporal graph evolution) | Interesting but harder to justify novelty; overlaps with general dynamic GNN literature |
| Scale-Adaptive Graph Construction | Somewhat incremental; SC-Net already handles multi-scale features |
| GeoAttri (geometric attribution) | Standard attribution methods applied to correspondence; less novel than CounterMatch |
| RuleMine-Corr (mining geometric rules) | Very hard to validate; extracted rules may not be meaningful or generalizable |

---

## Pilot Experiment Plan

| Priority | Idea | GPU | Est. Time | Key Metric | Success Threshold |
|----------|------|-----|-----------|------------|-------------------|
| 1 | Attn2Geo | CPU | 2-4 hr | Head-geometry correlation | r > 0.5 for at least 2 heads |
| 2 | CounterMatch | GPU 0 | 30-60 min | Perturbation interpretability | >70% cluster to single constraint |
| 3 | XGraph-Corr | GPU 1 | 1-2 hr | F-score vs MGCA-Net | +1% F-score on YFCC100M subset |
| 4 | ConsistentAttn | GPU 2 | 1-2 hr | Cross-descriptor mAP | +2% mAP@5deg on SuperPoint |

**Total estimated GPU time**: ~4-6 hours (well under 8-hour budget)

---

## Suggested Execution Order

1. **Start with Attn2Geo** (no GPU needed, lowest risk, immediate insights)
2. **Run CounterMatch pilot** if Attn2Geo shows head specialization (validates interpretability angle)
3. **Run XGraph-Corr pilot** as the main method contribution
4. **Run ConsistentAttn pilot** if cross-descriptor generalization is a priority

---

## Next Steps

- [ ] Run Attn2Geo analysis on pretrained MGCA-Net (no training needed)
- [ ] Obtain pretrained MGCA-Net weights or train from scratch
- [ ] Run CounterMatch and XGraph-Corr pilots in parallel on available GPUs
- [ ] If pilots succeed, invoke `/novelty-check` for deep validation of top idea
- [ ] If pilots succeed, invoke `/research-review` for external critical feedback
- [ ] If pilots succeed, invoke `/research-refine-pipeline` to develop full method and experiment plan
