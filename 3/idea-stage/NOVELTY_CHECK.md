# Novelty Check Report: IPG-SAD

## Proposed Method
Inlier-Prior Guided MoE with Scale-Aware Decomposition (IPG-SAD): Using predicted inlier probability to weight expert routing (instead of raw features), preserving sub-field size information via sigmoid+normalization (instead of softmax), and dynamically scaling expert capacity by sub-field size. Applied to motion field decomposition in two-view geometric matching.

## Core Claims Assessment

### Claim 1: Inlier-Probability Weighted Routing
- **Novelty**: HIGH
- **Closest work**: CorrMoE (arXiv 2507.11834) uses standard TopK+Softmax routing based on graph node features
- **Key difference**: CorrMoE routes by scene type (cross-domain/scene adaptation); IPG-SAD routes by inlier probability (within-scene geometric consistency). No prior work weights MoE routing by inlier probability in correspondence pruning.
- **Risk**: Low. Conceptually simple but never applied in this domain.

### Claim 2: Scale-Aware Motion Field Decomposition
- **Novelty**: HIGH
- **Closest work**: 
  - DeMatch (CVPR 2024) decomposes motion field but uses fixed-size sub-fields
  - GeoMoE (AAAI 2026) uses diff_Pool with softmax that discards size information
- **Key difference**: Both discard sub-field size. IPG-SAD explicitly preserves it via sigmoid + size normalization.
- **Risk**: Low. GeoMoE is the direct predecessor; this is a clear improvement.

### Claim 3: Dynamic Expert Capacity
- **Novelty**: MEDIUM-HIGH
- **Closest work**: 
  - Various LLM MoE papers use different expert sizes (e.g., Qwen3, DeepSeek)
  - "Mixture of Mini Experts" (2026) discusses capacity factors
- **Key difference**: None apply dynamic capacity based on data statistics (sub-field size) in real-time during inference. IPG-SAD scales hidden_dim by sub-field size.
- **Risk**: Medium. Dynamic capacity exists in other domains but not in geometric vision.

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| CorrMoE | 2025 | arXiv | MoE for correspondence pruning | Routes by scene type, not inlier prob; no size preservation |
| GeoMoE | 2026 | AAAI | MoE + motion field decomposition | Standard routing; softmax loses size; fixed expert capacity |
| DeMatch | 2024 | CVPR | Motion field decomposition | No MoE; fixed decomposition |
| DeMo | 2025 | AAAI | Motion field consensus | RKHS kernels, no MoE |
| EAQuant | 2025 | arXiv | Outlier-aware MoE (LLM) | Focus on quantization, not geometric matching |

## Overall Novelty Assessment

- **Score**: 8/10
- **Recommendation**: PROCEED
- **Key differentiator**: First to use inlier probability for MoE routing and preserve sub-field size in motion field decomposition
- **Risk**: CorrMoE/GeoMoE may cite this as incremental; need strong empirical results

## Suggested Positioning

Position as a **fundamental fix to MoE-based motion field decomposition** rather than just another MoE variant. Emphasize:
1. The verified failure of standard load balancing in high-outlier regimes
2. The information loss from softmax-based decomposition
3. The capacity bottleneck of fixed-size experts
4. All three fixes are minimal (3 lines each) but address root causes
