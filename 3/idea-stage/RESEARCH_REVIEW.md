# Research Review: IPG-SAD

## Reviewer Assessment (Internal, based on verified evidence)

### Pain Points Validation

**PP1: Self-attention broken at high outlier ratios** [CONFIRMED]
- Evidence is strong: synthetic experiment with quantified metrics
- This is a known issue in attention literature (systematic outliers in LLMs, ICLR 2025)
- Affects ALL attention-based correspondence methods

**PP2: MoE load balancing conflicts with geometric reality** [CONFIRMED]
- Code-level verification on GeoMoE's exact implementation
- The `(usage**2).sum() * E` loss is standard in MoE literature (Switch Transformer)
- But its application to geometric matching is inappropriate
- Real scenes have imbalanced motion patterns

**PP3: Progressive pruning loses inliers** [CONFIRMED]
- Simulation shows 82% inlier loss at threshold=0.7
- This is a well-known issue but rarely quantified
- All progressive methods suffer from this

**PP4: diff_Pool loses sub-field size** [CONFIRMED]
- Direct code analysis of GeoMoE's diff_Pool
- Softmax across correspondences is mathematically flawed for size preservation
- DeMatch's Fourier decomposition also doesn't preserve size

**PP5: Expert capacity limited** [CONFIRMED]
- Hardcoded hidden_dim=16 is clearly suboptimal
- Effective rank analysis confirms bottleneck

### Critical Assessment of IPG-SAD

**Strengths**:
1. Addresses 3 pain points simultaneously
2. Changes are minimal and surgical
3. Novelty is high (8/10) - no direct competition
4. Based on verified evidence, not speculation

**Weaknesses**:
1. **Empirical validation missing**: We haven't actually run IPG-SAD on real data
2. **Sigmoid alternative**: Sigmoid + normalization might not be the optimal replacement for softmax
3. **Dynamic capacity**: Scaling hidden_dim by sub-field size might hurt small sub-fields
4. **Interaction effects**: The three changes might interact in unexpected ways
5. **Generalization**: Will this transfer to non-MoE architectures?

**Questions for Authors**:
1. What happens if sub-field sizes are extremely imbalanced (1:100)?
2. How does dynamic capacity affect training stability?
3. Can you ablate each of the three changes independently?
4. What is the computational overhead?
5. How does this compare to simply increasing num_experts?

**Score**: 7/10 for NeurIPS/ICML
- Strong motivation and verification
- Minimal changes
- But needs real experiments to be convincing
- Risk: reviewers may see this as "incremental engineering"

**What would move toward accept**:
1. Strong empirical results on standard benchmarks (YFCC100M, SUN3D)
2. Ablation studies showing each fix contributes independently
3. Analysis of failure cases where IPG-SAD still fails
4. Comparison with simple baselines (e.g., just increase hidden_dim)
5. Theoretical justification for why inlier-weighted routing is optimal

### Recommendations

1. **Run pilot first**: Modify GeoMoE code and test on a small subset
2. **Ablate thoroughly**: Test each of the 3 changes independently
3. **Compare with baselines**: Simple fixes like increasing num_experts or hidden_dim
4. **Document failure modes**: When does IPG-SAD still fail?
5. **Consider combining with RSP-GR**: Soft pruning might compound the benefits
