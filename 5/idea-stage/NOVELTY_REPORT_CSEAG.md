# Novelty Check Report — Idea 2: Cross-Stage Edge Agreement Gating (CSEAG)

## Proposed Method
Before concatenating multi-stage graphs in CSMGC, compute edge-wise agreement between stages (count how many stages share the same k-NN edge). Weight edges by agreement score before MLP processing, turning naive concatenation into actual consensus voting.

## Core Claims

### Claim 1: Cross-stage graph consensus for correspondence pruning
**Novelty: LOW**

**Closest Prior Work**:
- **CLNet** (Zhao et al., ICCV 2021): "Progressive Correspondence Pruning by Consensus Learning." Explicitly constructs **local consensus graphs** and **global consensus graphs** across progressive stages. Uses annular convolution and spectral graph convolution to propagate consensus scores. This is the most direct prior art.
- **BCLNet** (Miao et al., AAAI 2024): "Bilateral Consensus Learning for Two-View Correspondence Pruning." Uses **Bilateral Consensus Mining Attention (BCMA)** and **Bilateral Consensus Recalibrate (BCR)** blocks across two pruning stages. Cross-stage feature propagation and consensus recalibration are central to the method.
- **GCT-Net** (AAAI 2024): "Graph Context Transformation Learning for Progressive Correspondence Pruning." Uses **multi-branch graph contexts** with self-attention and cross-attention across progressive stages, plus confidence-based sampling.

**Key Difference**: CSEAG uses a simple **edge-counting agreement score** (how many stages agree on an edge) rather than learned attention (BCLNet), spectral convolution (CLNet), or context transformation (GCT-Net). This is arguably a **simplification** rather than an innovation.

### Claim 2: Edge agreement gating before graph concatenation
**Novelty: MEDIUM-LOW**

**Closest Prior Work**:
- **BCLNet's BCMA block** performs graph grouping and global consensus via attention — implicitly gates information based on consensus.
- **GCT-Net's confidence-based sampling** screens high-confidence vertices before cross-attention fusion.
- **Various graph attention networks** (Veličković et al., 2018; SuperGlue, 2020) use attention scores as soft gates on edges.

**Key Difference**: CSEAG proposes a **hard/soft counting mechanism** (number of stages sharing an edge) rather than learned attention weights. The novelty lies in the simplicity and interpretability, but the functional behavior is similar to existing attention-based consensus mechanisms.

### Claim 3: Replacing concatenation with consensus-weighted fusion
**Novelty: LOW**

**Closest Prior Work**:
- MGCA-Net itself uses concatenation + MLP, which is indeed naive. However, fixing this naivety by adding consensus weighting is exactly what CLNet, BCLNet, and GCT-Net already do in their respective architectures. MGCA-Net's CSMGC module was published later (IJCAI 2025) but appears to have ignored these prior consensus-learning advances.

**Key Difference**: CSEAG is a minimal patch to MGCA-Net specifically, but the underlying concept (cross-stage graph consensus) is well-established.

## Closest Prior Work Table

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| CLNet (Zhao et al.) | 2021 | ICCV | Local-to-global consensus across stages | Uses spectral graph convolution and annular convolution for consensus propagation. More sophisticated than simple edge counting. |
| BCLNet (Miao et al.) | 2024 | AAAI | Bilateral consensus across two stages | Uses learned BCMA/BCR attention blocks for cross-stage consensus mining and recalibration. |
| GCT-Net | 2024 | AAAI | Multi-branch graph contexts + progressive pruning | Uses self/cross-attention and confidence-based sampling across stages. |
| VSFormer | 2023 | arXiv | Visual-spatial fusion with graph attention | Graph attention blocks for KNN graphs; not cross-stage consensus. |

## Overall Novelty Assessment

- **Score: 4/10**
- **Recommendation: PROCEED WITH CAUTION / ABANDON as standalone contribution**
- **Key differentiator**: The *specific* mechanism (edge-counting agreement) is simple and interpretable, but the *problem it solves* (cross-stage graph consensus) has been thoroughly addressed by CLNet (2021), BCLNet (2024), and GCT-Net (2024). A reviewer would almost certainly cite CLNet as prior art.
- **Risk**: HIGH. CSEAG is at high risk of being rejected as "already done by CLNet/BCLNet" or "an obvious simplification of existing attention-based consensus."
- **Suggested Positioning**:
  - **Option A**: Abandon CSEAG as a standalone contribution. Instead, mention it as an ablation or implementation detail within the RWGP paper.
  - **Option B**: Combine CSEAG with RWGP into a unified framework, positioning the joint contribution as "a minimal but principled two-fix solution to MGCA-Net's structural weaknesses." Even then, the novelty weight should be placed primarily on RWGP.
  - **Option C**: Enhance CSEAG with a learned component (e.g., differentiable edge agreement prediction) to create a stronger delta from CLNet/BCLNet.

## Concurrent Work Check (Last 6 Months)
- No new cross-stage consensus papers identified in 2025-2026 arXiv.
- However, the field is mature on this specific sub-problem.

## Conclusion
CSEAG is **not sufficiently novel** to stand alone. It should either be (1) dropped, (2) treated as a minor ablation/component within the RWGP paper, or (3) significantly enhanced with learned/differentiable mechanisms before being presented as a contribution.
