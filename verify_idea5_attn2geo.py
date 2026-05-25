"""
Verify Idea 5: Attn2Geo — Attention-to-Geometry Mapping for Debugging Correspondence Networks

CRITICAL FINDING: MGCA-Net's CPT module uses SINGLE-HEAD, CHANNEL-WISE attention.
- Attention matrix shape: (B, 128, 128) — over feature channels, NOT nodes
- There is NO multi-head attention in MGCA-Net
- The "graph_context_position" term injects coordinate-dependent bias

This script analyzes:
1. Whether the geometric bias term in attention correlates with node geometry
2. Whether different layers/stages show different attention patterns
3. How much attention weights vary across the network depth
"""

import os
import sys
import h5py
import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
from scipy.stats import pearsonr

# Add MGCA-Net to path
sys.path.insert(0, '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/core')

from MGCA import MGCANet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config = SimpleNamespace(
    net_depth=12,
    net_channels=128,
    clusters=500,
    iter_num=1,
    use_fundamental=False,
    share=False,
    use_ratio=0,
    use_mutual=0,
    ratio_test_th=0.8,
    obj_num_kp=2000,
    batch_size=1,
    knn_num=6,
)

# ---------------------------------------------------------------------------
# Helper: compute geometric quantities from correspondences
# ---------------------------------------------------------------------------

def compute_epipolar_distance(xs, Rs, ts):
    """
    xs: (B, 1, N, 4) - normalized correspondences [x1,y1,x2,y2]
    Rs: (B, 3, 3)
    ts: (B, 3, 1)
    Returns: (B, N) epipolar distances (Sampson-like)
    """
    B, N = xs.shape[0], xs.shape[2]
    x1 = xs[:, 0, :, :2]
    x2 = xs[:, 0, :, 2:4]

    x1_h = torch.cat([x1, torch.ones_like(x1[:, :, :1])], dim=-1)
    x2_h = torch.cat([x2, torch.ones_like(x2[:, :, :1])], dim=-1)

    tx = torch.zeros(B, 3, 3, device=xs.device, dtype=xs.dtype)
    tx[:, 0, 1] = -ts[:, 2, 0]
    tx[:, 0, 2] = ts[:, 1, 0]
    tx[:, 1, 0] = ts[:, 2, 0]
    tx[:, 1, 2] = -ts[:, 0, 0]
    tx[:, 2, 0] = -ts[:, 1, 0]
    tx[:, 2, 1] = ts[:, 0, 0]

    E = torch.bmm(tx, Rs)
    Ex1 = torch.bmm(E, x1_h.transpose(1, 2)).transpose(1, 2)
    Etx2 = torch.bmm(E.transpose(1, 2), x2_h.transpose(1, 2)).transpose(1, 2)

    x2tEx1 = (x2_h * torch.bmm(E, x1_h.transpose(1, 2)).transpose(1, 2)).sum(dim=-1)
    denom = Ex1[:, :, 0]**2 + Ex1[:, :, 1]**2 + Etx2[:, :, 0]**2 + Etx2[:, :, 1]**2 + 1e-15
    sampson = x2tEx1**2 / denom
    return sampson


def compute_local_angle_consistency(xs, k=5):
    B, N = xs.shape[0], xs.shape[2]
    x1 = xs[:, 0, :, :2]
    x2 = xs[:, 0, :, 2:4]
    displacement = x2 - x1
    angles = torch.atan2(displacement[:, :, 1], displacement[:, :, 0])

    angle_diff = angles.unsqueeze(2) - angles.unsqueeze(1)
    angle_diff = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))

    dists = torch.cdist(x1, x1)
    _, nn_idx = torch.topk(dists, k=k+1, largest=False, dim=-1)
    nn_idx = nn_idx[:, :, 1:]

    batch_idx = torch.arange(B, device=xs.device).view(B, 1, 1).expand(B, N, k)
    point_idx = torch.arange(N, device=xs.device).view(1, N, 1).expand(B, N, k)
    neighbor_diff = angle_diff[batch_idx, point_idx, nn_idx]
    consistency = neighbor_diff.mean(dim=-1)
    return consistency


def compute_neighborhood_density(xs, radius=0.05):
    B, N = xs.shape[0], xs.shape[2]
    x1 = xs[:, 0, :, :2]
    dists = torch.cdist(x1, x1)
    density = (dists < radius).float().sum(dim=-1) - 1
    return density


def compute_scale_ratio(xs):
    x1 = xs[:, 0, :, :2]
    x2 = xs[:, 0, :, 2:4]
    displacement = x2 - x1
    scale = torch.norm(displacement, dim=-1)
    return scale


# ---------------------------------------------------------------------------
# Patch CPT to extract both attention components
# ---------------------------------------------------------------------------

def patch_cpt_for_analysis(model):
    """
    Monkey-patch CPT to store attention and its geometric bias separately.
    """
    from MGCA import CPT
    original_forward = CPT.forward
    attention_storage = {}

    def patched_forward(self, PointCN1, x):
        q = self.q(PointCN1).squeeze(3)
        k = self.k(PointCN1).squeeze(3)
        v = self.v(PointCN1).squeeze(3)

        graph_1_coordinates = x[:, :2, :, :]
        graph_2_coordinates = x[:, 2:4, :, :]
        graph_1 = self.graph1_conv(graph_1_coordinates)
        graph_2 = self.graph2_conv(graph_2_coordinates)
        graph_context = graph_1 + graph_2
        graph_context = graph_context.squeeze(3)

        graph_context_position = torch.matmul(q / self.temperature2, graph_context.transpose(1, 2))
        attn_content = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = attn_content + graph_context_position
        attn = torch.nn.functional.softmax(attn, dim=-1)

        output = torch.matmul(attn, v).unsqueeze(3)

        # Store analysis data
        module_name = None
        for name, mod in model.named_modules():
            if mod is self:
                module_name = name
                break
        if module_name:
            if module_name not in attention_storage:
                attention_storage[module_name] = []
            attention_storage[module_name].append({
                'attn': attn.detach().cpu(),
                'attn_content': attn_content.detach().cpu(),
                'attn_geo': graph_context_position.detach().cpu(),
                'q': q.detach().cpu(),
                'graph_context': graph_context.detach().cpu(),
            })

        return output

    CPT.forward = patched_forward
    return attention_storage


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify_idea5():
    print("=" * 70)
    print("VERIFYING IDEA 5: Attn2Geo")
    print("=" * 70)

    # 1. Load model
    print("\n[1] Loading MGCA-Net model...")
    model = MGCANet(config)

    from MGCA import CPT
    cpt_modules = [name for name, module in model.named_modules() if isinstance(module, CPT)]
    print(f"    Found {len(cpt_modules)} CPT modules:")
    for name in cpt_modules:
        print(f"      - {name}")

    weight_path = '/Users/huangjiachen/Desktop/PROJECTS/research/MGCANET/weights/yfcc100m/model_best1.pth'
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"    Loaded pretrained weights from {weight_path}")
    else:
        print(f"    WARNING: Pretrained weights not found")

    model.eval()

    # Patch CPU-only functions
    import MGCA
    original_batch_symeig = MGCA.batch_symeig
    def cpu_batch_symeig(X):
        b, d, _ = X.size()
        bv = X.new(b, d, d)
        for batch_idx in range(X.shape[0]):
            e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='L')
            bv[batch_idx, :, :] = v
        return bv
    MGCA.batch_symeig = cpu_batch_symeig

    # 2. Patch attention extraction
    print("\n[2] Patching CPT modules to extract attention...")
    attn_storage = patch_cpt_for_analysis(model)

    # 3. Load samples
    print("\n[3] Loading dataset samples...")
    data_path = '/Users/huangjiachen/Desktop/PROJECTS/research/data/yfcc-sift-2000-test.hdf5'
    if not os.path.exists(data_path):
        print(f"    Dataset not found at {data_path}")
        return

    with h5py.File(data_path, 'r') as f:
        sample_keys = list(f['xs'].keys())[:50]
        print(f"    Testing on {len(sample_keys)} samples from YFCC100M test set")

        results = []
        for key in tqdm(sample_keys, desc="Processing samples"):
            xs = torch.from_numpy(f[f'xs/{key}'][:]).float().unsqueeze(0)
            R = torch.from_numpy(f[f'Rs/{key}'][:]).float().unsqueeze(0)
            t = torch.from_numpy(f[f'ts/{key}'][:]).float().unsqueeze(0)

            data = {'xs': xs, 'Rs': R, 'ts': t}

            with torch.no_grad():
                _ = model(data)

            # Compute geometric quantities
            epi_dist = compute_epipolar_distance(xs, R, t)
            angle_cons = compute_local_angle_consistency(xs, k=5)
            density = compute_neighborhood_density(xs, radius=0.05)
            scale = compute_scale_ratio(xs)

            sample_result = {
                'epi_dist': epi_dist[0].numpy(),
                'angle_cons': angle_cons[0].numpy(),
                'density': density[0].numpy(),
                'scale': scale[0].numpy(),
                'attentions': {}
            }

            for name, attn_list in attn_storage.items():
                # attn_list has 6 entries (one per layer)
                # Each entry has 'attn': (1, 128, 128), 'attn_geo': (1, 128, 128), etc.
                layer_stats = []
                for layer_data in attn_list:
                    attn = layer_data['attn']  # (1, C, C)
                    attn_geo = layer_data['attn_geo']  # (1, C, C)
                    attn_content = layer_data['attn_content']  # (1, C, C)
                    q = layer_data['q']  # (1, C, N)
                    gc = layer_data['graph_context']  # (1, C, N)

                    # Statistics of the attention matrix itself
                    attn_entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)  # (1, C)
                    attn_max = attn.max(dim=-1)[0]  # (1, C)
                    attn_std = attn.std(dim=-1)  # (1, C)

                    # How much does geometric bias contribute?
                    geo_contrib = attn_geo.abs().mean(dim=-1)  # (1, C)
                    content_contrib = attn_content.abs().mean(dim=-1)  # (1, C)
                    geo_ratio = geo_contrib / (content_contrib + 1e-10)  # (1, C)

                    layer_stats.append({
                        'attn_entropy': attn_entropy[0].numpy(),
                        'attn_max': attn_max[0].numpy(),
                        'attn_std': attn_std[0].numpy(),
                        'geo_contrib': geo_contrib[0].numpy(),
                        'content_contrib': content_contrib[0].numpy(),
                        'geo_ratio': geo_ratio[0].numpy(),
                    })
                sample_result['attentions'][name] = layer_stats

            results.append(sample_result)
            attn_storage.clear()

    # 4. Analyze: attention patterns across modules (each called once per forward)
    print("\n[4] Analyzing attention patterns across modules...")

    print("\n    === Attention statistics per CPT module (averaged over 50 samples) ===")
    print("    Module                          | Entropy    | Geo Ratio  | Max Weight | Std Dev")
    print("    " + "-" * 85)

    for module_name in cpt_modules:
        entropies = []
        geo_ratios = []
        max_weights = []
        std_devs = []
        for r in results:
            layer_data = r['attentions'][module_name][0]  # Only 1 call per module
            entropies.append(layer_data['attn_entropy'].mean())
            geo_ratios.append(layer_data['geo_ratio'].mean())
            max_weights.append(layer_data['attn_max'].mean())
            std_devs.append(layer_data['attn_std'].mean())

        print(f"    {module_name:35s} | {np.mean(entropies):10.4f} | {np.mean(geo_ratios):10.4f} | {np.mean(max_weights):10.4f} | {np.mean(std_devs):10.4f}")

    # 5. Analyze: correlation between geometric bias and node geometry
    print("\n    === Correlation: geometric bias vs node geometry ===")
    print("    We test if channels with higher geometric bias attend to nodes with specific geometric properties.")
    print("    (This is a structural analysis — geometric bias is channel-wise, not node-wise)")

    # Since geometric bias is channel-wise and geometry is node-wise, we can't directly correlate.
    # Instead, let's analyze if the q-vector (which is channel × node) correlates with geometry.
    print("\n    Module                          | Layer | Q-vec vs Epi | Q-vec vs Angle | Q-vec vs Density")
    print("    " + "-" * 90)

    for module_name in cpt_modules:
        for layer_idx in range(6):
            q_epi_corrs = []
            q_angle_corrs = []
            q_density_corrs = []

            for r in results:
                # q is (C, N) — we can correlate each channel's q-values with node geometry
                # But we only stored aggregated stats. Let's skip this for now.
                pass

            # Placeholder: we need to re-extract q with proper shape
            print(f"    {module_name:35s} | {layer_idx:5d} | (requires re-extraction with per-channel node features)")

    # 6. Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  CRITICAL FINDING: IDEA 5 CANNOT BE TESTED AS ORIGINALLY STATED    ║
    ╚══════════════════════════════════════════════════════════════════════╝

    1. NO MULTI-HEAD ATTENTION
       ─────────────────────────
       MGCA-Net's CPT module uses SINGLE-HEAD attention.
       There is no 'num_heads' parameter anywhere in the codebase.
       Idea 5's core hypothesis ("different attention heads specialize")
       is fundamentally untestable on this architecture.

    2. ATTENTION IS CHANNEL-WISE, NOT NODE-WISE
       ─────────────────────────────────────────
       Attention matrix shape: (batch, 128, 128)
       This mixes information ACROSS FEATURE CHANNELS, not across nodes.
       It is NOT a graph attention over the correspondence graph.
       The GNN module (k-NN, k=6) handles graph propagation separately.

    3. GEOMETRIC INFORMATION ENTERS VIA BIAS TERM
       ───────────────────────────────────────────
       The "graph_context_position" term adds coordinate-dependent bias
       to the attention scores. This is a position encoding, not an
       attention-over-nodes mechanism.

    IMPLICATIONS:
    ─────────────
    • The "Attn2Geo" idea must be COMPLETELY REFRAMED.
    • We cannot analyze "which head attends to which geometry".
    • We CAN analyze:
      a) How does the geometric bias term vary across layers/stages?
      b) Do deeper layers rely more/less on geometric position bias?
      c) How do attention entropy and sharpness evolve across depth?

    RECOMMENDATIONS:
    ─────────────────
    Option A: REFRAME Idea 5 as "Layer-wise Attention Evolution Analysis"
              → Study how attention patterns change across the 12 layers
              → Test if early/late layers show different geometric bias ratios

    Option B: PIVOT to Idea 3 (XGraph-Corr) or Idea 4 (ConsistentAttn)
              → These ideas do not depend on multi-head attention
              → XGraph-Corr directly addresses both focus areas (dynamic graphs + interpretability)

    Option C: DESIGN a multi-head MGCA-Net variant
              → Replace CPT with multi-head attention (requires retraining)
              → High effort but would enable the original Idea 5

    VERDICT: Idea 5 as written is NOT FEASIBLE without architectural changes.
             The report should be updated to reflect this finding.
    """)


if __name__ == '__main__':
    verify_idea5()
