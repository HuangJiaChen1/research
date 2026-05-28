"""
Verify MGCA-Net pain points using REAL YFCC100M data.

This script:
1. Loads the official MGCA-Net pretrained model (YFCC100M) on CPU.
2. Loads samples from ../data/yfcc-sift-2000-test.hdf5.
3. Runs forward pass with hooks to extract intermediate activations.
4. Measures pain point metrics on real correspondences.
"""

import sys
sys.path.insert(0, 'mgca-net-code/core')

import torch
import torch.nn.functional as F
import numpy as np
import h5py
from MGCA import MGCANet, knn, get_graph_feature
from config import get_config
from loss import batch_episym

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
config, _ = get_config()
config.iter_num = 2
config.data_te = '../data/yfcc-sift-2000-test.hdf5'
config.gpu_id = '-1'  # force CPU

# ------------------------------------------------------------------
# Load model on CPU
# ------------------------------------------------------------------
model = MGCANet(config)
model = model.cpu()
print('[INFO] Loading pretrained weights...')
ckpt = torch.load('mgca-net-code/weights/yfcc100m/model_best1.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# ------------------------------------------------------------------
# Hooks to capture intermediate activations
# ------------------------------------------------------------------
activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Register hooks on key modules
model.subnetwork_init.CGA1.CPA.register_forward_hook(get_hook('CPT_stage0'))
model.subnetwork[0].CGA1.CPA.register_forward_hook(get_hook('CPT_stage1'))
model.subnetwork[0].CGA2.CPA.register_forward_hook(get_hook('CPT_stage1_2'))

model.subnetwork_init.CGA1.MBFFN.register_forward_hook(get_hook('MBFFN_stage0'))
model.subnetwork[0].CGA1.MBFFN.register_forward_hook(get_hook('MBFFN_stage1'))

model.CSMGC.register_forward_hook(get_hook('CSMGC'))

# Hook GNN input to measure feature-space k-NN quality
gnn_inputs = {}
def get_pre_hook(name):
    def hook(module, input):
        gnn_inputs[name] = input[0].detach().cpu()
    return hook
model.subnetwork_init.GNN.register_forward_pre_hook(get_pre_hook('GNN_stage0'))
model.subnetwork[0].GNN.register_forward_pre_hook(get_pre_hook('GNN_stage1'))

# ------------------------------------------------------------------
# Load HDF5 data directly
# ------------------------------------------------------------------
print('[INFO] Opening HDF5 test file...')
hf = h5py.File(config.data_te, 'r')
xs_group = hf['xs']
ys_group = hf['ys']
Rs_group = hf['Rs']
ts_group = hf['ts']
sample_keys = list(xs_group.keys())
print(f'[INFO] Total samples: {len(sample_keys)}')

# ------------------------------------------------------------------
# Metrics storage
# ------------------------------------------------------------------
metrics = {
    'outlier_ratios': [],
    'knn_false_neighbor_ratio': [],
    'mbffn_avg_pool_shift': [],
    'mbffn_max_pool_shift': [],
    'csmgc_stage_consistency': [],
    'cpt_outlier_attention_mass': [],
}

# ------------------------------------------------------------------
# Main loop over samples
# ------------------------------------------------------------------
max_samples = 200
with torch.no_grad():
    for idx, key in enumerate(sample_keys[:max_samples]):
        xs = np.asarray(xs_group[key])   # [1, N, 4]
        ys = np.asarray(ys_group[key])   # [N, 1]  -- this is residual!
        R = np.asarray(Rs_group[key])    # [3,3]
        t = np.asarray(ts_group[key])    # [3,1]

        N = xs.shape[1]
        # ys is residual; inliers have very small residual
        outlier_mask = (ys.squeeze() > 1e-4).astype(np.float32)
        outlier_ratio = outlier_mask.mean()
        metrics['outlier_ratios'].append(outlier_ratio)

        # Prepare data dict
        data = {
            'xs': torch.from_numpy(xs).unsqueeze(1).float().cpu(),  # [1, 1, N, 4]
            'ys': torch.from_numpy(ys).float().unsqueeze(0).cpu(),  # [1, N, 1]
            'Rs': torch.from_numpy(R).unsqueeze(0).float().cpu(),
            'ts': torch.from_numpy(t).unsqueeze(0).float().cpu(),
        }

        # Forward pass
        try:
            res_weights, res_e_hat = model(data)
        except Exception as e:
            print(f'[WARN] Sample {key} forward failed: {e}')
            # Pad with zeros so arrays stay aligned
            metrics['knn_false_neighbor_ratio'].append(0.0)
            metrics['mbffn_avg_pool_shift'].append(0.0)
            metrics['mbffn_max_pool_shift'].append(0.0)
            metrics['csmgc_stage_consistency'].append(0.0)
            metrics['cpt_outlier_attention_mass'].append(0.0)
            continue

        # Get final logits
        logits_final = res_weights[-1].squeeze().cpu().numpy()  # [N]

        # ------------------------------
        # Pain Point 1: k-NN false neighbors in feature space
        # ------------------------------
        # We use the input feature to the first GNN as proxy.
        # The first stage's GNN input is x1_1 after CGA1 and PointCN.
        # We don't have direct hook, but we can compute k-NN on the
        # model's intermediate representation if we register a hook on GNN.
        # For now, use the coordinate-space k-NN as a lower-bound proxy.
        # ACTUALLY: let's hook GNN input in next version.
        # For this run, compute feature-space k-NN using first-stage output 'x1_1'.
        # We return x1_1 from sub_MGCANet but we don't hook it easily.
        # Simplification: use the final stage output features (out) as feature proxy.
        # We can get the last stage's 'out' from the returned tuple but it's not hooked.
        # Let's register a hook on subnetwork[0].GNN to capture its input.
        # (We didn't register above, but we can do it now for future runs.)
        # ------------------------------
        # Pain Point 1: k-NN false neighbors in feature space
        # ------------------------------
        if 'GNN_stage1' in gnn_inputs:
            features = gnn_inputs['GNN_stage1']  # [1, C, N, 1]
            features = features.squeeze(3)  # [1, C, N]
            k = model.subnetwork[0].GNN.knn_num
            knn_idx = knn(features, k=k)  # [1, N, k]
            # For inliers, count neighbors that are outliers
            inlier_indices = np.where(~outlier_mask.astype(bool))[0]
            if len(inlier_indices) > 0:
                neighbor_idx = knn_idx[0, inlier_indices, :].numpy()  # [n_inliers, k]
                false_neighbors = outlier_mask[neighbor_idx].mean()
                metrics['knn_false_neighbor_ratio'].append(false_neighbors)
            else:
                metrics['knn_false_neighbor_ratio'].append(0.0)
        else:
            metrics['knn_false_neighbor_ratio'].append(0.0)

        # ------------------------------
        # Pain Point 2: MB-FFN pooling contamination
        # ------------------------------
        e_hat = res_e_hat[-1]  # [1, 9]
        x1 = data['xs'][:, 0, :, :2]  # [1, N, 2]
        x2 = data['xs'][:, 0, :, 2:4]  # [1, N, 2]
        residual = batch_episym(x1, x2, e_hat).reshape(1, 1, N, 1).cpu()  # [1, 1, N, 1]

        if 'MBFFN_stage1' in activations:
            mbffn_out = activations['MBFFN_stage1']  # [1, C, N, 1]
            # Uniform weighting
            w_uniform = torch.ones_like(residual) / N
            avg_uniform = (mbffn_out * w_uniform).sum(dim=2)
            max_uniform = mbffn_out.max(dim=2)[0]

            # Residual-based weighting
            w_residual = torch.exp(-residual / 0.1)
            w_residual = w_residual / (w_residual.sum(dim=2, keepdim=True) + 1e-6)
            avg_weighted = (mbffn_out * w_residual).sum(dim=2)
            max_weighted = (mbffn_out * w_residual).max(dim=2)[0]

            shift_avg = (avg_uniform - avg_weighted).abs().mean().item()
            shift_max = (max_uniform - max_weighted).abs().mean().item()
            metrics['mbffn_avg_pool_shift'].append(shift_avg)
            metrics['mbffn_max_pool_shift'].append(shift_max)
        else:
            metrics['mbffn_avg_pool_shift'].append(0.0)
            metrics['mbffn_max_pool_shift'].append(0.0)

        # ------------------------------
        # Pain Point 3: CSMGC stage consistency
        # ------------------------------
        if 'CSMGC' in activations:
            csmgc_out = activations['CSMGC']  # [1, C, N, 1]
            metrics['csmgc_stage_consistency'].append(csmgc_out.var().item())
        else:
            metrics['csmgc_stage_consistency'].append(0.0)

        # ------------------------------
        # Pain Point 4: CPT outlier attention mass
        # ------------------------------
        if 'CPT_stage1' in activations:
            cpt_out = activations['CPT_stage1']  # [1, C, N, 1]
            # We don't have attention weights directly from hook.
            # As proxy: measure how much the output features of outliers
            # differ from inliers (high variance = outlier pollution).
            outlier_feat = cpt_out[:, :, outlier_mask.astype(bool), :]
            inlier_feat = cpt_out[:, :, (~outlier_mask.astype(bool)), :]
            if outlier_feat.numel() > 0 and inlier_feat.numel() > 0:
                diff = (outlier_feat.mean(dim=2) - inlier_feat.mean(dim=2)).abs().mean().item()
                metrics['cpt_outlier_attention_mass'].append(diff)
            else:
                metrics['cpt_outlier_attention_mass'].append(0.0)
        else:
            metrics['cpt_outlier_attention_mass'].append(0.0)

        if (idx + 1) % 50 == 0:
            print(f'[INFO] Processed {idx+1}/{max_samples} samples')

# ------------------------------------------------------------------
# Aggregate results
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('REAL DATA VERIFICATION RESULTS')
print('=' * 70)

ratios = np.array(metrics['outlier_ratios'])
shifts_avg = np.array(metrics['mbffn_avg_pool_shift'])
shifts_max = np.array(metrics['mbffn_max_pool_shift'])
consistencies = np.array(metrics['csmgc_stage_consistency'])
cpt_diffs = np.array(metrics['cpt_outlier_attention_mass'])

bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]

print('\nPain Point 1: k-NN False Neighbor Ratio (Feature Space)')
print('-' * 70)
knn_false = np.array(metrics['knn_false_neighbor_ratio'])
for low, high in bins:
    mask = (ratios >= low) & (ratios < high)
    if mask.sum() == 0:
        continue
    mean_false = knn_false[mask].mean()
    print(f'  Outlier [{low*100:.0f}%, {high*100:.0f}%)  N={mask.sum():3d}  '
          f'Avg false neighbor ratio: {mean_false*100:.2f}%')

print('\nPain Point 2: MB-FFN Pooling Shift (Uniform vs Residual-Weighted)')
print('-' * 70)
for low, high in bins:
    mask = (ratios >= low) & (ratios < high)
    if mask.sum() == 0:
        continue
    mean_shift_avg = shifts_avg[mask].mean()
    mean_shift_max = shifts_max[mask].mean()
    print(f'  Outlier [{low*100:.0f}%, {high*100:.0f}%)  N={mask.sum():3d}  '
          f'Avg-shift: {mean_shift_avg:.4f}  Max-shift: {mean_shift_max:.4f}')

print('\nPain Point 3: CSMGC Output Variance')
print('-' * 70)
for low, high in bins:
    mask = (ratios >= low) & (ratios < high)
    if mask.sum() == 0:
        continue
    mean_var = consistencies[mask].mean()
    print(f'  Outlier [{low*100:.0f}%, {high*100:.0f}%)  N={mask.sum():3d}  '
          f'Avg variance: {mean_var:.4f}')

print('\nPain Point 4: CPT Output Difference (Outlier vs Inlier Features)')
print('-' * 70)
for low, high in bins:
    mask = (ratios >= low) & (ratios < high)
    if mask.sum() == 0:
        continue
    mean_diff = cpt_diffs[mask].mean()
    print(f'  Outlier [{low*100:.0f}%, {high*100:.0f}%)  N={mask.sum():3d}  '
          f'Avg feat diff: {mean_diff:.4f}')

print('=' * 70)
hf.close()
