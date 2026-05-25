# LearnableConsensus 训练与评估完整指南

## 项目简介

本文档描述在 MGCA-Net 上训练 **LearnableConsensus** 模块的完整流程。

- **核心目标**：通过可学习的几何一致性过滤，提升高外点率（>95%）场景下的匹配精度。
- **训练策略**：冻结 backbone（subnetwork_init + subnetwork），仅微调 consensus_module + CSMGC。
- **评估指标**：全局 mAP / Precision / Recall / F1，以及按外点率分桶的细粒度指标。

---

## 1. 环境准备

### 1.1 依赖安装

```bash
cd MGCANET
pip install -r requirements.txt
```

**requirements.txt 内容：**

```
h5py==3.13.0
matplotlib==3.10.1
numpy==2.3.5
opencv_contrib_python==4.11.0.86
six==1.17.0
tensorboardX==2.6.2.2
torch==2.6.0
tqdm==4.67.1
```

### 1.2 CUDA 版本适配（RTX2080Ti）

如果目标机器 CUDA 版本与 PyTorch 预编译包不匹配，请手动安装对应版本：

```bash
# CUDA 11.8 示例
pip install torch==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 示例
pip install torch==2.6.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

验证 CUDA 可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 1.3 数据集准备

将 HDF5 数据集放在可访问路径。原始代码默认路径为 `../data_dump/`，若路径不同请在启动命令中指定。

| 数据集 | 默认路径 | 说明 |
|--------|----------|------|
| YFCC100M train | `../data_dump/yfcc-sift-2000-train.hdf5` | 训练集 |
| YFCC100M val | `../data_dump/yfcc-sift-2000-val.hdf5` | 验证集（选 best model） |
| YFCC100M test | `../data_dump/yfcc-sift-2000-test.hdf5` | 测试集（最终报告） |
| SUN3D test | `../data_dump/sun3d-test.hdf5` | 跨数据集验证 |

---

## 2. 代码修改说明

本次修改只涉及 `core/MGCA.py`，新增以下内容：

### 2.1 compute_epipolar_distance(E, xs)

计算每个 correspondence 到极线的几何距离。

- 输入：`E: [B, 9]`，`xs: [B, 1, N, 4]`
- 输出：`[B, N]` 距离值

### 2.2 LearnableConsensus(nn.Module)

可学习的跨阶段共识过滤模块，核心参数：

| 参数 | 形状 | 作用 |
|------|------|------|
| `stage_weights` | `[3]` | 3 个 stage 的加权 softmax |
| `sigma` | scalar | 几何敏感度（softplus 保证正） |
| `alpha` | scalar | 语义-几何平衡（sigmoid 保证 `[0,1]`） |

**核心设计**：log-domain 的 product 融合

```python
log_consensus = alpha * log(sem) + (1-alpha) * log(geo)
consensus = exp(log_consensus)
```

这保留了零-shot MVP 验证有效的 **AND-gate** 行为。

### 2.3 MGCANet 集成

在 `MGCANet.forward()` 中，CSMGC 之前插入共识过滤：

```python
consensus = self.consensus_module(res_weights, res_e_hat, data['xs'])
refined_stage2 = stage_out[2] * consensus.unsqueeze(1).unsqueeze(1)
sub_l_input = self.CSMGC(stage_out[0], stage_out[1], refined_stage2)
```

---

## 3. 训练脚本：`train_consensus.py`

### 3.1 基本用法

```bash
cd MGCANET/core

CUDA_VISIBLE_DEVICES=0 python train_consensus.py \
    --pretrained ../weights/yfcc100m/model_best1.pth \
    --data_tr ../data_dump/yfcc-sift-2000-train.hdf5 \
    --data_va ../data_dump/yfcc-sift-2000-val.hdf5 \
    --ablation learned \
    --log_dir ./log_consensus
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained` | 必填 | 预训练模型路径 |
| `--ablation` | `learned` | 消融模式（见下表） |
| `--train_iter` | `50000` | 训练步数 |
| `--val_intv` | `5000` | 验证间隔 |
| `--batch_size` | `32` | 批次大小 |
| `--lr_consensus` | `1e-4` | consensus 学习率 |
| `--lr_csmgc` | `1e-5` | CSMGC 学习率 |
| `--gpu_id` | `"0"` | GPU 编号 |

### 3.3 消融模式

| 模式 | Alpha | Stage Weights | Sigma | 说明 |
|------|-------|---------------|-------|------|
| `learned` | 可学习 | 可学习 | 可学习 | 完整版本 |
| `fixed_product` | 0.5（冻结） | 均匀（冻结） | 1.0（冻结） | 零-shot 上限 |
| `semantic_only` | ~1.0（冻结） | 冻结 | 冻结 | 仅用语义（基线） |
| `geo_only` | ~0.0（冻结） | 冻结 | 冻结 | 仅用几何一致性 |

### 3.4 恢复训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_consensus.py \
    --pretrained ../weights/yfcc100m/model_best1.pth \
    ... \
    --resume ./log_consensus/learned/checkpoint.pth
```

### 3.5 训练日志输出示例

```
============================================================
  GLOBAL  |  P=0.7234  R=0.8912  F1=0.7981  mAP=0.6543
------------------------------------------------------------
  0-50     n=  512  P=0.9123  R=0.9567  F1=0.9340
  50-75    n=  298  P=0.8234  R=0.9012  F1=0.8604
  75-90    n=  145  P=0.7123  R=0.8567  F1=0.7782
  90-95    n=   34  P=0.6789  R=0.8234  F1=0.7441
  >95      n=   11  P=0.5160  R=0.6240  F1=0.5654
============================================================
[BEST] >95% F1 = 0.5654  (global F1=0.7981)  -- saving model
```

---

## 4. 测试脚本：`test_consensus.py`

### 4.1 基本用法

```bash
cd MGCANET/core

CUDA_VISIBLE_DEVICES=0 python test_consensus.py \
    --checkpoint ./log_consensus/learned/model_best.pth \
    --data_te ../data_dump/yfcc-sift-2000-test.hdf5 \
    --out_json ./results/learned_yfcc.json \
    --out_txt ./results/learned_yfcc.txt
```

### 4.2 跨数据集验证（SUN3D）

```bash
CUDA_VISIBLE_DEVICES=0 python test_consensus.py \
    --checkpoint ./log_consensus/learned/model_best.pth \
    --data_te ../data_dump/sun3d-test.hdf5 \
    --out_json ./results/learned_sun3d.json \
    --out_txt ./results/learned_sun3d.txt
```

### 4.3 输出文件格式

**JSON 输出**（`results/learned_yfcc.json`）：

```json
{
  "checkpoint": "./log_consensus/learned/model_best.pth",
  "dataset": "../data_dump/yfcc-sift-2000-test.hdf5",
  "timestamp": "2026-05-25 17:30:00",
  "global": {
    "n": 4000,
    "mAP": 0.6543,
    "precision": 0.7234,
    "recall": 0.8912,
    "f1": 0.7981
  },
  "buckets": {
    "0-50": {"n": 2048, "precision": 0.9123, "recall": 0.9567, "f1": 0.9340},
    ">95": {"n": 44, "precision": 0.5160, "recall": 0.6240, "f1": 0.5654}
  }
}
```

**TXT 输出**（含 LaTeX 表格行）：

```
Checkpoint: ./log_consensus/learned/model_best.pth
Dataset:    ../data_dump/yfcc-sift-2000-test.hdf5
------------------------------------------------------------
GLOBAL    mAP=0.6543  P=0.7234  R=0.8912  F1=0.7981
------------------------------------------------------------
0-50      P=0.9123  R=0.9567  F1=0.9340  n=2048
...

LaTeX table row:
  & 0.723 & 0.891 & 0.798 & 0.934 & 0.860 & 0.778 & 0.744 & 0.565 \
```

---

## 5. 完整实验流程

### Phase 1：训练所有消融模式（并行）

```bash
cd MGCANET/core

for mode in fixed_product semantic_only geo_only learned; do
    CUDA_VISIBLE_DEVICES=0 python train_consensus.py \
        --pretrained ../weights/yfcc100m/model_best1.pth \
        --data_tr ../data_dump/yfcc-sift-2000-train.hdf5 \
        --data_va ../data_dump/yfcc-sift-2000-val.hdf5 \
        --ablation $mode \
        --log_dir ./log_consensus \
        --train_iter 50000 \
        --val_intv 5000 \
        --batch_size 32 &
done
wait
```

### Phase 2：YFCC100M 测试集评估

```bash
mkdir -p ./results

for mode in fixed_product semantic_only geo_only learned; do
    CUDA_VISIBLE_DEVICES=0 python test_consensus.py \
        --checkpoint ./log_consensus/$mode/model_best.pth \
        --data_te ../data_dump/yfcc-sift-2000-test.hdf5 \
        --out_json ./results/${mode}_yfcc.json \
        --out_txt ./results/${mode}_yfcc.txt
done
```

### Phase 3：SUN3D 跨数据集验证

```bash
CUDA_VISIBLE_DEVICES=0 python test_consensus.py \
    --checkpoint ./log_consensus/learned/model_best.pth \
    --data_te ../data_dump/sun3d-test.hdf5 \
    --out_json ./results/learned_sun3d.json \
    --out_txt ./results/learned_sun3d.txt
```

---

## 6. 预期结果对照表

基于 handoff 文档中的零-shot MVP 结果，训练后的预期：

| 方法 | 全局 F1 | >95% F1 | >95% Precision | >95% Recall |
|------|---------|---------|----------------|-------------|
| Baseline（无 consensus） | ~0.80 | **0.556** | 0.424 | 0.838 |
| Fixed Product（零-shot） | ~0.80 | **0.624** | 0.516 | 0.732 |
| Semantic Only | ~0.80 | ~0.556 | ~0.424 | ~0.838 |
| Geo Only | ~0.80 | ~0.572 | ~0.500 | ~0.650 |
| **Learned（目标）** | ~0.82 | **>0.635** | >0.530 | >0.750 |

**Phase 2 成功标准**：`learned` 模式在 >95% bucket 上达到 F1 > 0.635（比 zero-shot 的 0.624 再提升 ≥1pp）。

---

## 7. 常见问题

### Q1：训练时出现 CUDA out of memory？

减小 batch size：

```bash
python train_consensus.py ... --batch_size 16
```

RTX2080Ti（11GB）理论上可以跑 batch_size=32，若显存紧张可降至 16。

### Q2：可以加载原始 MGCA-Net 的 checkpoint 继续训练吗？

可以。`--pretrained` 支持原始 `model_best1.pth`，`strict=False` 会自动忽略新加的 consensus 参数。

### Q3：如何只评估已有的 checkpoint，不重新训练？

直接用 `test_consensus.py`：

```bash
python test_consensus.py --checkpoint <path> --data_te <path>
```

### Q4：TensorBoard 日志在哪里？

```bash
tensorboard --logdir ./log_consensus/learned
```

### Q5：训练需要多久？

RTX2080Ti 上约 **2-4 小时**（50K steps，frozen backbone）。

---

## 8. 文件变更总结

| 文件 | 操作 | 说明 |
|------|------|------|
| `core/MGCA.py` | 修改 | 新增 `LearnableConsensus`、`compute_epipolar_distance`，集成到 `MGCANet` |
| `core/train_consensus.py` | 新增 | Frozen backbone 训练脚本 |
| `core/test_consensus.py` | 新增 | Test 评估脚本 |
| `requirements.txt` | 修改 | 移除错误的 `python==3.12.9`，保留运行时依赖 |

---

## 9. 联系方式

如有问题，参考原项目的 `README.md` 或在当前工作目录下继续询问。
