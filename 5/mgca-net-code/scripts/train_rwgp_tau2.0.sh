#!/bin/bash
# ============================================================================
# RWGP with tau = 2.0 (softer weighting, mild outlier suppression)
# ============================================================================
# Usage: bash scripts/train_rwgp_tau2.0.sh [GPU_ID]

set -e

GPU_ID=${1:-0}
DATA_DIR=${DATA_DIR:-"../data_dump"}
LOG_BASE=${LOG_BASE:-"./log"}

mkdir -p ${LOG_BASE}

echo "=========================================="
echo "Training MGCA-Net + RWGP (tau=2.0)"
echo "GPU: ${GPU_ID}"
echo "Data: ${DATA_DIR}"
echo "Logs: ${LOG_BASE}/rwgp_tau2.0_yfcc"
echo "=========================================="

cd core

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --run_mode train \
    --data_tr ${DATA_DIR}/yfcc-sift-2000-train.hdf5 \
    --data_va ${DATA_DIR}/yfcc-sift-2000-val.hdf5 \
    --data_te ${DATA_DIR}/yfcc-sift-2000-test.hdf5 \
    --train_batch_size 8 \
    --train_iter 500000 \
    --train_lr 3e-4 \
    --val_intv 10000 \
    --save_intv 1000 \
    --net_depth 12 \
    --net_channels 128 \
    --clusters 500 \
    --iter_num 2 \
    --use_rwgp True \
    --rwgp_tau 2.0 \
    --log_base ${LOG_BASE} \
    --log_suffix rwgp_tau2.0_yfcc \
    --SummaryWriter_base ../tensorboardX_logs/ \
    --SummaryWriter_floder MGCA-Net-RWGP-tau2.0 \
    --num_processor 16

echo "RWGP (tau=2.0) training complete. Best model: ${LOG_BASE}/rwgp_tau2.0_yfcc/train/model_best1.pth"
