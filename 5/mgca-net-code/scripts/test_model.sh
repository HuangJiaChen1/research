#!/bin/bash
# ============================================================================
# Test a trained model on YFCC100M test set
# ============================================================================
# Usage: bash scripts/test_model.sh [MODEL_PATH] [GPU_ID]
# Example: bash scripts/test_model.sh ./log/rwgp_tau1.0_yfcc/train/model_best1.pth 0

set -e

MODEL_PATH=${1:-""}
GPU_ID=${2:-0}
DATA_DIR=${DATA_DIR:-"../data_dump"}

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Please provide the path to a trained model."
    echo "Usage: bash scripts/test_model.sh <model_path> [gpu_id]"
    echo "Example: bash scripts/test_model.sh ./log/rwgp_tau1.0_yfcc/train/model_best1.pth 0"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "Testing model: ${MODEL_PATH}"
echo "GPU: ${GPU_ID}"
echo "Data: ${DATA_DIR}/yfcc-sift-2000-test.hdf5"
echo "=========================================="

cd core

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --run_mode test \
    --data_te ${DATA_DIR}/yfcc-sift-2000-test.hdf5 \
    --model_path $(dirname ${MODEL_PATH}) \
    --use_rwgp True \
    --train_batch_size 1 \
    --net_depth 12 \
    --net_channels 128 \
    --clusters 500 \
    --iter_num 2

echo "Test complete. Results saved to: $(dirname ${MODEL_PATH})/../test/"
