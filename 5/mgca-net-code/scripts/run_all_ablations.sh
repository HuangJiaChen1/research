#!/bin/bash
# ============================================================================
# Run ALL ablation experiments in parallel (requires 5 GPUs)
# ============================================================================
# Usage: bash scripts/run_all_ablations.sh
# This will launch 5 training jobs simultaneously on GPUs 0-4.
# Adjust GPU assignments below if your setup differs.

set -e

DATA_DIR=${DATA_DIR:-"../data_dump"}
LOG_BASE=${LOG_BASE:-"./log"}

mkdir -p ${LOG_BASE}

echo "=========================================="
echo "Launching ALL ablation experiments"
echo "This requires 5 GPUs (or adjust script for fewer)"
echo "=========================================="

# Launch baseline on GPU 0
echo "[1/5] Launching BASELINE on GPU 0..."
bash scripts/train_baseline.sh 0 > ${LOG_BASE}/train_baseline.log 2>&1 &
PID_BASELINE=$!

# Launch RWGP tau=0.5 on GPU 1
echo "[2/5] Launching RWGP tau=0.5 on GPU 1..."
bash scripts/train_rwgp_tau0.5.sh 1 > ${LOG_BASE}/train_rwgp_tau0.5.log 2>&1 &
PID_TAU05=$!

# Launch RWGP tau=1.0 on GPU 2
echo "[3/5] Launching RWGP tau=1.0 on GPU 2..."
bash scripts/train_rwgp_tau1.0.sh 2 > ${LOG_BASE}/train_rwgp_tau1.0.log 2>&1 &
PID_TAU10=$!

# Launch RWGP tau=2.0 on GPU 3
echo "[4/5] Launching RWGP tau=2.0 on GPU 3..."
bash scripts/train_rwgp_tau2.0.sh 3 > ${LOG_BASE}/train_rwgp_tau2.0.log 2>&1 &
PID_TAU20=$!

# Launch RWGP tau=5.0 on GPU 4
echo "[5/5] Launching RWGP tau=5.0 on GPU 4..."
bash scripts/train_rwgp_tau5.0.sh 4 > ${LOG_BASE}/train_rwgp_tau5.0.log 2>&1 &
PID_TAU50=$!

echo ""
echo "=========================================="
echo "All jobs launched!"
echo "=========================================="
echo "Baseline      PID: ${PID_BASELINE}  Log: ${LOG_BASE}/train_baseline.log"
echo "RWGP tau=0.5  PID: ${PID_TAU05}     Log: ${LOG_BASE}/train_rwgp_tau0.5.log"
echo "RWGP tau=1.0  PID: ${PID_TAU10}     Log: ${LOG_BASE}/train_rwgp_tau1.0.log"
echo "RWGP tau=2.0  PID: ${PID_TAU20}     Log: ${LOG_BASE}/train_rwgp_tau2.0.log"
echo "RWGP tau=5.0  PID: ${PID_TAU50}     Log: ${LOG_BASE}/train_rwgp_tau5.0.log"
echo ""
echo "Monitor with: tail -f ${LOG_BASE}/train_*.log"
echo "Wait for all with: wait ${PID_BASELINE} ${PID_TAU05} ${PID_TAU10} ${PID_TAU20} ${PID_TAU50}"
