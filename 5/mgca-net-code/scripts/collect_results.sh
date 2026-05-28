#!/bin/bash
# ============================================================================
# Collect test results from all ablation experiments
# ============================================================================
# Usage: bash scripts/collect_results.sh
# Extracts metrics matching MGCA-Net paper format:
#   - Table 1: P / R / F1 (outlier rejection)
#   - Table 2: mAP@5, AUC@5, mAP@20, AUC@20 (pose estimation)
#   - Table 3: mAP@5, mAP@20 (ablation)

set -e

LOG_BASE=${LOG_BASE:-"./log"}
OUTPUT_FILE="${LOG_BASE}/ablation_summary.csv"

echo "experiment,tau,precision,recall,f1,mAP5,mAP10,mAP15,mAP20,AUC5,AUC10,AUC20" > ${OUTPUT_FILE}

EXPERIMENTS=(
    "baseline_yfcc:None"
    "rwgp_tau0.5_yfcc:0.5"
    "rwgp_tau1.0_yfcc:1.0"
    "rwgp_tau2.0_yfcc:2.0"
    "rwgp_tau5.0_yfcc:5.0"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name tau_val <<< "$exp"
    RES_DIR="${LOG_BASE}/${exp_name}/test"

    if [ ! -d "$RES_DIR" ]; then
        echo "Warning: Results directory not found: $RES_DIR"
        continue
    fi

    # --- Table 1: Outlier Rejection (P / R / F1) ---
    if [ -f "${RES_DIR}/precision_recall_f1.txt" ]; then
        prf_vals=($(tail -1 ${RES_DIR}/precision_recall_f1.txt))
        precision=${prf_vals[0]:-"N/A"}
        recall=${prf_vals[1]:-"N/A"}
        f1=${prf_vals[2]:-"N/A"}
    else
        precision="N/A"; recall="N/A"; f1="N/A"
    fi

    # --- Table 2: Pose Estimation (mAP) ---
    if [ -f "${RES_DIR}/map.txt" ]; then
        map_vals=($(tail -1 ${RES_DIR}/map.txt))
        mAP5=${map_vals[0]:-"N/A"}
        mAP10=${map_vals[1]:-"N/A"}
        mAP15=${map_vals[2]:-"N/A"}
        mAP20=${map_vals[3]:-"N/A"}
    else
        mAP5="N/A"; mAP10="N/A"; mAP15="N/A"; mAP20="N/A"
    fi

    # --- Table 2: Pose Estimation (AUC) ---
    if [ -f "${RES_DIR}/auc.txt" ]; then
        auc_vals=($(tail -1 ${RES_DIR}/auc.txt))
        AUC5=${auc_vals[0]:-"N/A"}
        AUC10=${auc_vals[1]:-"N/A"}
        AUC20=${auc_vals[2]:-"N/A"}
    else
        AUC5="N/A"; AUC10="N/A"; AUC20="N/A"
    fi

    echo "${exp_name},${tau_val},${precision},${recall},${f1},${mAP5},${mAP10},${mAP15},${mAP20},${AUC5},${AUC10},${AUC20}" >> ${OUTPUT_FILE}
    echo "Collected: ${exp_name}  P=${precision} R=${recall} F1=${f1}  mAP@5=${mAP5} mAP@20=${mAP20}  AUC@5=${AUC5} AUC@20=${AUC20}"
done

echo ""
echo "=========================================="
echo "Results summary saved to: ${OUTPUT_FILE}"
echo "=========================================="
cat ${OUTPUT_FILE}
