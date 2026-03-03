#!/bin/bash
set -e

# Pi0.5 Droid Hybrid pretraining (multi-node style, aligned with run_egodex_pretrain.sh).
NODE_RANK=${1:-0}
EXPERIMENT_NAME=${2:-"pi05_droid_hybrid"}
BATCH_SIZE=${3:-128}
NUM_STEPS=${4:-400000}
RESUME=${5:-false}
GPUS=${6:-"0,1,2,3,4,5,6,7"}
TRAIN_MODE=${7:-"full"}
ACTION_LOSS_WEIGHT=${8:-1.0}
TRACK_LOSS_WEIGHT=${9:-1.0}
VALIDATION_INTERVAL=${10:-1000}
VISUALIZE_TRACKS=${11:-true}

# Distributed defaults (same style as EgoDex script).
MASTER_IP=${MASTER_IP:-"10.40.0.83"}
MASTER_PORT=${MASTER_PORT:-"13218"}
NUM_NODES=${NUM_NODES:-4}

cd "$(dirname "$(dirname "$(realpath "$0")")")"

unset PYTHONPATH
export PYTHONPATH="$(pwd)/src"
export CUDA_VISIBLE_DEVICES="$GPUS"
export JAXTYPING_DISABLE=1
unset http_proxy
unset https_proxy

# Proxy + local tokenizer/checkpoint assets
#export http_proxy="http://192.168.32.28:18000"
#export https_proxy="http://192.168.32.28:18000"
# export HTTP_PROXY="$http_proxy"
# export HTTPS_PROXY="$https_proxy"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="/mnt/kevin/vlm_models/paligemma_tokenizer.model"
export OPENPI_FAST_TOKENIZER_PATH="/mnt/kevin/vlm_models/fast_tokenizer"

# JAX distributed
export JAX_COORDINATOR_ADDRESS="${MASTER_IP}:${MASTER_PORT}"
export JAX_PROCESS_COUNT="${NUM_NODES}"
export JAX_PROCESS_ID="${NODE_RANK}"

# WandB
export WANDB_ENTITY="kzyz"
export WANDB_API_KEY="8c5ea72862d197cebbf90dae91b2f3e869fc3169"

# Validation visuals
export OPENPI_VALIDATION_INTERVAL="${VALIDATION_INTERVAL}"
export OPENPI_VISUALIZE_TRACKS="${VISUALIZE_TRACKS}"

LOCAL_GPU_COUNT=$(echo "$GPUS" | awk -F',' '{print NF}')
# FSDP across all devices (4 nodes * 8 GPUs = 32)
FSDP_DEVICES=${FSDP_DEVICES:-8}

# For distributed runs, we want consistent experiment names across nodes.
# If EXPERIMENT_NAME is provided by user (arg 2), use it directly if it already looks like a full ID,
# or append date only (YYYYMMDD) to be safer across slight launch time diffs?
# Better: Just use the user-provided name exactly if provided, or date-stamped if not.
# The user usually provides "pi05_hybrid_run1".
# If we append seconds, it will differ.
if [ "$NUM_NODES" -gt 1 ]; then
    # In distributed mode, rely on user to provide a unique name or use date up to minute/hour to be safe?
    # Or just don't append date if user provided a name.
    if [ "$2" ]; then
        EXP_NAME="${EXPERIMENT_NAME}"
    else
        # Fallback if no name provided: use date but maybe less granular to avoid race?
        # Ideally user provides the name.
        EXP_NAME="${EXPERIMENT_NAME}_$(date +%Y%m%d)"
    fi
else
    # Single node: append full timestamp for uniqueness
    EXP_NAME="${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================"
echo "Pi0.5 Droid Hybrid Training (Distributed)"
echo "============================================"
echo "Experiment:      ${EXP_NAME}"
echo "Node rank:       ${NODE_RANK}/${NUM_NODES}"
echo "Master:          ${MASTER_IP}:${MASTER_PORT}"
echo "GPUs:            ${GPUS} (local=${LOCAL_GPU_COUNT})"
echo "FSDP devices:    ${FSDP_DEVICES}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Train steps:     ${NUM_STEPS}"
echo "Train mode:      ${TRAIN_MODE}"
echo "Loss weights:    action=${ACTION_LOSS_WEIGHT}, track=${TRACK_LOSS_WEIGHT}"
echo "Validation:      interval=${OPENPI_VALIDATION_INTERVAL}, visualize=${OPENPI_VISUALIZE_TRACKS}"
echo "============================================"

TRAIN_CMD="uv run python scripts/train_bypass.py pi05_droid_hybrid"
TRAIN_CMD="$TRAIN_CMD --exp-name \"${EXP_NAME}\""
TRAIN_CMD="$TRAIN_CMD --train-mode \"${TRAIN_MODE}\""
TRAIN_CMD="$TRAIN_CMD --fsdp-devices \"${FSDP_DEVICES}\""
TRAIN_CMD="$TRAIN_CMD --batch-size \"${BATCH_SIZE}\""
TRAIN_CMD="$TRAIN_CMD --num-train-steps \"${NUM_STEPS}\""
TRAIN_CMD="$TRAIN_CMD --action-loss-weight \"${ACTION_LOSS_WEIGHT}\""
TRAIN_CMD="$TRAIN_CMD --track-loss-weight \"${TRACK_LOSS_WEIGHT}\""
TRAIN_CMD="$TRAIN_CMD --validation-interval \"${VALIDATION_INTERVAL}\""
TRAIN_CMD="$TRAIN_CMD --visualize-tracks \"${VISUALIZE_TRACKS}\""
TRAIN_CMD="$TRAIN_CMD --wandb-enabled true"

if [ "$RESUME" = "true" ]; then
  TRAIN_CMD="$TRAIN_CMD --resume"
else
  TRAIN_CMD="$TRAIN_CMD --overwrite"
fi

echo -e "\nTraining command:\n${TRAIN_CMD}\n"
eval "${TRAIN_CMD}"
