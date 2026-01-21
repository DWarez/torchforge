#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CONFIG_NAME="${1}"
RES_DIR="/leonardo_scratch/fast/iGen_train/$USER/forge/logs/$SLURM_JOB_NAME"

export NUM_NODES=1
export GPUS_PER_NODE=4
export CPUS_PER_NODE=4

export HF_CACHE_DIR="/leonardo_scratch/fast/iGen_train/$USER/hf_cache"
export HF_HOME="/leonardo_scratch/fast/iGen_train/$USER/hf_cache"
export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# export NCCL_DEBUG=INFO 
export NCCL_TIMEOUT=600
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_LAUNCH_BLOCKING=1

export MASTER_PORT=9251


CONDA_PATH="/leonardo/home/userexternal/$USER/miniconda3"
FORGE_DIR="/leonardo_scratch/fast/iGen_train/$USER/forge"

export TORCH_COMPILE_DISABLE=1
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
export TORCHSTORE_RDMA_ENABLED=0

export MONARCH_LOG_LEVEL=DEBUG
export TMPDIR="/leonardo_scratch/fast/iGen_train/$USER/tmp"
export XDG_RUNTIME_DIR="/leonardo_scratch/fast/iGen_train/$USER/tmp"
export TEMPDIR="/leonardo_scratch/fast/iGen_train/$USER/tmp"
mkdir -p "$TMPDIR"


sbatch --verbose --job-name="${CONFIG_NAME}_controller" \
       --export=ALL,CONFIG_NAME="${CONFIG_NAME}" \
       --account=AIFAC_L07_016 \
       --qos=qos_llm_min \
       --partition=boost_usr_prod \
       --time=00:30:00 \
       --nodes=${NUM_NODES} \
       --gres=gpu:${GPUS_PER_NODE} \
       --cpus-per-task=${CPUS_PER_NODE} \
       --exclude="lrdn[1831-3456]" \
       --error="$RES_DIR/%x_%j.err" \
       --output="$RES_DIR/%x_%j.out" \
       experimental/slurm/custom_submit_sft.sh


# Usage:
# ./experimental/slurm/submit.sh qwen3_8b
# ./experimental/slurm/submit.sh qwen3_32b
# ./experimental/slurm/submit.sh qwen3_30b_a3b
