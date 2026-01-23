#!/bin/bash
echo "Starting SFT training job"

eval "$(conda shell.bash hook)"

conda activate forge

export TORCH_COMPILE_DISABLE=1
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
export TORCHSTORE_RDMA_ENABLED=0

cd /leonardo_scratch/fast/iGen_train/$USER/forge

srun python -m apps.sft.main --config experimental/slurm/${CONFIG_NAME}.yaml