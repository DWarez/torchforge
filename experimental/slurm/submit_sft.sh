#!/bin/bash

#SBATCH --qos=qos_llm_min
#SBATCH --account=AIFAC_L07_016
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclude="lrdn[1831-3456]"

echo "Starting SFT training job"

eval "$(conda shell.bash hook)"

conda activate forge

export TORCH_COMPILE_DISABLE=1
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
export TORCHSTORE_RDMA_ENABLED=0

cd /leonardo_scratch/fast/iGen_train/$USER/forge

srun python -m apps.sft.main --config experimental/slurm/${CONFIG_NAME}.yaml