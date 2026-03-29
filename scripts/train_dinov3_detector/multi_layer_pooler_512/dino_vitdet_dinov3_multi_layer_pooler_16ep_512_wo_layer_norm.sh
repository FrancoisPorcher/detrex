#!/usr/bin/env bash
#SBATCH --job-name=dino_vitdet_dinov3_multi_layer_pooler_16ep_512_wo_layer_norm
#SBATCH --output=/private/home/francoisporcher/detrex/experiments/logs/dino_vitdet_dinov3_multi_layer_pooler_16ep_512_wo_layer_norm/slurm_%j.out
#SBATCH --error=/private/home/francoisporcher/detrex/experiments/logs/dino_vitdet_dinov3_multi_layer_pooler_16ep_512_wo_layer_norm/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=80
#SBATCH --time=40:00:00

set -euo pipefail

cd /private/home/francoisporcher/detrex

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate detrex39

export PYTHONPATH="/private/home/francoisporcher/detrex:${PYTHONPATH:-}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
if [[ -z "${MASTER_ADDR}" ]]; then
    MASTER_ADDR=$(hostname -s)
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_DEBUG=INFO

NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
if [[ -z "${GPUS_PER_NODE}" || "${GPUS_PER_NODE}" == "(null)" ]]; then
    GPUS_PER_NODE=8
fi

CPUS_PER_GPU=10
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-$((CPUS_PER_GPU * GPUS_PER_NODE))}
NUM_WORKERS=${NUM_WORKERS:-3}

PYTHON_SCRIPT="/private/home/francoisporcher/detrex/projects/dino/train_net.py"
CONFIG_PATH="/private/home/francoisporcher/detrex/projects/dino/configs/dino-vitdet/dino_vitdet_dinov3_multi_layer_pooler_16ep_512_wo_layer_norm.py"

export NNODES
export GPUS_PER_NODE
export NUM_WORKERS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Nodes: ${NNODES}, GPUs per node: ${GPUS_PER_NODE}, CPUs per GPU: ${CPUS_PER_GPU}, CPUs per task: ${CPUS_PER_TASK}, num_workers per GPU: ${NUM_WORKERS}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Node list: ${SLURM_JOB_NODELIST}" >&2

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
    python "${PYTHON_SCRIPT}" \
        --config-file "${CONFIG_PATH}" \
        --num-gpus "${GPUS_PER_NODE}" \
        --num-machines "${NNODES}" \
        --machine-rank "${SLURM_NODEID}" \
        --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
        dataloader.train.num_workers="${NUM_WORKERS}" \
        dataloader.test.num_workers="${NUM_WORKERS}"
