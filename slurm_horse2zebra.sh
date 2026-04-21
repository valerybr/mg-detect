#!/bin/bash
#SBATCH --job-name=cut-horse2zebra
#SBATCH --output=logs/cut_h2z_%j.out
#SBATCH --error=logs/cut_h2z_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # adjust to your cluster's GPU partition name

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT="/path/to/horse2zebra"          # TODO: set this
OUTPUT_DIR="runs/cut_horse2zebra_${SLURM_JOB_ID}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── environment ────────────────────────────────────────────────────────────────
module load python/3/3.13
# packages installed via "pip install --user -r requirements.txt" are in ~/.local
# and picked up automatically — no venv activation needed

# -- OR with a venv (uncomment if using one) --
# source "/path/to/venv/bin/activate"

mkdir -p logs "${OUTPUT_DIR}"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU(s)      : ${CUDA_VISIBLE_DEVICES}"
echo "Data root   : ${DATA_ROOT}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Started at  : $(date)"

cd "${PROJECT_DIR}"

python train_cut_horse2zebra.py \
    --data-root   "${DATA_ROOT}" \
    --output-dir  "${OUTPUT_DIR}" \
    --img-size    256 \
    --batch-size  1 \
    --num-workers 8 \
    --n-epochs    200 \
    --n-epochs-decay 200 \
    --lr          2e-4 \
    --lambda-nce  1.0 \
    --lambda-idt  1.0 \
    --ngf         64 \
    --ndf         64 \
    --n-blocks    9 \
    --save-every  10

# To resume from a checkpoint, add:
#   --resume "${OUTPUT_DIR}/cut_ckpt_epoch_050.pt"

# To enable W&B logging, add:
#   --wandb --wandb-project cut-horse2zebra

echo "Finished at : $(date)"
