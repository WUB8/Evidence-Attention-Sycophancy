#!/bin/bash
# Pilot run on the cluster. Submit from the repo root:
#   sbatch slurm/run_pilot.sbatch
# Override the model:  sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct slurm/run_pilot.sbatch
#
# TODO: adjust partition / GPU type to what our allocation provides (needs ~24 GB+ GPU for 7-8B).
#SBATCH --job-name=sycoattn-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
source scripts/cluster_env.sh

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
nvidia-smi
python scripts/run_pilot.py --config configs/default.yaml \
  --set model.name="$MODEL" data.max_items=50 --run-name "pilot-$SLURM_JOB_ID"
