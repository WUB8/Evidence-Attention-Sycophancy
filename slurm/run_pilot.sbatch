#!/bin/bash
# Run on the cluster. Submit from the repo root:
#   sbatch slurm/run_pilot.sbatch                                   # pilot split (59 items), Qwen-7B
#   sbatch --export=ALL,SPLIT=main slurm/run_pilot.sbatch           # main experiment (273 items)
#   sbatch --export=ALL,SPLIT=main,MODEL=meta-llama/Llama-3.1-8B-Instruct slurm/run_pilot.sbatch
# The main split needs more time than the pilot: set --time from the pilot's run time (x ~4.6).
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
SPLIT=${SPLIT:-pilot}
nvidia-smi
python scripts/run_pilot.py --config configs/default.yaml \
  --set model.name="$MODEL" data.split="$SPLIT" data.max_items=null --run-name "$SPLIT-$SLURM_JOB_ID"
