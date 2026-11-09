#!/bin/bash
# Pilot run on the cluster. Submit from the repo root (logs/ must exist before sbatch,
# Slurm opens the log file before this script runs):
#   mkdir -p logs && sbatch slurm/run_pilot.sbatch
# Override the model:  sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct slurm/run_pilot.sbatch
# 14B pilot (set vars in the shell; --export splits on commas):
#   mkdir -p logs && MODEL=Qwen/Qwen2.5-14B-Instruct CONDITIONS='[A,C,B2,B5]' sbatch --export=ALL slurm/run_pilot.sbatch
# 14B in bf16 needs ~30 GB for weights. Explorer `gpu` partition (8 h limit) has H200 (141 GB) and
# A100 80 GB nodes; for the A100s use: sbatch --gres=gpu:a100:1 --constraint=a100@80g ...
#
#SBATCH --job-name=sycoattn-pilot
#SBATCH --account=cs6120.202710
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
source scripts/cluster_env.sh   # conda activation scripts break under set -u
set -u

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
CONDITIONS=${CONDITIONS:-[A,B,C,D,control]}
ITEMS=${ITEMS:-50}
nvidia-smi
# Fail fast if the pip torch wheel doesn't match the node's CUDA driver.
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python scripts/run_pilot.py --config configs/default.yaml \
  --set model.name="$MODEL" data.max_items="$ITEMS" dialogue.conditions="$CONDITIONS" --run-name "pilot-$SLURM_JOB_ID"
