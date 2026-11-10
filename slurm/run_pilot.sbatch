#!/bin/bash
# Run on the cluster (Explorer). Submit from the repo root; logs/ must exist before sbatch
# (Slurm opens the log file before this script runs). Set variables in the shell, not with
# --export=VAR=..., because --export splits on commas:
#   mkdir -p logs && sbatch slurm/run_pilot.sbatch                         # pilot split (59 items), Qwen-7B
#   mkdir -p logs && MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch --export=ALL slurm/run_pilot.sbatch
#   mkdir -p logs && SPLIT=main MODEL=Qwen/Qwen2.5-14B-Instruct sbatch --export=ALL slurm/run_pilot.sbatch
# The main split (273 items) takes about 4.6x the pilot's time: set --time from the pilot run
# (the `gpu` partition limit is 8 h). If a main run does not fit, the same person can split it
# into several jobs with CONDITIONS='[A,B3,B4]' etc. (all conditions of a model stay with one person).
# 14B in bf16 needs ~30 GB for weights. Explorer `gpu` partition has H200 (141 GB) and A100 80 GB nodes;
# for the A100s use: sbatch --gres=gpu:a100:1 --constraint=a100@80g ...
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
SPLIT=${SPLIT:-pilot}                    # pilot (tuning only) | main (reported results)
CONDITIONS=${CONDITIONS:-}               # empty = all conditions in configs/default.yaml
EXTRA=()
if [ -n "$CONDITIONS" ]; then EXTRA+=(dialogue.conditions="$CONDITIONS"); fi
nvidia-smi
# Fail fast if the pip torch wheel doesn't match the node's CUDA driver.
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python scripts/run_pilot.py --config configs/default.yaml \
  --set model.name="$MODEL" data.split="$SPLIT" data.max_items=null ${EXTRA[@]+"${EXTRA[@]}"} \
  --run-name "$SPLIT-$SLURM_JOB_ID"
