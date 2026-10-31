#!/bin/bash
# Create (first time) and activate the project environment on the cluster.
#   source scripts/cluster_env.sh
#
# TODO: adjust the module name to what `module avail` shows on our cluster.
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

ENV_NAME=sycoattn
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.11
  conda run -n "$ENV_NAME" pip install -e ".[dev]"
fi
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

# Keep model weights out of the small home quota. TODO: point at our scratch dir.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
