#!/bin/bash
# Create (first time) and activate the project environment on the cluster.
#   source scripts/cluster_env.sh
#
# Big files live in scratch, not home: the conda env, pip/conda download caches
# and Hugging Face model weights all go under /scratch/$USER.
module load anaconda3/2024.06   # Explorer; `module avail anaconda` lists others

SCRATCH=/scratch/$USER
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs

ENV_PREFIX=$SCRATCH/conda_envs/sycoattn
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
  conda run -p "$ENV_PREFIX" pip install -e ".[dev]"
fi
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"
