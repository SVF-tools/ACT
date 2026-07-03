#!/bin/bash
# VNN-COMP 2026 install_tool.sh for ACT.
# Arg: $1 = version string "v1". Runs once on the AWS instance; creates the
# act-py312 conda environment (installing Miniconda first if absent).
set -e

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "install_tool.sh: expected first argument '$VERSION_STRING', got '$1'"
    exit 1
fi

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# machine info (licensing / debugging), mirrors the CORA example toolkit
ip link show || true
echo "user: $USER"
nvidia-smi || true

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; installing Miniconda..."
    MC=/tmp/miniconda.sh
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MC"
    bash "$MC" -b -p "$HOME/miniconda3"
    export PATH="$HOME/miniconda3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Recent Miniconda gates channel use behind Anaconda Terms-of-Service acceptance, so
# 'conda env create' aborts non-interactively (CondaToSNonInteractiveError) and, under
# 'set -e', fails the whole install. Auto-accept it: the env var is the primary,
# network-free mechanism honoured by the conda-anaconda-tos plugin; the explicit
# accepts persist it to ~/.conda/tos and are best-effort.
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | grep -qE '/act-py312$'; then
    echo "creating act-py312 environment from environment.yml..."
    # Recent Miniconda can fail in conda-libmamba's sharded-repodata metadata
    # path before solving starts. Disable shards first (keeps libmamba's normal
    # fast solver path), then fall back to the classic solver after clearing the
    # index cache if metadata retrieval still fails.
    conda config --set repodata_use_shards false 2>/dev/null \
        || conda config --set plugins.use_sharded_repodata false 2>/dev/null \
        || true
    if ! conda env create -f "$REPO_DIR/environment.yml"; then
        echo "conda env create failed; cleaning index cache and retrying with classic solver..."
        conda clean --index-cache -y || true
        CONDA_REPODATA_USE_SHARDS=false CONDA_SOLVER=classic \
            conda env create --solver classic -f "$REPO_DIR/environment.yml"
    fi
fi

conda run -n act-py312 python -c "import torch, act; print('ACT import OK; torch', torch.__version__, 'cuda-build', torch.version.cuda, 'avail', torch.cuda.is_available())"
echo "install_tool.sh: done"
