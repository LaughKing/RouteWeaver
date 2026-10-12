#!/usr/bin/env bash
# Create the training environment. Install order matters.
#
#   bash setup_env.sh                        # creates conda env $ENV_NAME
#   FLASH_ATTN_WHEEL=... bash setup_env.sh   # use a prebuilt flash-attn wheel
set -euo pipefail

CONDA=${CONDA:-conda}
ENV_NAME=${ENV_NAME:-routeweaver}
PIP=${CONDA_ENVS:-$HOME/miniconda3/envs}/$ENV_NAME/bin/pip
# A prebuilt wheel if you have one; otherwise it is built from source below.
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-}

$CONDA create -n $ENV_NAME python=3.12.0 -y

# 1) vLLM first (brings torch and numpy).
$PIP install vllm==0.11.0

# 2) Pin the versions this code is written against.
$PIP install transformers==4.56.1 ray==2.56.0 tensordict==0.10.0

# 3) flash-attn (cu12 / torch 2.8 / cp312).
if [ -n "$FLASH_ATTN_WHEEL" ]; then
    $PIP install "$FLASH_ATTN_WHEEL"
else
    $PIP install flash-attn==2.8.1 --no-build-isolation
fi

# 4) verl v0.8.0, unpatched (downgrades numpy; restored in step 5).
$PIP install "git+https://github.com/volcengine/verl.git@v0.8.0"

# 5) Restore numpy 2.2.6 (verl declares numpy<2, vLLM needs numpy>=2).
$PIP install numpy==2.2.6

# requirements.lock.txt at the repo root is a full pip freeze of the result.
