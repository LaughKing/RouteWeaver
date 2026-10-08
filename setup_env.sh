#!/usr/bin/env bash
# Create the training environment. The ORDER matters: vLLM resolves torch and
# numpy first, the pins are then restored, verl goes in after that, and numpy
# is restored once more (step 5 explains the one accepted conflict).
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

# 1) vLLM first: brings torch 2.8.0+cu128 and numpy 2.2.6 (the resolver also
#    pulls transformers 5.x / ray 2.56.1, re-pinned in step 2).
$PIP install vllm==0.11.0

# 2) Pin the versions this code is written against.
$PIP install transformers==4.56.1 ray==2.56.0 tensordict==0.10.0

# 3) flash-attn (cu12 / torch 2.8 / cp312).
if [ -n "$FLASH_ATTN_WHEEL" ]; then
    $PIP install "$FLASH_ATTN_WHEEL"
else
    $PIP install flash-attn==2.8.1 --no-build-isolation
fi

# 4) verl v0.8.0, unpatched (installs as version string 0.8.0.dev0).
#    Side effect: downgrades numpy to 1.26.4, because verl declares
#    numpy<2.0.0. Undone in step 5.
$PIP install "git+https://github.com/volcengine/verl.git@v0.8.0"

# 5) Restore numpy. KNOWN DECLARED CONFLICT, accepted deliberately: `pip check`
#    reports verl requiring numpy<2.0.0 against the installed 2.2.6. vLLM
#    0.11.0's dependency tree (scipy, opencv-headless, cupy-cuda12x) requires
#    numpy>=2, so the two declarations cannot both be satisfied. numpy 2.2.6 is
#    the side that runs.
$PIP install numpy==2.2.6

# requirements.lock.txt at the repo root is a full pip freeze of the result.
