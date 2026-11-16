#!/usr/bin/env bash
# Sourced by run.sh before any Python/Torch commands. Paths can be overridden
# for another cluster; these defaults match the UF HiPerGator environment.
export CAESAR_GPU_STAGE="source ~/.bashrc"
echo "[GPU] $CAESAR_GPU_STAGE"
# Interactive shell initialization commonly ends with a false conditional or
# unavailable interactive command. Do not let that prevent explicit site setup.
if source "$HOME/.bashrc"; then
    :
else
    echo "WARNING: ~/.bashrc returned nonzero; continuing with explicit cluster setup" >&2
fi
CAESAR_ENV_FILE=${CAESAR_ENV_FILE:-/lustre/blue2/ranka/eklasky/data/test/set_env_ufl_caesar.sh}
CAESAR_VENV=${CAESAR_VENV:-/lustre/blue2/ranka/eklasky/caesar_venv}
if [[ ! -f "$CAESAR_ENV_FILE" ]]; then
    echo "ERROR: Missing environment setup: $CAESAR_ENV_FILE" >&2
    return 1
fi
export CAESAR_GPU_STAGE="source $CAESAR_ENV_FILE"
echo "[GPU] $CAESAR_GPU_STAGE"
if source "$CAESAR_ENV_FILE"; then
    :
else
    echo "ERROR: Cluster environment setup failed: $CAESAR_ENV_FILE" >&2
    return 1
fi
if [[ ! -f "$CAESAR_VENV/bin/activate" ]]; then
    echo "ERROR: Missing Python environment: $CAESAR_VENV/bin/activate" >&2
    return 1
fi
export CAESAR_GPU_STAGE="activate $CAESAR_VENV"
echo "[GPU] $CAESAR_GPU_STAGE"
if source "$CAESAR_VENV/bin/activate"; then
    :
else
    echo "ERROR: Python environment activation failed: $CAESAR_VENV" >&2
    return 1
fi
# Avoid importing a different Torch from ~/.local instead of the selected venv.
export PYTHONNOUSERSITE=1
