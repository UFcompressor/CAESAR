#!/usr/bin/env bash
# Sourced by run.sh before any Python/Torch commands. Paths can be overridden
# for another cluster; these defaults match the UF HiPerGator environment.
source "$HOME/.bashrc"
CAESAR_ENV_FILE=${CAESAR_ENV_FILE:-/lustre/blue2/ranka/eklasky/data/test/set_env_ufl_caesar.sh}
CAESAR_VENV=${CAESAR_VENV:-/lustre/blue2/ranka/eklasky/caesar_venv}
if [[ ! -f "$CAESAR_ENV_FILE" ]]; then
    echo "ERROR: Missing environment setup: $CAESAR_ENV_FILE" >&2
    return 1
fi
source "$CAESAR_ENV_FILE"
if [[ ! -f "$CAESAR_VENV/bin/activate" ]]; then
    echo "ERROR: Missing Python environment: $CAESAR_VENV/bin/activate" >&2
    return 1
fi
source "$CAESAR_VENV/bin/activate"
# Avoid importing a different Torch from ~/.local instead of the selected venv.
export PYTHONNOUSERSITE=1
