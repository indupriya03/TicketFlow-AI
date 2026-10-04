#!/bin/bash
# Don't rely on `conda deactivate` — conda's shell functions aren't
# available in a plain script. Strip conda's env vars and PATH entries
# directly instead, and call the venv's uvicorn by its exact path.
unset CONDA_SHLVL CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_EXE
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda3 | paste -sd ':' -)

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")"
exec .venv/bin/uvicorn src.api.main:app --port 8000
