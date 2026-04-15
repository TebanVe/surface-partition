#!/bin/bash
# Shared configuration for UPPMAX Pelle cluster submission scripts.
# Sourced by submit_relaxation.sh, submit_refinement.sh, submit_sweep.sh.
#
# FIRST-TIME SETUP ON PELLE:
#   1. Edit PROJECT_ID and PROJECT_BASE below to match your UPPMAX allocation
#   2. Clone the repo:  git clone <url> ~/surface-partition
#   3. Create venv:
#        module load Python/3.11.5-GCCcore-13.2.0
#        python -m venv ${HOME}/venvs/surface-partition
#        source ${HOME}/venvs/surface-partition/bin/activate
#        cd ~/surface-partition && pip install -e ".[all]"
#   4. Verify:  python -c "import numpy, scipy, h5py; print('OK')"
#
# Directory conventions on UPPMAX:
#   ~/                  — Code, repos, scripts (small, backed up, 32 GB quota)
#   /proj/<allocation>/ — Large data: results, HDF5 solutions (large quota, not backed up)

# --- Project Configuration (USER MUST EDIT) ---
PROJECT_ID="uppmax2025-2-XXX"
PROJECT_BASE="/proj/uppmax2025-2-XXX"
REPO_DIR="${HOME}/surface-partition"

# --- Python Environment ---
PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"
VENV_DIR="${HOME}/venvs/surface-partition"

# --- SLURM Defaults ---
DEFAULT_TIME="12:00:00"
DEFAULT_CPUS=4
DEFAULT_MEM="16G"

# --- Helpers ---

activate_env() {
    module load "${PYTHON_MODULE}"
    if [ -d "${VENV_DIR}" ]; then
        source "${VENV_DIR}/bin/activate"
    else
        echo "ERROR: venv not found at ${VENV_DIR}"
        echo "Create it with:"
        echo "  module load ${PYTHON_MODULE}"
        echo "  python -m venv ${VENV_DIR}"
        echo "  source ${VENV_DIR}/bin/activate"
        echo "  cd ${REPO_DIR} && pip install -e '.[all]'"
        exit 1
    fi
    export MPLBACKEND=Agg
}

abspath() {
    python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

extract_yaml() {
    local key="$1" file="$2"
    grep -E "^[[:space:]]*${key}:[[:space:]]*" "$file" | head -n1 \
        | sed 's/^[^:]*:[[:space:]]*//' \
        | sed 's/[[:space:]]*#.*//' \
        | tr -d '"' \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true
}
