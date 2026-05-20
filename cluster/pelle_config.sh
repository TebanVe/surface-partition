#!/bin/bash
# Shared configuration for UPPMAX Pelle cluster submission scripts.
# Sourced by submit_relaxation.sh, submit_refinement.sh, submit_sweep.sh.
#
# FIRST-TIME SETUP ON PELLE:
#   1. Clone the repo:  git clone <url> /home/teban66/projects/surface-partition
#   2. Create venv (first time only):
#        module load Python/3.11.5-GCCcore-13.3.0
#        python -m venv /home/teban66/venvs/surface-partition
#        source /home/teban66/venvs/surface-partition/bin/activate
#        pip install --upgrade pip
#   3. Install dependencies (cyipopt excluded — IPOPT not available on Pelle):
#        cd /home/teban66/projects/surface-partition && pip install -e ".[viz,implicit]"
#   4. Verify:  python -c "import numpy, scipy, h5py; print('OK')"
#
# Directory conventions on UPPMAX:
#   ~/                  — Code, repos, scripts (small, backed up, 32 GB quota)
#   /proj/<allocation>/ — Large data: results, HDF5 solutions (large quota, not backed up)

# --- Project Configuration ---
PROJECT_ID="uppmax2025-2-534"
PROJECT_BASE="/proj/snic2020-15-36/private/LINKED_LST_MANIFOLD"
REPO_DIR="/home/teban66/projects/surface-partition"
RESULTS_BASE="${PROJECT_BASE}/results"

# --- Python Environment ---
# Verify available modules on Pelle with: module spider Python
PYTHON_MODULE="Python/3.11.5-GCCcore-13.3.0"
VENV_DIR="/home/teban66/venvs/surface-partition"

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
        echo "  cd ${REPO_DIR} && pip install -e '.[viz,implicit]'"
        exit 1
    fi
    export MPLBACKEND=Agg
}

abspath() {
    if [[ "$1" = /* ]]; then echo "$1"; else echo "${PWD}/$1"; fi
}

extract_yaml() {
    local key="$1" file="$2"
    grep -E "^[[:space:]]*${key}:[[:space:]]*" "$file" | head -n1 \
        | sed 's/^[^:]*:[[:space:]]*//' \
        | sed 's/[[:space:]]*#.*//' \
        | tr -d '"' \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true
}
