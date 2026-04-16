#!/bin/bash
# Submit a single Phase 1 relaxation job to UPPMAX Pelle.
#
# Usage:
#   bash cluster/submit_relaxation.sh --config parameters/torus_10part.yaml
#   bash cluster/submit_relaxation.sh --config parameters/torus_10part.yaml --time 24:00:00 --cpus 8
#   bash cluster/submit_relaxation.sh --config parameters/torus_10part.yaml --resume-from results/run_.../solution/surface_....h5
#   bash cluster/submit_relaxation.sh --config parameters/torus_10part.yaml --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pelle_config.sh"

# --- Defaults ---
CONFIG=""
TIME_LIMIT="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
RESUME_FROM=""
SOLUTION_DIR=""
DRY_RUN=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2"; shift 2;;
        --time)         TIME_LIMIT="$2"; shift 2;;
        --cpus)         CPUS="$2"; shift 2;;
        --mem)          MEM="$2"; shift 2;;
        --resume-from)  RESUME_FROM="$2"; shift 2;;
        --solution-dir) SOLUTION_DIR="$2"; shift 2;;
        --dry-run)      DRY_RUN=true; shift;;
        *)              echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: bash cluster/submit_relaxation.sh --config <experiment.yaml> [options]"
    exit 1
fi

CONFIG_ABS="$(abspath "$CONFIG")"
if [[ "$DRY_RUN" == false && ! -f "$CONFIG_ABS" ]]; then
    echo "Error: config file not found: $CONFIG_ABS"
    exit 1
fi

# --- Extract YAML parameters for job naming ---
SURFACE="$(extract_yaml surface "$CONFIG_ABS")"
SURFACE="${SURFACE:-unknown}"
N_PARTITIONS="$(extract_yaml n_partitions "$CONFIG_ABS")"
N_PARTITIONS="${N_PARTITIONS:-3}"
LAMBDA="$(extract_yaml lambda_penalty "$CONFIG_ABS")"
LAMBDA="${LAMBDA:-0.0}"
SEED="$(extract_yaml seed "$CONFIG_ABS")"
SEED="${SEED:-42}"

# Resolution keys depend on surface type
NT="$(extract_yaml n_theta "$CONFIG_ABS")"
NP="$(extract_yaml n_phi "$CONFIG_ABS")"
NGX="$(extract_yaml n_grid_x "$CONFIG_ABS")"
NGY="$(extract_yaml n_grid_y "$CONFIG_ABS")"

if [[ -n "$NT" && -n "$NP" ]]; then
    RES_TAG="nt${NT}_np${NP}"
elif [[ -n "$NGX" && -n "$NGY" ]]; then
    RES_TAG="gx${NGX}_gy${NGY}"
else
    RES_TAG="res0"
fi

JOB_NAME="relax_${SURFACE}_npart${N_PARTITIONS}_${RES_TAG}_lam${LAMBDA}_s${SEED}"
# SLURM job names have a 128-char limit; truncate if needed
JOB_NAME="${JOB_NAME:0:128}"

# --- Build Python command ---
# Use absolute path for the script so it works regardless of working directory
PYTHON_CMD="python ${REPO_DIR}/scripts/find_surface_partition.py --config ${CONFIG_ABS}"
if [[ -n "$RESUME_FROM" ]]; then
    PYTHON_CMD+=" --resume-from $(abspath "$RESUME_FROM")"
fi
if [[ -n "$SOLUTION_DIR" ]]; then
    PYTHON_CMD+=" --solution-dir $(abspath "$SOLUTION_DIR")"
fi

# --- Build SLURM script ---
SLURM_LOGS="${REPO_DIR}/slurm_logs"

read -r -d '' SLURM_SCRIPT << SLURM_EOF || true
#!/bin/bash
#SBATCH -A ${PROJECT_ID}
#SBATCH -c ${CPUS}
#SBATCH --threads-per-core=1
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${SLURM_LOGS}/%x_%j.out
#SBATCH -e ${SLURM_LOGS}/%x_%j.err

source ${SCRIPT_DIR}/pelle_config.sh
activate_env
# cd to PROJECT_BASE so the default relative output "results/run_..."
# resolves to PROJECT_BASE/results/run_... instead of the home directory
mkdir -p ${PROJECT_BASE}
cd ${PROJECT_BASE}

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

${PYTHON_CMD}
SLURM_EOF

# --- Submit or dry-run ---
if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN — SLURM script that would be submitted ==="
    echo "$SLURM_SCRIPT"
    echo "======================================================="
else
    mkdir -p "$SLURM_LOGS"
    TMPFILE=$(mktemp)
    echo "$SLURM_SCRIPT" > "$TMPFILE"
    JOB_ID=$(sbatch "$TMPFILE" | awk '{print $4}')
    rm -f "$TMPFILE"
    echo "Submitted: ${JOB_NAME} (Job ID: ${JOB_ID})"
    echo "Logs: ${SLURM_LOGS}/${JOB_NAME}_${JOB_ID}.out"
fi
