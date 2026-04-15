#!/bin/bash
# Submit a single Phase 2 refinement job to UPPMAX Pelle.
#
# Usage:
#   bash cluster/submit_refinement.sh --solution results/run_.../solution/surface_....h5 --config parameters/torus_10part.yaml
#   bash cluster/submit_refinement.sh --solution results/run_.../refinement/ipopt_.../iteration_003.h5 --config parameters/torus_10part.yaml --method ipopt --exact-hessian
#   bash cluster/submit_refinement.sh --solution <h5> --config <yaml> --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pelle_config.sh"

# --- Defaults (refinement can be long) ---
SOLUTION=""
CONFIG=""
TIME_LIMIT="24:00:00"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
DRY_RUN=false
EXTRA_ARGS=()

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --solution)         SOLUTION="$2"; shift 2;;
        --config)           CONFIG="$2"; shift 2;;
        --time)             TIME_LIMIT="$2"; shift 2;;
        --cpus)             CPUS="$2"; shift 2;;
        --mem)              MEM="$2"; shift 2;;
        --dry-run)          DRY_RUN=true; shift;;
        # Passthrough flags for refine_perimeter.py
        --method)           EXTRA_ARGS+=("--method" "$2"); shift 2;;
        --max-iterations)   EXTRA_ARGS+=("--max-iterations" "$2"); shift 2;;
        --max-opt-iter)     EXTRA_ARGS+=("--max-opt-iter" "$2"); shift 2;;
        --tolerance)        EXTRA_ARGS+=("--tolerance" "$2"); shift 2;;
        --boundary-tol)     EXTRA_ARGS+=("--boundary-tol" "$2"); shift 2;;
        --lbfgs-memory)     EXTRA_ARGS+=("--lbfgs-memory" "$2"); shift 2;;
        --exact-hessian)    EXTRA_ARGS+=("--exact-hessian"); shift;;
        --best-iterate)     EXTRA_ARGS+=("--best-iterate"); shift;;
        --save-iterations)  EXTRA_ARGS+=("--save-iterations"); shift;;
        --allow-partial)    EXTRA_ARGS+=("--allow-partial"); shift;;
        *)                  echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "$SOLUTION" ]]; then
    echo "Error: --solution is required"
    echo "Usage: bash cluster/submit_refinement.sh --solution <h5> --config <yaml> [options]"
    exit 1
fi
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: bash cluster/submit_refinement.sh --solution <h5> --config <yaml> [options]"
    exit 1
fi

SOLUTION_ABS="$(abspath "$SOLUTION")"
CONFIG_ABS="$(abspath "$CONFIG")"

if [[ "$DRY_RUN" == false && ! -f "$SOLUTION_ABS" ]]; then
    echo "Error: solution file not found: $SOLUTION_ABS"
    exit 1
fi
if [[ "$DRY_RUN" == false && ! -f "$CONFIG_ABS" ]]; then
    echo "Error: config file not found: $CONFIG_ABS"
    exit 1
fi

# --- Extract YAML parameters for job naming ---
SURFACE="$(extract_yaml surface "$CONFIG_ABS")"
SURFACE="${SURFACE:-unknown}"
N_PARTITIONS="$(extract_yaml n_partitions "$CONFIG_ABS")"
N_PARTITIONS="${N_PARTITIONS:-3}"

JOB_NAME="refine_${SURFACE}_npart${N_PARTITIONS}"
JOB_NAME="${JOB_NAME:0:128}"

# --- Build Python command ---
PYTHON_CMD="python scripts/refine_perimeter.py --solution ${SOLUTION_ABS} --config ${CONFIG_ABS}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTHON_CMD+=" ${EXTRA_ARGS[*]}"
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
cd ${REPO_DIR}

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
