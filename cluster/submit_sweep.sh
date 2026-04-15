#!/bin/bash
# Submit a full parameter sweep to UPPMAX Pelle.
# Generates per-run configs via the sweep tool, then submits each as a
# separate SLURM job through submit_relaxation.sh.
#
# Usage:
#   bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml
#   bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml --time 12:00:00 --cpus 4
#   bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml --dry-run
#   bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml --auto-collect

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pelle_config.sh"

# --- Defaults ---
SWEEP_YAML=""
TIME_LIMIT=""
CPUS=""
MEM=""
OUTPUT_DIR=""
DRY_RUN=false
AUTO_COLLECT=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep)        SWEEP_YAML="$2"; shift 2;;
        --time)         TIME_LIMIT="$2"; shift 2;;
        --cpus)         CPUS="$2"; shift 2;;
        --mem)          MEM="$2"; shift 2;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2;;
        --dry-run)      DRY_RUN=true; shift;;
        --auto-collect) AUTO_COLLECT=true; shift;;
        *)              echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "$SWEEP_YAML" ]]; then
    echo "Error: --sweep is required"
    echo "Usage: bash cluster/submit_sweep.sh --sweep <sweep.yaml> [options]"
    exit 1
fi

SWEEP_YAML_ABS="$(abspath "$SWEEP_YAML")"
if [[ ! -f "$SWEEP_YAML_ABS" ]]; then
    echo "Error: sweep file not found: $SWEEP_YAML_ABS"
    exit 1
fi

# --- Generate per-run configs ---
echo "Generating per-run configs from sweep spec..."
GENERATE_CMD="python sweep/parameter_sweep.py --sweep ${SWEEP_YAML_ABS} --mode generate-only"
if [[ -n "$OUTPUT_DIR" ]]; then
    GENERATE_CMD+=" --output-dir $(abspath "$OUTPUT_DIR")"
fi

cd "${REPO_DIR}"
GENERATE_OUTPUT=$($GENERATE_CMD 2>&1)
echo "$GENERATE_OUTPUT"

# Extract generated config file paths (lines ending in .yaml that are files)
CONFIG_FILES=()
while IFS= read -r line; do
    trimmed=$(echo "$line" | xargs)
    if [[ "$trimmed" == *.yaml && -f "$trimmed" ]]; then
        CONFIG_FILES+=("$trimmed")
    fi
done <<< "$GENERATE_OUTPUT"

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "Error: no config files were generated. Check the sweep spec."
    exit 1
fi

echo ""
echo "Generated ${#CONFIG_FILES[@]} run config(s)."
echo ""

# --- Submit each config as a separate job ---
JOB_IDS=()
SUBMITTED=0

for cfg in "${CONFIG_FILES[@]}"; do
    SUBMIT_CMD="bash ${SCRIPT_DIR}/submit_relaxation.sh --config ${cfg}"
    if [[ -n "$TIME_LIMIT" ]]; then SUBMIT_CMD+=" --time ${TIME_LIMIT}"; fi
    if [[ -n "$CPUS" ]]; then SUBMIT_CMD+=" --cpus ${CPUS}"; fi
    if [[ -n "$MEM" ]]; then SUBMIT_CMD+=" --mem ${MEM}"; fi
    if [[ "$DRY_RUN" == true ]]; then SUBMIT_CMD+=" --dry-run"; fi

    OUTPUT=$($SUBMIT_CMD 2>&1)
    echo "$OUTPUT"

    if [[ "$DRY_RUN" == false ]]; then
        # Extract job ID from "Submitted: ... (Job ID: NNNN)"
        JID=$(echo "$OUTPUT" | grep -oP 'Job ID: \K[0-9]+' || true)
        if [[ -n "$JID" ]]; then
            JOB_IDS+=("$JID")
            SUBMITTED=$((SUBMITTED + 1))
        fi
    fi
    echo ""
done

# --- Summary ---
echo "========================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN complete. ${#CONFIG_FILES[@]} job(s) would be submitted."
else
    echo "Submitted ${SUBMITTED} job(s)."
    if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
        echo "Job IDs: ${JOB_IDS[*]}"
    fi
fi
echo "Sweep spec: ${SWEEP_YAML_ABS}"
echo "========================================"

# --- Optional: submit collector job ---
if [[ "$AUTO_COLLECT" == true && "$DRY_RUN" == false && ${#JOB_IDS[@]} -gt 0 ]]; then
    DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
    COLLECT_LOGS="${REPO_DIR}/slurm_logs"
    mkdir -p "$COLLECT_LOGS"

    SWEEP_NAME=$(basename "$SWEEP_YAML_ABS" .yaml)

    COLLECT_JID=$(sbatch --dependency=afterany:${DEPS} << COLLECT_EOF | awk '{print $4}'
#!/bin/bash
#SBATCH -A ${PROJECT_ID}
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 0:30:00
#SBATCH -J collect_${SWEEP_NAME}
#SBATCH -o ${COLLECT_LOGS}/%x_%j.out
#SBATCH -e ${COLLECT_LOGS}/%x_%j.err

source ${SCRIPT_DIR}/pelle_config.sh
activate_env
cd ${REPO_DIR}

python sweep/parameter_sweep.py --sweep ${SWEEP_YAML_ABS} --mode collect
COLLECT_EOF
    )
    echo ""
    echo "Collector job submitted (Job ID: ${COLLECT_JID}), depends on: ${DEPS}"
    echo "  Will run: python sweep/parameter_sweep.py --sweep ${SWEEP_YAML} --mode collect"
fi

# --- Manual collect command ---
if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo "To collect results manually after all jobs finish:"
    echo "  python sweep/parameter_sweep.py --sweep ${SWEEP_YAML} --mode collect"
fi
