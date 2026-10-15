#!/bin/bash
# Slurm CPU job: one light project tool (no training, no GPU). The historical execution policy runs project
# Python in batch jobs. Partition, QoS, time, CPUs and memory come from jobs/cluster.env
# through the launcher. Submit ONLY through:   bash jobs/run_launch.sh --cpu <tool> [args...]
# The tool runs from code/, so relative paths in [args] are relative to code/ (e.g. ../jobs/manifest_full.csv).
# --project-root is always passed by this script; do not repeat it.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

: "${PROJECT_ROOT:?}" "${CODE_DIR:?}" "${MOLECULEACE_ENV:?}" "${CONDA_INIT:?}" "${CUDA_MODULE:?}"
: "${SLURM_JOB_ID:?Submit through jobs/run_launch.sh --cpu <tool>}"
TOOL="${1:?Missing tool name}"; shift
case "$TOOL" in
    env_check|verify_datasets|verify_cliffs|make_manifest|task_status|aggregate_results|analyze_results) ;;
    *) echo "ERROR: '$TOOL' is not a CPU tool (allowed: env_check verify_datasets verify_cliffs make_manifest" \
            "task_status aggregate_results analyze_results)" >&2; exit 2 ;;
esac

set +u; source "$CONDA_INIT"; conda activate "$MOLECULEACE_ENV"; set -u
[[ "$CONDA_PREFIX" == "$MOLECULEACE_ENV" ]] || { echo "Wrong environment: $CONDA_PREFIX"; exit 2; }
source "$PROJECT_ROOT/jobs/load_cuda_module.sh"   # torch-sparse needs libcusparse.so.11 even without a GPU
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONUNBUFFERED=1

echo "host=$(hostname) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?} qos=${SLURM_JOB_QOS:-?} tool=$TOOL args=$* start=$(date -Is)"
cd "$CODE_DIR"
status=0
python "$TOOL.py" --project-root "$PROJECT_ROOT" "$@" || status=$?
echo "end=$(date -Is) status=$status"
exit $status
