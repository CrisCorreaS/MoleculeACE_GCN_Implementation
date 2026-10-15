#!/bin/bash
# One manifest row per batch job, submitted only through run_launch.sh.
# Production receives its manifest task ID as the only argument; the pilot uses its array index.
# Every job requests one GPU. All paths/resources come from the launcher; no nested job steps.
#SBATCH --job-name=gcn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
set -euo pipefail

: "${PROJECT_ROOT:?}" "${CODE_DIR:?}" "${MANIFEST:?}" "${CAMPAIGN:?}" "${MOLECULEACE_ENV:?}" "${CONDA_INIT:?}" "${CUDA_MODULE:?}"
: "${SLURM_JOB_ID:?Submit through jobs/run_launch.sh}"
if (( $# == 0 )) && [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    TASK_ID="$SLURM_ARRAY_TASK_ID"
elif (( $# == 1 )) && [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]; then
    TASK_ID="$1"
else
    echo "ERROR: Expected one numeric manifest task ID, or a pilot array index" >&2
    exit 2
fi

set +u; source "$CONDA_INIT"; conda activate "$MOLECULEACE_ENV"; set -u
[[ "$CONDA_PREFIX" == "$MOLECULEACE_ENV" ]] || { echo "Wrong environment: $CONDA_PREFIX"; exit 2; }
source "$PROJECT_ROOT/jobs/load_cuda_module.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONUNBUFFERED=1

echo "host=$(hostname) job=$SLURM_JOB_ID array_task=${SLURM_ARRAY_TASK_ID:-none} task_id=$TASK_ID campaign=$CAMPAIGN" \
     "partition=${SLURM_JOB_PARTITION:-?} qos=${SLURM_JOB_QOS:-?} restart_count=${SLURM_RESTART_COUNT:-0} start=$(date -Is)"
cd "$CODE_DIR"
timer=(); [[ -x /usr/bin/time ]] && timer=(/usr/bin/time -v)
status=0
"${timer[@]}" python run_benchmark.py --manifest "$MANIFEST" --campaign "$CAMPAIGN" \
    --task-id "$TASK_ID" --project-root "$PROJECT_ROOT" || status=$?
echo "end=$(date -Is) task_id=$TASK_ID status=$status"
exit "$status"
