#!/bin/bash
# Slurm GPU environment and allocation check; no tests or training are run here.
# Resources come from jobs/cluster.env through the launcher. Scientific CPU tests are documented in README.md.
# Submit only through: bash jobs/run_launch.sh --checks
#SBATCH --job-name=gcn_checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
set -euo pipefail

if (( $# != 0 )); then
    echo "ERROR: Usage: jobs/run_checks.sh (submit through jobs/run_launch.sh --checks)" >&2
    exit 2
fi

: "${PROJECT_ROOT:?}" "${CODE_DIR:?}" "${MOLECULEACE_ENV:?}" "${CONDA_INIT:?}" "${CUDA_MODULE:?}"
: "${SLURM_JOB_ID:?Submit through jobs/run_launch.sh --checks}"

set +u; source "$CONDA_INIT"; conda activate "$MOLECULEACE_ENV"; set -u
[[ "$CONDA_PREFIX" == "$MOLECULEACE_ENV" ]] || { echo "Wrong environment: $CONDA_PREFIX"; exit 2; }
source "$PROJECT_ROOT/jobs/load_cuda_module.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONUNBUFFERED=1

echo "host=$(hostname) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?} qos=${SLURM_JOB_QOS:-?} mode=environment start=$(date -Is)"
cd "$CODE_DIR"
status=0
python env_check.py --gpu --project-root "$PROJECT_ROOT" || status=$?
echo "end=$(date -Is) status=$status"
exit "$status"
