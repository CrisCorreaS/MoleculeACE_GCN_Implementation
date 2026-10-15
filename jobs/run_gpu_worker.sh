#!/bin/bash
# Compute-node worker. Submit only through run_launch.sh --workers.
# HPC's project-specific visibility exception is implemented by run_gpu_worker.py.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
set -euo pipefail
: "${PROJECT_ROOT:?}" "${CODE_DIR:?}" "${MANIFEST:?}" "${CAMPAIGN:?}" "${MOLECULEACE_ENV:?}" "${CONDA_INIT:?}" "${CUDA_MODULE:?}" "${SLURM_JOB_ID:?}"
set +u; source "$CONDA_INIT"; conda activate "$MOLECULEACE_ENV"; set -u
[[ "$CONDA_PREFIX" == "$MOLECULEACE_ENV" ]] || { echo "Wrong environment: $CONDA_PREFIX"; exit 2; }
source "$PROJECT_ROOT/jobs/load_cuda_module.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE_DIR"
exec python run_gpu_worker.py --project-root "$PROJECT_ROOT" --manifest "$MANIFEST" --campaign "$CAMPAIGN" "$@"
