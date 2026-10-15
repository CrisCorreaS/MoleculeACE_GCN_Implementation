#!/bin/bash
# Create the dedicated GCN environment on the cluster login node (installation only: conda/pip; no project Python,
# no training, no GPU). Run from the project root:   bash jobs/run_setup.sh
# Confirm that your site permits installation on its login node before running; see README.md.
set -euo pipefail

[[ -f jobs/cluster.env && -d code/MoleculeACE ]] || { echo "Run from the project root (jobs/cluster.env not found)"; exit 2; }
set -a; source jobs/cluster.env; set +a
[[ "$(pwd -P)" == "$(cd "$PROJECT_ROOT" && pwd -P)" ]] || { echo "cwd differs from PROJECT_ROOT=$PROJECT_ROOT"; exit 2; }
[[ "$MOLECULEACE_ENV" = /* && "$MOLECULEACE_ENV" != "$PROTECTED_ENV" && "${MOLECULEACE_ENV##*/}" != base ]] \
    || { echo "Refusing environment prefix $MOLECULEACE_ENV"; exit 2; }
[[ ! -e "$MOLECULEACE_ENV" ]] || { echo "$MOLECULEACE_ENV already exists: remove it deliberately or change cluster.env"; exit 2; }
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "Run on the login node, not inside a Slurm job"; exit 2; }

mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" results/environment
export PIP_CACHE_DIR CONDA_PKGS_DIRS PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1   # nothing written to code/

set +u; source "$CONDA_INIT"; set -u
conda env create --prefix "$MOLECULEACE_ENV" --file environments/training/environment.yml
set +u; conda activate "$MOLECULEACE_ENV"; set -u
[[ "$CONDA_PREFIX" == "$MOLECULEACE_ENV" ]] || { echo "Activation failed"; exit 2; }

PIP="python -m pip"
$PIP install --no-input -r environments/training/requirements.txt
$PIP install --no-input --no-deps torch==1.11.0+cu113 --index-url https://download.pytorch.org/whl/cu113
$PIP install --no-input --no-deps --only-binary=:all: torch-scatter==2.0.9 torch-sparse==0.6.13 \
    -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
$PIP install --no-input --no-deps torch-geometric==2.0.4
$PIP check

stamp=$(date +%Y%m%d_%H%M%S)
$PIP freeze > "results/environment/pip_freeze_${stamp}.txt"
conda list --explicit > "results/environment/conda_explicit_${stamp}.txt"
# The historical execution policy keeps project Python off the login node. The import/version check runs as a
# CPU job, the GPU checks as a debug-QoS GPU job.
echo "Environment installed: $MOLECULEACE_ENV (records in results/environment/)"
echo "Next: bash jobs/run_launch.sh --cpu env_check    then    bash jobs/run_launch.sh --checks"
