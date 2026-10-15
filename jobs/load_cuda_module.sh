# Sourced (never executed) AFTER conda activate by the Slurm jobs jobs/run_cpu.sh, run_checks.sh and run_gcn.sh:
#   source "$PROJECT_ROOT/jobs/load_cuda_module.sh"
# Loads $CUDA_MODULE (jobs/cluster.env), which provides the CUDA 11.3 libraries torch-sparse 0.6.13+cu113 needs
# (libcusparse.so.11; the torch 1.11 wheel only bundles cudart/nvrtc). Exits the caller if anything is missing:
# a job never continues without the module.

: "${CUDA_MODULE:?CUDA_MODULE is not set (jobs/cluster.env)}"
: "${CONDA_PREFIX:?Activate the conda environment before sourcing load_cuda_module.sh}"

# Lmod defines 'module' as a shell function; if the batch shell did not inherit it, use Lmod's own init file
if ! type module >/dev/null 2>&1 && [[ -n "${LMOD_PKG:-}" && -f "$LMOD_PKG/init/bash" ]]; then
    set +u; source "$LMOD_PKG/init/bash"; set -u
fi
type module >/dev/null 2>&1 \
    || { echo "ERROR: Lmod 'module' command not available; cannot load $CUDA_MODULE" >&2; exit 2; }

set +u
if ! module load "$CUDA_MODULE"; then
    set -u; echo "ERROR: 'module load $CUDA_MODULE' failed" >&2; exit 2
fi
set -u

# The module must actually expose libcusparse.so.11, and must not shadow the environment's python
cusparse_dir=""
IFS=: read -r -a _ld_dirs <<< "${LD_LIBRARY_PATH:-}"
for _dir in "${_ld_dirs[@]}"; do
    if [[ -n "$_dir" && -e "$_dir/libcusparse.so.11" ]]; then cusparse_dir="$_dir"; break; fi
done
[[ -n "$cusparse_dir" ]] \
    || { echo "ERROR: $CUDA_MODULE loaded but libcusparse.so.11 is not in LD_LIBRARY_PATH" >&2; exit 2; }
[[ "$(command -v python)" == "$CONDA_PREFIX/bin/python" ]] \
    || { echo "ERROR: after loading $CUDA_MODULE, python is $(command -v python), not $CONDA_PREFIX/bin/python" >&2; exit 2; }
echo "CUDA module $CUDA_MODULE loaded (libcusparse.so.11 in $cusparse_dir)"
unset _ld_dirs _dir
