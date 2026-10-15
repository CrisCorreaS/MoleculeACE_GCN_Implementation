#!/bin/bash
# Login-node launcher: Bash only. All Python runs inside Slurm compute jobs.
#   bash jobs/run_launch.sh --checks
#   bash jobs/run_launch.sh --cpu <tool> [args...]
#   bash jobs/run_launch.sh --workers jobs/manifest_full.csv gcn_full_v1 [task_ids]
# Pilot: one array element per row. Default production: one sbatch per selected manifest row (at most four).
# --workers: full campaign, at most two two-GPU jobs process all selected rows.
# The submission lock and queue guard count pending/running jobs before submitting.
# Each training process validates its manifest row and hashes on the compute node.
set -euo pipefail

[[ -f jobs/cluster.env && -d code/MoleculeACE ]] || { echo "Run from the project root (jobs/cluster.env not found)"; exit 2; }
set -a; source jobs/cluster.env; set +a
ROOT="$(pwd -P)"
[[ "$ROOT" == "$(cd "$PROJECT_ROOT" && pwd -P)" ]] || { echo "cwd $ROOT differs from PROJECT_ROOT=$PROJECT_ROOT"; exit 2; }
[[ -x "$MOLECULEACE_ENV/bin/python" ]] || { echo "Environment missing: run bash jobs/run_setup.sh first"; exit 2; }
COMMON="PROJECT_ROOT=$ROOT,CODE_DIR=$ROOT/code,MOLECULEACE_ENV=$MOLECULEACE_ENV,CONDA_INIT=$CONDA_INIT,CUDA_MODULE=${CUDA_MODULE:-}"

fail() { echo "ERROR: $*" >&2; exit 2; }
need() {
    local v
    for v in "$@"; do
        [[ -n "${!v:-}" ]] || fail "$v is empty in jobs/cluster.env (see README.md)"
    done
}
need CUDA_MODULE

# Keep the lock inode in place; close fd 9 in sbatch children while the parent retains it.
SUBMISSION_LOCK="$ROOT/results/.submission.lock"
mkdir -p "$ROOT/results"
exec 9>>"$SUBMISSION_LOCK"
flock -n 9 || fail "Another launcher is submitting a job; wait for it to finish and retry"

slurm_minutes() {   # MM, MM:SS, HH:MM:SS, D-HH[:MM[:SS]]; round seconds up
    local t="$1" d=0 h=0 m=0 s=0
    if [[ "$t" =~ ^([0-9]+)-([0-9]+)(:([0-9]+))?(:([0-9]+))?$ ]]; then
        d=${BASH_REMATCH[1]}; h=${BASH_REMATCH[2]}; m=${BASH_REMATCH[4]:-0}; s=${BASH_REMATCH[6]:-0}
        (( 10#$h < 24 && 10#$m < 60 && 10#$s < 60 )) || return 1
    elif [[ "$t" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        h=${BASH_REMATCH[1]}; m=${BASH_REMATCH[2]}; s=${BASH_REMATCH[3]}
        (( 10#$m < 60 && 10#$s < 60 )) || return 1
    elif [[ "$t" =~ ^([0-9]+):([0-9]+)$ ]]; then
        m=${BASH_REMATCH[1]}; s=${BASH_REMATCH[2]}
        (( 10#$s < 60 )) || return 1
    elif [[ "$t" =~ ^([0-9]+)$ ]]; then
        m=${BASH_REMATCH[1]}
    else
        return 1
    fi
    echo $(( (10#$d * 24 + 10#$h) * 60 + 10#$m + (10#$s > 0 ? 1 : 0) ))
}
check_time() {
    local t max
    t=$(slurm_minutes "${!1}") || fail "$1=${!1} is not a Slurm time"
    max=$(slurm_minutes "${!2}") || fail "$2=${!2} is not a Slurm time"
    (( t > 0 && t <= max )) || fail "$1=${!1} must be positive and at most $2=${!2}"
}

debug_profile() {
    need SLURM_GPU_DEBUG_PARTITION SLURM_GPU_DEBUG_QOS SLURM_GPU_DEBUG_TIME SLURM_GPU_DEBUG_MAX_TIME \
         SLURM_GPU_DEBUG_MAX_JOBS SLURM_GPU_DEBUG_CPUS SLURM_GPU_DEBUG_MEM
    check_time SLURM_GPU_DEBUG_TIME SLURM_GPU_DEBUG_MAX_TIME
    GPU_ARGS=(--partition="$SLURM_GPU_DEBUG_PARTITION" --qos="$SLURM_GPU_DEBUG_QOS" --time="$SLURM_GPU_DEBUG_TIME"
              --cpus-per-task="$SLURM_GPU_DEBUG_CPUS" --mem="$SLURM_GPU_DEBUG_MEM")
    PROFILE="qos=$SLURM_GPU_DEBUG_QOS time=$SLURM_GPU_DEBUG_TIME cpus=$SLURM_GPU_DEBUG_CPUS mem=$SLURM_GPU_DEBUG_MEM gpus=1"
}

prod_profile() {   # Default individual job, or explicitly requested two-GPU workers.
    need SLURM_GPU_PROD_ACCOUNT SLURM_GPU_PROD_PARTITION SLURM_GPU_PROD_QOS \
         SLURM_GPU_PROD_TIME SLURM_GPU_PROD_MAX_TIME SLURM_GPU_PROD_MAX_JOBS \
         SLURM_GPU_PROD_CPUS SLURM_GPU_PROD_MEM
    check_time SLURM_GPU_PROD_TIME SLURM_GPU_PROD_MAX_TIME
    local limit
    for limit in SLURM_GPU_PROD_CPUS SLURM_GPU_PROD_MAX_JOBS; do
        [[ "${!limit}" =~ ^[1-9][0-9]*$ ]] || fail "$limit must be a positive integer"
    done
    (( WORKERS == 1 || COUNT <= SLURM_GPU_PROD_MAX_JOBS )) \
        || fail "Select at most $SLURM_GPU_PROD_MAX_JOBS production task ids per launch (e.g. 0-3); selected $COUNT. No jobs submitted."
    GPU_ARGS=(--account="$SLURM_GPU_PROD_ACCOUNT" --partition="$SLURM_GPU_PROD_PARTITION"
              --qos="$SLURM_GPU_PROD_QOS" --time="$SLURM_GPU_PROD_TIME" --nodes=1 --ntasks=1
              --gres=gpu:1 --cpus-per-task="$SLURM_GPU_PROD_CPUS" --mem="$SLURM_GPU_PROD_MEM")
    PROFILE="account=$SLURM_GPU_PROD_ACCOUNT partition=$SLURM_GPU_PROD_PARTITION qos=$SLURM_GPU_PROD_QOS time=$SLURM_GPU_PROD_TIME cpus=$SLURM_GPU_PROD_CPUS mem=$SLURM_GPU_PROD_MEM gpus=1"
    if (( WORKERS == 1 )); then
        need SLURM_GPU_WORKER_TIME SLURM_GPU_WORKER_MEM SLURM_GPU_PROD_MAX_CONCURRENT SLURM_GPU_PROD_MAX_GPUS
        for limit in SLURM_GPU_PROD_MAX_CONCURRENT SLURM_GPU_PROD_MAX_GPUS; do
            [[ "${!limit}" =~ ^[1-9][0-9]*$ ]] || fail "$limit must be a positive integer"
        done
        check_time SLURM_GPU_WORKER_TIME SLURM_GPU_PROD_MAX_TIME
        (( COUNT >= 2 )) || fail "Workers require at least two selected tasks"
        WORKER_COUNT=1
        (( COUNT < 4 )) || WORKER_COUNT=2
        (( WORKER_COUNT <= SLURM_GPU_PROD_MAX_JOBS &&
           WORKER_COUNT <= SLURM_GPU_PROD_MAX_CONCURRENT &&
           2 * WORKER_COUNT <= SLURM_GPU_PROD_MAX_GPUS )) || fail "Worker layout exceeds configured limits"
        GPU_ARGS=(--account="$SLURM_GPU_PROD_ACCOUNT" --partition="$SLURM_GPU_PROD_PARTITION"
                  --qos="$SLURM_GPU_PROD_QOS" --time="$SLURM_GPU_WORKER_TIME" --nodes=1 --ntasks=1
                  --gres=gpu:2 --cpus-per-task="$((2 * SLURM_GPU_PROD_CPUS))" --mem="$SLURM_GPU_WORKER_MEM")
        TASK_TIMEOUT=$(( $(slurm_minutes "$SLURM_GPU_PROD_TIME") * 60 ))
        PROFILE="account=$SLURM_GPU_PROD_ACCOUNT partition=$SLURM_GPU_PROD_PARTITION qos=$SLURM_GPU_PROD_QOS time=$SLURM_GPU_WORKER_TIME cpus=$((2 * SLURM_GPU_PROD_CPUS)) mem=$SLURM_GPU_WORKER_MEM gpus=2 cpus_per_training=$SLURM_GPU_PROD_CPUS task_timeout=$TASK_TIMEOUT"
    fi
    if [[ -n "${SLURM_GPU_PROD_EXCLUDE:-}" ]]; then
        GPU_ARGS+=(--exclude="$SLURM_GPU_PROD_EXCLUDE")
        PROFILE+=" exclude=$SLURM_GPU_PROD_EXCLUDE"
    fi
}

debug_slots() {
    local current
    current=$(squeue --me --noheader --array --qos="$SLURM_GPU_DEBUG_QOS" --format=%i | wc -l) \
        || fail "squeue failed; cannot check the $SLURM_GPU_DEBUG_QOS cap"
    (( current + $1 <= SLURM_GPU_DEBUG_MAX_JOBS )) \
        || fail "$SLURM_GPU_DEBUG_QOS cap: $current job(s) already queued/running + $1 new > $SLURM_GPU_DEBUG_MAX_JOBS." \
                "Wait for them to finish, or submit fewer task ids (e.g. '0,1' then '2,3')"
}
production_slots() {
    local queue current queued_id queued_name task
    queue=$(squeue --me --noheader --array --qos="$SLURM_GPU_PROD_QOS" --format='%i|%j') \
        || fail "squeue failed; cannot check the $SLURM_GPU_PROD_QOS cap"
    current=$(printf '%s\n' "$queue" | awk 'NF {n++} END {print n+0}')
    if (( WORKERS == 1 )); then
        (( current == 0 )) || fail "Wait for all production jobs to finish before launching workers"
        return 0
    fi
    [[ "$queue" != *"_worker0"* && "$queue" != *"_worker1"* ]] \
        || fail "Production workers are still queued/running; do not overlap direct submissions"
    (( current + COUNT <= SLURM_GPU_PROD_MAX_JOBS )) \
        || fail "$SLURM_GPU_PROD_QOS cap: $current job(s) already queued/running + $COUNT new > $SLURM_GPU_PROD_MAX_JOBS. Wait or select fewer task ids."
    while IFS='|' read -r queued_id queued_name; do
        for task in "${TASK_IDS[@]}"; do
            [[ "$queued_name" != "${CAMPAIGN}_t${task}" ]] \
                || fail "Task $task already queued/running as job $queued_id; do not submit it twice"
        done
    done <<< "$queue"
}

select_tasks() {   # Expand ranges once; overlapping selections must not dispatch duplicate folds.
    local item a b id items
    local -A seen=()
    TASK_IDS=()
    [[ "$1" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || return 1
    IFS=, read -r -a items <<< "$1"
    for item in "${items[@]}"; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            a=$((10#${BASH_REMATCH[1]})); b=$((10#${BASH_REMATCH[2]}))
        else
            a=$((10#$item)); b=$a
        fi
        (( a <= b && b < $2 )) || return 1
        for ((id=a; id<=b; id++)); do
            [[ ! -v seen[$id] ]] || return 1
            seen[$id]=1
            TASK_IDS+=("$id")
        done
    done
    COUNT=${#TASK_IDS[@]}
    TASK_LIST=$(IFS=,; echo "${TASK_IDS[*]}")
}

if [[ "${1:-}" == "--checks" ]]; then
    CHECK_MODE=environment
    [[ $# -eq 1 ]] || fail "Usage: bash jobs/run_launch.sh --checks"
    debug_profile
    debug_slots 1
    LOGS="$ROOT/logs/checks"
    mkdir -p "$LOGS"
    job=$(sbatch --parsable "${GPU_ARGS[@]}" --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
          --export="ALL,$COMMON" jobs/run_checks.sh 9>&-)
    echo "$(date -Is) checks job=$job mode=$CHECK_MODE partition=$SLURM_GPU_DEBUG_PARTITION $PROFILE" >> "$LOGS/submissions.log"
    echo "Submitted checks job $job (mode=$CHECK_MODE; $PROFILE); logs in $LOGS"
    exit 0
fi

if [[ "${1:-}" == "--cpu" ]]; then
    TOOL="${2:-}"
    [[ "$TOOL" =~ ^[a-z_]+$ && -f "code/$TOOL.py" ]] \
        || fail "Usage: bash jobs/run_launch.sh --cpu <tool> [args...]  (tools: see jobs/run_cpu.sh)"
    need SLURM_CPU_PARTITION SLURM_CPU_QOS SLURM_CPU_CPUS SLURM_CPU_MEM SLURM_CPU_TIME
    LOGS="$ROOT/logs/cpu"
    mkdir -p "$LOGS"
    job=$(sbatch --parsable --partition="$SLURM_CPU_PARTITION" --qos="$SLURM_CPU_QOS" --time="$SLURM_CPU_TIME" \
          --cpus-per-task="$SLURM_CPU_CPUS" --mem="$SLURM_CPU_MEM" --job-name="cpu_$TOOL" \
          --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
          --export="ALL,$COMMON" jobs/run_cpu.sh "${@:2}" 9>&-)
    echo "$(date -Is) job=$job tool=$TOOL args=${*:3} qos=$SLURM_CPU_QOS time=$SLURM_CPU_TIME cpus=$SLURM_CPU_CPUS mem=$SLURM_CPU_MEM" \
        >> "$LOGS/submissions.log"
    echo "Submitted CPU job $job ($TOOL); logs in $LOGS/cpu_${TOOL}_${job}.out/.err"
    exit 0
fi

WORKERS=0
if [[ "${1:-}" == --workers ]]; then WORKERS=1; shift; fi
[[ $# -ge 2 && $# -le 3 ]] || fail "Usage: bash jobs/run_launch.sh [--workers] <manifest> <campaign> [task_ids]  |  --checks  |  --cpu <tool> [args]"
MANIFEST="$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")"
CAMPAIGN="$2"
[[ "$CAMPAIGN" =~ ^gcn_(pilot|demo|full)_[A-Za-z0-9_-]+$ ]] || fail "Invalid campaign name: $CAMPAIGN"
[[ "$MANIFEST" == "$ROOT/jobs/"* && -f "$MANIFEST" ]] || fail "Manifest must be an existing file in jobs/"

# Plain CSV from make_manifest.py. Full scientific validation remains inside each compute job.
column() {
    awk -F, -v col="$1" 'NR == 1 { sub(/\r$/, ""); for (i = 1; i <= NF; i++) if ($i == col) c = i; if (!c) exit 3; next }
                         { sub(/\r$/, ""); if ($0 != "") print $c }' "$MANIFEST"
}
N=$(column task_id | wc -l) || fail "manifest has no task_id column"
campaigns=$(column campaign | sort -u) || fail "manifest has no campaign column"
stages=$(column stage | sort -u) || fail "manifest has no stage column"
[[ "$N" -gt 0 ]] || fail "manifest has no tasks"
[[ "$campaigns" == "$CAMPAIGN" ]] || fail "campaign $CAMPAIGN does not match the manifest ($campaigns)"
ARRAY="${3:-0-$((N - 1))}"
select_tasks "$ARRAY" "$N" || fail "Invalid or duplicate task ids '$ARRAY' (manifest rows 0..$((N - 1)))"
if (( WORKERS == 1 )); then
    [[ "$stages" == full ]] || fail "--workers is restricted to the approved full campaign"
fi
case "$stages" in
    pilot)     debug_profile
               debug_slots "$COUNT" ;;
    demo|full) prod_profile
               production_slots ;;
    *)         fail "manifest must have exactly one stage (pilot|demo|full), found: $stages" ;;
esac

LOGS="$ROOT/logs/$CAMPAIGN"
mkdir -p "$LOGS"
commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)
manifest_hash=$(sha256sum "$MANIFEST" | cut -c1-64)
if [[ "$stages" == pilot ]]; then
    job=$(sbatch --parsable --array="${ARRAY}%${SLURM_GPU_DEBUG_MAX_JOBS}" --job-name="$CAMPAIGN" "${GPU_ARGS[@]}" --open-mode=append \
          --output="$LOGS/%x_%A_%a.out" --error="$LOGS/%x_%A_%a.err" \
          --export="ALL,$COMMON,MANIFEST=$MANIFEST,CAMPAIGN=$CAMPAIGN" jobs/run_gcn.sh 9>&-)
    echo "$(date -Is) job=$job array=$ARRAY task_ids=$TASK_LIST stage=$stages $PROFILE manifest=$MANIFEST sha256=$manifest_hash commit=$commit" >> "$LOGS/submissions.log"
    echo "Submitted $CAMPAIGN as pilot array $job; logs in $LOGS"
    exit 0
fi

if (( WORKERS == 1 )); then
    for ((worker=0; worker<WORKER_COUNT; worker++)); do
        worker_ids=()
        for ((position=worker; position<COUNT; position+=WORKER_COUNT)); do
            worker_ids+=("${TASK_IDS[position]}")
        done
        worker_list=$(IFS=,; echo "${worker_ids[*]}")
        if ! job=$(sbatch --parsable --job-name="${CAMPAIGN}_worker${worker}" "${GPU_ARGS[@]}" --open-mode=append \
              --output="$LOGS/worker_${worker}_%j.out" --error="$LOGS/worker_${worker}_%j.err" \
              --export="ALL,$COMMON,MANIFEST=$MANIFEST,CAMPAIGN=$CAMPAIGN" jobs/run_gpu_worker.sh \
              --task-ids "$worker_list" --cpus-per-training "$SLURM_GPU_PROD_CPUS" --task-timeout "$TASK_TIMEOUT" 9>&-); then
            fail "Worker $worker submission failed. Earlier jobs remain submitted; review $LOGS/submissions.log before retrying."
        fi
        echo "$(date -Is) job=$job worker=$worker task_ids=$worker_list stage=$stages $PROFILE manifest=$MANIFEST sha256=$manifest_hash commit=$commit" >> "$LOGS/submissions.log"
        echo "Submitted worker=$worker as job $job ($PROFILE); task_ids=$worker_list; logs in $LOGS"
    done
    exit 0
fi

# Separate submissions can partially succeed. Record and print every accepted ID immediately.
# If a later sbatch fails, leave earlier jobs intact and do not automatically retry the selection.
submitted=0
for task in "${TASK_IDS[@]}"; do
    if ! job=$(sbatch --parsable --job-name="${CAMPAIGN}_t${task}" "${GPU_ARGS[@]}" --open-mode=append \
          --output="$LOGS/task_${task}_%j.out" --error="$LOGS/task_${task}_%j.err" \
          --export="ALL,$COMMON,MANIFEST=$MANIFEST,CAMPAIGN=$CAMPAIGN" jobs/run_gcn.sh "$task" 9>&-); then
        fail "sbatch failed for task_id=$task after $submitted accepted job(s). Earlier jobs remain submitted; review $LOGS/submissions.log before retrying."
    fi
    submitted=$((submitted + 1))
    echo "$(date -Is) job=$job task_id=$task stage=$stages $PROFILE manifest=$MANIFEST sha256=$manifest_hash commit=$commit" >> "$LOGS/submissions.log"
    echo "Submitted task_id=$task as job $job ($PROFILE); logs: $LOGS/task_${task}_%j.out/.err"
done
