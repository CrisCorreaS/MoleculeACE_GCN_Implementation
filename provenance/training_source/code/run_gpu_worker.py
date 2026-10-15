"""Two fixed GPU lanes inside one batch allocation; HPC exception documented in docs/gpu_workers.md.
Only run on a compute node through jobs/run_gpu_worker.sh. No scheduler commands or nested submissions.
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

from MoleculeACE.benchmark.artifacts import atomic_write_json, sha256_file
from MoleculeACE.benchmark.campaign import load_manifest, resolve_project_root, validate_task_outputs
from MoleculeACE.benchmark.gpu_allocation import allocation_report, worker_gpu_map


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def split_lanes(task_ids):
    if len(task_ids) < 2 or len(set(task_ids)) != len(task_ids):
        raise ValueError('A worker needs at least two distinct manifest task IDs')
    return [list(task_ids[0::2]), list(task_ids[1::2])]


def cpu_lanes(count):
    available = sorted(os.sched_getaffinity(0))
    if count < 1 or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 2 * count or len(available) != 2 * count:
        raise ValueError('CPU affinity must match the requested batch CPU count exactly; refusing an ambiguous CPU allocation')
    return [available[:count], available[count:2 * count]]


def child_environment(mapping, lane, cpus):
    env = dict(os.environ)
    # The only visibility override: a child gets a full UUID resolved from SLURM_JOB_GPUS.
    env.update(CUDA_VISIBLE_DEVICES=mapping[lane]['uuid'], GCN_GPU_WORKER='hpc_visibility_v1',
               GCN_GPU_MAP=json.dumps(mapping), GCN_GPU_LANE=str(lane),
               GCN_PARENT_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
               GCN_CPU_AFFINITY=json.dumps(cpus), OMP_NUM_THREADS=str(len(cpus)),
               MKL_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS=str(len(cpus)))
    # Do not overwrite scheduler variables: they continue to describe the two-GPU parent.
    return env


def start_process(command, env, cpus, base):
    out = Path(str(base) + '.out').open('a', encoding='utf-8')
    err = Path(str(base) + '.err').open('a', encoding='utf-8')
    try:
        # A stdlib-only child sets affinity before exec; avoid preexec_fn after numeric-library imports.
        shim = str(Path(__file__).with_name('gpu_worker_child.py'))
        process = subprocess.Popen([sys.executable, shim, json.dumps(cpus)] + command,
                                   env=env, stdout=out, stderr=err, start_new_session=True)
    except BaseException:
        out.close()
        err.close()
        raise
    return process, (out, err)


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_lanes(lanes, launch, validate, timeout, poll_seconds=0.5):
    """Stop filling both lanes on any failure; allow already-running siblings to finish.
    launch(lane, task) returns (Popen, open log handles). Always reap children, including on interruption.
    """
    pending = [deque(ids) for ids in lanes]
    active = {}
    completed, failed = [], []
    try:
        while any(pending) or active:
            if not failed:
                for lane in range(2):
                    if lane not in active and pending[lane]:
                        task = pending[lane].popleft()
                        try:
                            process, handles = launch(lane, task)
                        except Exception as exc:
                            failed.append(dict(task_id=task, lane=lane, exit_code=None, error=str(exc)))
                            print(f'FAILED to launch lane={lane} task_id={task}: {exc}', flush=True)
                            break
                        active[lane] = (task, process, handles, time.monotonic())
            for lane, (task, process, handles, started) in list(active.items()):
                rc = process.poll()
                if rc is None and time.monotonic() - started >= timeout:
                    terminate(process)
                    rc = 124
                if rc is None:
                    continue
                for handle in handles:
                    handle.close()
                del active[lane]
                error = None
                if rc == 0:
                    try:
                        validate(lane, task)
                    except Exception as exc:
                        error = str(exc)
                if rc != 0 or error is not None:
                    failed.append(dict(task_id=task, lane=lane, exit_code=rc, error=error))
                    print(f'FAILED lane={lane} task_id={task} exit_code={rc} error={error}', flush=True)
                else:
                    completed.append(task)
                    print(f'COMPLETED at={timestamp()} lane={lane} task_id={task}', flush=True)
            if failed and not active:
                break
            if active:
                time.sleep(poll_seconds)
    finally:
        for task, process, handles, started in active.values():
            terminate(process)
            for handle in handles:
                handle.close()
    return dict(completed=completed, failed=failed, not_started=[task for queue in pending for task in queue])


def probe(path):
    # Runs in a fresh process whose visibility/CPU affinity was fixed before Python started.
    import torch
    report, problems = allocation_report(torch, check_process=False)
    if not problems:
        tensor = torch.ones(1, device='cuda')
        torch.cuda.synchronize()
        report, problems = allocation_report(torch, check_process=True)
        del tensor
    report['problems'] = problems
    report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    atomic_write_json(path, report)
    if problems:
        raise ValueError('; '.join(problems))


class WorkerInterrupted(BaseException):
    pass


def interrupted(signum, frame):
    raise WorkerInterrupted(f'Worker interrupted by signal {signum}')


def run(args):
    root = resolve_project_root(args.project_root)
    tasks = load_manifest(args.manifest)
    if tasks[0]['campaign'] != args.campaign or tasks[0]['stage'] != 'full':
        raise ValueError('Workers require the approved full-campaign manifest and matching campaign')
    ids = [int(value) for value in args.task_ids.split(',')]
    by_id = {task['task_id']: task for task in tasks}
    if any(task not in by_id for task in ids):
        raise ValueError('Unknown manifest task ID')
    lanes = split_lanes(ids)
    mapping = worker_gpu_map()
    cpus = cpu_lanes(args.cpus_per_training)
    environments = [child_environment(mapping, lane, cpus[lane]) for lane in range(2)]
    logs = root / 'logs' / args.campaign
    logs.mkdir(parents=True, exist_ok=True)
    job = os.environ['SLURM_JOB_ID']
    attempt = os.environ.get('SLURM_RESTART_COUNT', '0')
    prefix = logs / f'worker_{job}_restart_{attempt}_{uuid.uuid4().hex}'
    manifest_hash = sha256_file(args.manifest)
    plan = dict(job_id=job, host=os.uname().nodename, started=timestamp(),
                campaign=args.campaign, gpu_map=mapping, cpu_lanes=cpus,
                parent_visible=os.environ['CUDA_VISIBLE_DEVICES'], lanes=lanes,
                manifest_sha256=manifest_hash, task_timeout=args.task_timeout)
    atomic_write_json(str(prefix) + '_plan.json', plan)
    print(json.dumps(plan), flush=True)
    probe_paths = [Path(str(prefix) + f'_probe_{lane}.json') for lane in range(2)]

    def launch_probe(lane, task):
        return start_process([sys.executable, str(Path(__file__).resolve()), '--probe', str(probe_paths[lane])],
                             environments[lane], cpus[lane], str(prefix) + f'_probe_{lane}')

    def validate_probe(lane, task):
        report = json.loads(probe_paths[lane].read_text(encoding='utf-8'))
        if (report['problems'] or report['allocated_gpu']['uuid'] != mapping[lane]['uuid']
                or report['cpu_affinity'] != cpus[lane]
                or report['process_gpu_bus_ids'] != [report['allocated_gpu']['pci.bus_id']]):
            raise ValueError('GPU isolation probe did not match the assigned GPU and CPUs')

    probes = run_lanes([[0], [1]], launch_probe, validate_probe, timeout=120)
    if probes['failed'] or len(probes['completed']) != 2:
        atomic_write_json(str(prefix) + '_outcome.json', dict(probes=probes, status='failed'))
        raise ValueError('GPU preflight failed; no training tasks started (see probe logs)')
    print('Both GPU isolation probes passed; starting training lanes', flush=True)

    def launch_task(lane, task):
        if sha256_file(args.manifest) != manifest_hash:
            raise ValueError('Manifest changed during worker execution')
        command = [sys.executable, str(root / 'code' / 'run_benchmark.py'), '--project-root', str(root),
                   '--manifest', str(args.manifest), '--campaign', args.campaign, '--task-id', str(task)]
        print(f'START at={timestamp()} lane={lane} task_id={task} gpu={mapping[lane]["uuid"]}', flush=True)
        return start_process(command, environments[lane], cpus[lane], logs / f'task_{task}_{job}')

    def validate_task(lane, task):
        validate_task_outputs(root, by_id[task])

    outcome = run_lanes(lanes, launch_task, validate_task, args.task_timeout)
    outcome['finished'] = timestamp()
    outcome['status'] = 'completed' if not outcome['failed'] and not outcome['not_started'] else 'failed'
    atomic_write_json(str(prefix) + '_outcome.json', outcome)
    print(json.dumps(outcome), flush=True)
    return 0 if outcome['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--project-root')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--campaign')
    parser.add_argument('--task-ids')
    parser.add_argument('--cpus-per-training', type=int, default=10)
    parser.add_argument('--task-timeout', type=int, default=1800)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.probe:
            probe(args.probe)
            return 0
        if not args.manifest or not args.campaign or not args.task_ids or args.task_timeout <= 0:
            raise ValueError('Missing worker arguments or invalid task timeout')
        args.manifest = args.manifest.resolve()
        return run(args)
    except (WorkerInterrupted, KeyboardInterrupt) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
