"""Read-only checks that a job uses only the GPU that Slurm allocated to it (DANA GPU Allocation Policy).

nvidia-smi can list every GPU on a node even inside a job with --gres=gpu:1. That listing is not the job's
allocation. This module never changes device visibility. Each process gets one GPU. The approved full-campaign
worker may narrow its two Slurm GPUs to one assigned UUID per child (HPC exception, docs/gpu_workers.md).
Legacy step evidence remains readable; current job scripts do not launch steps.

    allocation_report(torch)  -> (report, problems)   environment, torch.cuda.device_count(), nvidia-smi of the
                                                       allocated GPU and the GPUs where this process holds a context
"""
import json
import os
import subprocess

EXPECTED_GPUS = 1
ENV_VARIABLES = ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_STEP_ID',
                 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
                 'GPU_DEVICE_ORDINAL', 'SLURM_GPUS_ON_NODE', 'GCN_GPU_WORKER',
                 'GCN_GPU_MAP', 'GCN_GPU_LANE', 'GCN_PARENT_VISIBLE_DEVICES', 'GCN_CPU_AFFINITY')
GPU_FIELDS = ('index', 'pci.bus_id', 'uuid', 'name', 'driver_version', 'memory.total')


def _nvidia_smi(args):
    try:
        return subprocess.check_output(['nvidia-smi'] + args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except subprocess.CalledProcessError as exc:
        return f"error: {(exc.output or '').strip()}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc!r}"


def visible_devices():
    """ Entries of CUDA_VISIBLE_DEVICES as set by Slurm (None if unset) """
    value = os.environ.get('CUDA_VISIBLE_DEVICES')
    return None if value is None else [v.strip() for v in value.split(',') if v.strip()]


def _gpu_ids(value):
    return None if value is None else [v.strip() for v in value.split(',') if v.strip()]


def allocation_report(torch, check_process=True):
    """ Environment, device count and nvidia-smi of the allocated GPU only. Call after torch has used the GPU (so the
    process holds a CUDA context) when check_process is True. Returns (report, problems); never raises for a
    failed check, the caller decides """
    report = dict(env={k: os.environ.get(k) for k in ENV_VARIABLES},
                  expected_gpus=EXPECTED_GPUS, torch_device_count=None)
    problems = []
    if not report['env']['SLURM_JOB_ID']:
        problems.append('GPU checks require a Slurm GPU allocation')
    visible = visible_devices()
    if visible is None or len(visible) != EXPECTED_GPUS:
        problems.append(f"CUDA_VISIBLE_DEVICES={report['env']['CUDA_VISIBLE_DEVICES']!r}: Slurm should expose exactly "
                        f"{EXPECTED_GPUS} GPU to this process")
    step_id = report['env']['SLURM_STEP_ID']
    in_step = bool(step_id and step_id.isdecimal())  # batch/extern are not isolated srun training steps
    step_gpus = _gpu_ids(report['env']['SLURM_STEP_GPUS'])
    job_gpus = _gpu_ids(report['env']['SLURM_JOB_GPUS'])
    node_gpus = report['env']['SLURM_GPUS_ON_NODE']
    report['allocation_scope'] = 'step' if in_step else 'job'
    worker = report['env']['GCN_GPU_WORKER']
    expected_uuid = None
    if worker is not None:
        report['allocation_scope'] = 'hpc_worker_lane'
        allocated_ids = []
        try:
            mapping = json.loads(report['env']['GCN_GPU_MAP'] or 'null')
            lane = int(report['env']['GCN_GPU_LANE'])
            parent_visible = _gpu_ids(report['env']['GCN_PARENT_VISIBLE_DEVICES'])
            if (worker != 'hpc_visibility_v1' or in_step or node_gpus != '2'
                    or not job_gpus or len(job_gpus) != 2 or len(set(job_gpus)) != 2
                    or not isinstance(mapping, list) or len(mapping) != 2 or lane not in (0, 1)
                    or [g['index'] for g in mapping] != job_gpus
                    or len({g['uuid'] for g in mapping}) != 2
                    or not all(g['uuid'].startswith('GPU-') for g in mapping)
                    or not parent_visible or len(parent_visible) != 2 or len(set(parent_visible)) != 2
                    or step_gpus not in (None, job_gpus)):
                raise ValueError('inconsistent parent allocation')
            expected_uuid = mapping[lane]['uuid']
            if visible != [expected_uuid]:
                raise ValueError('child visibility does not match its assigned lane UUID')
            allocated_ids = [mapping[lane]['index']]
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            problems.append(f'Invalid HPC worker GPU assignment: {exc}')
    elif in_step:
        # A parent job may own two GPUs. Require independent scheduler evidence that this step owns one;
        # hiding a GPU with CUDA_VISIBLE_DEVICES alone is insufficient.
        if step_gpus is None or len(step_gpus) != EXPECTED_GPUS:
            problems.append(f"SLURM_STEP_GPUS={report['env']['SLURM_STEP_GPUS']!r}: "
                            'an srun training step must be allocated exactly one GPU')
        if node_gpus is not None and (not node_gpus.isdecimal() or int(node_gpus) < 1):
            problems.append(f"Invalid SLURM_GPUS_ON_NODE={node_gpus!r}")
        allocated_ids = step_gpus
    else:
        if node_gpus not in (None, str(EXPECTED_GPUS)):
            problems.append(f"SLURM_GPUS_ON_NODE={node_gpus} != {EXPECTED_GPUS}; "
                            'multi-GPU jobs require an isolated single-GPU srun step')
        for name, gpu_ids in (('SLURM_JOB_GPUS', job_gpus), ('SLURM_STEP_GPUS', step_gpus)):
            if gpu_ids is not None and len(gpu_ids) != EXPECTED_GPUS:
                problems.append(f"{name}={report['env'][name]!r}: expected a single-GPU job")
        allocated_ids = job_gpus if job_gpus is not None else (step_gpus or visible)
    if problems:
        return report, problems

    report['allocated_gpu_id'] = allocated_ids[0]
    report['torch_device_count'] = torch.cuda.device_count()
    if report['torch_device_count'] != EXPECTED_GPUS:
        problems.append(f"torch.cuda.device_count()={report['torch_device_count']} != {EXPECTED_GPUS} allocated GPU")
        return report, problems

    # Slurm GPU IDs are global; CUDA_VISIBLE_DEVICES may use a cgroup-local index (e.g. 0 for physical GPU 1).
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so prefer the scheduler's GPU ID for the selected scope.
    allocated_id = report['allocated_gpu_id']
    allocated = _nvidia_smi(['-i', allocated_id, f"--query-gpu={','.join(GPU_FIELDS)}", '--format=csv,noheader'])
    report['nvidia_smi_allocated'] = allocated
    fields = [] if allocated.startswith('error') or '\n' in allocated else [f.strip() for f in allocated.split(',')]
    if len(fields) != len(GPU_FIELDS):
        problems.append(f"nvidia-smi -i {allocated_id} did not describe exactly one GPU: {allocated!r}")
        return report, problems
    report['allocated_gpu'] = dict(zip(GPU_FIELDS, fields))
    if expected_uuid is not None and (report['allocated_gpu']['uuid'] != expected_uuid
                                     or report['allocated_gpu']['index'] != allocated_id):
        problems.append('Assigned UUID does not match the Slurm GPU reported by the driver')
        return report, problems

    if check_process:
        # GPUs on which THIS process holds a CUDA context: must be the allocated one and nothing else
        apps = _nvidia_smi(['--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'])
        pid = str(os.getpid())
        report['nvidia_smi_compute_apps_error'] = apps if apps.startswith('error') else None
        buses = sorted({line.split(',')[1].strip() for line in apps.splitlines()
                        if line.count(',') == 1 and line.split(',')[0].strip() == pid})
        report['process_gpu_bus_ids'] = buses
        if buses != [report['allocated_gpu']['pci.bus_id']]:
            problems.append(f"this process (pid {pid}) uses GPUs {buses}, expected only the allocated "
                            f"{report['allocated_gpu']['pci.bus_id']}")
    return report, problems


def worker_gpu_map():
    """Resolve only the two Slurm-assigned global GPU IDs to full UUIDs; no torch, no visibility change."""
    env = os.environ
    ids = _gpu_ids(env.get('SLURM_JOB_GPUS'))
    visible = visible_devices()
    if (not env.get('SLURM_JOB_ID') or env.get('SLURM_GPUS_ON_NODE') != '2'
            or env.get('SLURM_STEP_ID', '').isdecimal()
            or not ids or len(ids) != 2 or len(set(ids)) != 2
            or not all(i.isdecimal() for i in ids)
            or not visible or len(visible) != 2 or len(set(visible)) != 2
            or _gpu_ids(env.get('SLURM_STEP_GPUS')) not in (None, ids)):
        raise ValueError('Worker requires exactly two Slurm-assigned GPUs and original two-device visibility')
    mapping = []
    for gpu_id in ids:
        answer = _nvidia_smi(['-i', gpu_id, '--query-gpu=index,uuid', '--format=csv,noheader'])
        parts = [p.strip() for p in answer.split(',')]
        if ('\n' in answer or len(parts) != 2 or parts[0] != gpu_id
                or not parts[1].startswith('GPU-')):
            raise ValueError(f'Cannot identify Slurm GPU {gpu_id}: {answer}')
        mapping.append(dict(index=gpu_id, uuid=parts[1]))
    if mapping[0]['uuid'] == mapping[1]['uuid']:
        raise ValueError('Slurm GPU IDs resolved to the same physical GPU')
    return mapping
