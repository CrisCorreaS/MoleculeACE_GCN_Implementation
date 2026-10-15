"""Check the pinned historical training environment inside an allocated Slurm job.

Inputs are installed package versions, graph-feature shapes and scheduler/device
information. Both modes require the training stack; this is not the local CPU
analysis-environment checker. --gpu additionally exercises the allocated CUDA
device. It checks compatibility without training a model.

The CLI writes results/environment/env_check_<host>_<job>.json and returns exit
status two for failed checks. From code/ inside an existing Slurm GPU job:
    python env_check.py --gpu

For the launcher and environment prerequisites, see docs/reproduction.md.
The command rejects execution without SLURM_JOB_ID.
"""
import argparse
import importlib
import inspect
import json
import os
import platform
import socket
import sys

from MoleculeACE.benchmark.artifacts import atomic_write_json
from MoleculeACE.benchmark.campaign import resolve_project_root
from MoleculeACE.benchmark.gpu_allocation import allocation_report

EXPECTED = {'python': '3.8', 'torch': '1.11.0+cu113', 'torch_geometric': '2.0.4', 'torch_scatter': '2.0.9',
            'torch_sparse': '0.6.13', 'numpy': '1.23.5', 'pandas': '1.5.3', 'sklearn': '1.1.3', 'rdkit': '2022.09.5'}


def main():
    """Validate the historical stack and write its environment/device report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', action='store_true', help='Also check CUDA (only inside a Slurm GPU allocation)')
    parser.add_argument('--project-root', default=None)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Environment checks run only inside Slurm; use jobs/run_launch.sh --cpu env_check or --checks')
    root = resolve_project_root(args.project_root)
    report, problems = dict(host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                            platform=platform.platform(), python=platform.python_version()), []

    if not report['python'].startswith(EXPECTED['python'] + '.'):
        problems.append(f"python {report['python']} != {EXPECTED['python']}.x")
    versions = {}
    for module in ('torch', 'torch_geometric', 'torch_scatter', 'torch_sparse', 'numpy', 'pandas', 'sklearn',
                   'rdkit', 'yaml', 'Levenshtein', 'matplotlib'):
        try:
            versions[module] = getattr(importlib.import_module(module), '__version__', 'unknown')
        except Exception as exc:  # report every broken import, then fail below
            versions[module] = None
            problems.append(f"cannot import {module}: {exc!r}")
    report['versions'] = versions
    for module, expected in EXPECTED.items():
        if module in versions and versions[module] is not None and versions[module] != expected:
            problems.append(f"{module} {versions[module]} != pinned {expected}")

    if versions.get('torch_geometric'):
        from torch_geometric.nn import GraphMultisetTransformer
        params = list(inspect.signature(GraphMultisetTransformer.__init__).parameters)
        report['gmt_signature'] = params
        if not {'in_channels', 'hidden_channels', 'out_channels', 'Conv', 'num_heads'} <= set(params):
            problems.append('GraphMultisetTransformer lacks the API used by models/gcn.py')
    if versions.get('torch') and versions.get('rdkit'):
        from MoleculeACE.benchmark.featurization import Featurizer
        graph = Featurizer().graphs(['c1ccccc1C(=O)N'])[0]
        report['graph_features'] = [int(graph.x.shape[1]), int(graph.edge_attr.shape[1])]
        if report['graph_features'] != [37, 6]:
            problems.append(f"graph features {report['graph_features']} != [37, 6]")

    if args.gpu and versions.get('torch'):
        import torch
        # Validate Slurm visibility before creating any CUDA tensor.
        report['gpu_allocation'], allocation_problems = allocation_report(torch, check_process=False)
        problems.extend(allocation_problems)
        if not problems:
            x = torch.randn(1024, 1024, device='cuda')
            report['gpu'] = dict(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                                 torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                 arch_list=torch.cuda.get_arch_list(), matmul_ok=bool(torch.isfinite(x @ x).all()))
            if 'sm_80' not in report['gpu']['arch_list']:
                problems.append('This torch build has no sm_80 (A100) kernels')
            if not report['gpu']['matmul_ok']:
                problems.append('GPU matrix multiplication produced non-finite values')
            # After computation, also check that this process uses its allocated GPU only.
            report['gpu_allocation'], allocation_problems = allocation_report(torch)
            problems.extend(allocation_problems)

    report['problems'] = problems
    out = root / 'results' / 'environment' / f"env_check_{report['host']}_{report['slurm_job_id'] or 'login'}.json"
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2, default=str))
    if problems:
        print('FAILED: ' + '; '.join(problems), file=sys.stderr)
        sys.exit(2)
    print(f"OK -> {out}")


if __name__ == '__main__':
    main()
