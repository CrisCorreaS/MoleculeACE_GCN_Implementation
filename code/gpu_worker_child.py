"""Bind a Linux worker child to its allocated CPUs before replacing the process.

This internal shim uses only the standard library. run_gpu_worker.py supplies
a JSON CPU list as the first argument and an executable command thereafter;
the shim checks affinity, applies it, then calls os.execv. It writes no files
itself. Invoke through jobs/run_gpu_worker.sh inside a Slurm allocation,
as described in docs/reproduction.md, never directly on a login node.
"""
import json
import os
import sys


def bind_and_exec(cpus, command):
    """Validate a distinct CPU subset, set affinity and execute command.

    Parameters
    ----------
    cpus : list of int
        Non-empty subset of the process's current Linux CPU affinity.
    command : list of str
        Executable path followed by its arguments.

    Returns
    -------
    None
        Successful os.execv replaces this process and does not return. Invalid
        inputs raise ValueError; operating-system errors propagate to the caller.
    """
    if (not cpus or not all(type(cpu) is int for cpu in cpus)
            or len(set(cpus)) != len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0))):
        raise ValueError('Child CPU set is not a distinct subset of its Slurm allocation')
    if not command:
        raise ValueError('Missing child command')
    os.sched_setaffinity(0, cpus)
    os.execv(command[0], command)


if __name__ == '__main__':
    bind_and_exec(json.loads(sys.argv[1]), sys.argv[2:])
