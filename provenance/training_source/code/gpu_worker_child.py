"""Stdlib-only affinity shim for a GPU worker's child; never run on the login node."""
import json
import os
import sys


def bind_and_exec(cpus, command):
    if (not cpus or not all(type(cpu) is int for cpu in cpus)
            or len(set(cpus)) != len(cpus) or not set(cpus).issubset(os.sched_getaffinity(0))):
        raise ValueError('Child CPU set is not a distinct subset of its Slurm allocation')
    if not command:
        raise ValueError('Missing child command')
    os.sched_setaffinity(0, cpus)
    os.execv(command[0], command)


if __name__ == '__main__':
    bind_and_exec(json.loads(sys.argv[1]), sys.argv[2:])
