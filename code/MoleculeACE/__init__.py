"""Expose the data and metric interfaces for the bounded GCN_GRAPH package.

Importing MoleculeACE loads NumPy, pandas and PyYAML through benchmark.utils.
Torch, PyG and RDKit are loaded by the submodules that use them, not here.
Import the model with 'from MoleculeACE.models import GCN'; use
run_benchmark.py for explicit training. Package import creates no result files.

The 30 corrected datasets, published configurations and retained implementation
are described in docs/protocol.md and docs/upstream_changes.md.
"""
from MoleculeACE.benchmark.const import Algorithms, Descriptors, RANDOM_SEED, datasets, WORKING_DIR, \
    CONFIG_PATH_GCN
from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse, get_config, write_config, \
    get_benchmark_config
