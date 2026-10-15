"""MoleculeACE reduced to the GCN_GRAPH strategy (see docs/upstream_changes.md).

Upstream imported every model here (TensorFlow, Transformers, ...). Only lightweight names are exported now;
heavy dependencies (torch, torch_geometric, rdkit) are imported by the submodules that need them, e.g.
``from MoleculeACE.models import GCN`` or ``from MoleculeACE.benchmark.featurization import Featurizer``.
"""
from MoleculeACE.benchmark.const import Algorithms, Descriptors, RANDOM_SEED, datasets, WORKING_DIR, \
    CONFIG_PATH_GCN
from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse, get_config, write_config, \
    get_benchmark_config
