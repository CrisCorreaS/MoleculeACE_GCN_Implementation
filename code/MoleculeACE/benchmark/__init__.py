"""Expose benchmark constants, data partitions and metric/configuration helpers.

These exports require NumPy, pandas and PyYAML. ActivityCliffs and Featurizer
are imported explicitly from their own modules when needed, avoiding an eager
Torch/PyG/RDKit dependency. Importing this namespace does not run verification,
training or file generation.
"""
from MoleculeACE.benchmark.const import Algorithms, Descriptors, RANDOM_SEED, datasets, WORKING_DIR, CONFIG_PATH_GCN

from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse, get_config, write_config, get_benchmark_config
