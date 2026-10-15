"""
This file holds many variables that are used throughout MoleculeACE

Author: Derek van Tilborg -- TU/e -- March 2022

Reduced to GCN_GRAPH (see docs/upstream_changes.md): only the GCN algorithm, the GRAPH descriptor and ECFP
(needed by the activity-cliff definition) remain. Helpers that created log/working folders inside the package were
removed, because nothing may be written into code/.
"""

import os
from enum import Enum
from pathlib import Path


def get_project_root() -> Path:
    """ Folder that contains the MoleculeACE package (code/ in this project). """
    return Path(__file__).parent.parent.parent

###
# Benchmark parameters
###


RANDOM_SEED = 42


class Algorithms(Enum):
    GCN = 'GCN'


class Descriptors(Enum):
    ECFP = 'ECFP'
    GRAPH = 'Graph'


package_path = os.path.join(get_project_root(), 'MoleculeACE')

WORKING_DIR = os.path.join(package_path, "Data")
DATA_PATH = os.path.join(WORKING_DIR, "benchmark_data")
CONFIG_PATH = os.path.join(WORKING_DIR, "configures")


###
# Default configures path (search space of the original Bayesian optimisation; not used by the benchmark)
###

CONFIG_PATH_GCN = os.path.join(CONFIG_PATH, 'default', 'GCN.yml')


datasets = ['CHEMBL4203_Ki', 'CHEMBL2034_Ki', 'CHEMBL233_Ki', 'CHEMBL4616_EC50', 'CHEMBL287_Ki', 'CHEMBL218_EC50',
            'CHEMBL264_Ki', 'CHEMBL219_Ki', 'CHEMBL2835_Ki', 'CHEMBL2147_Ki', 'CHEMBL231_Ki', 'CHEMBL3979_EC50',
            'CHEMBL237_EC50', 'CHEMBL244_Ki', 'CHEMBL4792_Ki', 'CHEMBL1871_Ki', 'CHEMBL237_Ki', 'CHEMBL262_Ki',
            'CHEMBL2047_EC50', 'CHEMBL239_EC50', 'CHEMBL2971_Ki', 'CHEMBL204_Ki', 'CHEMBL214_Ki', 'CHEMBL1862_Ki',
            'CHEMBL234_Ki', 'CHEMBL238_Ki', 'CHEMBL235_EC50', 'CHEMBL4005_Ki', 'CHEMBL236_Ki', 'CHEMBL228_Ki']
