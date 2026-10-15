"""Load corrected benchmark partitions, configurations and response metrics.

Data reads existing CSV rows without replacing the published split or cliff
labels. Responses use y = -log10(concentration in nM); pKi/pEC50 is y + 9.
molecular_folds subdivides only the published training responses. Configuration
helpers read the supplied YAML/JSON; metric helpers return values without files.

Import from MoleculeACE.benchmark.utils. NumPy, pandas and PyYAML are required
at import; sklearn, RDKit and graph dependencies are loaded only by functions
that need them. get_config/read helpers do not train models; write_config
explicitly writes its requested path. See docs/protocol.md and
docs/upstream_changes.md for the retained protocol and known limitations.

Author: Derek van Tilborg -- TU/e -- 23-05-2022. The package is reduced to
GCN_GRAPH; unrelated models, tokenisation and augmentation are omitted.
"""

from MoleculeACE.benchmark.const import Descriptors, datasets, DATA_PATH, RANDOM_SEED, CONFIG_PATH
from yaml import load, Loader, dump
from typing import List, Union
import pandas as pd
import numpy as np
import json
import os


class Data:
    """Hold aligned published train/test responses and optional molecular features."""
    def __init__(self, file: Union[str, pd.DataFrame]):
        """Read the published partitions without fitting a transformation.

        Parameters
        ----------
        file : str or pandas.DataFrame
            Dataset identifier, CSV filename or frame containing smiles, y, split and
            cliff_mol. The response is y = -log10(concentration in nM); split values
            select the published train/test rows and cliff_mol is an evaluation label.

        Notes
        -----
        Row order is retained separately in each partition. Graph features are created
        only when featurize_data is called; cliff labels are never model inputs.
        CSV/schema validation belongs to the campaign's input checks.
        """

        # Either load a .csv file or use a provided dataframe
        if type(file) is str:
            if file in datasets:
                file = os.path.join(DATA_PATH, f"{file}.csv")
            df = pd.read_csv(file)
        else:
            df = file

        self.smiles_train = df[df['split'] == 'train']['smiles'].tolist()
        self.y_train = df[df['split'] == 'train']['y'].tolist()
        self.cliff_mols_train = df[df['split'] == 'train']['cliff_mol'].tolist()

        self.smiles_test = df[df['split'] == 'test']['smiles'].tolist()
        self.y_test = df[df['split'] == 'test']['y'].tolist()
        self.cliff_mols_test = df[df['split'] == 'test']['cliff_mol'].tolist()

        # Upstream built the Featurizer here; it is now created on first use so that loading data needs no rdkit/torch
        self.featurizer = None

        self.x_train = None
        self.x_test = None

        self.featurized_as = 'Nothing'

    def featurize_data(self, descriptor: Descriptors, **kwargs):
        """Populate aligned training and test features from their SMILES.

        Parameters
        ----------
        descriptor : Descriptors
            GRAPH for the benchmark or ECFP for the retained fingerprint interface.
        **kwargs
            Options passed to the selected featurisation method.

        Returns
        -------
        None
            Sets x_train, x_test and featurized_as in place without changing labels.
            Graph/ECFP construction does not fit a transform to test responses.
        """
        if self.featurizer is None:
            from MoleculeACE.benchmark.featurization import Featurizer
            self.featurizer = Featurizer()

        self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, **kwargs)
        self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, **kwargs)
        self.featurized_as = descriptor.name

    def __call__(self, descriptor: Descriptors, **kwargs):
        self.featurize_data(descriptor, **kwargs)

    def __repr__(self):
        return f"Data object with molecules as: {self.featurized_as}. {len(self.y_train)} train/{len(self.y_test)} test"


def get_config(file: str):
    """Read a non-empty YAML or JSON configuration mapping.

    Parameters
    ----------
    file : str
        Path ending in .yml, .yaml or .json.

    Returns
    -------
    dict
        Parsed mapping, without validating model-specific parameter names or values.

    Raises
    ------
    ValueError
        If the extension is unsupported or the parsed content is not a non-empty map.

    Notes
    -----
    This inherited reader uses yaml.Loader. Read only trusted, hash-checked
    published configurations; the reader is not a safe loader for arbitrary input.
    """
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    elif file.endswith('.json'):
        with open(file, 'r', encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {file}")
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Configuration is empty or not a mapping: {file}")
    return config


def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    args = {k: v.item() if isinstance(v, np.generic) else v for k, v in args.items()}
    with open(filename, 'w') as file:
        dump(args, file)


def get_benchmark_config(dataset: str, algorithm='GCN', descriptor='GRAPH'):
    """ Get the published (optimised) hyperparameters of GCN_GRAPH for one of the 30 datasets

    :param dataset: (str) one of the 30 datasets included in MoleculeACE
    :param algorithm: only 'GCN' (or the GCN class) is kept in this reduced version
    :param descriptor: only 'GRAPH' (or Descriptors.GRAPH) is kept in this reduced version
    :return: (dict) hyperparameters, exactly as stored in the YAML (including the 'epochs' key)
    """
    if dataset not in datasets:
        raise ValueError(f"Chosen dataset is not included, please pick from: {datasets}")
    algorithm = algorithm if type(algorithm) is str else algorithm.__name__
    descriptor = descriptor if type(descriptor) is str else descriptor.name
    if (algorithm, descriptor) != ('GCN', 'GRAPH'):
        raise ValueError('This reduced MoleculeACE only contains GCN_GRAPH')

    config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{algorithm}_{descriptor}.yml")
    return get_config(config_path)


def calc_rmse(true, pred):
    """Calculate RMSE between aligned one-dimensional response arrays.

    Parameters
    ----------
    true, pred : array-like, shape (n_molecules,)
        Non-empty true and predicted responses in the same scale. Benchmark values
        are y = -log10(concentration in nM). Callers must ensure matching shapes.

    Returns
    -------
    numpy floating scalar
        Root mean squared error in the response units. The campaign uses the stricter
        artifacts.fold_metrics interface to validate lengths and finite values.
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """Calculate RMSE on an explicitly supplied activity-cliff molecule mask.

    Parameters
    ----------
    y_test_pred, y_test : array-like, shape (n_test,)
        Aligned predictions and responses. Benchmark values use
        y = -log10(concentration in nM).
    cliff_mols_test : list of int or bool
        Binary mask in test-row order. It must be supplied explicitly; use the
        published mask for benchmark evaluation.
    smiles_test, smiles_train : list of str, optional
        Retained for call compatibility; not used to infer cliff membership.
    y_train : array-like, optional
        Retained for call compatibility; not used by this metric.
    **kwargs
        Retained for call compatibility; no cliff-recomputation options are used.

    Returns
    -------
    float or None
        Test-cliff RMSE, or None when the supplied mask contains no cliff molecules.

    Raises
    ------
    ValueError
        If cliff_mols_test is omitted, including when legacy structure and training
        arguments are supplied.

    Notes
    -----
    This helper never passes logarithmic responses to a concentration-ratio
    calculation. Compute any new cliff labels separately from positive
    concentrations. The completed campaign uses published masks through the
    stricter artifacts.fold_metrics interface.
    """

    if cliff_mols_test is None:
        raise ValueError('cliff_mols_test must be supplied explicitly; compute cliff labels '
                         'separately from positive concentrations')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]
    if not cliff_test_idx:
        return None

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)


def molecular_folds(y_train: List[float], n_folds: int = 5, seed: int = RANDOM_SEED):
    """Partition the published training responses into internal validation folds.

    Parameters
    ----------
    y_train : list of float
        Only published training responses, in y = -log10(concentration in nM).
    n_folds : int
        Number of internal folds; five in the benchmark.
    seed : int
        Random state for shuffled StratifiedKFold; 42 in the completed campaign.

    Returns
    -------
    list of dict
        train_idx and val_idx arrays indexing the supplied training rows. Each row
        validates once. The strata are y_train >= median(y_train), and no external
        test response participates. These folds are not independent random-seed runs.
    """
    from sklearn.model_selection import StratifiedKFold

    ss = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cutoff = np.median(y_train)
    labels = [0 if i < cutoff else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    return splits
