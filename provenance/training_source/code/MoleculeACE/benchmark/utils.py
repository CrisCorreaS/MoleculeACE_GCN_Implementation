"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

A collection of random functions that are used in different places

    - Data:                     Class that holds all data and featurizes it
        - featurize_data()
    - get_config():             function to load a dict of parameters from a .yml or .json
    - write_config():           function to write a dict of parameters to a .yml
    - calc_rmse():              calculate Root Mean Square Error
    - calc_cliff_rmse():        calculate Root Mean Square Error on activity cliff compounds
    - molecular_folds():        the internal train/validation split of upstream cross_validate()

Reduced to GCN_GRAPH (see docs/upstream_changes.md). Removed: TensorFlow/Transformers imports, SMILES augmentation,
tokenisation, shuffle(), load_model(). cross_validate() is replaced by molecular_folds() plus one task per fold in
code/run_benchmark.py; the split lines are unchanged. Heavy imports (rdkit, sklearn, featurizer) are lazy.
"""

from MoleculeACE.benchmark.const import Descriptors, datasets, DATA_PATH, RANDOM_SEED, CONFIG_PATH
from yaml import load, Loader, dump
from typing import List, Union
import pandas as pd
import numpy as np
import json
import os


class Data:
    def __init__(self, file: Union[str, pd.DataFrame]):
        """ Data class to easily load and featurize molecular bioactivity data

        :param file: 1. (str) path to .csv file with the following columns; 'smiles': (str) SMILES strings
                                                                                'y': (float) bioactivity values
                                                                                'cliff_mol': (int) 1 if cliff else 0
                                                                                'split': (str) 'train' or 'test'
                     2. (str) name of a dataset provided with the benchmark; see MoleculeACE.benchmark.const.datasets
                     3. (pandas.DataFrame) pandas dataframe with columns similar to 1. (see above)
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
        """ Encode your molecules with Descriptors.ECFP or Descriptors.GRAPH """
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
    """ Load a yml config file. Fails loudly (upstream returned an unbound name for other extensions) """
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
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
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
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds, or None if the test set has no cliff compounds
             (upstream returned NaN with a RuntimeWarning in that case)
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        from MoleculeACE.benchmark.cliffs import ActivityCliffs
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]
    if not cliff_test_idx:
        return None

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)


def molecular_folds(y_train: List[float], n_folds: int = 5, seed: int = RANDOM_SEED):
    """ Internal train/validation folds, copied line by line from upstream cross_validate():
    StratifiedKFold(shuffle=True, random_state=seed) stratified on y_train >= median(y_train).

    :return: (list) [{'train_idx': array, 'val_idx': array}, ...] indices into the training set
    """
    from sklearn.model_selection import StratifiedKFold

    ss = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cutoff = np.median(y_train)
    labels = [0 if i < cutoff else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    return splits
