"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Utility functions that are used by several different models.

    - GNN():                 Base GNN parent class for all graph neural network models
    - graphs_to_loader():    Turn lists of molecular graphs and their bioactivities into a dataloader
    - squeeze_if_needed():   if the input is a squeezable tensor, squeeze it

Reduced to GCN_GRAPH and corrected (see docs/upstream_changes.md):
    - train() now delegates the epoch loop to MoleculeACE.models.training.fit(): model.train() for training epochs,
      model.eval() for validation and prediction, selectable early-stopping mode, checkpoint at an explicit path
      (upstream: './best_model.pkl' in the working directory, shared by every run), and the best state is restored
      also when the epoch limit is reached (upstream: only after early stopping). In that case the last-epoch
      state, which upstream evaluated, is saved to final_checkpoint_path so both can be evaluated without retraining.
    - The checkpoint is a state_dict written atomically with torch.save (upstream: pickle of the whole module).
    - Unchanged: loaders (train batch 64 without shuffling, prediction batch 32), MSE loss, validation loss computed
      on the predictions of the whole validation fold, self.epochs overwritten by the model constructor.
    - Removed: NN base class, NumpyDataset, numpy_loader, plot_loss, scatter (other models / plotting only).
"""

from MoleculeACE.models.training import EarlyStopping, UPSTREAM_CONSECUTIVE, evaluating, fit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List
import torch
import os
import uuid


class GNN:
    """ Base GNN class that takes care of training, testing, predicting for all graph-based methods """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = None       # upstream: os.path.join('.', 'best_model.pkl')
        self.training_record = None

        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None

    def train(self, x_train: List[Data], y_train: List[float], x_val: List[Data] = None, y_val: List[float] = None,
              early_stopping_patience: int = None, epochs: int = None, print_every_n: int = 100,
              early_stopping_mode: str = UPSTREAM_CONSECUTIVE, checkpoint_path: str = None,
              final_checkpoint_path: str = None):
        """ Train a graph neural network.

        :param x_train: (List[Data]) a list of graph objects for training
        :param y_train: (List[float]) a list of bioactivites for training
        :param x_val: (List[Data]) a list of graph objects for validation
        :param y_val: (List[float]) a list of bioactivites for validation
        :param early_stopping_patience: (int) patience of the early-stopping rule
        :param epochs: (int) train the model for at most n epochs (default: self.epochs, as upstream)
        :param print_every_n: (int) printout training progress every n epochs
        :param early_stopping_mode: 'upstream_consecutive' or 'paper_nonconsecutive' (see models/training.py)
        :param checkpoint_path: (str) file for the best state_dict; required (no implicit working-directory file)
        :param final_checkpoint_path: (str) file for the last-epoch state_dict, written only when upstream would have
                                      kept the last weights (record['official_state'] == 'last_epoch')
        :return: (dict) training record: history, best epoch, stop reason, early-stopping state
        """
        if epochs is None:
            epochs = self.epochs
        if x_val is None or y_val is None or early_stopping_patience is None:
            raise ValueError('The benchmark protocol needs a validation fold and an early-stopping patience')
        checkpoint_path = checkpoint_path or self.save_path
        if not checkpoint_path:
            raise ValueError('An explicit checkpoint_path is required')
        self.save_path = checkpoint_path

        train_loader = graphs_to_loader(x_train, y_train)
        y_val_tensor = torch.tensor(y_val)

        def train_one_epoch():
            loss = self._one_epoch(train_loader)
            self.train_losses.append(loss)
            return loss

        def validation_loss():
            val_pred = self.predict(x_val)
            val_loss = float(self.loss_fn(squeeze_if_needed(val_pred), y_val_tensor))
            self.val_losses.append(val_loss)
            self.epoch += 1
            return val_loss

        def save_final(epoch):
            if not final_checkpoint_path:
                raise ValueError('final_checkpoint_path is required when upstream would keep the last weights')
            self.save_checkpoint(final_checkpoint_path, epoch)

        stopper = EarlyStopping(early_stopping_patience, early_stopping_mode)
        self.training_record = fit(self.model, epochs, train_one_epoch, validation_loss,
                                   lambda epoch: self.save_checkpoint(checkpoint_path, epoch),
                                   lambda: self.load_checkpoint(checkpoint_path), stopper,
                                   log_every=print_every_n, save_final=save_final)
        return self.training_record

    def save_checkpoint(self, path: str, epoch: int):
        """ Atomic write of the current state_dict: a unique temporary file in the same folder, then os.replace """
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        torch.save(dict(epoch=epoch, state_dict=state), tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str):
        """ Load a state_dict saved by save_checkpoint(); returns its epoch """
        saved = torch.load(path, map_location=self.device)
        self.model.load_state_dict(saved['state_dict'])
        return saved['epoch']

    def _one_epoch(self, train_loader):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: mean batch loss of the epoch (upstream returned the loss of the last batch; only used for logging)
        """
        losses = []
        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            batch.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y))

            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

            losses.append(float(loss.detach()))

        return sum(losses) / len(losses)

    def test(self,  x_test: List[Data], y_test: List[float]):
        """ Perform testing

        :param x_test: (List[Data]) a list of graph objects for testing
        :param y_test: (List[float]) a list of bioactivites for testing
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        data_loader = graphs_to_loader(x_test, y_test, shuffle=False)
        y_pred, y = [], []
        with evaluating(self.model), torch.no_grad():
            for batch in data_loader:
                batch.to(self.device)
                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                    y.extend(squeeze_if_needed(batch.y).tolist())
                else:
                    y_pred.append(y_hat)
                    y.append(squeeze_if_needed(batch.y).tolist())

        return torch.tensor(y_pred), torch.tensor(y)

    def predict(self, x, batch_size: int = 32):
        """ Predict bioactivity on molecular graphs (in eval() mode; the previous mode is restored afterwards)

        :param x_test: (List[Data]) a list of graph objects for testing
        :param batch_size: (int) batch size for the prediction loader
        :return: A 1D-tensors of predicted values
        """
        loader = DataLoader(x, batch_size=batch_size, shuffle=False)
        y_pred = []
        with evaluating(self.model), torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)

        return torch.tensor(y_pred)

    def __repr__(self):
        return 'Basic Graph Neural Network Class'


def graphs_to_loader(x: List[Data], y: List[float], batch_size: int = 64, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)

    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor
