"""Train and evaluate the retained graph regressor with explicit checkpoints.

Inputs are molecular graphs and responses y = -log10(concentration in nM).
GNN delegates the stopping/state-selection protocol to models.training, uses
the pinned Torch/PyG loaders, and writes checkpoints only during explicit
training calls. It returns predictions in input order and records epoch losses.

Use this module through GCN and run_benchmark.py in the historical training
environment; importing it requires Torch and PyG. See docs/protocol.md and
docs/upstream_changes.md. Loaders retain batches of 64 without training
shuffling and prediction batches of 32. Validation MSE covers the complete
internal validation fold; the external test set does not control stopping.

Author: Derek van Tilborg -- TU/e -- 23-05-2022.
The adaptation uses train/eval modes, task-local atomic state dictionaries,
and separate official/secondary checkpoint selection. The inherited protocol
keeps last-epoch weights at the ceiling; best-validation weights are also
retained for the clearly labelled secondary result.
"""

from MoleculeACE.models.training import EarlyStopping, UPSTREAM_CONSECUTIVE, evaluating, fit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List
import torch
import os
import uuid


class GNN:
    """Manage graph-regression fitting, checkpoint persistence and ordered inference."""
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
        """Fit one internal fold and restore its best-validation checkpoint.

        Parameters
        ----------
        x_train, x_val : list of torch_geometric.data.Data
            Non-empty training and validation graphs, with each label in the same order.
            Validation graphs must belong to the internal training split, not the test set.
        y_train, y_val : list of float
            Responses y = -log10(concentration in nM), not pKi/pEC50 values.
        early_stopping_patience : int
            Required positive patience for the selected stopping rule.
        epochs : int, optional
            Epoch ceiling; defaults to the model's configured maximum.
        print_every_n : int
            Logging interval; zero disables periodic epoch output.
        early_stopping_mode : str
            'upstream_consecutive' resets patience on improvement;
            'paper_nonconsecutive' accumulates non-improving epochs.
        checkpoint_path : str
            Required destination for the best-validation state dictionary.
        final_checkpoint_path : str, optional
            Required if the upstream rule selects the last epoch rather than the best.

        Returns
        -------
        dict
            History, best epoch, stopping state and official-state identity. The model
            itself finishes in its best-validation state and evaluation mode. The caller
            must load the final checkpoint when the record selects last_epoch officially.

        Notes
        -----
        Checkpoint files are atomically replaced but not explicitly fsynced. Ties in
        validation loss count as improvements. Optimiser state is not saved for resuming
        an interrupted run; the campaign retries an unfinished task from the beginning.
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
        """Return predictions and supplied responses in their original graph order.

        Parameters
        ----------
        x_test : list of torch_geometric.data.Data
            Graphs to evaluate with the current model weights.
        y_test : list of float
            Aligned responses y = -log10(concentration in nM).

        Returns
        -------
        tuple of torch.Tensor
            One-dimensional CPU tensors (predictions, responses), each of length
            len(x_test), in the same response scale. This method does not calculate
            or write benchmark metrics and restores the model's previous mode.
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
        """Predict logarithmic responses without gradients or changing model mode.

        Parameters
        ----------
        x : list of torch_geometric.data.Data
            Graphs in the desired output order.
        batch_size : int
            Number of graphs per prediction batch; the campaign uses 32.

        Returns
        -------
        torch.Tensor, shape (n_graphs,)
            CPU predictions of y = -log10(concentration in nM). Prediction uses eval()
            and restores the previous training/evaluation mode afterwards.
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
    """Attach response labels to graphs and construct the training loader.

    Parameters
    ----------
    x : list of torch_geometric.data.Data
        Graphs modified in place by assigning each graph's y attribute.
    y : list of float
        Aligned responses y = -log10(concentration in nM); callers must supply
        exactly one response per graph.
    batch_size : int
        Graphs per batch; 64 in the retained training protocol.
    shuffle : bool
        Whether to shuffle graphs; False in the published-configuration campaign.

    Returns
    -------
    torch_geometric.loader.DataLoader
        Loader over the supplied graph objects.
    """
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)

    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def squeeze_if_needed(tensor):
    """Squeeze a Torch tensor whose second dimension is one.

    Parameters
    ----------
    tensor : torch.Tensor
        Model output or response tensor.

    Returns
    -------
    torch.Tensor
        The tensor after squeeze() when the inherited condition matches. A
        single-graph (1, 1) output becomes a scalar; other inputs are unchanged.
    """
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor
