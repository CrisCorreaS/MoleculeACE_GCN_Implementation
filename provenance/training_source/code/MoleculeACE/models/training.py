"""
Training protocol shared by the GCN (not part of upstream MoleculeACE; see docs/upstream_changes.md).

Kept free of torch so that it can be tested locally with small fake models:

    - EarlyStopping:   the two patience rules, selectable and always recorded
    - evaluating():    context manager that puts a model in eval() mode and restores its previous mode
    - fit():           epoch loop = train() epoch -> eval() validation -> checkpoint on improvement ->
                       stop rule -> save the last-epoch state if upstream would have kept it (official result)
                       -> restore the best state (secondary result when it differs)

Early-stopping modes (patience P = 10 in the benchmark):

    'upstream_consecutive'  Upstream code since commit 141ebc9 (2023-03-16), i.e. the code of the published corrected
                            results (45cf46d, 2023-09-29): the counter is reset to 0 on every improvement, so training
                            stops after P consecutive epochs without improvement.
    'paper_nonconsecutive'  Wording of the 2023 correction (DOI 10.1021/acs.jcim.3c00423) and upstream code before
                            141ebc9: the counter is never reset, so training stops after P epochs without improvement
                            in total, consecutive or not.

The two rules are NOT equivalent: 'paper_nonconsecutive' never trains longer, and usually stops earlier.
In both modes an epoch "improves" when val_loss <= best so far (upstream: val_loss <= min(self.val_losses)); ties count.
"""

import math
from contextlib import contextmanager

UPSTREAM_CONSECUTIVE = 'upstream_consecutive'
PAPER_NONCONSECUTIVE = 'paper_nonconsecutive'
EARLY_STOPPING_MODES = (UPSTREAM_CONSECUTIVE, PAPER_NONCONSECUTIVE)


class EarlyStopping:
    def __init__(self, patience: int, mode: str):
        if mode not in EARLY_STOPPING_MODES:
            raise ValueError(f"Unknown early-stopping mode {mode!r}; choose from {EARLY_STOPPING_MODES}")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError(f"Patience must be a positive integer, got {patience!r}")
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.best_epoch = None
        self.nonfinite_epochs = []

    def update(self, val_loss: float, epoch: int):
        """ Register the validation loss of a finished epoch.

        :return: (improved, stop) -- improved: save a checkpoint now; stop: do not train another epoch
        """
        val_loss = float('nan') if val_loss is None else float(val_loss)
        if not math.isfinite(val_loss):
            # Upstream compared NaN with <=, which is False: a non-finite loss is never an improvement
            self.nonfinite_epochs.append(epoch)
            improved = False
        else:
            improved = self.best is None or val_loss <= self.best

        if improved:
            self.best, self.best_epoch = val_loss, epoch
            if self.mode == UPSTREAM_CONSECUTIVE:
                self.counter = 0
        else:
            self.counter += 1

        return improved, self.counter >= self.patience

    def state(self):
        return dict(mode=self.mode, patience=self.patience, counter=self.counter, best_val_loss=self.best,
                    best_epoch=self.best_epoch, nonfinite_val_epochs=list(self.nonfinite_epochs))


@contextmanager
def evaluating(model):
    """ Temporarily switch a torch.nn.Module-like model to eval() mode, then restore its previous mode """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


def fit(model, epochs: int, train_one_epoch, validation_loss, save_checkpoint, restore_checkpoint,
        stopper: EarlyStopping, log_every: int = 0, log=print, save_final=None):
    """ Train with early stopping; finish with the best validated state and say which state upstream would keep.

    Upstream (models/utils.py, GNN.train) checked the patience at the START of each epoch and only then restored the
    best model. So it restored the best state only if the patience ran out before the last allowed epoch; when the
    epoch limit was reached (including patience running out exactly at the last epoch) it kept the last weights.
    The official state (record['official_state']) reproduces that rule:
        'best_validation'  upstream restored the best state, or the last epoch was the best one (identical)
        'last_epoch'       upstream kept the last weights; save_final(epoch) persists them before the best state
                           is restored, so both can be evaluated from the same training run (no retraining)

    :param model: object with .training, .train(mode=True) and .eval() (a torch.nn.Module)
    :param epochs: maximum number of epochs
    :param train_one_epoch: callable() -> mean training loss; called with the model in train() mode
    :param validation_loss: callable() -> validation loss; called with the model in eval() mode
    :param save_checkpoint: callable(epoch) that persists the current state (only called on improvement)
    :param restore_checkpoint: callable() that loads the best persisted state
    :param stopper: EarlyStopping instance
    :param save_final: callable(epoch) persisting the last-epoch state; required if the official state is the last
    :return: dict with the training record (history, best epoch, stop reason, official state, ...)
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")

    history = []
    stop_reason = 'epoch_limit'
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = float(train_one_epoch())
        with evaluating(model):
            val_loss = float(validation_loss())

        improved, stop = stopper.update(val_loss, epoch)
        if improved:
            save_checkpoint(epoch)
        # Non-finite losses are stored as None so that the record stays valid JSON
        history.append(dict(epoch=epoch, train_loss=train_loss if math.isfinite(train_loss) else None,
                            val_loss=val_loss if math.isfinite(val_loss) else None, improved=improved))
        if log_every and epoch % log_every == 0:
            log(f"Epoch {epoch} | Train Loss {train_loss} | Val Loss {val_loss}")
        if stop:
            stop_reason = 'early_stopping'
            break

    if stopper.best_epoch is None:
        raise RuntimeError('No finite validation loss was obtained; there is no state to restore')

    epochs_run = len(history)
    upstream_restores = stop_reason == 'early_stopping' and epochs_run < epochs
    official = 'last_epoch' if (not upstream_restores and stopper.best_epoch != epochs_run) else 'best_validation'
    if official == 'last_epoch':
        if save_final is None:
            raise ValueError('save_final is required: the official (upstream) state is the last epoch')
        save_final(epochs_run)

    # The model always ends in the best validated state (secondary result when official == 'last_epoch')
    restore_checkpoint()
    model.eval()

    return dict(epochs_max=epochs, epochs_run=epochs_run, best_epoch=stopper.best_epoch,
                best_val_loss=stopper.best, stop_reason=stop_reason, restored_best=True,
                upstream_would_restore_best=upstream_restores, last_epoch_is_best=stopper.best_epoch == epochs_run,
                official_state=official, early_stopping=stopper.state(), history=history)
