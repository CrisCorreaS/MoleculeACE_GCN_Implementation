"""Apply early stopping and choose official and secondary model states.

Inputs are training/validation callbacks, a model-like object and checkpoint
callbacks. fit returns the epoch history and state-selection record; only the
supplied callbacks perform training or persistence. This module imports no
Torch and supports small fake-model tests. Package initialisation still needs
NumPy, pandas and PyYAML. See docs/protocol.md and docs/upstream_changes.md.

The completed campaign uses upstream_consecutive: reset the patience counter
on improvement. paper_nonconsecutive accumulates all non-improving epochs.
Both count validation ties as improvements. Attribution of the corrected
published results to the consecutive implementation is inferred from upstream
commit chronology, not from published execution manifests.

At the epoch ceiling the upstream rule keeps the last weights, including when
patience is exhausted on that final epoch. Otherwise early stopping selects the
best validation state. fit always restores the best state before returning;
the caller separately evaluates saved last-epoch weights when they are official.
This is not a resumable optimiser checkpoint or a new training experiment.
"""

import math
from contextlib import contextmanager

UPSTREAM_CONSECUTIVE = 'upstream_consecutive'
PAPER_NONCONSECUTIVE = 'paper_nonconsecutive'
EARLY_STOPPING_MODES = (UPSTREAM_CONSECUTIVE, PAPER_NONCONSECUTIVE)


class EarlyStopping:
    """Track validation improvements under either retained patience convention."""
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
        """Record a validation loss and advance the patience counter.

        Parameters
        ----------
        val_loss : float or None
            Validation MSE in squared logarithmic response units for this campaign.
            None and non-finite losses cannot improve the best state.
        epoch : int
            Completed epoch number, supplied in increasing order by fit.

        Returns
        -------
        tuple of bool
            (improved, stop). A finite tie improves; consecutive mode resets the
            counter on improvement, whereas non-consecutive mode does not.
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
        """Return a fresh dictionary of the stopping rule and recorded best state."""
        return dict(mode=self.mode, patience=self.patience, counter=self.counter, best_val_loss=self.best,
                    best_epoch=self.best_epoch, nonfinite_val_epochs=list(self.nonfinite_epochs))


@contextmanager
def evaluating(model):
    """Temporarily switch a model to evaluation mode and restore its prior mode.

    Parameters
    ----------
    model : object
        Object exposing training, eval() and train(mode).

    Yields
    ------
    object
        The supplied model in evaluation mode. Restoration also occurs on exceptions;
        this context manager itself does not disable gradient calculation.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


def fit(model, epochs: int, train_one_epoch, validation_loss, save_checkpoint, restore_checkpoint,
        stopper: EarlyStopping, log_every: int = 0, log=print, save_final=None):
    """Run fold training and preserve the upstream official-state selection.

    Parameters
    ----------
    model : object
        Model exposing training, train() and eval().
    epochs : int
        Positive maximum number of epochs.
    train_one_epoch, validation_loss : callable
        Zero-argument callbacks returning training loss and whole-fold validation
        loss. fit supplies train/eval mode; the callbacks control gradients.
    save_checkpoint, restore_checkpoint : callable
        Save the best state with its epoch, and restore it without arguments.
    stopper : EarlyStopping
        Fresh stopping state for this training run.
    log_every : int
        Positive logging interval, or zero to disable progress output.
    log : callable
        Destination for epoch progress messages.
    save_final : callable, optional
        Save last-epoch weights before restoration; required when they are official.

    Returns
    -------
    dict
        History, epoch counts, patience state and official_state. The supplied model
        is in its best-validation state and evaluation mode; last_epoch as the
        official state requires the caller to load the separately saved final state.

    Raises
    ------
    ValueError
        If the epoch count is invalid or required final-state persistence is absent.
    RuntimeError
        If validation never produces a finite loss.

    Notes
    -----
    Only validation loss controls checkpoint selection. Ties improve the best
    state. Patience exhausted exactly at the ceiling still follows the upstream
    last-epoch rule; best and last are equivalent when the best epoch is the last.
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
