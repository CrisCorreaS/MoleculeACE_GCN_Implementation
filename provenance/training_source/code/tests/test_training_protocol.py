"""Early stopping, train()/eval() switching and best-state restoration (torch-free, local)."""
import math
import unittest

from MoleculeACE.models.training import (PAPER_NONCONSECUTIVE, UPSTREAM_CONSECUTIVE, EarlyStopping, evaluating,
                                         fit)
from helpers import FakeModel, scripted_fit


class EarlyStoppingModes(unittest.TestCase):
    # improvements at epochs 1, 3, 5, 7 ... separated by single non-improving epochs
    ALTERNATING = [10, 11, 9, 11, 8, 11, 7, 11, 6, 11, 5, 11, 4, 11, 3, 11, 2, 11, 1, 11, 0.5, 11, 0.4, 11]

    def epochs_until_stop(self, losses, mode, patience):
        stopper = EarlyStopping(patience, mode)
        for epoch, loss in enumerate(losses, 1):
            _, stop = stopper.update(loss, epoch)
            if stop:
                return epoch, stopper
        return None, stopper

    def test_modes_differ_on_interleaved_improvements(self):
        stop_up, _ = self.epochs_until_stop(self.ALTERNATING, UPSTREAM_CONSECUTIVE, 3)
        stop_paper, s = self.epochs_until_stop(self.ALTERNATING, PAPER_NONCONSECUTIVE, 3)
        self.assertIsNone(stop_up)            # never 3 bad epochs in a row
        self.assertEqual(stop_paper, 6)       # 3rd non-improving epoch in total
        self.assertEqual(s.best_epoch, 5)

    def test_modes_agree_on_a_plateau(self):
        losses = [5, 4, 3] + [3.5] * 20
        for mode in (UPSTREAM_CONSECUTIVE, PAPER_NONCONSECUTIVE):
            stop, s = self.epochs_until_stop(losses, mode, 10)
            self.assertEqual((stop, s.best_epoch), (13, 3))

    def test_ties_count_as_improvement_like_upstream(self):
        stop, s = self.epochs_until_stop([2, 2, 2, 2], UPSTREAM_CONSECUTIVE, 1)
        self.assertIsNone(stop)
        self.assertEqual(s.best_epoch, 4)

    def test_nonfinite_losses_never_improve(self):
        stop, s = self.epochs_until_stop([1.0, float('nan'), None, float('inf')], UPSTREAM_CONSECUTIVE, 3)
        self.assertEqual((stop, s.best_epoch, s.nonfinite_epochs), (4, 1, [2, 3, 4]))

    def test_invalid_arguments(self):
        for patience, mode in [(0, UPSTREAM_CONSECUTIVE), (True, UPSTREAM_CONSECUTIVE), (10, 'consecutive')]:
            with self.assertRaises(ValueError):
                EarlyStopping(patience, mode)


class FitLoop(unittest.TestCase):
    def test_train_and_eval_modes(self):
        record, model, events = scripted_fit([3, 2, 1], epochs=3, mode=UPSTREAM_CONSECUTIVE)
        self.assertTrue(all(training for kind, training in events if kind == 'train_epoch'))
        self.assertFalse(any(training for kind, training in events if kind == 'validate'))
        self.assertFalse(model.training)                     # finishes in eval() mode, ready to predict

    def test_epoch_limit_official_is_last_epoch_secondary_is_best(self):
        # best at epoch 2, limit 4 reached before patience: upstream kept epoch-4 weights (official);
        # the last state is saved before the best one is restored (secondary), without retraining
        record, _, events = scripted_fit([3, 1, 2, 2.5], epochs=4, mode=UPSTREAM_CONSECUTIVE, patience=10)
        self.assertEqual((record['stop_reason'], record['best_epoch'], record['epochs_run']), ('epoch_limit', 2, 4))
        self.assertFalse(record['upstream_would_restore_best'])
        self.assertEqual(record['official_state'], 'last_epoch')
        self.assertEqual([e for e in events if e[0] in ('save', 'final', 'restore')],
                         [('save', 1), ('save', 2), ('final', 4), ('restore', True)])

    def test_patience_exhausted_at_the_last_epoch_is_upstream_last_epoch(self):
        # upstream checked the patience at the start of the next epoch, which never came: no restore
        record, _, events = scripted_fit([1, 2, 3], epochs=3, mode=UPSTREAM_CONSECUTIVE, patience=2)
        self.assertEqual((record['stop_reason'], record['epochs_run']), ('early_stopping', 3))
        self.assertFalse(record['upstream_would_restore_best'])
        self.assertEqual(record['official_state'], 'last_epoch')
        self.assertIn(('final', 3), events)

    def test_best_last_epoch_needs_no_final_checkpoint(self):
        record, _, events = scripted_fit([3, 2, 1], epochs=3, mode=UPSTREAM_CONSECUTIVE)
        self.assertEqual(record['official_state'], 'best_validation')
        self.assertNotIn('final', [e[0] for e in events])

    def test_final_checkpoint_callback_required_when_official_is_last(self):
        with self.assertRaises(ValueError):
            fit(FakeModel(), 2, lambda: 1, iter([1, 2]).__next__, lambda e: None, lambda: None,
                EarlyStopping(10, UPSTREAM_CONSECUTIVE))

    def test_restores_best_after_early_stopping(self):
        record, _, events = scripted_fit([1, 2, 3, 4, 5], epochs=300, mode=UPSTREAM_CONSECUTIVE, patience=2)
        self.assertEqual((record['stop_reason'], record['epochs_run'], record['best_epoch']), ('early_stopping', 3, 1))
        self.assertTrue(record['upstream_would_restore_best'])
        self.assertEqual(record['official_state'], 'best_validation')
        self.assertNotIn('final', [e[0] for e in events])
        self.assertEqual(events[-1][0], 'restore')

    def test_no_finite_validation_loss_fails(self):
        with self.assertRaises(RuntimeError):
            scripted_fit([float('nan')] * 3, epochs=3, mode=UPSTREAM_CONSECUTIVE)

    def test_history_is_json_safe(self):
        record, _, _ = scripted_fit([1.0, float('nan'), 0.5], epochs=3, mode=UPSTREAM_CONSECUTIVE)
        self.assertIsNone(record['history'][1]['val_loss'])

    def test_evaluating_restores_previous_mode_on_exception(self):
        model = FakeModel()
        model.train(True)
        with self.assertRaises(KeyError):
            with evaluating(model):
                self.assertFalse(model.training)
                raise KeyError('boom')
        self.assertTrue(model.training)
        model.eval()
        with evaluating(model):
            pass
        self.assertFalse(model.training)

    def test_invalid_epochs(self):
        with self.assertRaises(ValueError):
            fit(FakeModel(), 0, lambda: 1, lambda: 1, lambda e: None, lambda: None,
                EarlyStopping(10, UPSTREAM_CONSECUTIVE))


if __name__ == '__main__':
    unittest.main()
