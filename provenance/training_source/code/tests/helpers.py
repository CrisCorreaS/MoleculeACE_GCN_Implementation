"""Synthetic campaigns for local tests: real manifests and source CSVs, fake training (no torch)."""
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from MoleculeACE.benchmark.campaign import build_manifest, source_test_rows, task_dir, write_manifest
from MoleculeACE.benchmark.artifacts import canonical_json_sha256
from MoleculeACE.models.training import EarlyStopping, fit


class FakeModel:
    """ Mimics torch.nn.Module.train()/eval() and records the mode seen by each phase """
    def __init__(self):
        self.training = True
        self.calls = []

    def train(self, mode=True):
        self.training = mode
        self.calls.append('train' if mode else 'eval')
        return self

    def eval(self):
        return self.train(False)


def scripted_fit(val_losses, epochs, mode, patience=10):
    """ Run the real fit() loop with scripted validation losses; returns (record, model, events) """
    model = FakeModel()
    losses = iter(val_losses)
    events = []

    def train_one_epoch():
        events.append(('train_epoch', model.training))
        return 1.0

    def validation_loss():
        events.append(('validate', model.training))
        return next(losses)

    record = fit(model, epochs, train_one_epoch, validation_loss,
                 save_checkpoint=lambda epoch: events.append(('save', epoch)),
                 restore_checkpoint=lambda: events.append(('restore', model.training)),
                 stopper=EarlyStopping(patience, mode),
                 save_final=lambda epoch: events.append(('final', epoch)))
    return record, model, events


def synthetic_record(task, rng, n=None):
    """ A task record whose training history follows the manifest's early-stopping rule """
    n = rng.randint(15, 40) if n is None else n
    losses = [1.0 / (1 + e) for e in range(n)] + [2.0] * 400      # improves n epochs, then plateaus
    record, _, _ = scripted_fit(losses, task['max_epochs'], task['early_stopping_mode'], task['patience'])
    effective = dict(dataset=task['dataset'], fold=task['fold'], epochs=task['max_epochs'])
    return dict(published_config={'lr': 5e-05}, yaml_epochs_ignored=123, effective_config=effective,
                effective_config_sha256=canonical_json_sha256(effective), seed=task['seed'],
                training=record, timing=dict(total_seconds=float(rng.randint(10, 100))),
                resources=dict(peak_vram_allocated_bytes=10 ** 9, peak_rss_bytes=4 * 10 ** 9),
                provenance=dict(python='3.8.18', versions={'torch': '1.11.0+cu113'}, source_sha256={'a.py': 'x'}))


def synthetic_training(folder, task, rng, noise=0.5, n_improving=None, tag=b''):
    """ What train_task() leaves and returns, without torch: fake checkpoints in folder, (official, best, record) """
    record = synthetic_record(task, rng, n_improving)
    tag = tag + task['dataset'].encode() + bytes([task['fold']])
    (folder / 'best_model.pt').write_bytes(b'fake best checkpoint ' + tag)
    rows = source_test_rows(task['dataset'])
    best = [(csv_row, smiles, y, y + rng.gauss(0, noise), cliff) for csv_row, smiles, y, cliff in rows]
    official = best
    if record['training']['official_state'] == 'last_epoch':
        (folder / 'final_model.pt').write_bytes(b'fake last-epoch checkpoint ' + tag)
        official = [(csv_row, smiles, y, y + rng.gauss(0, noise), cliff) for csv_row, smiles, y, cliff in rows]
    return official, best, record


def write_synthetic_task(root, task, rng, noise=0.5, n_improving=None):
    """ Fake checkpoints and predictions; official != best-validation predictions when upstream kept the last epoch """
    from run_benchmark import write_task_outputs
    folder = task_dir(root, task)
    folder.mkdir(parents=True, exist_ok=True)
    official, best, record = synthetic_training(folder, task, rng, noise, n_improving)
    return write_task_outputs(root, task, official, best, record)


class CampaignTestCase(unittest.TestCase):
    """ Temporary project root with results/ and jobs/; the package data stays the real one """
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='moleculeace_test_'))
        (self.root / 'jobs').mkdir()
        self.rng = random.Random(0)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def make_campaign(self, campaign, stage, names, folds=None, max_epochs=None, write=True):
        rows = build_manifest(campaign, stage, names, folds=folds, max_epochs=max_epochs)
        path = self.root / 'jobs' / f"manifest_{campaign}.csv"
        write_manifest(path, rows)
        if write:
            for task in rows:
                write_synthetic_task(self.root, task, self.rng)
        return path, rows
