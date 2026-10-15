"""
Artifact helpers for campaign outputs (not part of upstream MoleculeACE). Standard library only.

    - sha256_file() / canonical_json_sha256()
    - atomic_write_bytes() / atomic_write_json() / atomic_write_csv():  unique temporary file + os.replace
    - read_json():       strict JSON (NaN/Infinity rejected)
    - fold_metrics():    RMSE on the whole test set and RMSE_cliff on its cliff molecules (each molecule once);
                         RMSE_cliff is None when the test set has no cliff molecule (never 0)
    - read_predictions(): parse and type-check a predictions.csv
"""

import csv
import hashlib
import io
import json
import math
import os
import uuid
from pathlib import Path

ACTIVITY_SCALE = 'y=-log10(nM)'

PREDICTION_FIELDS = ['id', 'csv_row', 'smiles', 'y_true', 'y_pred', 'cliff_mol', 'fold', 'dataset', 'campaign',
                     'config_sha256', 'effective_config_sha256', 'activity_scale']


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj) -> str:
    """ Deterministic JSON; raises ValueError on NaN/Infinity """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


def canonical_json_sha256(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode('utf-8'))


def atomic_write_bytes(path, data: bytes):
    """ Write to a unique temporary file in the destination folder, fsync, then atomically replace """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path, obj):
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))


def atomic_write_csv(path, rows, fieldnames):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator='\n', extrasaction='raise')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write_bytes(path, buffer.getvalue().encode('utf-8'))


def _reject_constant(name):
    raise ValueError(f"Non-finite JSON constant {name}")


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {value}")
    return value


def fold_metrics(y_true, y_pred, cliff_mol):
    """ RMSE and RMSE_cliff of one model on the test set.

    :param y_true: true activities (y = -log10 nM)
    :param y_pred: predicted activities, same order
    :param cliff_mol: 0/1 labels of the test molecules (evaluation only)
    :return: dict(n_test, n_cliff, rmse, rmse_cliff) -- rmse_cliff is None if n_cliff == 0
    """
    if not (len(y_true) == len(y_pred) == len(cliff_mol)) or len(y_true) == 0:
        raise ValueError('y_true, y_pred and cliff_mol must be non-empty and of equal length')
    squared, squared_cliff = [], []
    for t, p, c in zip(y_true, y_pred, cliff_mol):
        t, p = _finite(t, 'y_true'), _finite(p, 'y_pred')
        if c not in (0, 1) or isinstance(c, bool):
            raise ValueError(f"cliff_mol must be 0 or 1, got {c!r}")
        squared.append((t - p) ** 2)
        if c == 1:
            squared_cliff.append((t - p) ** 2)
    rmse = math.sqrt(math.fsum(squared) / len(squared))
    rmse_cliff = math.sqrt(math.fsum(squared_cliff) / len(squared_cliff)) if squared_cliff else None
    return dict(n_test=len(squared), n_cliff=len(squared_cliff), rmse=rmse, rmse_cliff=rmse_cliff)


def read_predictions(path):
    """ Read predictions.csv with exact columns; returns typed rows. Raises ValueError on any malformed value """
    with open(path, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise ValueError(f"Unexpected prediction columns in {path}: {reader.fieldnames}")
        rows = []
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f"Ragged row in {path}")
            row = dict(raw)
            row['csv_row'] = int(raw['csv_row'])
            row['fold'] = int(raw['fold'])
            row['cliff_mol'] = int(raw['cliff_mol'])
            row['y_true'] = _finite(raw['y_true'], 'y_true')
            row['y_pred'] = _finite(raw['y_pred'], 'y_pred')
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty predictions: {path}")
    return rows
