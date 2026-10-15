"""Bounded inference diagnostics on a frozen test model; never changes the benchmark protocol.

The caller supplies graphs selected only from the published training split and their ordered identities.
Reports describe numerical variation; no CPU/GPU accuracy threshold is imposed here.
"""
import copy
import io
import traceback
from pathlib import Path

from MoleculeACE.benchmark.artifacts import atomic_write_bytes, atomic_write_json, sha256_file


def diagnose_repeatability(model, graphs, output_dir, metadata, repeats=20):
    """Save the frozen model, probe inputs and repeated float32 predictions.

    metadata requires best_epoch and probe_identities (one identity per graph). The caller supplies provenance,
    including versions and source hashes. output_dir must not already exist. Exceptions leave a partial report
    and propagate; TF32 flags, CPU thread count and the CPU RNG state are restored even after failure.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 2 <= repeats <= 20:
        raise ValueError('Diagnostic repeats must be an integer from 2 to 20')
    if not 1 <= len(graphs) <= 40:
        raise ValueError('Diagnostic probes must contain between 1 and 40 graphs')
    if (isinstance(metadata.get('best_epoch'), bool) or not isinstance(metadata.get('best_epoch'), int)
            or metadata['best_epoch'] < 1):
        raise ValueError('metadata.best_epoch must identify the restored checkpoint epoch')
    identities = metadata.get('probe_identities')
    if not isinstance(identities, list) or len(identities) != len(graphs):
        raise ValueError('metadata.probe_identities must align with the probe graphs')

    if any(not isinstance(identity, dict) or identity.get('split') != 'train' for identity in identities):
        raise ValueError('Every probe identity must refer to the published training split')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / 'report.json'
    original_flags = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                          cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    original_threads = torch.get_num_threads()
    # DataLoader iterators consume CPU RNG state even with shuffle=False. Preserve the caller's state.
    original_cpu_rng = torch.get_rng_state().clone()
    report = dict(status='running', scientific=False, purpose='frozen-model inference diagnostics',
                  metadata=metadata, probe_identities=identities, n_graphs=len(graphs), batch_size=32,
                  requested_gpu_repeats=repeats, requested_cpu_repeats=3,
                  original_tf32=original_flags, original_cpu_threads=original_threads,
                  artifacts={}, conditions={})

    def save_report():
        atomic_write_json(report_path, report)

    def flags():
        return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)

    def snapshot_state(module):
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def changed_state(module, expected):
        actual = module.state_dict()
        changed = sorted(set(actual) ^ set(expected))
        for key in sorted(set(actual) & set(expected)):
            value = actual[key].detach().cpu()
            reference = expected[key]
            if (value.dtype != reference.dtype or value.shape != reference.shape
                    or not torch.equal(value, reference)):
                changed.append(key)
        return changed

    def graph_changes(probes, expected_devices):
        changed = []
        if len(probes) != len(graph_snapshot):
            changed.append(dict(expected_graph_count=len(graph_snapshot), actual_graph_count=len(probes)))
        for index, (graph, expected) in enumerate(zip(probes, graph_snapshot)):
            for key in ('x', 'edge_index', 'edge_attr'):
                value = getattr(graph, key, None)
                reference = expected[key]
                if (not isinstance(value, torch.Tensor) or value.dtype != reference.dtype
                        or value.shape != reference.shape or str(value.device) != expected_devices[index][key]
                        or not torch.equal(value.detach().cpu(), reference)):
                    changed.append(dict(probe_index=index, tensor=key))
        return changed

    def invariants(module, probes, expected_devices):
        changed = changed_state(module, state_snapshot)
        source_changed = changed_state(model.model, state_snapshot)
        inputs_changed = graph_changes(probes, expected_devices)
        source_inputs_changed = graph_changes(graphs, graph_devices)
        training_modules = [name or '<root>' for name, child in module.named_modules() if child.training]
        return dict(state_unchanged=not changed, changed_state_keys=changed,
                    source_gpu_state_unchanged=not source_changed, changed_source_gpu_state_keys=source_changed,
                    graph_inputs_unchanged=not inputs_changed, changed_graph_inputs=inputs_changed,
                    source_graph_inputs_unchanged=not source_inputs_changed,
                    changed_source_graph_inputs=source_inputs_changed,
                    all_modules_eval=not training_modules, training_modules=training_modules)

    def require_invariants(record, condition):
        if not all(record[key] for key in ('state_unchanged', 'source_gpu_state_unchanged',
                                           'graph_inputs_unchanged', 'source_graph_inputs_unchanged',
                                           'all_modules_eval')):
            raise AssertionError('Diagnostic invariants failed in ' + condition + ': ' + repr(record))

    def condition(name, module, predict, count, probes, expected_devices):
        item = dict(status='running', repeats_requested=count, tf32=flags(),
                    cpu_threads=torch.get_num_threads(), before=invariants(module, probes, expected_devices),
                    predictions=[], prediction_checks=[])
        report['conditions'][name] = item
        save_report()
        print(f'Repeatability: {name}, {count} predictions; TF32={item["tf32"]}', flush=True)
        try:
            require_invariants(item['before'], name + ' before inference')
            for iteration in range(count):
                predictions = predict()
                is_tensor = isinstance(predictions, torch.Tensor)
                check = dict(iteration=iteration, is_tensor=is_tensor)
                item['prediction_checks'].append(check)
                if not is_tensor:
                    raise AssertionError(name + ': prediction output is not a tensor')
                values = predictions.detach().cpu()
                check.update(shape=list(values.shape), dtype=str(values.dtype),
                             finite=bool(torch.isfinite(values).all()),
                             all_modules_eval=all(not child.training for child in module.modules()))
                if (tuple(values.shape) != (len(graphs),) or values.dtype != torch.float32
                        or not check['finite'] or not check['all_modules_eval']):
                    # Preserve invalid values exactly in a tensor artifact; strict JSON cannot contain NaN/Inf.
                    buffer = io.BytesIO()
                    torch.save(values, buffer)
                    invalid_path = output_dir / (name + '_invalid_predictions.pt')
                    atomic_write_bytes(invalid_path, buffer.getvalue())
                    report['artifacts'][invalid_path.name] = sha256_file(invalid_path)
                    raise AssertionError(name + ': invalid prediction output: ' + repr(check))
                item['predictions'].append(values.tolist())

            per_graph = list(zip(*item['predictions']))
            item['per_graph_min'] = [min(values) for values in per_graph]
            item['per_graph_max'] = [max(values) for values in per_graph]
            item['per_graph_span'] = [high - low for low, high in
                                      zip(item['per_graph_min'], item['per_graph_max'])]
            item['max_pairwise_span'] = max(item['per_graph_span'])
            item['per_graph_mean'] = [sum(values) / len(values) for values in per_graph]
            item['after'] = invariants(module, probes, expected_devices)
            require_invariants(item['after'], name + ' after inference')
            item['status'] = 'completed'
            print(f'Repeatability: {name} completed; max_pairwise_span={item["max_pairwise_span"]:.9g}; '
                  'weights/inputs unchanged, evaluation mode verified', flush=True)
        except BaseException as exc:
            item['status'] = 'failed'
            item['error'] = repr(exc)
            item['traceback'] = traceback.format_exc()
            item['after'] = invariants(module, probes, expected_devices)
            raise
        finally:
            save_report()

    save_report()
    try:
        try:
            state_snapshot = snapshot_state(model.model)
            graph_snapshot, graph_devices = [], []
            for graph in graphs:
                tensors, devices = {}, {}
                for key in ('x', 'edge_index', 'edge_attr'):
                    value = getattr(graph, key, None)
                    if not isinstance(value, torch.Tensor):
                        raise ValueError('Every probe graph must provide tensor ' + key)
                    tensors[key] = value.detach().cpu().clone()
                    devices[key] = str(value.device)
                graph_snapshot.append(tensors)
                graph_devices.append(devices)

            checkpoint_path = output_dir / 'model_checkpoint.pt'
            model.save_checkpoint(str(checkpoint_path), metadata['best_epoch'])
            report['artifacts'][checkpoint_path.name] = sha256_file(checkpoint_path)
            probe_path = output_dir / 'probe_graphs.pt'
            buffer = io.BytesIO()
            torch.save(dict(probe_identities=identities, graphs=graph_snapshot), buffer)
            atomic_write_bytes(probe_path, buffer.getvalue())
            report['artifacts'][probe_path.name] = sha256_file(probe_path)
            save_report()

            condition('gpu_original', model.model, lambda: model.predict(graphs, batch_size=32), repeats,
                      graphs, graph_devices)
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                condition('gpu_tf32_disabled', model.model, lambda: model.predict(graphs, batch_size=32), repeats,
                          graphs, graph_devices)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = original_flags['matmul_allow_tf32']
                torch.backends.cudnn.allow_tf32 = original_flags['cudnn_allow_tf32']

            cpu_model = copy.deepcopy(model.model).cpu().eval()
            cpu_graphs = [Data(**{key: value.clone() for key, value in tensors.items()})
                          for tensors in graph_snapshot]
            cpu_generator = torch.Generator()

            def predict_cpu():
                loader = DataLoader(cpu_graphs, batch_size=32, shuffle=False, generator=cpu_generator)
                predictions = []
                with torch.no_grad():
                    for batch in loader:
                        values = cpu_model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                        predictions.append(values.reshape(-1).detach().cpu())
                return torch.cat(predictions)

            try:
                torch.set_num_threads(1)
                cpu_devices = [{key: 'cpu' for key in tensors} for tensors in graph_snapshot]
                condition('cpu_float32', cpu_model, predict_cpu, 3, cpu_graphs, cpu_devices)
            finally:
                torch.set_num_threads(original_threads)

            means = {name: item['per_graph_mean'] for name, item in report['conditions'].items()}
            comparisons = {}
            for left, right in (('gpu_original', 'cpu_float32'), ('gpu_tf32_disabled', 'cpu_float32'),
                                ('gpu_original', 'gpu_tf32_disabled')):
                difference = [a - b for a, b in zip(means[left], means[right])]
                comparisons[left + '_minus_' + right] = dict(per_graph_mean_difference=difference,
                                                            max_abs_mean_difference=max(map(abs, difference)))
            report['comparisons'] = comparisons
            report['comparison_note'] = 'Descriptive differences only; CPU is not treated as ground truth.'
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_flags['matmul_allow_tf32']
            torch.backends.cudnn.allow_tf32 = original_flags['cudnn_allow_tf32']
            torch.set_num_threads(original_threads)
            torch.set_rng_state(original_cpu_rng)
            report['restored_tf32'] = flags()
            report['restored_cpu_threads'] = torch.get_num_threads()
            report['cpu_rng_restored'] = bool(torch.equal(torch.get_rng_state(), original_cpu_rng))
            save_report()
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        save_report()
        raise
    report['status'] = 'completed'
    save_report()
    return report
