"""Report Base-only feature scale and optional RQ-VAE reconstruction baselines.

Does not encode items, normalize features, fit a model, or inspect held-out data.
The RQ-VAE MSE is taken from its saved report, not independently recomputed.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from genrec.temporal_content import file_sha256


def feature_scale(values):
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError('Expected a nonempty matrix')
    count, dim = values.shape
    sums = np.zeros(dim, dtype=np.float64)
    norms, squared_sum = [], 0.
    for start in range(0, count, 512):
        block = np.asarray(values[start:start+512], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite feature values')
        energy = np.einsum('ij,ij->i', block, block)
        sums += block.sum(0)
        squared_sum += float(energy.sum())
        norms.extend(np.sqrt(energy).tolist())
    mean = sums / count
    centered_sum = 0.
    for start in range(0, count, 512):
        centered = np.asarray(values[start:start+512], dtype=np.float64) - mean
        centered_sum += float(np.einsum('ij,ij->', centered, centered))
    return dict(items=count, dimension=dim, dtype=str(values.dtype),
        norm_percentiles={str(q): float(np.percentile(norms, q)) for q in (0, 25, 50, 75, 95, 99, 100)},
        zero_predictor_mse=squared_sum / (count*dim),
        mean_predictor_mse=centered_sum / (count*dim),
        baseline_scope='Base train catalog only; per-item per-dimension squared error')


def report(args):
    features = Path(args.features_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite report: ' + str(output))
    summary_path = features / 'summary.json'
    source = json.loads(summary_path.read_text())
    if source.get('stage') != 'E0_fixed_content_embeddings' or source.get('execution_status') != 'completed':
        raise ValueError('Expected completed full feature export')
    path = features / 'base_train/embeddings.npy'
    checksum = file_sha256(path)
    if source['artifact_files_sha256']['base_train/embeddings.npy'] != checksum:
        raise ValueError('Base feature checksum differs from its saved identity')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if list(values.shape) != source['stages']['base_train']['shape']:
        raise ValueError('Feature shape differs from its summary')
    stats = feature_scale(values)
    result = dict(execution_status='completed', features_dir=str(features),
        feature_summary_sha256=file_sha256(summary_path), base_embeddings_sha256=checksum,
        input=stats, rqvae=None, feature_files_modified=False, numpy_version=np.__version__)
    if args.rqvae_dir:
        trained_path = Path(args.rqvae_dir).resolve() / 'summary.json'
        trained = json.loads(trained_path.read_text())
        if (trained.get('stage') != 'E0_base_content_rqvae' or trained.get('execution_status') != 'completed'
                or Path(trained['features_dir']).resolve() != features
                or trained['binding']['feature_summary_sha256'] != result['feature_summary_sha256']
                or trained['feature_dim'] != stats['dimension'] or trained['training_items'] != stats['items']):
            raise ValueError('RQ-VAE report is not bound to these Base features')
        mse = trained['final_train_catalog_metrics']['reconstruction_loss']
        initial_mse = trained['initial_train_catalog_metrics']['reconstruction_loss']
        if not all(math.isfinite(v) and v >= 0 for v in (mse, initial_mse)):
            raise ValueError('Invalid reported reconstruction MSE')
        mean_mse = stats['mean_predictor_mse']
        result['rqvae'] = dict(directory=str(trained_path.parent), summary_sha256=file_sha256(trained_path),
            optimizer_steps=trained['optimizer_steps'], initial_reconstruction_mse=initial_mse,
            reconstruction_mse=mse,
            seed=trained.get('config', {}).get('seed'), runtime_seconds=trained.get('runtime_seconds'),
            relative_reduction_from_initial=1-mse/initial_mse if initial_mse > 0 else None,
            mse_over_mean_predictor=mse/mean_mse if mean_mse > 0 else None,
            relative_reduction_vs_mean_predictor=1-mse/mean_mse if mean_mse > 0 else None,
            collision_statistics=trained['collision_statistics'],
            checkpoint_reload_assignments_equal=trained['checkpoint_reload_assignments_equal'],
            mse_source='saved full Base train catalog evaluation; not independently recomputed here')
    root = Path(__file__).resolve().parent
    result['commit'] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    result['report_source_sha256'] = file_sha256(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (output / 'summary.json').write_text(text)
    (output / 'compact.txt').write_text(text)
    print(text, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir', required=True)
    p.add_argument('--rqvae-dir', help='Optional completed Base RQ-VAE run')
    p.add_argument('--output', required=True)
    report(p.parse_args())
