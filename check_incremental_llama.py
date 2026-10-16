"""Small, separate Llama inference check; never exports a training feature set.

Reuses the feature branch's input validation, model loading and pooling without
changing its source-bound encoder or touching any existing .incomplete run.
"""
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import prepare_incremental_llama as llama


def run(args, *, encoder_factory=llama.LocalLlamaEncoder):
    started = time.monotonic()
    if (args.resume or not 1 <= args.items <= 8
            or min(args.batch_size, args.max_length, args.expected_dim) < 1
            or not 0 <= args.seed < 2**32):
        raise ValueError('Use positive sizes and a new smoke output, without --resume')
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.with_name(output.name + '.incomplete').exists():
        raise FileExistsError('Smoke output exists; preserve it and choose a new directory')
    binding, registries, rows, texts = llama.checked_inputs(args)
    identity = llama.model_identity(args.model_dir, args.expected_dim)
    chosen = rows['update'][:args.items]
    if not chosen:
        raise ValueError('No eligible items for smoke inference')
    root = Path(__file__).resolve().parent
    config = dict(command=vars(args), binding=binding, encoder=identity,
        commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        source_files_sha256={name: llama.file_sha256(root / name) for name in (
            'prepare_incremental_llama.py', 'check_incremental_llama.py',
            'genrec/temporal.py', 'genrec/temporal_content.py')})
    output.mkdir(parents=True, exist_ok=False)
    llama.write_json(output / 'config.json', config)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if encoder_factory is llama.LocalLlamaEncoder:
            import torch
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        encoder = encoder_factory(args)
        batches = []
        for start in range(0, len(chosen), args.batch_size):
            subset = chosen[start:start+args.batch_size]
            batch = encoder.encode([texts[r['raw_item_id']] for r in subset])
            if (batch.shape != (len(subset), args.expected_dim) or batch.dtype != np.float32
                    or not np.isfinite(batch).all() or (np.linalg.norm(batch, axis=1) == 0).any()):
                raise ValueError('Invalid smoke embeddings')
            batches.append(batch)
        values = np.concatenate(batches)
        np.save(output / 'sample_embeddings.npy', values, allow_pickle=False)
        llama.write_json(output / 'sample_rows.json', chosen)
        report = dict(stage='llama_content_smoke_v1', execution_status='passed',
            encoded_items=len(chosen), shape=list(values.shape), dtype=str(values.dtype),
            norm_min=float(np.linalg.norm(values, axis=1).min()),
            norm_max=float(np.linalg.norm(values, axis=1).max()),
            expected_catalog_rows={s: len(rows[s]) for s in llama.STAGES},
            actual_encoder_runtime=encoder.info, **encoder.memory(),
            pooling='padding-zeroed sum / fixed max_length', max_length=args.max_length,
            full_catalog_encoded=len(chosen) == len(rows['update']), training_status='not_run',
            usable_as_training_features=False, runtime_seconds=time.monotonic()-started)
    except Exception as exc:
        report = dict(stage='llama_content_smoke_v1', execution_status='failed',
            error=str(exc), full_catalog_encoded=False, training_status='not_run',
            usable_as_training_features=False)
        llama.write_json(output / 'summary.json', report)
        raise
    report['artifact_files_sha256'] = {p.name: llama.file_sha256(p)
        for p in output.iterdir() if p.is_file()}
    llama.write_json(output / 'summary.json', report)
    compact = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (output / 'compact.txt').write_text(compact)
    print(compact, flush=True)
    return report


def parser():
    p = llama.parser()
    p.description = __doc__
    p.add_argument('--items', type=int, default=2, choices=range(1, 9), help='Small smoke sample only')
    p.add_argument('--seed', type=int, default=2026)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
