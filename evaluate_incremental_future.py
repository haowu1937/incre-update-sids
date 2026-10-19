"""Evaluate Base/Plain tokenizer snapshots before and after Update TIGER finetuning."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

from genrec.future_evaluation import parser, run

if __name__ == '__main__':
    import json
    report = run(parser().parse_args())
    compact = {key: report[key] for key in ('execution_status', 'config', 'pair_checks', 'future',
        'plain_minus_base', 'catalog_policy', 'evaluation_target_policy', 'future_used_for_selection',
        'selection_locked_before_future', 'original_checkpoints_unchanged', 'runtime_seconds',
        'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
    compact['training'] = {arm: {key: value[key] for key in ('update_optimizer_steps',
        'processed_examples', 'best_update_optimizer_step', 'best_update_validation', 'full_update_validation',
        'exact_base_warm_start', 'selected_checkpoint_reload_equal', 'runtime_seconds')}
        for arm, value in report['training'].items()}
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
