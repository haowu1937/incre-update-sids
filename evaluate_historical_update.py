"""Manual entry point for matched evaluation of a historical-compatible Update."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
from gensid.historical_incremental_content import evaluation_parser, evaluate_update

if __name__ == '__main__':
    evaluate_update(evaluation_parser().parse_args())
