"""Manual entry point for a historical-compatible content RQ-VAE Update."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
from gensid.historical_incremental_content import parser, run

if __name__ == '__main__':
    run(parser().parse_args())
