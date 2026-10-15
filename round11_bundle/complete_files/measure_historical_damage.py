"""Paired E1 loss measurement; all trained models remain frozen."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

from genrec.historical_damage import parser, run

if __name__ == '__main__':
    run(parser().parse_args())
