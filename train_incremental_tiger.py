"""Fixed-step Base content-SID TIGER training with strict temporal evaluation."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

from genrec.temporal_tiger import parser, run

if __name__ == '__main__':
    run(parser().parse_args())
