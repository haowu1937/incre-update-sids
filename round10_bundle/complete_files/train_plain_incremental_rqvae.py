"""Ordinary Update content RQ-VAE, initialized from the frozen Base checkpoint."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from gensid.plain_incremental_content import parser, run

if __name__ == '__main__':
    run(parser().parse_args())
