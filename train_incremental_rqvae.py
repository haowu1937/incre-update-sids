"""Train the Base content RQ-VAE without graph imports or future-item data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from gensid.incremental_content import parser, run

if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), indent=2), flush=True)
