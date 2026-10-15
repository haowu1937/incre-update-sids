"""Prepare frozen Base SID mappings and causal TIGER tensors."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from genrec.temporal_sids import export_base_sids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-dir',required=True)
    parser.add_argument('--features-dir',required=True)
    parser.add_argument('--rqvae-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--suffix-capacity',type=int,default=256)
    parser.add_argument('--expected-checkpoint-sha256')
    args = parser.parse_args()
    result = export_base_sids(args.protocol_dir,args.features_dir,args.rqvae_dir,args.output,
        args.device,args.suffix_capacity,args.expected_checkpoint_sha256)
    print(json.dumps(result,indent=2),flush=True)
