"""Prepare whitelisted metadata, then fixed features for pure-content SID runs."""
import argparse
import json
from genrec.temporal_content import prepare_content, encode_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    meta = sub.add_parser('metadata')
    meta.add_argument('--metadata', required=True)
    meta.add_argument('--protocol-dir', required=True)
    meta.add_argument('--output', required=True)
    meta.add_argument('--missing-policy', choices=['error', 'placeholder'], default='error')
    emb = sub.add_parser('encode')
    emb.add_argument('--content-dir', required=True)
    emb.add_argument('--output', required=True)
    emb.add_argument('--model', default='sentence-transformers/sentence-t5-base')
    emb.add_argument('--model-revision', default=None)
    emb.add_argument('--device', default='cpu')
    emb.add_argument('--batch-size', type=int, default=32)
    emb.add_argument('--max-length', type=int, default=256)
    emb.add_argument('--expected-dim', type=int, default=768)
    emb.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'metadata':
        result = prepare_content(args.metadata, args.protocol_dir, args.output, args.missing_policy)
    else:
        result = encode_content(args.content_dir, args.output, args.model, args.model_revision,
            args.device, args.batch_size, args.max_length, args.expected_dim, args.local_files_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
