"""Prepare temporal artifacts from an existing Amazon Reviews 2014 gzip file."""
import argparse
import json
from genrec.temporal_artifacts import prepare_artifacts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', required=True, help='Existing reviews_Beauty_5.json.gz')
    parser.add_argument('--output', required=True, help='New incremental artifact directory')
    parser.add_argument('--max-history', type=int, default=50)
    parser.add_argument('--release', choices=['2014_beauty_5core', 'synthetic'],
                        default='2014_beauty_5core')
    parser.add_argument('--source-url', default=None, help='Declared download origin, if known')
    args = parser.parse_args()
    result = prepare_artifacts(args.reviews, args.output, args.max_history,
                               args.release, args.source_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
