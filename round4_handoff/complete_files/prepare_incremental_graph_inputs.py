"""Export verified training-window graph inputs, without building graphs."""
import argparse
import json
from genrec.temporal_graph_inputs import prepare_graph_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', required=True)
    parser.add_argument('--protocol-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_graph_inputs(args.reviews, args.protocol_dir, args.output),
                     ensure_ascii=False, indent=2))
