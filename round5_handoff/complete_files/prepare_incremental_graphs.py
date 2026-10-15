import argparse
import json
from genrec.temporal_graphs import prepare_graphs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build stage-specific graphs with the existing gensid builder')
    parser.add_argument('--graph-inputs-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_graphs(args.graph_inputs_dir, args.output), indent=2))
