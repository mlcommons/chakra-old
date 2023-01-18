#!/usr/bin/env python3

import argparse
import sys
sys.path.append("../eg_format/")
sys.path.append("../third_party/utils/")

from execution_graph_def_pb2 import *
from protolib import openFileRd, decodeMessage

def main():
    parser = argparse.ArgumentParser(
            description="Execution Graph Visualizer"
    )
    parser.add_argument(
            "--input_filename",
            type=str,
            default=None,
            required=True,
            help="Input Chakra execution graph filename"
    )
    parser.add_argument(
            "--output_filename",
            type=str,
            default=None,
            required=True,
            help="Output Graphviz graph filename"
    )
    args = parser.parse_args()

    chakra_eg = openFileRd(args.input_filename)
    node = ChakraNode()
    with open(args.output_filename, "w") as f:
        f.write("digraph taskgraph {\n")
        while decodeMessage(chakra_eg, node):
            label = "%s" % (node.name)
            f.write("  node%d [label=\"{%s}\", shape=\"record\"]\n"\
                    % (node.id, label))
            for dep in node.dep:
                f.write("  node%d->node%d\n" % (dep, node.id))
        f.write("}")
    chakra_eg.close()

if __name__ == "__main__":
    main()
