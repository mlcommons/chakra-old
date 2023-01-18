#!/usr/bin/env python3

import argparse
import sys

from Text2ChakraConverter import *

def main():
    parser = argparse.ArgumentParser(
            description="Execution Graph Converter"
    )
    parser.add_argument(
            "--input_type",
            type=str,
            default=None,
            required=True,
            help="Input execution graph type"
    )
    parser.add_argument(
            "--input_filename",
            type=str,
            default=None,
            required=True,
            help="Input execution graph filename"
    )
    parser.add_argument(
            "--output_filename",
            type=str,
            default=None,
            required=True,
            help="Output Chakra execution graph filename"
    )
    parser.add_argument(
            "--num_npus",
            type=int,
            default=None,
            required="Text" in sys.argv,
            help="Number of NPUs in a system"
    )
    parser.add_argument(
            "--num_dims",
            type=int,
            default=None,
            required=True,
            help="Number of dimensions in the network topology"
    )
    parser.add_argument(
            "--num_passes",
            type=int,
            default=None,
            required="Text" in sys.argv,
            help="Number of training passes"
    )
    args = parser.parse_args()

    if args.input_type == "Text":
        converter = Text2ChakraConverter(
                args.input_filename,
                args.output_filename,
                args.num_npus,
                args.num_dims,
                args.num_passes)
        converter.convert()
    else:
        print("%s unsupported" % (args.input_type))

if __name__ == "__main__":
    main()
