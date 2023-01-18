#!/usr/bin/env python3

import argparse

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
    args = parser.parse_args()

    print("Unsupported")

if __name__ == "__main__":
    main()
