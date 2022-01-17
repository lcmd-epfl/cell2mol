#!/usr/bin/env python

import numpy as np
import argparse

def parsing_arguments ():
    parser = argparse.ArgumentParser(
        prog="cell2mol",
        description="Generator cell object from an input file",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="filename",
        type=str,
        required=True,
        help="Filename of Input (.info or .cif file)",
    )

    parser.add_argument(
        "-s",
        "--step",
        dest="step",
        type=int,
        required=True,
        help="Generate cell by the step of cell reconstruction (1), only charge assignment (2), or cell reconstruction + charge assignment (3)",
    )

    args = parser.parse_args()

    return args.filename, args.step

if __name__ == '__main__':
    parsing_arguments ()



