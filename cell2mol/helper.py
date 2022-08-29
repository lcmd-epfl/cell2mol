#!/usr/bin/env python

import argparse


def parsing_arguments():
    parser = argparse.ArgumentParser(
        prog="cell2mol", description="Interprets the crystallography file (.cif) of a molecular crystal, and stores the information in a python cell object"
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
        help="Executes (1) only cell reconstruction, (2) only charge assignment, or (3) both cell reconstruction and charge assignment",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        #dest="verbose",
        help="Extended output for debugging.",
        action="store_true",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        #dest="quiet",
        help="Suppress all screen output. Overrides --verbose flag.",
        action="store_true",
    )

    args = parser.parse_args()

    return args.filename, args.step, args.verbose, args.quiet


if __name__ == "__main__":
    parsing_arguments()
