#!/usr/bin/env python

import argparse


def parsing_arguments():
    """Parses the arguments of the command line.
    
    Returns:
        filename (str): filename of the input file
        step (int): step of the program to be executed
        verbose (bool): verbose flag
        quiet (bool): quiet flag
    """
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

    return args.filename, args.verbose, args.quiet


if __name__ == "__main__":
    parsing_arguments()
