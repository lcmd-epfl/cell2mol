#!/usr/bin/env python

import argparse
import numpy as np

def parsing_arguments():
    """Parses the arguments of the command line.
    
    Returns:
        filename (str): filename of the input file
        step (int): step of the program to be executed
        verbose (bool): verbose flag
        quiet (bool): quiet flag
    """
    parser = argparse.ArgumentParser(
        prog="cell2mol", 
        description="Interprets the crystallography file (.cif) of a molecular crystal, and stores the information in a python cell object",
        add_help=True
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="filename",
        type=str,
        required=True,
        help="Filename of Input (.cif or .xyz file)",
    )

    parser.add_argument(
        "-t",
        "--type",
        dest="system_type",
        type=str,
        choices=["reference", "unitcell", "molecule"],
        required=True,
        help="Type of information in the input file ('reference', 'unitcell' or 'molecule')",
    )

    parser.add_argument(
        "--cell-para",
        dest="cell_para",
        type=float,
        nargs=6,
        help="Cell parameters (a, b, c, alpha, beta, gamma) for .xyz file",
    )
    parser.add_argument(
        "--charge",
        dest="charge",
        type=int,
        help="Total charge of a molecule in .xyz file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Extended output for debugging.",
        action="store_true",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        help="Suppress all screen output. Overrides --verbose flag.",
        action="store_true",
    )

    args = parser.parse_args()

    cell_para = None
    if args.filename.endswith(".xyz") and args.system_type == "unitcell":
        if args.cell_para is None:
            parser.error("Cell parameters must be provided for .xyz file of an unit cell")
        cell_para = np.array(args.cell_para)

    # if args.filename.endswith(".xyz") and args.system_type == "molecule":
    #     if args.charge is None:
    #         parser.error("Total charge must be provided for .xyz file of a molecule")

    debug_mode = determine_debug_level(args.verbose, args.quiet)
    return args.filename, args.system_type, cell_para, args.charge, debug_mode

def determine_debug_level(isverbose, isquiet):
    if isverbose and not isquiet:
        return 2
    elif isverbose and isquiet:
        return 0
    elif not isverbose and isquiet:
        return 0
    elif not isverbose and not isquiet:
        return 1


if __name__ == "__main__":
    parsing_arguments()
