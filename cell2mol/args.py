#!/usr/bin/env python

import argparse
import logging
import os
import numpy as np


def parsing_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (argparse.Namespace)
    """

    parser = argparse.ArgumentParser(
        prog="cell2mol",
        description=(
            "Interprets a crystallography file and extracts molecular information"
        ),
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="filepath",
        type=str,
        required=True,
        help="Path to the input file (.cif or .xyz)",
    )

    parser.add_argument(
        "-t",
        "--type",
        dest="system_type",
        type=str,
        choices=["reference", "unitcell", "molecule"],
        required=True,
        help="Type of information in the input file",
    )

    parser.add_argument(
        "--cell-para",
        dest="cell_para",
        type=float,
        nargs=6,
        help="Cell parameters (a, b, c, alpha, beta, gamma) for unit cell in .xyz file",
    )

    parser.add_argument(
        "--charge",
        dest="charge",
        type=int,
        help="Total charge of a molecule in .xyz file",
    )

    parser.add_argument(
        "--cif-bond-info",
        dest="cif_bond_info",
        action="store_true",
        help="Use CIF _geom_bond information to generate adjacency matrix",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )

    args = parser.parse_args()

    # ----------------------------
    # Validation logic
    # ----------------------------
    ext = os.path.splitext(args.filepath)[1].lower()

    if ext not in (".cif", ".xyz"):
        parser.error(
            "Invalid input file format. Only .cif and .xyz files are supported."
        )

    cell_para = None
    if args.filepath.endswith(".xyz") and args.system_type == "unitcell":
        if args.cell_para is None:
            parser.error("Cell parameters must be provided for unit cell in .xyz file")
        cell_para = np.array(args.cell_para)

    args.cell_para = cell_para

    # ----------------------------
    # Logging configuration
    # ----------------------------
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )

    return args


if __name__ == "__main__":
    args = parsing_arguments()
