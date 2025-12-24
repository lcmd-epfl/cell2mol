#!/usr/bin/env python

import os
import logging

from cell2mol.args import parsing_arguments
from cell2mol.refcell import process_refcell
from cell2mol.unitcell import process_unitcell
from cell2mol.molecule import process_molecule
from cell2mol.read_write import (
    prefilter_cif,
    exit_with_error_input,
    exit_with_error_exception,
)

logger = logging.getLogger("cell2mol")
VERSION = "2.0"


def main():
    args = parsing_arguments()

    current_dir = os.getcwd()
    input_path = os.path.normpath(args.filepath)
    name, extension = os.path.splitext(os.path.basename(input_path))

    if not os.path.exists(input_path):
        exit_with_error_input(f"Input file not found: {input_path}")

    if extension == ".cif":
        handle_cif_file(
            input_path=input_path,
            system_type=args.system_type,
            name=name,
            current_dir=current_dir,
            cif_bond_info=args.cif_bond_info,
        )

    elif extension == ".xyz":
        handle_xyz_file(
            input_path=input_path,
            system_type=args.system_type,
            name=name,
            input_charge=args.charge,
            current_dir=current_dir,
        )

    else:
        exit_with_error_input(f"Invalid file extension: {input_path}")


def handle_cif_file(
    input_path,
    system_type,
    name,
    current_dir,
    cif_bond_info,
):
    logger.info("Processing CIF file: %s", input_path)

    cif_okay, error_message = prefilter_cif(input_path)
    if not cif_okay:
        exit_with_error_input(
            f"CIF file is not suitable for processing: {error_message}"
        )

    try:
        if system_type == "reference":
            logger.info("Processing reference (Wyckoff sites)")
            process_refcell(
                input_path=input_path,
                name=name,
                current_dir=current_dir,
                cif_bond_info=cif_bond_info,
            )

        elif system_type == "unitcell":
            logger.info("Processing unit cell")
            process_unitcell(
                input_path=input_path,
                name=name,
                current_dir=current_dir,
                cif_bond_info=cif_bond_info,
            )

        else:
            exit_with_error_input(
                "Invalid system type for .cif file",
                {"system_type": system_type},
            )

    except Exception as exc:
        exit_with_error_exception(exc)


def handle_xyz_file(
    input_path,
    system_type,
    name,
    input_charge,
    current_dir,
):
    logger.info("Processing XYZ file: %s", input_path)
    try:
        if system_type == "molecule":
            process_molecule(
                input_path=input_path,
                name=name,
                input_charge=input_charge,
                current_dir=current_dir,
            )
        else:
            exit_with_error_input(
                "Invalid system type for .xyz file",
                {"system_type": system_type},
            )

    except Exception as exc:
        exit_with_error_exception(exc)


if __name__ == "__main__":
    main()
