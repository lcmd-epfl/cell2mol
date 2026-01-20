#!/usr/bin/env python

import os
import logging
import warnings
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.process_reference import interpret_reference
from cell2mol.process_unitcell import interpret_unitcell
from cell2mol.process_xyz import interpret_molecule
from cell2mol.read_cif import prefilter_cif
from cell2mol.write_results import exit_with_error_input, exit_with_error_exception

warnings.filterwarnings(
    "ignore",
    message="crystal system .* is not interpreted for space group",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="scaled_positions .* are equivalent",
    category=UserWarning,
)

logger = logging.getLogger(__name__)


def main():
    args = parsing_arguments()

    # --- set global runtime config ---
    if args.cif_bond_info:
        config.USE_BOND_INFO = True

    if args.print_config:
        print(config.dump())
        return

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
):
    logger.info("Processing CIF file")

    cif_okay, error_message = prefilter_cif(input_path)
    if not cif_okay:
        exit_with_error_input(
            f"CIF file is not suitable for processing: {error_message}"
        )

    try:
        if system_type == "reference":
            logger.info("Processing reference (Wyckoff sites)")
            interpret_reference(
                input_path=input_path,
                name=name,
                current_dir=current_dir,
            )

        elif system_type == "unitcell":
            logger.info("Processing unit cell")
            interpret_unitcell(
                input_path=input_path,
                name=name,
                current_dir=current_dir,
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
            interpret_molecule(
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
