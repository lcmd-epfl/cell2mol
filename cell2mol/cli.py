#!/usr/bin/env python

import os
import sys
import logging
import warnings
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.process_reference import interpret_reference
from cell2mol.process_unitcell import interpret_unitcell
from cell2mol.process_xyz import interpret_molecule
from cell2mol.write_results import write_input_error_log
from cell2mol.utils.processing import run_cli
from rdkit import RDLogger

try:
    from cell2mol.utils.limits import set_memory_limit
except ImportError:
    # Fallback if the file doesn't exist yet, to prevent import errors
    def set_memory_limit(max_mem_gb):
        pass


RDLogger.DisableLog("rdApp.warning")  # pyright: ignore[reportAttributeAccessIssue]
RDLogger.DisableLog("rdApp.error")  # pyright: ignore[reportAttributeAccessIssue]
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

    # --- set memory limit ---
    try:
        set_memory_limit(config.MAX_MEM_GB)
        logger.info(f"System memory limit set to {config.MAX_MEM_GB} GB")
    except Exception as e:
        logger.warning(f"Could not set memory limit: {e}")

    # --- set global runtime config ---
    if args.cif_bond_info:
        config.USE_BOND_INFO = True

    if args.print_config:
        print(config.dump())
        return

    current_dir = os.getcwd()
    input_path = os.path.normpath(args.filepath)
    name, extension = os.path.splitext(os.path.basename(input_path))
    extension = extension.lower()

    if not os.path.exists(input_path):
        write_input_error_log(f"Input file not found: {input_path}")
        sys.exit(config.ERR_INPUT)

    if extension == ".cif":
        code = handle_cif_file(
            input_path=input_path,
            system_type=args.system_type,
            name=name,
            current_dir=current_dir,
        )

    elif extension == ".xyz":
        code = handle_xyz_file(
            input_path=input_path,
            system_type=args.system_type,
            name=name,
            input_charge=args.charge,
            current_dir=current_dir,
        )

    else:
        write_input_error_log(f"Invalid file extension: {input_path}")
        code = config.ERR_INPUT

    sys.exit(code)


def handle_cif_file(
    input_path,
    system_type,
    name,
    current_dir,
):
    logger.info("Processing CIF file: %s", input_path)
    if system_type == "reference":
        process = interpret_reference
    elif system_type == "unitcell":
        process = interpret_unitcell
    else:
        write_input_error_log(
            "Invalid system type for .cif file", {"system_type": system_type}
        )
        return config.ERR_INPUT
    return run_cli(process, input_path=input_path, name=name, current_dir=current_dir)


def handle_xyz_file(
    input_path,
    system_type,
    name,
    input_charge,
    current_dir,
):
    logger.info("Processing XYZ file: %s", input_path)
    if system_type != "molecule":
        write_input_error_log(
            "Invalid system type for .xyz file", {"system_type": system_type}
        )
        return config.ERR_INPUT
    return run_cli(
        interpret_molecule,
        input_path=input_path,
        name=name,
        input_charge=input_charge,
        current_dir=current_dir,
    )


if __name__ == "__main__":
    main()
