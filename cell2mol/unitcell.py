#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.args import parsing_arguments
from cell2mol.refcell import process_refcell
from cell2mol.cell_reconstruction import reconstruct_unitcell

# from cell2mol.final_c2m_module import cell2mol_mode
from cell2mol.read_cif import get_cell_parameters
from cell2mol.write_results import (
    write_refmoleclist,
    write_unique_species,
    get_reference_error_message,
)
import copy
from cell2mol.utils import config

logger = logging.getLogger(__name__)

# Constants
VERSION = "2.0"
COV_FACTOR = 1.0
METAL_FACTOR = 1.0


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def process_unitcell(input_path, name, current_dir):
    """
    Process the molecules from a CIF file and generate a unit cell object.
    Args:
        input_path (str): Path to the CIF file (downloaded from CSD).
        name (str): CSD refcode.
        current_dir (str): Current working directory.
    Returns:
        cells (object): Cells object containing reference and unit cell information.
    """

    # Set up file paths
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    cell_fname = os.path.join(current_dir, f"Cell_{name}.cell")

    structure = read(input_path)
    cell_vector, cell_param, sym_ops = get_cell_parameters(structure)

    # Process reference cell
    logger.info("Starting the cell2mol process for reference (Wyckoff sites)")
    cells = process_refcell(input_path, name, current_dir)
    refcell = cells.reference

    if refcell.error_case != 0:
        logging.error("Error encountered while processing the reference cell")
        return cells
    if (
        refcell.disagree_with_cif_formula is not None
        and refcell.disagree_with_cif_formula
    ):
        logger.info("Discrepancies found between refcell and CIF.")
        logger.info("This will cause errors in the charge prediction!")

    unitcell = cells.unitcell
    unitcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
    unitcell.has_isolated_H = refcell.has_isolated_H
    unitcell.has_missing_H = refcell.has_missing_H
    unitcell.error_get_poscharges = refcell.error_get_poscharges
    logger.info("Starting the cell2mol process for the unit cell")

    if refcell.error_case == 0:
        # Step-by-step molecule reconstruction and error assessment
        unitcell = reconstruct_unitcell(refcell, unitcell, sym_ops)
        unitcell.assess_errors(mode="reconstruction")
        logger.info(f"Unitcell error case: {unitcell.error_case}")

        # if unitcell.error_case == 0:
        #     mode = "charge_assignment"
        #     cell2mol_mode(unitcell, refcell, sym_ops, mode, debug)
        #     unitcell.assess_errors(mode=mode)
    else:
        logger.info(
            f"Error occurred in processing refcell: error case {refcell.error_case}"
        )

    refcell.save(ref_cell_fname)
    unitcell.save(cell_fname)
    cells.reference = refcell
    cells.unitcell = unitcell

    cells.save(os.path.join(current_dir, f"Cells_{name}.json"), format="json")
    # cells.save(os.path.join(current_dir, f"Cells_{name}.cell"), format="pickle")

    # Summary
    summary_fname = os.path.join(current_dir, "reference_summary.out")
    with open(summary_fname, "w") as f:
        print(name, file=f)
        write_refmoleclist(refcell, file=f)
        write_unique_species(refcell, file=f)
        print(get_reference_error_message(refcell.error_case), file=f)

    return cells


def _main():
    args = parsing_arguments()
    if args.cif_bond_info:
        config.USE_BOND_INFO = True

    if args.print_config:
        print(config.dump())
        return

    input_path = os.path.normpath(args.filepath)
    name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(args.filepath)[1].lower()

    if ext != ".cif":
        raise ValueError("Invalid input file format. Only .cif files are supported.")

    process_unitcell(
        input_path=input_path,
        name=name,
        current_dir=os.getcwd(),
    )


if __name__ == "__main__":
    _main()
