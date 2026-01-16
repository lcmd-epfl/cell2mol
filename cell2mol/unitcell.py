#!/usr/bin/env python

import os
import logging
import copy
from ase.io import read
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.reference import process_reference
from cell2mol.construction import construct_unitcell
from cell2mol.charge.charge_balancer import balance_unitcell_charge
from cell2mol.read_cif import get_cell_parameters
from cell2mol.write_results import (
    write_cell_molecules_info,
    write_unique_species,
    write_possible_charges,
    get_reference_error_message,
    get_unitcell_error_message,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def interpret_unitcell(input_path, name, current_dir):
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
    cells = process_reference(input_path, name, current_dir)
    refcell = cells.reference
    unitcell = cells.unitcell

    if refcell.error_case != 0:
        logging.error("Error encountered while processing the reference cell")
        return cells

    # Process unit cell
    logger.info("Starting the cell2mol process for the unit cell")

    # Unit cell construction and error assessment
    unitcell = construct_unitcell(refcell, unitcell, sym_ops)
    unitcell.assess_errors(mode="reconstruction")
    logger.info(f"Unitcell error case: {unitcell.error_case}")

    # Charge assignment
    refcell.get_selected_cs()
    refcell, unitcell = balance_unitcell_charge(refcell, unitcell)
    refcell.assign_charges()

    unitcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
    unitcell.unique_species = copy.deepcopy(refcell.unique_species)

    unitcell.assign_charges()
    unitcell.check_charge_neutrality()
    unitcell.assess_errors(mode="charge_assignment")
    logger.info(f"Unitcell error case after charge assignment: {unitcell.error_case}")

    refcell.assign_spin()
    unitcell.assign_spin()

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
        write_cell_molecules_info(refcell, file=f)
        write_unique_species(refcell, file=f)
        write_possible_charges(refcell, file=f)
        print(get_reference_error_message(refcell.error_case), file=f)

    # Summary of unit cell
    summary_unitcell_fname = os.path.join(current_dir, "unitcell_summary.out")
    with open(summary_unitcell_fname, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(unitcell, file=f)
        print(get_unitcell_error_message(unitcell.error_case), file=f)

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

    interpret_unitcell(
        input_path=input_path,
        name=name,
        current_dir=os.getcwd(),
    )


if __name__ == "__main__":
    _main()
