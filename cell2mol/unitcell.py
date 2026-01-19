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

    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    cell_fname = os.path.join(current_dir, f"Cell_{name}.cell")
    cells_json = os.path.join(current_dir, f"Cells_{name}.json")
    cells_pickle = os.path.join(current_dir, f"Cells_{name}.cell")

    cells = None
    refcell = None
    unitcell = None

    try:
        structure = read(input_path)
        _, _, sym_ops = get_cell_parameters(structure)

        logger.info("Starting the cell2mol process for reference (Wyckoff sites)")
        cells = process_reference(input_path, name, current_dir)

        refcell = cells.reference
        unitcell = cells.unitcell

        if refcell.error_case != 0:
            logger.error("Error while processing the reference cell")
            return cells

        logger.info("Starting the cell2mol process for the unit cell")
        unitcell = construct_unitcell(refcell, unitcell, sym_ops)
        unitcell.assess_errors(mode="reconstruction")

        if unitcell.error_case != 0:
            logger.error("Error while unit cell reconstruction")
            return cells

        refcell.get_selected_cs()
        refcell.assess_errors(mode="possible_charges")
        if refcell.error_case != 0:
            logger.error(
                "Error while retrieving possible charges for unique species in the reference cell."
            )
            return cells

        refcell, unitcell = balance_unitcell_charge(refcell, unitcell)
        unitcell.assess_errors(mode="balance_charges")

        if unitcell.error_case != 0:
            logger.error("Error while balancing charges in the unit cell")
            return cells

        refcell.assign_charges()
        refcell.assess_errors(mode="charge_assignment")
        if refcell.error_case != 0:
            logger.error("Error while assigning charges in the reference cell")
            return cells

        unitcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
        unitcell.unique_species = copy.deepcopy(refcell.unique_species)
        unitcell.assign_charges()
        unitcell.assess_errors(mode="charge_assignment")
        if unitcell.error_case != 0:
            logger.error("Error while assigning charges in the unit cell")
            return cells

        unitcell.check_charge_neutrality()

        refcell.assign_spin()
        unitcell.assign_spin()
        refcell.assess_errors(mode="spin_assignment")
        unitcell.assess_errors(mode="spin_assignment")

    except Exception as exc:
        logger.exception(f"Unhandled exception while processing {name}: {exc}")

    finally:
        # Always try to save what exists
        if refcell is not None:
            try:
                refcell.save(ref_cell_fname)
            except Exception:
                logger.exception("Failed to save reference cell")

        if unitcell is not None:
            try:
                unitcell.save(cell_fname)
            except Exception:
                logger.exception("Failed to save unit cell")

        if cells is not None:
            try:
                cells.reference = refcell
                cells.unitcell = unitcell
                cells.save(cells_json, format="json")
                cells.save(cells_pickle, format="pickle")
            except Exception:
                logger.exception("Failed to save cells object")

        # Reference summary
        if refcell is not None:
            try:
                summary_fname = os.path.join(current_dir, "reference_summary.out")
                with open(summary_fname, "w") as f:
                    print(name, file=f)
                    write_cell_molecules_info(refcell, file=f)
                    write_unique_species(refcell, file=f)
                    write_possible_charges(refcell, file=f)
                    print(
                        get_reference_error_message(refcell.error_case),
                        file=f,
                    )
            except Exception:
                logger.exception("Failed to write reference summary")

        # Unit cell summary
        if unitcell is not None:
            try:
                summary_unitcell_fname = os.path.join(
                    current_dir, "unitcell_summary.out"
                )
                with open(summary_unitcell_fname, "w") as f:
                    print(name, file=f)
                    write_cell_molecules_info(unitcell, file=f)
                    print(
                        get_unitcell_error_message(unitcell.error_case),
                        file=f,
                    )
            except Exception:
                logger.exception("Failed to write unit cell summary")

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
