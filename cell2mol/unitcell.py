#!/usr/bin/env python

import os
import logging
import argparse
from ase.io import read
from cell2mol.refcell import process_refcell, get_error_case_message
from cell2mol.final_c2m_module import cell2mol_mode
from cell2mol.read_write import (
    write_refmoleclist,
    write_unique_species,
    get_cell_parameters,
)
import copy

logger = logging.getLogger("cell2mol")

# Constants
VERSION = "2.0"
COV_FACTOR = 1.0
METAL_FACTOR = 1.0


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def process_unitcell(input_path, name, current_dir, cif_bond_info):
    """
    Process the molecules from a CIF file and generate a unit cell object.
    Args:
        input_path (str): Path to the CIF file (downloaded from CSD).
        name (str): CSD refcode.
        current_dir (str): Current working directory.
        cif_bond_info (bool): Whether to use bond information reported in the CIF (_geom_bond block)
    Returns:
        newcell (object): Unit cell object containing the molecules and their information.
    """

    # Set up file paths
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    cell_fname = os.path.join(current_dir, f"Cell_{name}.cell")

    structure = read(input_path)
    (
        cell_labels,
        cell_pos,
        cell_fracs,
        cell_vector,
        cell_param,
        sym_ops,
    ) = get_cell_parameters(structure)

    # Process reference cell
    logger.info("Starting the cell2mol process for reference (Wyckoff sites)")
    cells = process_refcell(input_path, name, current_dir, cif_bond_info)
    refcell = cells.reference

    if refcell.error_case != 0:
        logging.error("Error encountered while processing the reference cell")
        return cells
    if (
        refcell.disagree_with_cif_formula is not None
        and refcell.disagree_with_cif_formula
    ):
        logging.info(
            "Discrepancies found between refcell and CIF. This will cause errors in the charge prediction!"
        )

    unitcell = cells.unitcell
    unitcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
    unitcell.has_isolated_H = refcell.has_isolated_H
    unitcell.has_missing_H = refcell.has_missing_H
    unitcell.error_get_poscharges = refcell.error_get_poscharges
    logging.info("Starting the cell2mol process for the unit cell")

    debug = 1
    if refcell.error_case == 0:
        # Step-by-step molecule reconstruction and error assessment
        mode = "reconstruction"
        cell2mol_mode(unitcell, refcell, sym_ops, mode, debug)
        unitcell.assess_errors(mode=mode)

        if unitcell.error_case == 0:
            mode = "charge_assignment"
            cell2mol_mode(unitcell, refcell, sym_ops, mode, debug)
            unitcell.assess_errors(mode=mode)
    else:
        logger.info(
            f"Error occurred in processing refcell: error case {refcell.error_case}"
        )

    refcell.save(ref_cell_fname)
    unitcell.save(cell_fname)
    cells.reference = refcell
    cells.unitcell = unitcell

    cells.save(os.path.join(current_dir, f"Cells_{name}.json"), format="json")
    cells.save(os.path.join(current_dir, f"Cells_{name}.cell"), format="pickle")

    # Summary
    summary_fname = os.path.join(current_dir, "reference_summary.out")
    with open(summary_fname, "w") as f:
        print(name, file=f)
        write_refmoleclist(refcell, file=f)
        write_unique_species(refcell, file=f)

        if refcell.refmoleclist == []:
            refcell.error_case = "X"
        elif (
            refcell.disagree_with_cif_formula is not None
            and refcell.disagree_with_cif_formula
        ):
            refcell.error_case = 9
        error = get_error_case_message(refcell.error_case)
        print(f"Error case: {refcell.error_case} - {error}", file=f)

    return cells


def _main():
    parser = argparse.ArgumentParser(
        description="Process reference object from a CIF file"
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="filepath",
        type=str,
        required=True,
        help="Path to the input file (.cif)",
    )
    parser.add_argument(
        "--cif-bond-info",
        action="store_true",
        help="Use CIF _geom_bond information to generate adjacency matrix",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    ext = os.path.splitext(args.filepath)[1].lower()

    # Validation
    if ext != ".cif":
        parser.error("Invalid input file format. Only .cif files are supported.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )

    input_path = os.path.normpath(args.filepath)
    name = os.path.splitext(os.path.basename(input_path))[0]

    process_unitcell(
        input_path=input_path,
        name=name,
        current_dir=os.getcwd(),
        cif_bond_info=args.cif_bond_info,
    )


if __name__ == "__main__":
    _main()
