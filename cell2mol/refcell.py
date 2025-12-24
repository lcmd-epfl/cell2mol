#!/usr/bin/env python

import os
import time
import logging
import argparse

from ase.io import read

from cell2mol.classes import Cell, Cells
from cell2mol.operations import frac2cart_fromparam
from cell2mol.elementdata import ElementData
from cell2mol.read_write import (
    extract_refmoleclist_xyz,
    get_wyckoff_positions,
    get_geom_bond,
    remove_disorder_atoms,
    get_cell_parameters,
    compare_reference_with_cif,
    write_refmoleclist,
    write_unique_species,
)

logger = logging.getLogger("cell2mol")

# Constants
VERSION = "2.0"
COV_FACTOR = 1.0
METAL_FACTOR = 1.0

elemdatabase = ElementData()


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def process_refcell(input_path, name, current_dir, cif_bond_info):
    """
    Process the reference molecules from a CIF file and generate a reference cell object.
    Args:
        input_path (str): Path to the CIF file (downloaded from CSD).
        name (str): CSD refcode.
        current_dir (str): Current working directory.
        cif_bond_info (bool): Whether to use bond information reported in the CIF (_geom_bond block)
    Returns:
        refcell (object): Reference cell object containing the reference molecules and their information.
    """

    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")

    logger.info("cell2mol version %s", VERSION)
    logger.info("Input CIF: %s", input_path)
    logger.info("Use CIF bond information: %s", cif_bond_info)

    structure = read(input_path)
    (
        cell_labels,
        cell_pos,
        cell_fracs,
        cell_vector,
        cell_param,
        sym_ops,
    ) = get_cell_parameters(structure)

    print(f"{sym_ops=}\n{type(sym_ops)=}")

    # Unit cell
    unitcell = Cell.from_positional(
        name, cell_labels, cell_pos, cell_fracs, cell_vector, cell_param
    )
    unitcell.set_subtype("unitcell")

    # Reference cell
    refcell = create_reference(
        input_path,
        name,
        cell_vector,
        cell_param,
        cif_bond_info,
    )

    extract_refmoleclist_xyz(current_dir, refcell.refmoleclist, name)

    if refcell.error_case == 0:
        refcell.get_unique_species()
        logger.info(
            "Unique species: %s",
            [s.formula for s in refcell.unique_species],
        )
        logger.info(
            "Species list: %s",
            [s.formula for s in refcell.species_list],
        )
    else:
        logger.error(
            "Error occurred while processing reference cell: error case %d",
            refcell.error_case,
        )

    refcell.save(ref_cell_fname)

    cells = Cells(
        name=name,
        reference=refcell,
        unitcell=unitcell,
        cell_vector=cell_vector,
        cell_param=cell_param,
    )
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

    # return refcell
    return cells


def create_reference(input_path, name, cell_vector, cell_param, cif_bond_info):
    """
    Create the Reference cell object.
    Args:
        input_path (str): Path to the CIF file.
        name (str): CSD refcode.
        cell_vector (list): Cell vectors.
        cell_param (list): Cell parameters.
        cif_bond_info (bool): Whether to use bond information reported in the CIF (_geom_bond block)
    Returns:
        refcell (object): Reference cell object.
    """

    start_time = time.time()

    atom_site_labels, ref_labels, ref_fracs = get_wyckoff_positions(input_path)
    atom_site_labels, ref_labels, ref_fracs = remove_disorder_atoms(
        atom_site_labels,
        ref_labels,
        ref_fracs,
    )

    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)

    refcell = Cell.from_positional(
        name,
        ref_labels,
        ref_pos,
        ref_fracs,
        cell_vector,
        cell_param,
    )
    refcell.set_atom_site_labels(atom_site_labels)
    refcell.set_subtype("reference")

    geom_bond_cif, moiety_list_cif = get_geom_bond(input_path)
    refcell.get_cif_bond_moiety(
        cif_bond_info,
        geom_bond_cif,
        moiety_list_cif,
    )

    logger.debug(
        "exist_cif_bond_moiety: %s",
        refcell.exist_cif_bond_moiety,
    )

    if cif_bond_info:
        refcell.get_reference_molecules_from_moiety(
            ref_labels,
            ref_fracs,
            cov_factor=COV_FACTOR,
            metal_factor=METAL_FACTOR,
        )
    else:
        refcell.get_reference_molecules(
            ref_labels,
            ref_fracs,
            cov_factor=COV_FACTOR,
            metal_factor=METAL_FACTOR,
        )

    if not refcell.refmoleclist:
        logger.warning("No reference molecules found in the CIF file")
        return refcell

    compare_reference_with_cif(input_path, refcell)
    refcell.check_missing_H()
    refcell.assess_errors(mode="hydrogens")

    elapsed = time.time() - start_time
    logger.debug(
        "Reference molecules generated in %.2f seconds",
        elapsed,
    )

    return refcell


def get_error_case_message(error_case):
    """
    Return an error message for a given error case.
    """
    if error_case == 0:
        return "No errors found"
    elif error_case in (2, 3, 4):
        if error_case == 2:
            return "Missing Hydrogens in Water Molecules"
        elif error_case == 3:
            return "Missing Hydrogens in Coordinated Water Molecules"
        elif error_case == 4:
            return "Missing Hydrogens in Carbon Atoms"

    elif error_case == 9:
        return (
            "Missing elements in Reference Molecules compared to "
            "moieties reported in CIF"
        )

    elif error_case == "X":
        return "Empty Reference Molecules list"

    else:
        return f"Unhandled error case: {error_case}"


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

    process_refcell(
        input_path=input_path,
        name=name,
        current_dir=os.getcwd(),
        cif_bond_info=args.cif_bond_info,
    )


if __name__ == "__main__":
    _main()
