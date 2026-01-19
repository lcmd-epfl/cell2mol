#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.classes import Cell, Cells
from cell2mol.operations import frac2cart_fromparam
from cell2mol.read_cif import (
    get_cell_atoms,
    get_cell_parameters,
    get_wyckoff_positions,
    get_geom_bond,
    extract_info_from_cif,
    compare_cif_with_reference,
)
from cell2mol.write_results import (
    extract_refmoleclist_xyz,
    write_cell_molecules_info,
    write_unique_species,
    get_reference_error_message,
)

logger = logging.getLogger(__name__)


def process_reference(input_path, name, current_dir):
    """
    Process the reference molecules from a CIF file and generate a reference cell object.
    Args:
        input_path (str): Path to the CIF file (downloaded from CSD).
        name (str): CSD refcode.
        current_dir (str): Current working directory.
    Returns:
        cells (object): Cells object containing reference and unit cell information.
    """

    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    cells_json = os.path.join(current_dir, f"Cells_{name}.json")

    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input CIF: %s", input_path)
    logger.info("Use CIF bond information: %s", config.USE_BOND_INFO)

    refcell = None
    unitcell = None
    cells = None

    try:
        structure = read(input_path)

        cell_labels, cell_pos, cell_fracs = get_cell_atoms(structure)
        cell_vector, cell_param, sym_ops = get_cell_parameters(structure)

        # Reference cell
        refcell = create_reference(input_path, name, cell_vector, cell_param)

        if refcell.error_case == 0:
            refcell.get_unique_species()
        else:
            logger.error(
                "Error occurred while processing reference cell: error case %d",
                refcell.error_case,
            )

        # Unit cell (always constructed)
        unitcell = Cell.from_positional(
            name,
            cell_labels,
            cell_pos,
            cell_fracs,
            cell_vector,
            cell_param,
        )
        unitcell.set_subtype("unitcell")

        if refcell is not None:
            unitcell.has_isolated_H = refcell.has_isolated_H
            unitcell.has_missing_H = refcell.has_missing_H

        cells = Cells(
            name=name,
            reference=refcell,
            unitcell=unitcell,
            cell_vector=cell_vector,
            cell_param=cell_param,
        )

    except Exception as exc:
        logger.exception(
            "Unhandled exception while processing reference for %s: %s",
            name,
            exc,
        )

    finally:
        # Always save what exists
        if refcell is not None:
            try:
                refcell.save(ref_cell_fname)
                if logger.isEnabledFor(logging.DEBUG):
                    extract_refmoleclist_xyz(current_dir, refcell.refmoleclist, name)
            except Exception:
                logger.exception("Failed to save reference cell")

        if cells is not None:
            try:
                cells.save(cells_json, format="json")
            except Exception:
                logger.exception("Failed to save Cells JSON")

        # Summary (only if reference exists)
        if refcell is not None:
            try:
                summary_fname = os.path.join(current_dir, "reference_summary.out")
                with open(summary_fname, "w") as f:
                    print(name, file=f)
                    write_cell_molecules_info(refcell, file=f)
                    write_unique_species(refcell, file=f)
                    print(
                        get_reference_error_message(refcell.error_case),
                        file=f,
                    )
            except Exception:
                logger.exception("Failed to write reference summary")

    return cells


def create_reference(input_path, name, cell_vector, cell_param):
    """
    Create the Reference cell object.
    Args:
        input_path (str): Path to the CIF file.
        name (str): CSD refcode.
        cell_vector (list): Cell vectors.
        cell_param (list): Cell parameters.
    Returns:
        refcell (object): Reference cell object.
    """

    atom_site_labels, ref_labels, ref_fracs = get_wyckoff_positions(input_path)
    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)

    refcell = Cell.from_positional(
        name=name,
        labels=ref_labels,
        pos=ref_pos,
        frac_coord=ref_fracs,
        cell_vector=cell_vector,
        cell_param=cell_param,
    )
    refcell.set_atom_site_labels(atom_site_labels)
    refcell.set_subtype("reference")

    # Get CIF bond moiety information
    geom_bond_cif, moiety_list_cif = get_geom_bond(input_path)
    refcell.set_cif_bond_moiety(geom_bond_cif, moiety_list_cif)
    logger.info("CIF has bond moiety information: %s", refcell.exist_cif_bond_moiety)

    refcell.get_reference_molecules()

    if not refcell.refmoleclist:
        refcell.error_case = -1
        logger.warning("No reference molecules found in the CIF file")
        return refcell

    chemical_name, reported_metal_os, moiety_dicts = extract_info_from_cif(input_path)
    refcell.set_additional_cif_info(chemical_name, reported_metal_os, moiety_dicts)
    compare_cif_with_reference(refcell)
    refcell.check_hydrogens()
    refcell.assess_errors(mode="hydrogens")

    logger.info("Reference molecules generated")

    return refcell


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

    process_reference(
        input_path=input_path,
        name=name,
        current_dir=os.getcwd(),
    )


if __name__ == "__main__":
    _main()
