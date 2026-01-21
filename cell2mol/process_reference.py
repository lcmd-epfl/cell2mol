#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.classes import Reference, Cells
from cell2mol.operations import (
    frac2cart_fromparam,
    is_polynuclear_over_limit,
    has_mixed_metal_types,
)
from cell2mol.read_cif import (
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
    get_reference_warning_messages,
)
from cell2mol.connectivity import is_mismatch_adjacency
from cell2mol.write_results import exit_with_error_exception

logger = logging.getLogger(__name__)


def interpret_reference(input_path, name, current_dir):
    """
    Process the reference molecules from a CIF file and generate a reference cell object.
    Args:
        input_path (str): Path to the CIF file (downloaded from CSD).
        name (str): CSD refcode.
        current_dir (str): Current working directory.
    Returns:
        cells (object): Cells object containing reference and unit cell information.
    """
    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input CIF: %s", input_path)

    cells = None
    refcell = None

    try:
        try:
            structure = read(input_path, format="cif")
        except (AssertionError, Exception) as e:
            logger.error(f"ASE failed to parse {input_path}: {e}")
            raise
        cell_vector, cell_param, _ = get_cell_parameters(structure)
        refcell = create_reference(input_path, name, cell_vector, cell_param)

        if refcell.has_error():
            logger.error("Error in reference cell: case %d", refcell.error_case)
            return None

        refcell.get_unique_species()

        cells = Cells(
            name=name,
            reference=refcell,
            unitcell=None,
            cell_vector=cell_vector,
            cell_param=cell_param,
        )

    except Exception as exc:
        exit_with_error_exception(exc)

    finally:
        _handle_reference_outputs(name, current_dir, cells, refcell)

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

    refcell = Reference.from_positional(
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

    # Extract additional CIF information
    chemical_name, reported_metal_os, moiety_dicts = extract_info_from_cif(input_path)
    refcell.set_additional_cif_info(chemical_name, reported_metal_os, moiety_dicts)

    # Generate reference molecules
    refcell.get_reference_molecules()

    if not refcell.refmoleclist:
        refcell.error_case = -1
        logger.warning("No reference molecules found in the CIF file")
        return refcell

    # Check for potential warnings
    cif_mismatch = compare_cif_with_reference(moiety_dicts, refcell.refmoleclist)
    over_polynuclear_limit = any(
        is_polynuclear_over_limit(ref.labels, max_metal_centers=config.MAX_METALS)
        for ref in refcell.refmoleclist
    )
    mixed_metals = any(
        has_mixed_metal_types(ref.labels) for ref in refcell.refmoleclist
    )
    is_mismatch_adj = is_mismatch_adjacency(
        ref_labels, ref_pos, atom_site_labels, geom_bond_cif
    )
    refcell.set_potential_warning(
        cif_mismatch, over_polynuclear_limit, mixed_metals, is_mismatch_adj
    )

    # Check missing hydrogens and assess errors
    refcell.check_hydrogens()
    refcell.assess_errors(mode="hydrogens")
    logger.info("Reference molecules generated")

    return refcell


def _handle_reference_outputs(name, current_dir, cells, refcell):
    """Manages saving files and writing summaries."""
    if refcell is None:
        return

    # 1. Write the .out summary file
    summary_path = os.path.join(current_dir, "reference_summary.out")
    _safe_run(
        lambda: _write_ref_detailed_summary(name, refcell, summary_path),
        "Failed to write reference summary",
    )

    # 2. Save the refcell object (.cell)
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")

    def save_ref():
        refcell.save(ref_cell_fname)
        # if logger.isEnabledFor(logging.DEBUG) and not refcell.error_case == -1:
        #     extract_refmoleclist_xyz(current_dir, refcell.refmoleclist, name)

    _safe_run(save_ref, "Failed to save reference cell")

    # # 3. Save the Cells container (.json)
    # if cells:
    #     cells_json = os.path.join(current_dir, f"Cells_{name}.json")
    #     _safe_run(
    #         lambda: cells.save(cells_json, format="json"), "Failed to save Cells JSON"
    #     )


def _write_ref_detailed_summary(name, refcell, summary_path):
    """Writes the molecules info, species, errors, and warnings to file and log."""
    error_message = get_reference_error_message(refcell.error_case)
    warnings = get_reference_warning_messages(refcell)

    # Write to File
    with open(summary_path, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(refcell, file=f)
        write_unique_species(refcell, file=f)
        print(f"ERROR: {error_message}", file=f)
        for msg in warnings:
            print(f"WARNING: {msg}", file=f)

    # Write to Logger
    logger.info("Reference Summary: %s", error_message)
    if not warnings:
        logger.info("No potential issues detected.")
    else:
        logger.warning("Potential issues detected:")
        for msg in warnings:
            logger.warning("  - %s", msg)


def _safe_run(func, error_msg):
    """Utility to wrap save operations in try-except."""
    try:
        func()
    except Exception:
        logger.exception(error_msg)


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

    interpret_reference(
        input_path=input_path,
        name=name,
        current_dir=os.getcwd(),
    )


if __name__ == "__main__":
    _main()
