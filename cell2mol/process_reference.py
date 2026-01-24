#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.classes import Reference
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
    write_cell_molecules_info,
    write_unique_species,
    write_possible_charges,
    get_reference_error_message,
    get_reference_warning_messages,
)
from cell2mol.connectivity import is_mismatch_adjacency
from cell2mol.write_results import exit_with_error_exception
from cell2mol.utils.limits import ProcessingTimeoutError, set_time_limit
import sys
import gc

logger = logging.getLogger(__name__)

# Error Codes
ERR_MEMORY = config.ERR_MEMORY
ERR_TIMEOUT = config.ERR_TIMEOUT
ERR_GENERAL = config.ERR_GENERAL


def interpret_reference(input_path, name, current_dir):
    """
    Process the reference molecules from a CIF file.
    """
    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input CIF: %s", input_path)
    logger.debug("Timeout set to %d seconds", config.TIMEOUT)

    refcell = None
    exit_code = 0

    try:
        # Enforce timeout on the heavy lifting
        with set_time_limit(config.TIMEOUT):
            try:
                structure = read(input_path, format="cif")
            except (AssertionError, Exception) as e:
                logger.error(f"ASE failed to parse {input_path}: {e}")
                raise  # Re-raise to be caught by the outer Exception block

            cell_vector, cell_param, _ = get_cell_parameters(structure)
            refcell = create_reference(input_path, name, cell_vector, cell_param)

            # Check missing hydrogens and assess errors
            refcell.check_hydrogens()
            refcell.assess_errors(mode="hydrogens")
            if refcell.has_error():
                logger.error(
                    f"Error generating reference molecules (case={refcell.error_case})"
                )

            refcell.get_unique_species()

            # Save intermediate success
            _handle_reference_outputs(name, current_dir, refcell, mode="hydrogens")

            # --- Possible charge analysis ---
            refcell.get_selected_cs()
            refcell.assess_errors(mode="possible_charges")

            if refcell.has_error():
                logger.error(
                    f"Error retrieving possible charges (case={refcell.error_case})"
                )

    # 1. Handle Memory Errors First
    except MemoryError as exc:
        # CRITICAL: Delete large objects and force GC *before* doing anything else
        if "structure" in locals():
            del structure
        gc.collect()

        logger.error("Memory limit reached. RAM cleared.")

        if refcell is not None:
            refcell.error_case = ERR_MEMORY

        # Now it is safe(r) to call the exit helper
        exit_with_error_exception(exc)
        exit_code = ERR_MEMORY

    # 2. Handle Timeout Errors Second
    except ProcessingTimeoutError as exc:
        logger.error(f"Processing timed out after {config.TIMEOUT} seconds.")

        if refcell is not None:
            refcell.error_case = ERR_TIMEOUT

        exit_with_error_exception(exc)
        exit_code = ERR_TIMEOUT

    # 3. Handle All Other Errors
    except Exception as exc:
        logger.error(f"Unhandled error: {exc}")
        if refcell is not None:
            refcell.error_case = ERR_GENERAL

        exit_with_error_exception(exc)
        exit_code = ERR_GENERAL

    finally:
        logger.info("Executing final output handling...")
        # Ensure we try to save whatever valid data we have (refcell might be None)
        _handle_reference_outputs(name, current_dir, refcell, mode="possible_charges")

        if exit_code != 0:
            logger.info(f"Process exiting with code {exit_code}")
            sys.exit(exit_code)

    return refcell


def create_reference(input_path, name, cell_vector, cell_param):
    """
    Create the Reference Cell object.
    Args:
        input_path (str): Path to the CIF file.
        name (str): CSD refcode.
        cell_vector (np.ndarray): Cell vectors.
        cell_param (np.ndarray): Cell parameters.
    Returns:
        refcell (object): Reference Cell object.
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
    return refcell


def _handle_reference_outputs(name, current_dir, refcell, mode=None):
    """Manages saving files and writing summaries."""
    if refcell is None:
        return

    # 1. Write the .out summary file
    summary_path = os.path.join(current_dir, "reference_summary.out")
    _safe_run(
        lambda: _write_ref_detailed_summary(name, refcell, summary_path, mode=mode),
        "Failed to write reference summary",
    )

    # 2. Save the refcell object (.cell)
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")

    def save_ref():
        refcell.save(ref_cell_fname)
        # from cell2mol.write_results import extract_refmoleclist_xyz
        # if logger.isEnabledFor(logging.DEBUG) and not refcell.error_case == -1:
        #     extract_refmoleclist_xyz(current_dir, refcell.refmoleclist, name)

    _safe_run(save_ref, "Failed to save reference cell")


def _write_ref_detailed_summary(name, refcell, summary_path, mode=None):
    """Writes the molecules info, species, errors, and warnings to file and log."""
    error_message = get_reference_error_message(refcell.error_case)
    warnings = get_reference_warning_messages(refcell)

    # Write to File
    with open(summary_path, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(refcell, file=f)
        write_unique_species(refcell, file=f)
        if mode == "possible_charges":
            write_possible_charges(refcell, file=f)
        print(f"ERROR: {error_message}", file=f)
        for msg in warnings:
            print(f"WARNING: {msg}", file=f)

    # Write to Logger
    logger.info("Reference Error (mode=%s): %s", mode, error_message)
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
