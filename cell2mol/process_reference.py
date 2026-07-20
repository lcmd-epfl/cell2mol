#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.standardize_metal_os import standardize_reported_metal_os
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.classes import Reference
from cell2mol.operations import (
    frac2cart_fromparam,
    is_polynuclear_over_limit,
    has_mixed_metal_types,
    has_different_metal_coordination,
)
from cell2mol.read_cif import (
    prefilter_cif,
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
    exit_with_error_input,
    exit_with_error_exception,
)
from cell2mol.connectivity import is_mismatch_adjacency
from cell2mol.utils.limits import ProcessingTimeoutError, set_time_limit
from cell2mol.utils.exceptions import ASEParseError
import sys
import gc

logger = logging.getLogger(__name__)

# Error Codes
ERR_CELL2MOL = config.ERR_CELL2MOL
ERR_ASE_PARSE = config.ERR_ASE_PARSE
ERR_INPUT = config.ERR_INPUT
ERR_GENERAL = config.ERR_GENERAL
ERR_TIMEOUT = config.ERR_TIMEOUT
ERR_MEMORY = config.ERR_MEMORY


def interpret_reference(input_path, name, current_dir):
    """
    Process the reference molecules from a CIF file.
    """
    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input CIF: %s", input_path)
    logger.debug("Timeout set to %d seconds", config.TIMEOUT)

    refcell = None
    exit_code = 0
    process_failure = False
    mode = None
    try:
        logger.info("Processing CIF file")

        cif_okay, error_message = prefilter_cif(input_path)
        if not cif_okay:
            for err_msg in error_message.splitlines():
                logger.info(err_msg)
            exit_with_error_input(
                f"CIF file is not suitable for processing\n{error_message}"
            )
            exit_code = ERR_INPUT
            return None
        # Enforce timeout on the heavy lifting
        with set_time_limit(config.TIMEOUT):
            try:
                structure = read(input_path, format="cif")
            except Exception as exc:
                logger.error(f"ASE failed to parse {input_path}: {exc}")
                raise ASEParseError from exc

            cell_vector, cell_param, _ = get_cell_parameters(structure)
            refcell = create_reference(input_path, name, cell_vector, cell_param)

            # Check missing hydrogens and assess errors
            if refcell.has_error():
                logger.error(
                    "Fails generating reference molecules (case=%s)",
                    refcell.error_cases.get("ref_molecules"),
                )
                process_failure = True
                mode = "ref_molecules"
                return refcell

            refcell.check_hydrogens()
            refcell.assess_errors(mode="hydrogens")
            if refcell.has_error():
                logger.error(
                    "Detects missing hydrogens (case=%s)",
                    refcell.error_cases.get("hydrogens"),
                )
                process_failure = True
                mode = "hydrogens"
                return refcell

            # Save intermediate success
            _handle_reference_outputs(name, current_dir, refcell, mode="hydrogens")

            # Possible charge states for unique species
            refcell.get_unique_species()
            refcell.get_selected_cs()
            refcell.assess_errors(mode="possible_charges")

            if refcell.has_error():
                logger.error(
                    "Fails retrieving possible charges (case=%s)",
                    refcell.error_cases.get("possible_charges"),
                )
                process_failure = True
                mode = "possible_charges"
                return refcell

    except ASEParseError as exc:
        logger.error("ASE parsing failed.")
        exit_with_error_exception(exc)
        exit_code = ERR_ASE_PARSE

    except MemoryError as exc:
        # CRITICAL: Delete large objects and force GC *before* doing anything else
        if "structure" in locals():
            del structure
        gc.collect()

        logger.error("Memory limit reached. RAM cleared.")

        if refcell is not None:
            if refcell.error_cases is None:
                refcell.error_cases = {}
            refcell.error_cases["memory"] = ERR_MEMORY

        # Now it is safe(r) to call the exit helper
        exit_with_error_exception(exc)
        exit_code = ERR_MEMORY

    # 2. Handle Timeout Errors Second
    except ProcessingTimeoutError as exc:
        logger.error(f"Processing timed out after {config.TIMEOUT} seconds.")

        if refcell is not None:
            if refcell.error_cases is None:
                refcell.error_cases = {}
            refcell.error_cases["timeout"] = ERR_TIMEOUT

        exit_with_error_exception(exc)
        exit_code = ERR_TIMEOUT

    # 3. Handle All Other Errors
    except Exception as exc:
        logger.error(f"Unhandled error: {exc}")
        if refcell is not None:
            if refcell.error_cases is None:
                refcell.error_cases = {}
            refcell.error_cases["general"] = ERR_GENERAL
        exit_with_error_exception(exc)
        exit_code = ERR_GENERAL

    finally:
        logger.info("Executing final output handling...")
        # Ensure we try to save whatever valid data we have (refcell might be None)
        if mode is None:
            mode = "possible_charges"  # Default to hydrogens if no mode set
        _handle_reference_outputs(name, current_dir, refcell, mode=mode)

        if exit_code == 0 and process_failure:
            exit_code = ERR_CELL2MOL

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

    # Extract additional CIF information
    chemical_name, reported_metal_os, moiety_dicts = extract_info_from_cif(input_path)
    refcell.set_additional_cif_info(chemical_name, reported_metal_os, moiety_dicts)

    # Generate reference molecules
    refcell.get_reference_molecules()

    if refcell.error_cases is None:
        refcell.error_cases = {}
    if not refcell.refmoleclist:
        logger.warning("No reference molecules could be generated.")
        refcell.error_cases["ref_molecules"] = -1
        return refcell

    metal_label_list = []
    for ref in refcell.refmoleclist:
        if ref.metals:
            for metal in ref.metals:
                metal_label_list.append(metal.label)

    logger.debug(
        f"Reference Metals found: {metal_label_list} with reported oxidation states: {reported_metal_os}"
    )

    matched_list, confidence = standardize_reported_metal_os(
        metal_label_list, reported_metal_os
    )
    logger.info(
        f"Reported metal oxidation states standardized to {matched_list} with confidence {confidence:.2f}"
    )
    refcell.reported_metal_os_matched = matched_list
    refcell.metal_os_match_confidence = confidence

    logger.info("Generated %d reference molecules.", len(refcell.refmoleclist))
    refcell.error_cases["ref_molecules"] = 0

    # --------------------------------------------------
    # Gather warnings into a dictionary
    # --------------------------------------------------
    warnings_payload = {
        "cif_mismatch": compare_cif_with_reference(moiety_dicts, refcell.refmoleclist),
        "over_polynuclear_limit": any(
            is_polynuclear_over_limit(ref.labels, max_metal_centers=config.MAX_METALS)
            for ref in refcell.refmoleclist
        ),
        "mixed_metals": any(
            has_mixed_metal_types(ref.labels) for ref in refcell.refmoleclist
        ),
        "is_mismatch_adj": is_mismatch_adjacency(
            ref_labels, ref_pos, atom_site_labels, geom_bond_cif
        ),
        "is_mismatch_metal_adj": is_mismatch_adjacency(
            ref_labels, ref_pos, atom_site_labels, geom_bond_cif, metal_only=True
        ),
        "metal_coord_diff": has_different_metal_coordination(
            refcell.refmoleclist,
            geom_bond_cif,
            name,
            # report_csv="coordination_report.csv",
        ),
    }

    # Set the warnings in the refcell object
    refcell.set_potential_warning(warnings_payload)

    return refcell


def _handle_reference_outputs(name, current_dir, refcell, mode=None):
    """Manages saving files and writing summaries."""
    if refcell is None:
        return

    # 1. Write the .out summary file
    summary_path = os.path.join(current_dir, "reference_summary.txt")
    _safe_run(
        lambda: _write_ref_detailed_summary(name, refcell, summary_path, mode=mode),
        "Failed to write reference summary",
    )

    # 2. Save the refcell object (.cell)
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")

    def save_ref():
        refcell.save(ref_cell_fname)
        # from cell2mol.write_results import extract_refmoleclist_xyz
        # if logger.isEnabledFor(logging.DEBUG) and not refcell.error_cases.get("ref_molecules", 0) == -1:
        #     extract_refmoleclist_xyz(current_dir, refcell.refmoleclist, name)

    _safe_run(save_ref, "Failed to save reference cell")


def _write_ref_detailed_summary(name, refcell, summary_path, mode=None):
    """Writes the molecules info, species, errors, and warnings to file and log."""
    # Retrieve pre-formatted messages (Warnings for True, INFO for None)
    warning_messages = get_reference_warning_messages(refcell)

    # --- Write to File ---
    with open(summary_path, "w") as f:
        print(name, file=f)
        print(
            f"Total charge comparison result: {getattr(refcell, 'total_charge_comparison')}",
            file=f,
        )
        print(
            f"Metal oxidation state comparison result: {getattr(refcell, 'metal_os_comparison')}",
            file=f,
        )
        reported_os = getattr(refcell, "reported_metal_os", None)
        matched_os = getattr(refcell, "reported_metal_os_matched", None)
        os_confidence = getattr(refcell, "metal_os_match_confidence", None)
        confidence_str = f"{os_confidence:.2f}" if os_confidence is not None else "N/A"
        print(f"  - Reported:   {reported_os}", file=f)
        print(f"  - Matched:    {matched_os}", file=f)
        print(f"  - Confidence: {confidence_str}", file=f)

        write_cell_molecules_info(refcell, file=f)

        if mode == "possible_charges":
            write_unique_species(refcell, file=f)
            write_possible_charges(refcell, file=f)

        # Print step-specific reference errors
        if refcell.error_cases:
            for err_mode, code in refcell.error_cases.items():
                msg = get_reference_error_message(code)
                print(f"Reference Error (mode={err_mode}): {msg}", file=f)

        # Print potential issues and data status
        if warning_messages:
            print("\nPotential issues and data status detected:", file=f)
            for msg in warning_messages:
                # Apply appropriate prefix based on the message content
                prefix = "  " if msg.startswith("Skipped:") else "  Warning: "
                print(f"{prefix}{msg}", file=f)

    # --- Write to Logger ---
    # Log the specific error for the current operation mode
    current_err_code = refcell.error_cases.get(mode, 0)
    current_err_msg = get_reference_error_message(current_err_code)
    logger.info("Reference Error (mode=%s): %s", mode, current_err_msg)

    if not warning_messages:
        logger.info("No potential issues or skipped checks detected.")
    else:
        logger.warning("Potential issues and data status detected:")
        for msg in warning_messages:
            if msg.startswith("Skipped:"):
                # Log skipped checks as INFO level to avoid cluttering warnings
                logger.info("  - %s", msg)
            else:
                # Log actual mismatches as WARNING level
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
