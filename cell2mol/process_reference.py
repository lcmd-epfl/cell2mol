#!/usr/bin/env python

import os
import logging
import numpy as np
from ase.io import read
from cell2mol.standardize_metal_os import (
    is_metallocene,
    standardize_reported_metal_oxidation_states,
)
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
    extract_cell_parameters,
    extract_atom_sites,
    extract_bonds_and_moieties,
    extract_cif_metadata,
    check_cif_formula_mismatch,
)
from cell2mol.write_results import (
    extract_refmoleclist_xyz,
    write_cell_molecule_details,
    write_unique_species,
    write_plausible_charges,
    get_reference_error_messages,
    get_reference_warning_messages,
    write_input_error_log,
    write_exception_error_log,
)
from cell2mol.connectivity import has_adjacency_mismatch
from cell2mol.utils.limits import set_time_limit
import sys
from cell2mol.utils.processing import (
    UnsupportedInput,
    read_structure,
    raise_processing_error,
    save_partial_results,
    run_cli,
)
import gc

logger = logging.getLogger(__name__)


def interpret_reference(input_path, name, current_dir):
    """Return reference results; raise ProcessingError on input or runtime failure."""
    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input CIF: %s", input_path)
    logger.debug("Timeout set to %d seconds", config.TIMEOUT)

    refcell = None
    structure = None

    try:
        logger.info("Processing CIF file")

        cif_okay, error_message = prefilter_cif(input_path)
        if not cif_okay:
            for err_msg in error_message.splitlines():
                logger.info(err_msg)
            write_input_error_log(
                f"Input is not supported by cell2mol\n{error_message}"
            )
            raise UnsupportedInput(error_message)
        # Enforce timeout on the heavy lifting
        with set_time_limit(config.TIMEOUT):
            structure = read_structure(read, input_path, "cif")

            cell_vector, cell_param, _ = extract_cell_parameters(structure)
            refcell = create_reference(input_path, name, cell_vector, cell_param)

            if refcell.has_error():
                logger.error(
                    "Fails generating reference molecules (case=%s)",
                    (refcell.error_cases or {}).get("ref_molecules"),
                )
                return refcell

            if logger.isEnabledFor(logging.DEBUG):
                extract_refmoleclist_xyz(current_dir, refcell.refmoleclist, name)

            # Save intermediate success
            _handle_reference_outputs(name, current_dir, refcell)

            # Identify unique species and full species list
            refcell.collect_unique_species()

            # Check for missing hydrogens
            refcell.check_hydrogens()

            # Identify the plausible charge states for species in the reference molecules
            refcell.get_plausible_charges()

            refcell.assess_errors(mode="hydrogens")
            if refcell.has_error("hydrogens"):
                logger.error(
                    "Fails checking hydrogens (case=%s)",
                    (refcell.error_cases or {}).get("hydrogens"),
                )

            refcell.assess_errors(mode="plausible_charges")
            if refcell.has_error("plausible_charges"):
                logger.error(
                    "Fails checking plausible charges (case=%s)",
                    (refcell.error_cases or {}).get("plausible_charges"),
                )

    except UnsupportedInput:
        raise
    except Exception as exc:
        if isinstance(exc, MemoryError):
            structure = None
            gc.collect()
        logger.error("Reference processing failed: %s", exc)
        write_exception_error_log(exc)
        raise_processing_error(exc, refcell)

    finally:
        save_partial_results(
            lambda: _handle_reference_outputs(name, current_dir, refcell), refcell
        )

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

    atom_site_labels, ref_labels, ref_fracs = extract_atom_sites(input_path)
    ref_pos = np.asarray(frac2cart_fromparam(ref_fracs, cell_param))

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
    geom_bond_cif, moiety_list_cif = extract_bonds_and_moieties(input_path)
    refcell.set_cif_bond_moiety(geom_bond_cif, moiety_list_cif)

    # Extract additional CIF information
    chemical_name, reported_metal_os, moiety_dicts = extract_cif_metadata(input_path)
    refcell.set_cif_metadata(chemical_name, reported_metal_os, moiety_dicts)

    # Generate reference molecules
    refcell.build_reference_molecules()

    if refcell.error_cases is None:
        refcell.error_cases = {}
    if not refcell.refmoleclist:
        logger.warning("No reference molecules could be generated.")
        refcell.error_cases["ref_molecules"] = -1
        return refcell

    metal_label_list = []
    metallocene_list = []
    for ref in refcell.refmoleclist:
        if ref.metals:
            for metal in ref.metals:
                metal_label_list.append(metal.label)
                metallocene_list.append(is_metallocene(metal))

    logger.debug(
        f"Reference Metals found: {metal_label_list} with reported oxidation states: {reported_metal_os}"
    )

    matched_list, confidence = standardize_reported_metal_oxidation_states(
        metal_label_list, reported_metal_os, metallocene_list
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
        "cif_mismatch": check_cif_formula_mismatch(moiety_dicts, refcell.refmoleclist),
        "over_polynuclear_limit": any(
            is_polynuclear_over_limit(ref.labels, max_metal_centers=config.MAX_METALS)
            for ref in refcell.refmoleclist
        ),
        "mixed_metals": any(
            has_mixed_metal_types(ref.labels) for ref in refcell.refmoleclist
        ),
        "is_mismatch_adj": has_adjacency_mismatch(
            ref_labels, ref_pos, atom_site_labels, geom_bond_cif
        ),
        "is_mismatch_metal_adj": has_adjacency_mismatch(
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
    refcell.set_potential_warnings(warnings_payload)

    return refcell


def _handle_reference_outputs(name, current_dir, refcell):
    """Manages saving files and writing summaries."""
    if refcell is None:
        return
    """Handles all file writing and summary generation."""
    paths = {
        "refcell": os.path.join(current_dir, f"Ref_Cell_{name}"),
        "ref_sum": os.path.join(current_dir, "reference_summary.txt"),
    }

    _safe_run(
        lambda: refcell.save(paths["refcell"] + ".cell", format="pickle"),
        "Failed to save reference as pickle",
    )
    _safe_run(
        lambda: refcell.save(paths["refcell"] + ".json", format="json"),
        "Failed to save reference as JSON",
    )
    _safe_run(
        lambda: _write_ref_detailed_summary(name, refcell, paths["ref_sum"]),
        "Failed to write reference summary",
    )


def _write_ref_detailed_summary(name, refcell, summary_path):
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

        write_cell_molecule_details(refcell, file=f)

        write_unique_species(refcell, file=f)
        write_plausible_charges(refcell, file=f)

        # Print step-specific reference errors
        if refcell.error_cases:
            all_codes = refcell.error_cases_all or {}
            for err_mode, code in refcell.error_cases.items():
                msg = get_reference_error_messages(all_codes.get(err_mode), code)
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
    all_codes = refcell.error_cases_all or {}
    for err_mode, code in refcell.error_cases.items():
        current_err_msg = get_reference_error_messages(all_codes.get(err_mode), code)
        logger.info("Reference Error (mode=%s): %s", err_mode, current_err_msg)

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
        raise


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
        sys.exit(config.ERR_INPUT)

    sys.exit(
        run_cli(
            interpret_reference,
            input_path=input_path,
            name=name,
            current_dir=os.getcwd(),
        )
    )


if __name__ == "__main__":
    _main()
