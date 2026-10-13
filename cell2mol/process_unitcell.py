#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.classes import Reference, UnitCell, Cells
from cell2mol.process_reference import interpret_reference
from cell2mol.construction import construct_unitcell
from cell2mol.charge.balancing import balance_unitcell_charge
from cell2mol.read_cif import extract_cell_atoms, extract_cell_parameters
from cell2mol.write_results import (
    write_cell_molecule_details,
    write_unique_species,
    write_plausible_charges,
    get_reference_error_message,
    get_reference_error_messages,
    get_unitcell_error_message,
    write_exception_error_log,
    get_reference_warning_messages,
)
from cell2mol.utils.limits import set_time_limit
from cell2mol.compare import compare_metal_oxidation_states, compare_total_charge
import sys
from cell2mol.utils.processing import (
    ProcessingError,
    read_structure,
    raise_processing_error,
    save_partial_results,
    run_cli,
)
import gc

logger = logging.getLogger(__name__)


def interpret_unitcell(input_path: str, name: str, current_dir: str):
    """Return cell results; raise ProcessingError on input or runtime failure."""

    refcell: Reference | None = None
    unitcell = None
    cells = None
    structure = None

    try:
        refcell = interpret_reference(input_path, name, current_dir)
        if refcell is None:
            raise ProcessingError(
                "Failed to create reference cell", config.ERR_CELL2MOL
            )
        if refcell.has_error():
            return None

        with set_time_limit(config.TIMEOUT):
            structure = read_structure(read, input_path, "cif")
            cell_labels, cell_pos, cell_fracs = extract_cell_atoms(structure)
            cell_vector, cell_param, sym_ops = extract_cell_parameters(structure)

            unitcell = UnitCell.from_positional(
                name=name,
                labels=cell_labels,
                pos=cell_pos,
                frac_coord=cell_fracs,
                cell_vector=cell_vector,
                cell_param=cell_param,
            )

            unitcell.set_subtype("unitcell")

            cells = Cells.from_positional(
                name=name,
                reference=refcell,
                unitcell=unitcell,
                cell_vector=cell_vector,
                cell_param=cell_param,
            )
            # -------------------------------
            # Run the Processing Pipeline
            # -------------------------------
            success = _process_cell_logic(refcell, unitcell, sym_ops)
            cells.reference = refcell
            cells.unitcell = unitcell
            # Update cells object with processed data
            if success:
                logger.info("cell2mol process completed successfully.")
                total_charge_comparison = compare_total_charge(refcell)
                metal_os_comparison = compare_metal_oxidation_states(refcell)
                logger.info(
                    f" - Total charge comparison result: {total_charge_comparison}"
                )
                logger.info(
                    f" - Metal oxidation states comparison result: {metal_os_comparison}"
                )
                refcell.set_comparison_results(
                    total_charge_comparison, metal_os_comparison
                )
                unitcell.set_comparison_results(
                    total_charge_comparison, metal_os_comparison
                )
            else:
                logger.error("cell2mol process encountered errors.")
                logger.debug(" - Reference error case: %s", refcell.error_cases)
                logger.debug(" - Unit cell error case: %s", unitcell.error_cases)

    except Exception as exc:
        if isinstance(exc, ProcessingError) and refcell is None:
            refcell = exc.result
        if isinstance(exc, MemoryError):
            structure = None
            gc.collect()
        logger.error("Unit-cell processing failed: %s", exc)
        write_exception_error_log(exc)
        raise_processing_error(exc, cells, refcell, unitcell)

    finally:
        save_partial_results(
            lambda: _save_cell_outputs(name, current_dir, refcell, unitcell, cells),
            cells,
            refcell,
            unitcell,
        )

    return cells


def _process_cell_logic(refcell, unitcell, sym_ops) -> bool:
    """Executes the scientific logic: reconstruction, charge balancing, and assignment."""
    if refcell.has_error():
        logger.error(
            "Processing the reference cell failed. Aborting unit cell processing."
        )
        return False

    logger.info("Starting the cell2mol process for the unit cell")

    # Reconstruction
    unitcell = construct_unitcell(refcell, unitcell, sym_ops)
    if _has_step_failed(unitcell, "reconstruction"):
        return False

    # Charge Balancing
    refcell, unitcell = balance_unitcell_charge(refcell, unitcell)
    if _has_step_failed(unitcell, "balance_charges"):
        return False

    # Reference Assignment
    refcell.assign_charges()
    if _has_step_failed(refcell, "charge_assignment"):
        return False

    refcell.assign_spin()
    if _has_step_failed(refcell, "spin_assignment"):
        return False

    # Unit Cell Assignment
    unitcell.assign_charges(refmoleclist=refcell.refmoleclist)
    unitcell.check_charge_neutrality()
    if _has_step_failed(unitcell, "charge_assignment"):
        return False

    unitcell.assign_spin()
    return not _has_step_failed(unitcell, "spin_assignment")


def _has_step_failed(obj, mode):
    """Assess errors for a specific processing step and log them."""
    obj.assess_errors(mode=mode)

    error_message = ""
    if obj.subtype == "reference":
        code = obj.error_cases.get(mode, 0)
        error_message = get_reference_error_message(code)
        logger.info("Reference Error (mode=%s): %s", mode, error_message)

    elif obj.subtype == "unitcell":
        code = obj.error_cases.get(mode, 0)
        error_message = get_unitcell_error_message(code)
        logger.info("UnitCell Error (mode=%s): %s", mode, error_message)

    if obj.has_error(mode):
        logger.error(error_message)
        return True
    return False


def _save_cell_outputs(name, current_dir, refcell, unitcell, cells):
    """Handles all file writing and summary generation."""
    paths = {
        "refcell": os.path.join(current_dir, f"Ref_Cell_{name}"),
        "unitcell": os.path.join(current_dir, f"Cell_{name}"),
        "cells": os.path.join(current_dir, f"Cells_{name}"),
        "ref_sum": os.path.join(current_dir, "reference_summary.txt"),
        "unit_sum": os.path.join(current_dir, "unitcell_summary.txt"),
    }

    if refcell:
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

    if unitcell:
        _safe_run(
            lambda: unitcell.save(paths["unitcell"] + ".cell", format="pickle"),
            "Failed to save unit cell as pickle",
        )
        _safe_run(
            lambda: unitcell.save(paths["unitcell"] + ".json", format="json"),
            "Failed to save unit cell as JSON",
        )
        _safe_run(
            lambda: _write_unit_summary(name, unitcell, paths["unit_sum"]),
            "Failed to write unit cell summary",
        )

    # Retrieve pre-formatted messages (Warnings for True, INFO for None)
    if refcell:
        warning_messages = get_reference_warning_messages(refcell)
        if warning_messages:
            for summary_path in [paths["ref_sum"], paths["unit_sum"]]:
                with open(summary_path, "a") as f:
                    print("\nPotential issues and data status detected:", file=f)
                    for msg in warning_messages:
                        # Apply appropriate prefix based on the message content
                        prefix = "  " if msg.startswith("Skipped:") else "  Warning: "
                        print(f"{prefix}{msg}", file=f)
    if cells:
        _safe_run(
            lambda: cells.save(paths["cells"] + ".cells", format="pickle"),
            "Failed to save cells (reference + unit cell) as pickle",
        )
        _safe_run(
            lambda: cells.save(paths["cells"] + ".json", format="json"),
            "Failed to save cells (reference + unit cell) as JSON",
        )


def _write_ref_detailed_summary(name, refcell, summary_path):
    """Writes the molecules info, species, errors, and warnings to file and log."""
    # Write to File
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


def _write_unit_summary(name: str, unitcell, summary_path: str):
    """Writes the reconstruction results and unit cell info to a text file."""
    with open(summary_path, "w") as f:
        print(name, file=f)
        print(
            f"Total charge comparison result: {getattr(unitcell, 'total_charge_comparison', 'N/A')}",
            file=f,
        )
        print(
            f"Metal oxidation state comparison result: {getattr(unitcell, 'metal_os_comparison', 'N/A')}",
            file=f,
        )
        write_cell_molecule_details(unitcell, file=f)

        if unitcell.error_cases:
            # Print step-specific unit cell errors
            for err_mode in unitcell.error_cases.keys():
                code = unitcell.error_cases.get(err_mode, 0)
                msg = get_unitcell_error_message(code)
                print(f"UnitCell Error (mode={err_mode}): {msg}", file=f)


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
            interpret_unitcell,
            input_path=input_path,
            name=name,
            current_dir=os.getcwd(),
        )
    )


if __name__ == "__main__":
    _main()
