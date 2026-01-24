#!/usr/bin/env python

import os
import logging
from ase.io import read
from cell2mol.utils import config
from cell2mol.args import parsing_arguments
from cell2mol.classes import Reference, UnitCell, Cells
from cell2mol.process_reference import interpret_reference
from cell2mol.construction import construct_unitcell
from cell2mol.charge.charge_balancer import balance_unitcell_charge
from cell2mol.read_cif import get_cell_atoms, get_cell_parameters
from cell2mol.write_results import (
    write_cell_molecules_info,
    write_unique_species,
    write_possible_charges,
    get_reference_error_message,
    get_unitcell_error_message,
    exit_with_error_exception,
)
from cell2mol.utils.limits import ProcessingTimeoutError, set_time_limit

import sys
import gc

logger = logging.getLogger(__name__)

# Error Codes
ERR_CELL2MOL = config.ERR_CELL2MOL
ERR_GENERAL = config.ERR_GENERAL
ERR_TIMEOUT = config.ERR_TIMEOUT
ERR_MEMORY = config.ERR_MEMORY


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def interpret_unitcell(input_path: str, name: str, current_dir: str):
    """Orchestrates the cell2mol process for a CIF file."""

    refcell = None
    unitcell = None
    sym_ops = None
    exit_code = 0

    try:
        # Enforce timeout on the heavy lifting
        with set_time_limit(config.TIMEOUT):
            # -------------------------------
            # Initialize and Load Data
            # -------------------------------
            refcell, unitcell, sym_ops = _initialize_cells(
                input_path, name, current_dir
            )
            if not refcell or not unitcell:
                logger.error("Failed to initialize reference or unit cell.")
                exit_code = ERR_CELL2MOL
                return None
            # -------------------------------
            # Run the Processing Pipeline
            # -------------------------------
            success = _process_cell_logic(refcell, unitcell, sym_ops)

            # Update cells object with processed data
            if success:
                logger.info("cell2mol process completed successfully.")
            else:
                logger.error("cell2mol process encountered errors.")
                logger.debug(" - Reference error case: %s", refcell.error_cases)
                logger.debug(" - Unit cell error case: %s", unitcell.error_cases)
                exit_code = ERR_CELL2MOL

    # -------------------------------
    # Memory errors
    # -------------------------------
    except MemoryError as exc:
        logger.error("Memory limit reached. Attempting cleanup.")
        gc.collect()

        if refcell is not None:
            refcell.error_cases["memory"] = ERR_MEMORY
        if unitcell is not None:
            unitcell.error_cases["memory"] = ERR_MEMORY

        exit_with_error_exception(exc)
        exit_code = ERR_MEMORY

    # -------------------------------
    # Timeout errors
    # -------------------------------
    except ProcessingTimeoutError as exc:
        logger.error(f"Processing timed out after {config.TIMEOUT} seconds.")

        if refcell is not None:
            refcell.error_cases["timeout"] = ERR_TIMEOUT
        if unitcell is not None:
            unitcell.error_cases["timeout"] = ERR_TIMEOUT

        exit_with_error_exception(exc)
        exit_code = ERR_TIMEOUT

    # -------------------------------
    # All other runtime errors
    # -------------------------------
    except Exception as exc:
        logger.error(f"Unhandled error: {exc}")

        if refcell is not None:
            refcell.error_cases["general"] = ERR_GENERAL
        if unitcell is not None:
            unitcell.error_cases["general"] = ERR_GENERAL

        exit_with_error_exception(exc)
        exit_code = ERR_GENERAL

    finally:
        logger.info("Executing final output handling...")
        _save_cell_outputs(name, current_dir, refcell, unitcell)

        if exit_code != 0:
            logger.info(f"Process exiting with code {exit_code}")
            sys.exit(exit_code)

    return unitcell


def _initialize_cells(input_path, name, current_dir):
    """Handles file reading and initial object instantiation."""
    logger.info("Starting the cell2mol process for reference (Wyckoff sites)")

    refcell: Reference = interpret_reference(input_path, name, current_dir)

    if not refcell:
        logger.error("Failed to create reference cell from CIF.")
        return None, None, None

    try:
        structure = read(input_path, format="cif")
    except (AssertionError, Exception) as e:
        logger.error(f"ASE failed to parse {input_path}: {e}")
        raise
    cell_labels, cell_pos, cell_fracs = get_cell_atoms(structure)
    cell_vector, cell_param, sym_ops = get_cell_parameters(structure)

    unitcell = UnitCell.from_positional(
        name=name,
        labels=cell_labels,
        pos=cell_pos,
        frac_coord=cell_fracs,
        cell_vector=cell_vector,
        cell_param=cell_param,
    )

    unitcell.set_subtype("unitcell")

    return refcell, unitcell, sym_ops


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


def _save_cell_outputs(name, current_dir, refcell, unitcell):
    """Handles all file writing and summary generation."""
    paths = {
        "ref": os.path.join(current_dir, f"Ref_Cell_{name}.cell"),
        "cell": os.path.join(current_dir, f"Cell_{name}.cell"),
        "json": os.path.join(current_dir, f"Cells_{name}.json"),
        "pickle": os.path.join(current_dir, f"Cells_{name}.cell"),
        "ref_sum": os.path.join(current_dir, "reference_summary.out"),
        "unit_sum": os.path.join(current_dir, "unitcell_summary.out"),
    }

    if refcell:
        _safe_run(lambda: refcell.save(paths["ref"]), "Failed to save reference cell")
        _safe_run(
            lambda: _write_ref_detailed_summary(name, refcell, paths["ref_sum"]),
            "Failed to write ref summary",
        )

    if unitcell:
        _safe_run(lambda: unitcell.save(paths["cell"]), "Failed to save unit cell")
        _safe_run(
            lambda: _write_unit_summary(name, unitcell, paths["unit_sum"]),
            "Failed to write unit summary",
        )

    cells = Cells.from_positional(
        name=name,
        reference=refcell,
        unitcell=unitcell,
        cell_vector=refcell.cell_vector,
        cell_param=refcell.cell_param,
    )

    if cells:
        _safe_run(
            lambda: cells.save(paths["json"], format="json"), "Failed to save JSON"
        )
        _safe_run(
            lambda: cells.save(paths["pickle"], format="pickle"),
            "Failed to save pickle",
        )


def _write_ref_detailed_summary(name, refcell, summary_path):
    """Writes the molecules info, species, errors, and warnings to file and log."""
    # Write to File
    with open(summary_path, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(refcell, file=f)
        write_unique_species(refcell, file=f)
        write_possible_charges(refcell, file=f)

        # Print step-specific reference errors
        if refcell.error_cases:
            for err_mode, code in refcell.error_cases.items():
                msg = get_reference_error_message(code)
                print(f"Reference Error (mode={err_mode}): {msg}", file=f)


def _write_unit_summary(name: str, unitcell, summary_path: str):
    """Writes the reconstruction results and unit cell info to a text file."""
    with open(summary_path, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(unitcell, file=f)

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
