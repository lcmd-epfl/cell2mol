#!/usr/bin/env python

import os
import logging
import copy
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
    get_reference_warning_messages,
    get_unitcell_error_message,
    exit_with_error_exception,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def interpret_unitcell(input_path: str, name: str, current_dir: str):
    """Orchestrates the cell2mol process for a CIF file."""
    try:
        # 1. Initialize and Load Data
        cells, refcell, unitcell, sym_ops = _initialize_cells(
            input_path, name, current_dir
        )
        if not cells or not unitcell:
            return None

        # 2. Run the Processing Pipeline
        success = _process_cell_logic(refcell, unitcell, sym_ops)

        # Update cells object with processed data
        if success:
            cells.reference = refcell
            cells.unitcell = unitcell
            return cells

    except Exception as exc:
        exit_with_error_exception(exc)
    finally:
        # 3. Handle Saving and Summaries (Always runs even on error)
        _save_cell_outputs(name, current_dir, cells, refcell, unitcell)

    return None


def _initialize_cells(input_path, name, current_dir):
    """Handles file reading and initial object instantiation."""
    logger.info("Starting the cell2mol process for reference (Wyckoff sites)")
    cells: Cells = interpret_reference(input_path, name, current_dir)
    refcell: Reference = cells.reference

    try:
        structure = read(input_path, format="cif")
    except (AssertionError, Exception) as e:
        logger.error(f"ASE failed to parse {input_path}: {e}")
        raise
    cell_labels, cell_pos, cell_fracs = get_cell_atoms(structure)
    cell_vector, cell_param, sym_ops = get_cell_parameters(structure)

    unitcell = UnitCell.from_positional(
        name, cell_labels, cell_pos, cell_fracs, cell_vector, cell_param
    )
    unitcell.set_subtype("unitcell")

    return cells, refcell, unitcell, sym_ops


def _process_cell_logic(refcell, unitcell, sym_ops) -> bool:
    """Executes the scientific logic: reconstruction, charge balancing, and assignment."""
    if refcell.has_error():
        logger.error("Error while processing the reference cell")
        return False

    logger.info("Starting the cell2mol process for the unit cell")

    # Reconstruction
    unitcell = construct_unitcell(refcell, unitcell, sym_ops)
    if _has_step_failed(
        unitcell, "reconstruction", "Error while unit cell reconstruction"
    ):
        return False

    # Charge Retrieval & Balancing
    refcell.get_selected_cs()
    if _has_step_failed(
        refcell, "possible_charges", "Error retrieving possible charges"
    ):
        return False

    refcell, unitcell = balance_unitcell_charge(refcell, unitcell)
    if _has_step_failed(unitcell, "balance_charges", "Error while balancing charges"):
        return False

    # Reference Assignment
    refcell.assign_charges()
    if _has_step_failed(
        refcell, "charge_assignment", "Error assigning reference charges"
    ):
        return False

    refcell.assign_spin()
    if _has_step_failed(refcell, "spin_assignment", "Error assigning reference spins"):
        return False

    # Unit Cell Assignment
    unitcell.assign_charges(refmoleclist=refcell.refmoleclist)
    unitcell.check_charge_neutrality()
    if _has_step_failed(
        unitcell, "charge_assignment", "Error assigning unit cell charges"
    ):
        return False

    unitcell.assign_spin()
    return not _has_step_failed(
        unitcell, "spin_assignment", "Error assigning unit cell spins"
    )


def _has_step_failed(obj, mode, error_msg):
    """Helper to assess errors and log them."""
    obj.assess_errors(mode=mode)
    if obj.has_error():
        logger.error(error_msg)
        return True
    return False


def _save_cell_outputs(name, current_dir, cells, refcell, unitcell):
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
    error_message = get_reference_error_message(refcell.error_case)
    warnings = get_reference_warning_messages(refcell)

    # Write to File
    with open(summary_path, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(refcell, file=f)
        write_unique_species(refcell, file=f)
        write_possible_charges(refcell, file=f)
        print(f"ERROR: {error_message}", file=f)
        for msg in warnings:
            print(f"WARNING: {msg}", file=f)


def _write_unit_summary(name: str, unitcell, summary_path: str):
    """Writes the reconstruction results and unit cell info to a text file."""
    with open(summary_path, "w") as f:
        print(name, file=f)
        write_cell_molecules_info(unitcell, file=f)
        error_message = get_unitcell_error_message(unitcell.error_case)
        print(f"ERROR: {error_message}", file=f)


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
