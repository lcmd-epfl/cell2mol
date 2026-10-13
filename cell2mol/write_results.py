#!/usr/bin/env python

"""Write result summaries, error logs, and coordination reports."""

from cell2mol.utils import config
import sys
import os
import traceback
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def write_input_error_log(message, details=None):
    """Write input-error details to error_InputFile.out without exiting."""
    error_log_path = os.path.join(os.getcwd(), "error_InputFile.out")
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error: {message}\n")
        if details is not None:
            error_log.write(f"Details: {details}\n")


def write_exception_error_log(exc, error_log_path=None):
    """Write the active exception traceback, omitting it for MemoryError."""
    exc_type = type(exc).__name__
    if error_log_path is None:
        error_log_path = os.path.join(os.getcwd(), f"error_{exc_type}.out")

    try:
        with open(error_log_path, "w") as error_log:
            error_log.write(f"Error Type: {exc_type}\n")

            # Avoid allocating a traceback after memory exhaustion.
            if isinstance(exc, MemoryError):
                error_log.write(
                    "Memory limit exceeded. Full traceback omitted to save RAM.\n"
                )
            else:
                error_log.write(traceback.format_exc())

    except (MemoryError, OSError):
        # Report log-write failures on stderr.
        sys.stderr.write(
            f"FATAL: Could not write to {error_log_path} due to {exc_type}\n"
        )


def get_reference_error_message(error_case):
    """Return the message for a reference-processing error code."""

    # Input and runtime failures
    if error_case == config.ERR_INPUT:
        return "Processing failed: Invalid or inaccessible input"

    elif error_case == config.ERR_ASE_PARSE:
        return "Processing failed: ASE could not parse the input"

    elif error_case == config.ERR_MEMORY:
        return "Process terminated: Memory limit exceeded"

    elif error_case == config.ERR_TIMEOUT:
        return f"Process terminated: Execution timed out (> {config.TIMEOUT}s)"

    elif error_case == config.ERR_GENERAL:
        return "Process terminated: General error occurred"

    elif error_case == 0:
        return "No errors found"

    elif error_case == 1:
        return "Isolated hydrogens found"

    elif error_case == 2:
        return "Missing hydrogens in water molecules"

    elif error_case == 3:
        return "Missing hydrogens on a coordinated O (water/hydroxide) or N (ammonia)"

    elif error_case == 4:
        return "Missing hydrogens in carbon atoms"

    elif error_case == 5:
        return "Some unique species have no plausible charge states"

    elif error_case == 8:
        return "Error in assigning charges"

    elif error_case == 9:
        return "Error in creating bonds"

    elif error_case == 10:
        return "Error in assigning spin multiplicity"

    elif error_case == -1:
        return "Empty reference molecules list"

    else:
        return f"Unhandled error case: {error_case}"


def get_reference_error_messages(codes, fallback_code=0):
    """Join messages for all codes, using fallback_code when the list is empty."""
    if not codes:
        return get_reference_error_message(fallback_code)
    return "; ".join(get_reference_error_message(code) for code in codes)


def get_reference_warning_messages(refcell):
    """Return messages for active warnings (True) and skipped checks (None)."""

    # Active warning messages
    warning_map = {
        "cif_mismatch": "CIF formula mismatch detected",
        "over_polynuclear_limit": f"Polynuclear complex exceeds metal center limit (> max_metals {config.MAX_METALS})",
        "mixed_metals": "Mixed metal types detected",
        "is_mismatch_adj": "Adjacency matrix does not match CIF bond connectivity.",
        "is_mismatch_metal_adj": "Metal adjacency does not match CIF bond connectivity.",
        "metal_coord_diff": "Final metal coordination differs from CIF bond connectivity.",
    }

    # Skipped-check messages
    skip_map = {
        "cif_mismatch": "CIF formula check skipped (No _chemical_formula_moiety in CIF)",
        "is_mismatch_adj": "Adjacency check skipped (No _geom_bond in CIF)",
        "is_mismatch_metal_adj": "Metal adjacency check skipped (No _geom_bond in CIF)",
        "metal_coord_diff": "Metal coordination check skipped (No _geom_bond in CIF)",
    }

    messages = []

    if not refcell.potential_warnings:
        return messages

    for key, status in refcell.potential_warnings.items():
        if status is True:
            messages.append(warning_map.get(key, f"Warning: {key}"))

        elif status is None:
            if key in skip_map:
                messages.append(f"Skipped: {skip_map[key]}")
            else:
                messages.append(
                    f"Skipped: Check for {key} was skipped due to missing reference data."
                )

    return messages


def get_unitcell_error_message(error_case):
    """Return the message for a unit-cell processing error code."""

    # Input and runtime failures
    if error_case == config.ERR_INPUT:
        return "Processing failed: Invalid or inaccessible input"

    elif error_case == config.ERR_ASE_PARSE:
        return "Processing failed: ASE could not parse the input"

    elif error_case == config.ERR_MEMORY:
        return "Process terminated: Memory limit exceeded"

    elif error_case == config.ERR_TIMEOUT:
        return f"Process terminated: Execution timed out (> {config.TIMEOUT}s)"

    elif error_case == config.ERR_GENERAL:
        return "Process terminated: General error occurred"

    elif error_case == 0:
        return "No errors found"

    elif error_case == 3:
        return "Error in reconstructing fragments"

    elif error_case == 4:
        return "Error in unit cell construction"

    elif error_case == 6:
        return "Multiple valid charge distributions detected"

    elif error_case == 7:
        return "No valid charge distribution detected"

    elif error_case == 8:
        return "Error in assigning charges"

    elif error_case == 9:
        return "Error in creating bonds"

    elif error_case == 10:
        return "Error in assigning spin multiplicity"

    else:
        return f"Unhandled error case: {error_case}"


def get_molecule_error_message(error_case):
    """Return the message for a molecule-input processing error code."""

    # Input and runtime failures
    if error_case == config.ERR_INPUT:
        return "Processing failed: Invalid or inaccessible input"

    elif error_case == config.ERR_ASE_PARSE:
        return "Processing failed: ASE could not parse the input"

    elif error_case == config.ERR_MEMORY:
        return "Process terminated: Memory limit exceeded"

    elif error_case == config.ERR_TIMEOUT:
        return f"Process terminated: Execution timed out (> {config.TIMEOUT}s)"

    elif error_case == config.ERR_GENERAL:
        return "Process terminated: General error occurred"

    elif error_case == 0:
        return "No errors found"

    elif error_case == 1:
        return "Isolated hydrogens found"

    elif error_case == 2:
        return "Missing hydrogens in water molecules"

    elif error_case == 3:
        return "Missing hydrogens on a coordinated O (water/hydroxide) or N (ammonia)"

    elif error_case == 4:
        return "Missing hydrogens in carbon atoms"

    elif error_case == 5:
        return "Some unique species have no plausible charge states"

    elif error_case == 6:
        return "Multiple valid charge distributions detected"

    elif error_case == 7:
        return "No valid charge distribution detected"

    elif error_case == 8:
        return "Error in assigning charges"

    elif error_case == 9:
        return "Error in creating bonds"

    elif error_case == 10:
        return "Error in assigning spin multiplicity"

    else:
        return f"Unhandled error case: {error_case}"


def get_molecule_error_messages(codes, fallback_code=0):
    """Join molecule error messages, using fallback_code when the list is empty."""
    if not codes:
        return get_molecule_error_message(fallback_code)
    return "; ".join(get_molecule_error_message(code) for code in codes)


def write_cell_molecule_details(cell, file=None):
    """Write numbered molecule details for a reference or unit cell."""
    if cell.subtype == "reference":
        cell_type = "Reference"
        molecule_list = getattr(cell, "refmoleclist", None)
    elif cell.subtype == "unitcell":
        cell_type = "Unitcell"
        molecule_list = getattr(cell, "moleclist", None)
    else:
        return

    if molecule_list is None:
        print(f"\nNo molecules found in the {cell_type} cell.", file=file)
        return

    print(f"\nMolecules in {cell.subtype}:", file=file)

    for i, mol in enumerate(molecule_list):
        write_molecule_details(mol, file=file, index=i)


def write_molecule_details(mol, file=None, index=None):
    """Write molecule, metal, ligand, and donor-group details to a stream or stdout."""
    if mol is None:
        print("\nNo molecule object.", file=file)
        return

    prefix = f"Molecule {index}:" if index is not None else "Molecule:"
    mol_info_parts = [f"{prefix} {mol.formula}"]

    if mol.is_metal_complex:
        mol_info_parts.append("(Metal Complex)")
    if mol.has_s_block_metal:
        mol_info_parts.append("(Complex with Alkali or Alkaline metals)")
    if mol.is_post_transition_metal_complex:
        mol_info_parts.append("(Complex with Post-Transition metals)")
    if mol.is_metal_free_molecule:
        mol_info_parts.append("(Non-complex)")

    if mol.totcharge is not None:
        mol_info_parts.append(f"totcharge={mol.totcharge}")
    if getattr(mol, "totcharge_cif", None) is not None:
        mol_info_parts.append(f"totcharge_cif={mol.totcharge_cif}")
    if getattr(mol, "totcharge_agree", None) is not None:
        mol_info_parts.append(f"totcharge_agree={mol.totcharge_agree}")
    if getattr(mol, "spin", None) is not None:
        mol_info_parts.append(f"spin_multiplicity={mol.spin}")
    if mol.smiles is not None:
        mol_info_parts.append(f"smiles={mol.smiles}")

    print(" ".join(mol_info_parts), file=file)

    # Metal coordination details
    if not mol.is_metal_free_molecule and hasattr(mol, "metals"):
        for met in mol.metals:
            met_info = f"\t{met.formula} ({met.subtype})"

            labels = getattr(met, "atom_site_label", None)
            if labels:
                met_info += f" atom_site_label={labels}"

            if met.charge is not None:
                met_info += f" metal_OS={met.charge}"
            elif getattr(met, "plausible_os", None) is not None:
                met_info += f" metal_plausible_OS={met.plausible_os}"

            if getattr(met, "spin", None) is not None:
                met_info += f" metal_spin={met.spin}"

            print(met_info, file=file)

            if getattr(met, "coord_sphere_formula", None):
                print(
                    f"\t|--coord_sphere_formula={met.coord_sphere_formula}", file=file
                )

            if all(
                hasattr(met, attr)
                for attr in ("coord_nr", "coord_geometry", "geom_deviation")
            ):
                print(
                    f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}",
                    file=file,
                )

            if getattr(met, "groups", None):
                for group in met.groups:
                    group_info = f"\t|--(group) {group.labels}"
                    group_atom_site_labels = [
                        a.atom_site_label
                        for a in group.atoms
                        if a.atom_site_label is not None
                    ]
                    if group_atom_site_labels:
                        group_info += f" atom_site_labels={group_atom_site_labels}"
                    if hasattr(group, "denticity"):
                        group_info += f" denticity={group.denticity}"
                    if getattr(group, "is_haptic", False):
                        group_info += f" haptic_type={group.haptic_type}"

                    print(group_info, file=file)

            # Metal–metal links
            if hasattr(met, "metals") and hasattr(met, "coord_nr_with_metal_bonds"):
                bonded_metals = [m.label for m in met.metals]
                if bonded_metals:
                    print(
                        f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} "
                        f"coord_geometry_with_metal_bonds={getattr(met, 'coord_geometry_with_metal_bonds', 'N/A')} "
                        f"geom_deviation_with_metal_bonds={getattr(met, 'geom_deviation_with_metal_bonds', 'N/A')}",
                        file=file,
                    )

    # Ligand and donor-group details
    if not mol.is_metal_free_molecule and getattr(mol, "ligands", None):
        for lig in mol.ligands:
            lig_info = f"\t{lig.formula} ({lig.subtype})"
            for attr in ("smiles", "denticity", "totcharge"):
                val = getattr(lig, attr, None)
                if val is not None:
                    lig_info += f" {attr}={val}"
            if (
                getattr(lig, "plausible_charge_states", None) is not None
                and getattr(lig, "totcharge", None) is None
            ):
                status = "Exists" if lig.plausible_charge_states else "Does not exist"
                lig_info += f" lig.plausible_charge_states {status}"

            print(lig_info, file=file)

            if getattr(lig, "groups", None):
                for group in lig.groups:
                    group_info = f"\t|--(group) {group.labels}"
                    group_atom_site_labels = [
                        a.atom_site_label
                        for a in group.atoms
                        if a.atom_site_label is not None
                    ]
                    if group_atom_site_labels:
                        group_info += f" atom_site_labels={group_atom_site_labels}"
                    if hasattr(group, "denticity"):
                        group_info += f" denticity={group.denticity}"
                    if getattr(group, "is_haptic", False):
                        group_info += f" haptic_type={group.haptic_type}"

                    if getattr(group, "metals", None):
                        labels = [
                            getattr(m, "atom_site_label", None) or m.label
                            for m in group.metals
                        ]
                        group_info += f" connected_metals={labels}"
                    print(group_info, file=file)


def write_unique_species(object, file):
    """Write representative species, their indices, and available charge details."""

    unique_species = getattr(object, "unique_species", None)
    if not unique_species:
        print(f"\nNo unique species found in the {object.subtype} object.", file=file)
        return

    print(f"\nUnique Species in {object.subtype}:", file=file)

    for specie in unique_species:
        parts = [
            f"unique_index={specie.unique_index}",
            f"{specie.formula}",
            f"({specie.subtype})",
        ]

        if specie.subtype == "metal":
            parts.append(f"coord_sphere_formula={specie.coord_sphere_formula}")
            if getattr(specie, "charge", None) is not None:
                parts.append(f"charge={specie.charge}")

        elif specie.subtype == "ligand":
            parts.append(f"denticity={specie.denticity}")
            if specie.is_haptic:
                parts.append(f"haptic_type={specie.haptic_type}")
            if getattr(specie, "smiles", None) is not None:
                parts.append(f"smiles={specie.smiles}")
            if getattr(specie, "totcharge", None) is not None:
                parts.append(f"totcharge={specie.totcharge}")
            if getattr(specie, "groups", None) is not None:
                parts.append(f"groups={[group.formula for group in specie.groups]}")

        else:
            if getattr(specie, "smiles", None) is not None:
                parts.append(f"smiles={specie.smiles}")
            if getattr(specie, "totcharge", None) is not None:
                parts.append(f"totcharge={specie.totcharge}")

        print("\t" + " ".join(parts), file=file)


def _get_charge_state_failure_reason(specie) -> str:
    """Explain skipped charge enumeration or the absence of valid charge states."""
    if getattr(specie, "has_missing_H", False):
        kinds = [
            name
            for flag, name in (
                ("missing_H_in_Carbon", "carbon"),
                ("missing_H_on_CoordDonor", "coordinated donor"),
                ("missing_H_in_Water", "water"),
            )
            if getattr(specie, flag, False)
        ]
        detail = ", ".join(kinds) if kinds else "unspecified site"
        if detail == "coordinated donor":
            detail += f" {specie.formula}"
        return f"SKIPPED: missing hydrogens on {detail}"

    warning = getattr(specie, "protonation_warning", None)
    if warning:
        return f"SKIPPED: protonation not auto-handled ({warning})"

    # Report the final valence-search limit before earlier perception limits.
    if getattr(specie, "valence_search_too_large", False):
        return "SKIPPED: valence search space too large, enumeration terminated"

    if getattr(specie, "bond_perception_capped", False):
        return "SKIPPED: bond order assignment exceeded its iteration limit"

    return "NO PLAUSIBLE CHARGE STATES FOUND"


def write_plausible_charges(object, file=None):
    """Write charge candidates, missing-state reasons, and inconsistencies between copies."""
    if object.species_list is not None:
        print(f"\nPossible charges of species in {object.subtype}:", file=file)
        for specie in object.species_list:
            info = f"\tunique_index={specie.unique_index} {specie.formula} ({specie.subtype})"

            if specie.subtype == "metal":
                info += f" coord_sphere_formula={getattr(specie, 'coord_sphere_formula', 'N/A')}"

            if specie.subtype == "metal":
                plausible = getattr(specie, "plausible_os", None)
            else:
                plausible = getattr(specie, "plausible_charge_states", None)

            if plausible is not None:
                if specie.subtype == "metal":
                    info += f" plausible_os={plausible}"
                else:
                    info += f"\n\tplausible_charge_states={plausible}"
            else:
                info += f" {_get_charge_state_failure_reason(specie)}"

            print(info, file=file)
    else:
        print("\nNo species list found in the cell object.", file=file)

    # Different candidates for equivalent copies can affect charge selection.
    inconsistent = getattr(object, "inconsistent_plausible_charges", None)
    if inconsistent:
        print(
            "\nWARNING: plausible charges disagree between copies of the same specie:",
            file=file,
        )
        for unique_index, variants in sorted(inconsistent.items()):
            formula = next(
                (
                    spec.formula
                    for spec in (object.species_list or [])
                    if getattr(spec, "unique_index", None) == unique_index
                ),
                "?",
            )
            print(
                f"\tunique_index={unique_index} {formula} enumerated to {variants}",
                file=file,
            )

    print("", file=file)


def log_charge_diagnostics(newcell, refcell) -> None:
    """Log reference charge candidates and check reference/unit-cell index mappings."""

    logger.debug("#### Charge State & Index Validation ####")

    if hasattr(refcell, "plausible_charges") and refcell.plausible_charges:
        for specie, options in zip(refcell.unique_species, refcell.plausible_charges):
            logger.debug(
                "Unique Specie %s (Formula: %s):", specie.unique_index, specie.formula
            )
            logger.debug("  > Plausible Charge Options: %s", options)
    else:
        logger.warning("RefCell has no 'plausible_charges' populated to display.")

    _validate_species_indices(refcell, "Reference Cell")
    _validate_species_indices(newcell, "Unit Cell")


def _validate_species_indices(cell, label: str) -> None:
    """Log mismatches between species unique_index values and their mapping list."""
    logger.debug("--- Validating %s Indices ---", label)

    if not hasattr(cell, "species_list") or not hasattr(cell, "unique_indices"):
        logger.warning(
            "Skipping validation: %s missing species list or indices.", label
        )
        return

    for i, (specie, mapped_idx) in enumerate(
        zip(cell.species_list, cell.unique_indices)
    ):
        if mapped_idx != specie.unique_index:
            logger.warning(
                "MISMATCH at list index %d: Species %s claims UniqueID %s, "
                "but mapping list has %s.",
                i,
                specie.formula,
                specie.unique_index,
                mapped_idx,
            )
        else:
            logger.debug(
                "\t[OK] %s (UniqueID: %s)", specie.formula, specie.unique_index
            )


def write_coordination_report(report, output_csv):
    """Write a coordination-discrepancy CSV with fixed columns; skip empty reports."""
    if not report:
        logger.info("No coordination discrepancy report to save.")
        return None

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "refcode",
        "molecule_index",
        "formula",
        "metal",
        "coord_atom",
        "metal_site_label",
        "coord_atom_site_label",
        "distance",
        "status",
        "bond_change",
        "removed",
        "final_match",
        "final_match_wo_mm",
        "haptic_type",
        "is_haptic",
        "n_metals",
    ]

    df = pd.DataFrame(report)

    # Preserve column order and fill absent fields.
    df = df.reindex(columns=fieldnames)

    df.to_csv(output_csv, index=False)

    logger.info("Saved coordination discrepancy report to %s", output_csv)
    return output_csv
