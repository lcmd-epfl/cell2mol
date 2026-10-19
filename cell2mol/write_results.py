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


def write_cell_molecule_details(cell, file=None, *, detailed=True):
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
        write_molecule_details(mol, file=file, index=i, detailed=detailed)


def _coordination_atom_label(atom):
    """Use CIF site labels when available, otherwise the element symbol."""
    return atom.atom_site_label or atom.label


def _coordination_mode_text(ligand, mode, metal=None):
    """Format one interaction without exposing internal atom indices."""
    atoms = ligand.atoms or []
    parts = []
    if mode["kappa_atoms"]:
        labels = [_coordination_atom_label(atoms[i]) for i in mode["kappa_atoms"]]
        parts.append(f"kappa{len(labels)} [{', '.join(labels)}]")
    for site in mode["eta_groups"]:
        labels = [_coordination_atom_label(atoms[i]) for i in site]
        text = f"eta{len(site)} [{', '.join(labels)}]"
        site_atoms = {id(atoms[i]) for i in site}
        groups = (
            metal.groups
            if metal is not None and metal.groups is not None
            else ligand.groups
        )
        for group in groups or []:
            if (
                group.is_haptic
                and group.haptic_type
                and {id(atom) for atom in group.atoms or []} == site_atoms
            ):
                text += f" haptic_type={group.haptic_type}"
                break
        parts.append(text)
    return "; ".join(parts) or "no donor contacts"


def write_molecule_details(mol, file=None, index=None, *, detailed=False):
    """Write compact coordination summaries using site labels or elements.

    L1, L2, ... identify ligand copies in mol.ligands. Internal coordination
    indices stay unchanged. Detailed output adds SMILES, donor totals and
    legacy geometry-point counts.
    Reporting does not perceive bonds or refresh cached results.
    """
    if mol is None:
        print("\nNo molecule object.", file=file)
        return

    prefix = f"Molecule {index}:" if index is not None else "Molecule:"
    mol_info_parts = [f"{prefix} {mol.formula}"]

    if mol.is_metal_complex:
        mol_info_parts.append("(Contains d/f-block metals)")
    if mol.has_s_block_metal:
        mol_info_parts.append("(Contains alkali or alkaline-earth metals)")
    if mol.is_post_transition_metal_complex:
        mol_info_parts.append("(Post-transition-metal-only category)")
    if mol.is_metal_free_molecule:
        mol_info_parts.append("(Metal-free molecule)")

    if mol.totcharge is not None:
        mol_info_parts.append(f"totcharge={mol.totcharge}")
    if getattr(mol, "totcharge_cif", None) is not None:
        mol_info_parts.append(f"totcharge_cif={mol.totcharge_cif}")
    if getattr(mol, "totcharge_agree", None) is not None:
        mol_info_parts.append(f"totcharge_agree={mol.totcharge_agree}")
    if getattr(mol, "spin", None) is not None:
        mol_info_parts.append(f"spin_multiplicity={mol.spin}")
    if detailed and mol.smiles is not None:
        mol_info_parts.append(f"smiles={mol.smiles}")

    print(" ".join(mol_info_parts), file=file)

    if mol.is_metal_free_molecule:
        return

    metals = mol.metals or []
    ligands = mol.ligands or []
    metal_by_index = {met.get_parent_index("molecule"): met for met in metals}
    # if metals or ligands:
    #     print(
    #         "\tCoordination: CIF site labels / XYZ element symbols; "
    #         "L1, L2, ... identify ligand copies. "
    #         "Atom contacts and sites exclude metal-metal bonds.",
    #         file=file,
    #     )

    for met in metals:
        parts = [f"\tMetal {_coordination_atom_label(met)}"]
        if met.charge is not None:
            parts.append(f"OS={met.charge}")
        elif met.plausible_os is not None:
            parts.append(f"plausible_OS={met.plausible_os}")
        if met.spin is not None:
            parts.append(f"spin_multiplicity={met.spin}")
        print(" ".join(parts), file=file)
        coordination = met.coordination
        contact_info = (
            f"\t  ligands=[{', '.join(f'L{i + 1}' for i in coordination)}] "
            f"atom_contacts={met.coordination_number_atoms}"
        )
        if met.coord_sphere_formula:
            contact_info += f" coord_sphere_formula={met.coord_sphere_formula}"
        print(contact_info, file=file)
        geometry_parts = [f"coordination_points={met.coordination_number_sites}"]
        if met.coord_geometry is not None:
            geometry_parts.append(f"geometry={met.coord_geometry}")
        if met.geom_deviation is not None:
            geometry_parts.append(f"deviation={met.geom_deviation}")
        if geometry_parts:
            print("\t  " + " ".join(geometry_parts), file=file)
        for ligand_index, mode in coordination.items():
            print(
                f"\t  L{ligand_index + 1}: "
                f"{_coordination_mode_text(ligands[ligand_index], mode, met)}",
                file=file,
            )
        if met.metals:
            labels = [_coordination_atom_label(other) for other in met.metals]
            print(
                f"\t  metal-metal=[{', '.join(labels)}] "
                f"coordination_points_with_metal_bonds={met.coord_nr_with_metal_bonds} "
                f"geometry={met.coord_geometry_with_metal_bonds} "
                f"deviation={met.geom_deviation_with_metal_bonds}",
                file=file,
            )

    for ligand_index, lig in enumerate(ligands):
        coordination = lig.coordination
        has_kappa = any(mode["kappa_atoms"] for mode in coordination.values())
        has_eta = any(mode["eta_groups"] for mode in coordination.values())
        parts = [f"\tLigand L{ligand_index + 1}: {lig.formula}"]
        if lig.totcharge is not None:
            parts.append(f"charge={lig.totcharge}")
        elif lig.plausible_charge_states is not None:
            parts.append(
                f"charge_candidates={'available' if lig.plausible_charge_states else 'none'}"
            )
        parts.extend(
            [
                f"mu={len(coordination)}",
                f"bridging={len(coordination) > 1}",
                f"haptic={has_eta}",
                f"mixed={has_kappa and has_eta}",
            ]
        )
        print(" ".join(parts), file=file)
        for metal_index, mode in coordination.items():
            metal = metal_by_index.get(metal_index)
            if metal is None:
                metal = next(
                    (
                        candidate
                        for candidate in lig.metals or []
                        if candidate.get_parent_index("molecule") == metal_index
                    ),
                    None,
                )
            metal_label = (
                _coordination_atom_label(metal)
                if metal is not None
                else "unknown metal"
            )
            print(
                f"\t  -> {metal_label}: {_coordination_mode_text(lig, mode, metal)}",
                file=file,
            )
        if not coordination:
            print("\t  No coordination contacts available.", file=file)
        if detailed and lig.smiles is not None:
            print(f"\t  smiles={lig.smiles}", file=file)


def write_unique_species(object, file):
    """Write representative species, their indices, and available charge details."""

    unique_species = getattr(object, "unique_species", None)
    if not unique_species:
        print(f"\nNo unique species found in the {object.subtype} object.", file=file)
        return

    print(f"\nUnique Species in {object.subtype}:", file=file)

    for species in unique_species:
        parts = [
            f"unique_index={species.unique_index}",
            f"{species.formula}",
            f"({species.subtype})",
        ]

        if species.subtype == "metal":
            parts.append(f"coord_sphere_formula={species.coord_sphere_formula}")
            if getattr(species, "charge", None) is not None:
                parts.append(f"charge={species.charge}")

        elif species.subtype == "ligand":
            metal_labels = {
                metal.get_parent_index("molecule"): _coordination_atom_label(metal)
                for metal in species.metals or []
            }
            modes = []
            for metal_index, mode in species.coordination.items():
                counts = []
                if mode["kappa_atoms"]:
                    counts.append(f"kappa{len(mode['kappa_atoms'])}")
                counts.extend(f"eta{len(site)}" for site in mode["eta_groups"])
                modes.append(
                    f"{metal_labels.get(metal_index, 'unknown metal')}:"
                    + "+".join(counts)
                )
            parts.append(f"coordination=[{', '.join(modes)}]")
            if species.is_haptic:
                parts.append(f"haptic_type={species.haptic_type}")
            if getattr(species, "smiles", None) is not None:
                parts.append(f"smiles={species.smiles}")
            if getattr(species, "totcharge", None) is not None:
                parts.append(f"totcharge={species.totcharge}")
            if getattr(species, "groups", None) is not None:
                parts.append(f"groups={[group.formula for group in species.groups]}")

        else:
            if getattr(species, "smiles", None) is not None:
                parts.append(f"smiles={species.smiles}")
            if getattr(species, "totcharge", None) is not None:
                parts.append(f"totcharge={species.totcharge}")

        print("\t" + " ".join(parts), file=file)


def _get_charge_state_failure_reason(species) -> str:
    """Explain skipped charge enumeration or the absence of valid charge states."""
    if getattr(species, "has_missing_H", False):
        kinds = [
            name
            for flag, name in (
                ("missing_H_in_Carbon", "carbon"),
                ("missing_H_on_CoordDonor", "coordinated donor"),
                ("missing_H_in_Water", "water"),
            )
            if getattr(species, flag, False)
        ]
        detail = ", ".join(kinds) if kinds else "unspecified site"
        if detail == "coordinated donor":
            detail += f" {species.formula}"
        return f"SKIPPED: missing hydrogens on {detail}"

    warning = getattr(species, "protonation_warning", None)
    if warning:
        return f"SKIPPED: protonation not auto-handled ({warning})"

    # Report the final valence-search limit before earlier perception limits.
    if getattr(species, "valence_search_too_large", False):
        return "SKIPPED: valence search space too large, enumeration terminated"

    if getattr(species, "bond_perception_capped", False):
        return "SKIPPED: bond order assignment exceeded its iteration limit"

    return "NO PLAUSIBLE CHARGE STATES FOUND"


def write_plausible_charges(object, file=None):
    """Write charge candidates, missing-state reasons, and inconsistencies between copies."""
    if object.species_list is not None:
        print(f"\nPossible charges of species in {object.subtype}:", file=file)
        for species in object.species_list:
            info = f"\tunique_index={species.unique_index} {species.formula} ({species.subtype})"

            if species.subtype == "metal":
                info += f" coord_sphere_formula={getattr(species, 'coord_sphere_formula', 'N/A')}"

            if species.subtype == "metal":
                plausible = getattr(species, "plausible_os", None)
            else:
                plausible = getattr(species, "plausible_charge_states", None)

            if plausible is not None:
                if species.subtype == "metal":
                    info += f" plausible_os={plausible}"
                else:
                    info += f"\n\tplausible_charge_states={plausible}"
            else:
                info += f" {_get_charge_state_failure_reason(species)}"

            print(info, file=file)
    else:
        print("\nNo species list found in the cell object.", file=file)

    # Different candidates for equivalent copies can affect charge selection.
    inconsistent = getattr(object, "inconsistent_plausible_charges", None)
    if inconsistent:
        print(
            "\nWARNING: plausible charges disagree between copies of the same species:",
            file=file,
        )
        for unique_index, variants in sorted(inconsistent.items()):
            formula = next(
                (
                    species.formula
                    for species in (object.species_list or [])
                    if getattr(species, "unique_index", None) == unique_index
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
        for species, options in zip(refcell.unique_species, refcell.plausible_charges):
            logger.debug(
                "Unique Species %s (Formula: %s):",
                species.unique_index,
                species.formula,
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

    for i, (species, mapped_idx) in enumerate(
        zip(cell.species_list, cell.unique_indices)
    ):
        if mapped_idx != species.unique_index:
            logger.warning(
                "MISMATCH at list index %d: Species %s claims UniqueID %s, "
                "but mapping list has %s.",
                i,
                species.formula,
                species.unique_index,
                mapped_idx,
            )
        else:
            logger.debug(
                "\t[OK] %s (UniqueID: %s)", species.formula, species.unique_index
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
