#!/usr/bin/env python

from cell2mol.utils import config
import sys
import re
import os
import traceback
from collections import Counter
from ase.io import read
from cell2mol.elementdata import ElementData
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


def exit_with_error_input(message):
    """Logs the error message to a file and exits the program."""
    error_log_path = os.path.join(os.getcwd(), "error_input.out")
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error: {message}\n")
    sys.exit(message)


def exit_with_error_exception(exc, error_log_path=None):
    exc_type = type(exc).__name__
    if error_log_path is None:
        error_log_path = os.path.join(os.getcwd(), f"error_{exc_type}.out")

    try:
        with open(error_log_path, "w") as error_log:
            error_log.write(f"Error Type: {exc_type}\n")

            # Handle MemoryError specifically to avoid allocating traceback string
            if isinstance(exc, MemoryError):
                error_log.write(
                    "Memory limit exceeded. Full traceback omitted to save RAM.\n"
                )
            else:
                error_log.write(traceback.format_exc())

    except (MemoryError, OSError):
        # Fallback to stderr if file writing fails
        sys.stderr.write(
            f"FATAL: Could not write to {error_log_path} due to {exc_type}\n"
        )


def compare_formula_xyz_vs_cif(xyzfile: str, formula_str_from_cif: str) -> dict:
    """
    Compare the elements from CIF with the elements in the XYZ file.
    """
    element_pattern = r"([A-Z][a-z]*)(\d*)"
    parsed_formula_cif = dict()
    for element, count in re.findall(element_pattern, formula_str_from_cif):
        parsed_formula_cif[element] = int(count) if count else 1
    mol = read(xyzfile, format="xyz")
    element_list = mol.get_chemical_symbols()
    element_list_count = dict(Counter(element_list))
    comparison_result = {
        "in_formula_cif": parsed_formula_cif,
        "in_list_xyz": element_list_count,
        "missing_in_xyz": {
            k: v - element_list_count.get(k, 0)
            for k, v in parsed_formula_cif.items()
            if element_list_count.get(k, 0) < v
        },
        "extra_in_xyz": {
            k: v - parsed_formula_cif.get(k, 0)
            for k, v in element_list_count.items()
            if parsed_formula_cif.get(k, 0) < v
        },
    }
    return comparison_result


def readxyz(file):
    labels = []
    pos = []
    xyz = open(file, "r")
    n_atoms = xyz.readline()
    title = xyz.readline()
    for line in xyz:
        line_data = line.split()
        if len(line_data) == 4:
            label, x, y, z = line.split()
            pos.append([float(x), float(y), float(z)])
            labels.append(label)
        else:
            print("I can't read the xyz. It has =/ than 4 columns")
    xyz.close()

    return labels, pos


def printxyz(labels, pos):
    print(len(labels))
    print("")
    for idx, l in enumerate(labels):
        print("%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]))


def writexyz(fdir, fname, labels, pos, charge: int = 0, spin: int = 1, info: str = ""):
    """Writes an XYZ file with given labels and positions."""
    os.makedirs(fdir, exist_ok=True)

    fullname = os.path.join(fdir, fname)
    natoms = len(labels)

    with open(fullname, "w") as fil:
        print(natoms, file=fil)
        print(f"{charge=} {spin=} {info}", file=fil)
        for label, (x, y, z) in zip(labels, pos):
            fil.write(f"{label:<2}  {x:15.8f}  {y:15.8f}  {z:15.8f}\n")


def extract_refmoleclist_xyz(fdir, refmoleclist, name: str):
    """Extracts reference molecules to XYZ files.

    Args:
        fdir (str): Directory to save the XYZ files.
        refmoleclist (list): List of reference molecule objects.
        name (str): Base name for the output files.
    """
    for i, ref in enumerate(refmoleclist):
        if ref.iscomplex:
            if ref.totcharge_cif is not None:
                N = 0
                for atom in ref.labels:
                    N += elemdatabase.elementnr[atom]
                N -= ref.totcharge_cif
                if N % 2 == 0:
                    spin = 1
                else:
                    spin = 2
                writexyz(
                    fdir,
                    f"{name}_Ref_{i}_{ref.formula}_charge_{ref.totcharge_cif}_lowspin_{spin}.xyz",
                    ref.labels,
                    ref.coord,
                    charge=ref.totcharge_cif,
                    spin=spin,
                )
                logger.debug(
                    "Ref molecule %s %s total charge %s lowest spin multiplicity %s",
                    i,
                    ref.formula,
                    ref.totcharge_cif,
                    spin,
                )
            else:
                writexyz(
                    fdir,
                    f"{name}_Ref_{i}_{ref.formula}.xyz",
                    ref.labels,
                    ref.coord,
                    charge="",
                    spin="",
                )
                logger.debug(
                    "Ref molecule %s %s without charge and spin information",
                    i,
                    ref.formula,
                )


def get_reference_error_message(error_case):
    """
    Return an error message for a given error case.
    """

    # --- Critical System Errors ---
    if error_case == config.ERR_MEMORY:
        return "Process terminated: Memory limit exceeded"

    elif error_case == config.ERR_TIMEOUT:
        return f"Process terminated: Execution timed out (> {config.TIMEOUT}s)"

    elif error_case == config.ERR_GENERAL:
        return "Process terminated: General error occurred"

    # --- Standard Processing Errors ---
    elif error_case == 0:
        return "No errors found"

    elif error_case == 1:
        return "Isolated hydrogens found"

    elif error_case == 2:
        return "Missing hydrogens in water molecules"

    elif error_case == 3:
        return "Missing hydrogens in coordinated water molecules"

    elif error_case == 4:
        return "Missing hydrogens in carbon atoms"

    elif error_case == 5:
        return "Some unique species have no possible charge states"

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


def get_reference_warning_messages(refcell):
    """
    Returns a list of messages based on potential_warnings.
    Triggers 'Warning' for True and 'Skipped' for None.
    """

    # Mapping for triggered warnings (status is True)
    warning_map = {
        "cif_mismatch": "CIF formula mismatch detected",
        "over_polynuclear_limit": f"Polynuclear complex exceeds metal center limit (> max_metals {config.MAX_METALS})",
        "mixed_metals": "Mixed metal types detected",
        "is_mismatch_adj": "Adjacency matrix does not match CIF bond connectivity.",
        "metal_coord_diff": "Final metal coordination differs from CIF bond connectivity.",
    }

    # Mapping for skipped checks (status is None)
    skip_map = {
        "cif_mismatch": "CIF formula check skipped (No _chemical_formula_moiety in CIF)",
        "is_mismatch_adj": "Adjacency check skipped (No _geom_bond in CIF)",
        "metal_coord_diff": "Metal coordination check skipped (No _geom_bond in CIF)",
    }

    messages = []

    if not refcell.potential_warnings:
        return messages

    for key, status in refcell.potential_warnings.items():
        # Case 1: The check failed (Warning)
        if status is True:
            messages.append(warning_map.get(key, f"Warning: {key}"))

        # Case 2: The check was skipped (None)
        elif status is None:
            # Only add a message if the key is in our skip_map
            if key in skip_map:
                messages.append(f"Skipped: {skip_map[key]}")
            else:
                messages.append(
                    f"Skipped: Check for {key} was skipped due to missing reference data."
                )

    return messages


def get_unitcell_error_message(error_case):
    """
    Return an error message for a given error case.
    """

    # --- Critical System Errors ---
    if error_case == config.ERR_MEMORY:
        return "Process terminated: Memory limit exceeded"

    elif error_case == config.ERR_TIMEOUT:
        return f"Process terminated: Execution timed out (> {config.TIMEOUT}s)"

    elif error_case == config.ERR_GENERAL:
        return "Process terminated: General error occurred"

    # --- Standard Processing Errors ---
    elif error_case == 0:
        return "No errors found"

    # elif error_case == 1:
    #     return "Isolated hydrogens found"

    # elif error_case == 2:
    #     return "Missing hydrogens"

    elif error_case == 3:
        return "Error in reconstructing fragments"

    elif error_case == 4:
        return "Error in unit cell construction"

    # elif error_case == 5:
    #     return "Some unique species have no possible charge states"

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
    """
    Return an error message for a given error case.
    """

    # --- Critical System Errors ---
    if error_case == config.ERR_MEMORY:
        return "Process terminated: Memory limit exceeded"

    elif error_case == config.ERR_TIMEOUT:
        return f"Process terminated: Execution timed out (> {config.TIMEOUT}s)"

    elif error_case == config.ERR_GENERAL:
        return "Process terminated: General error occurred"

    # --- Standard Processing Errors ---
    elif error_case == 0:
        return "No errors found"

    elif error_case == 5:
        return "Some unique species have no possible charge states"

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


def write_cell_molecules_info(cell, file=None):
    """
    Write molecule information within a cell to a file-like object.
    """
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
        # Pass the index to keep the output numbered
        write_molecule_info(mol, file=file, index=i)


def write_molecule_info(mol, file=None, index=None):
    """
    Writes detailed molecule information to a file-like object or the console.
    """
    if mol is None:
        print("\nNo molecule object.", file=file)
        return

    # Start header
    prefix = f"Molecule {index}:" if index is not None else "Molecule:"
    mol_info_parts = [f"{prefix} {mol.formula}"]

    # Append classification tags
    if mol.iscomplex:
        mol_info_parts.append("(TM Complex)")
    if mol.has_ia_iia:
        mol_info_parts.append("(Complex with Alkali or Alkaline metals)")
    if mol.has_post_transition_metal:
        mol_info_parts.append("(Complex with Post-Transition metals)")
    if mol.is_non_complex_molecule:
        mol_info_parts.append("(Non-complex)")

    # Append properties
    if mol.totcharge is not None:
        mol_info_parts.append(f"totcharge={mol.totcharge}")
    if getattr(mol, "totcharge_cif", None) is not None:
        mol_info_parts.append(f"totcharge_cif={mol.totcharge_cif}")
    if getattr(mol, "spin", None) is not None:
        mol_info_parts.append(f"spin_multiplicity={mol.spin}")
    if mol.smiles is not None:
        mol_info_parts.append(f"smiles={mol.smiles}")

    print(" ".join(mol_info_parts), file=file)

    # --------------------
    # Metals (Coordination Centers)
    # --------------------
    if not mol.is_non_complex_molecule and hasattr(mol, "metals"):
        for met in mol.metals:
            met_info = f"\t{met.formula} ({met.subtype})"

            labels = getattr(met, "atom_site_label", None)
            if labels:
                met_info += f" atom_site_label={labels}"

            # Oxidation state
            if met.charge is not None:
                met_info += f" metal_OS={met.charge}"
            elif getattr(met, "possible_cs", None) is not None:
                met_info += f" metal_possible_OS={met.possible_cs}"

            if getattr(met, "spin", None) is not None:
                met_info += f" metal_spin={met.spin}"

            print(met_info, file=file)

            # Coordination details
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

            # Metal-Metal Bonds
            if hasattr(met, "metals") and hasattr(met, "coord_nr_with_metal_bonds"):
                bonded_metals = [m.label for m in met.metals]
                if bonded_metals:
                    print(
                        f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} "
                        f"coord_geometry_with_metal_bonds={getattr(met, 'coord_geometry_with_metal_bonds', 'N/A')} "
                        f"geom_deviation_with_metal_bonds={getattr(met, 'geom_deviation_with_metal_bonds', 'N/A')}",
                        file=file,
                    )

    # --------------------
    # Ligands
    # --------------------
    if not mol.is_non_complex_molecule and getattr(mol, "ligands", None):
        for lig in mol.ligands:
            lig_info = f"\t{lig.formula} ({lig.subtype})"
            for attr in ("smiles", "denticity", "totcharge"):
                val = getattr(lig, attr, None)
                if val is not None:
                    lig_info += f" {attr}={val}"
            if (
                getattr(lig, "possible_cs", None) is not None
                and getattr(lig, "totcharge", None) is None
            ):
                status = "Exists" if lig.possible_cs else "Does not exist"
                lig_info += f" lig.possible_cs {status}"

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
    """
    Write unique species information to a file-like object.

    Args:
        object: Object containing unique_species.
        file: File-like object opened for writing.
    """

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


def write_possible_charges(object, file=None):
    """
    Write the possible charges for each species in the object.

    Args:
        object: Object containing species_list.
        file: File-like object (e.g., opened file). Defaults to None (console).
    """
    if object.species_list is not None:
        print(f"\nPossible charges of species in {object.subtype}:", file=file)
        for specie in object.species_list:
            # Base species information
            info = f"\tunique_index={specie.unique_index} {specie.formula} ({specie.subtype})"

            # Add coordination sphere for metals
            if specie.subtype == "metal":
                info += f" coord_sphere_formula={getattr(specie, 'coord_sphere_formula', 'N/A')}"

            # Add charge state information
            if getattr(specie, "possible_cs", None) is not None:
                if specie.subtype == "metal":
                    info += f" possible_cs={specie.possible_cs}"
                else:
                    # Non-metals use a new line for charge states as per original logic
                    info += f"\n\tpossible_cs={specie.possible_cs}"
            else:
                info += " NO POSSIBLE CHARGE STATES FOUND"

            print(info, file=file)
    else:
        print("\nNo species list found in the cell object.", file=file)
    print("", file=file)


def write_possible_and_selected_cs(newcell, refcell, file=None):
    """
    Write possible charge states and selected charge state for unique species.

    Args:
        newcell: Unit cell object.
        refcell: Reference cell object.
        file: File-like object. Defaults to None.
    """

    print(
        "\nPossible charge states and charge of selected charge state for unique species:",
        file=file,
    )

    # Check unique species and their selected charge states
    for idx, (specie, select) in enumerate(
        zip(refcell.unique_species, refcell.selected_cs)
    ):
        print(
            f"Unique unique_index={specie.unique_index} formula={specie.formula}",
            file=file,
        )
        print(
            f"charge of selected charge state={select}\npossible_cs={specie.possible_cs}\n",
            file=file,
        )

    # Reference cell species vs unique index mapping
    print("Species list in the reference and their unique indices:", file=file)
    for specie, idx in zip(refcell.species_list, refcell.unique_indices):
        print(
            f"\tformula={specie.formula} unique_index={specie.unique_index}", file=file
        )
        if idx != specie.unique_index:
            print(
                f"\tWARNING: formula={specie.formula} unique_index={specie.unique_index} "
                f"differs from refcell unique indices {idx=}",
                file=file,
            )

    # Unit cell species vs unique index mapping
    print("Species list in the unit cell and their unique indices:", file=file)
    for specie, idx in zip(newcell.species_list, newcell.unique_indices):
        print(
            f"\tformula={specie.formula} unique_index={specie.unique_index}", file=file
        )
        if idx != specie.unique_index:
            print(
                f"\tWARNING: formula={specie.formula} unique_index={specie.unique_index} "
                f"differs from newcell unique indices {idx=}",
                file=file,
            )


def log_charge_state_details(newcell, refcell) -> None:
    """
    Logs the available charge candidates for unique species and validates
    consistency between species objects and the unique_indices mapping.
    """

    logger.debug("#### Charge State & Index Validation ####")

    # --- 1. Log Charge Options for Unique Species ---
    if hasattr(refcell, "selected_cs") and refcell.selected_cs:
        for specie, options in zip(refcell.unique_species, refcell.selected_cs):
            logger.debug(
                "Unique Specie %s (Formula: %s):", specie.unique_index, specie.formula
            )
            logger.debug("  > Possible Charge Options: %s", options)
            # logger.debug("  > Full Possible States:     %s", specie.possible_cs)
    else:
        logger.warning("RefCell has no 'selected_cs' populated to display.")

    # --- 2. Validate Indices in Reference and Unit Cells ---
    _validate_species_indices(refcell, "Reference Cell")
    _validate_species_indices(newcell, "Unit Cell")


def _validate_species_indices(cell, label: str) -> None:
    """Helper to check consistency between species objects and cell mapping lists."""
    logger.debug("--- Validating %s Indices ---", label)

    if not hasattr(cell, "species_list") or not hasattr(cell, "unique_indices"):
        logger.warning(
            "Skipping validation: %s missing species list or indices.", label
        )
        return

    for i, (specie, mapped_idx) in enumerate(
        zip(cell.species_list, cell.unique_indices)
    ):
        # Check consistency: The index stored in the list must match the specie's internal ID
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
            # Verbose logging only if needed
            logger.debug(
                "\t[OK] %s (UniqueID: %s)", specie.formula, specie.unique_index
            )
