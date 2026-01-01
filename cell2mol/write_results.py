#!/usr/bin/env python

import sys
import re
import os
import traceback
from collections import Counter
from ase.io import read
from contextlib import redirect_stdout
from cell2mol.elementdata import ElementData
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


#######################
def exit_with_error_input(message):
    """Logs the error message to a file and exits the program."""
    error_log_path = os.path.join(os.getcwd(), "error_input.out")
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error: {message}\n")
    sys.exit(message)


#######################
def exit_with_error_exception(e):
    """Logs the error details to a file and exits the program."""
    error_details = traceback.format_exc()
    error_log_path = os.path.join(os.getcwd(), f"error_{type(e).__name__}.out")

    # Write the full error details to the log file
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error message: {type(e).__name__} - {str(e)}\n")
        error_log.write(f"Error details:\n{error_details}")

    # Print the error details to the console
    logger.error("An error occurred. Details have been logged to  %s", error_log_path)
    logger.error("Error details:\n%s", error_details)

    sys.exit(e)


##########################
def compare_formula_xyz_vs_cif(xyzfile: str, formula_str_from_cif: str) -> dict:
    """
    Compare the elements from CIF with the elements in the XYZ file.
    """
    element_pattern = r"([A-Z][a-z]*)(\d*)"
    parsed_formula_cif = dict()
    for element, count in re.findall(element_pattern, formula_str_from_cif):
        parsed_formula_cif[element] = int(count) if count else 1
    mol = read(xyzfile)
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


##############
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


################################
def printxyz(labels, pos):
    print(len(labels))
    print("")
    for idx, l in enumerate(labels):
        print("%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]))


################################
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


################################
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
                print(
                    f"Ref molecule {i} {ref.formula} total charge {ref.totcharge_cif} lowest spin multiplicity {spin}"
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
                print(
                    f"Ref molecule {i} {ref.formula} without charge and spin information"
                )


################################
def print_error_case_reference(error_case, error_fname):
    """Prints the error case to a file."""

    with open(error_fname, "w") as error_output:
        with redirect_stdout(error_output):
            if error_case == 2 or error_case == 3 or error_case == 4:
                handle_error(2)
                if error_case == 2:
                    print("    - Missing Hydrogens in Water Molecules")
                elif error_case == 3:
                    print("    - Missing Hydrogens in Coordinated Water Molecules")
                elif error_case == 4:
                    print("    - Missing Hydrogens in Carbon Atoms")
            elif error_case == 9:
                handle_error(9)
                print(
                    "    - Missing elements in Reference Molecules compared to moieties reported in CIF"
                )
            elif error_case == "X":
                print("    - Empty Reference Molecules list")
            else:
                handle_error(error_case)
    return


def get_reference_error_message(error_case):
    """
    Return an error message for a given error case.
    If error_fname exists, print the message in the file instead.
    """

    if error_case == 0:
        return "No errors found"

    elif error_case == 2:
        return "Missing Hydrogens in Water Molecules"

    elif error_case == 3:
        return "Missing Hydrogens in Coordinated Water Molecules"

    elif error_case == 4:
        return "Missing Hydrogens in Coordinated Carbon Atoms"

    elif error_case == 8:
        return "Error in Creating Bonds"

    elif error_case == "X":
        return "Empty Reference Molecules list"

    else:
        return f"Unhandled error case: {error_case}"


######################################################
def write_refmoleclist(cell, file):
    """
    Write reference molecule information to a file-like object.

    Args:
        cell: Reference Cell object.
        file: File-like object opened for writing.
    """

    for i, ref in enumerate(cell.refmoleclist):
        ref_info = f"Reference Molecule {i}: {ref.formula} "

        if ref.iscomplex:
            ref_info += "(TM Complex) "
        if ref.has_IA_IIA:
            ref_info += "(Complex with Alkali or Alkaline metals) "
        if ref.has_post_transition_metal:
            ref_info += "(Complex with Post-Transition metals) "
        if (
            not ref.iscomplex
            and not ref.has_IA_IIA
            and not ref.has_post_transition_metal
        ):
            ref_info += "(Non-complex)"

        if ref.totcharge is not None:
            ref_info += f" totcharge={ref.totcharge}"
        if ref.totcharge_cif is not None:
            ref_info += f" totcharge_cif={ref.totcharge_cif}"
        if ref.smiles is not None:
            ref_info += f" smiles={ref.smiles}"

        print(ref_info, file=file)

        # --------------------
        # Metals
        # --------------------
        if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
            for met in ref.metals:
                met_info = (
                    f"\t{met.formula} ({met.subtype}) "
                    f"atom_site_label={met.atom_site_label}"
                )

                if met.charge is not None:
                    met_info += f" metal_OS={met.charge}"
                elif met.possible_cs is not None:
                    met_info += f" metal_possible_OS={met.possible_cs}"

                print(met_info, file=file)

                if met.coord_sphere_formula is not None:
                    print(
                        f"\t|--coord_sphere_formula={met.coord_sphere_formula}",
                        file=file,
                    )

                if all(
                    hasattr(met, attr)
                    for attr in ("coord_nr", "coord_geometry", "geom_deviation")
                ):
                    print(
                        f"\t|--coord_nr={met.coord_nr} "
                        f"coord_geometry={met.coord_geometry} "
                        f"geom_deviation={met.geom_deviation}",
                        file=file,
                    )

                if all(
                    hasattr(met, attr)
                    for attr in (
                        "coord_nr_with_metal_bonds",
                        "coord_geometry_with_metal_bonds",
                        "geom_deviation_with_metal_bonds",
                        "metals",
                    )
                ):
                    bonded_metals = [m.label for m in met.metals]
                    if bonded_metals:
                        print(
                            f"\t|--bonded metals={bonded_metals} "
                            f"coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} "
                            f"coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} "
                            f"geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}",
                            file=file,
                        )

        # --------------------
        # Ligands
        # --------------------
        for lig in ref.ligands:
            lig_info = f"\t{lig.formula} ({lig.subtype})"

            for attr in ("smiles", "denticity", "totcharge"):
                if hasattr(lig, attr):
                    lig_info += f" {attr}={getattr(lig, attr)}"

            if lig.possible_cs is not None and lig.totcharge is None:
                lig_info += (
                    " lig.possible_cs Exists"
                    if lig.possible_cs
                    else " lig.possible_cs Does not exist"
                )

            print(lig_info, file=file)

            if lig.groups:
                for group in lig.groups:
                    group_info = f"\t|--(group) {group.labels}"

                    if hasattr(group, "denticity"):
                        group_info += f" denticity={group.denticity}"

                    if getattr(group, "is_haptic", False):
                        group_info += f" haptic_type={group.haptic_type}"

                    if group.metals:
                        group_info += (
                            f" connected_metals="
                            f"{[m.atom_site_label for m in group.metals]}"
                        )

                    print(group_info, file=file)


######################################################
def write_unique_species(cell, file):
    """
    Write unique species information to a file-like object.

    Args:
        cell: Cell object containing unique_species.
        file: File-like object opened for writing.
    """

    unique_species = getattr(cell, "unique_species", None)
    if not unique_species:
        print("\nNo unique species found in the cell object.", file=file)
        return

    print(f"\nUnique Species in {cell.subtype}:", file=file)

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


######################################################
def print_refmoleclist(cell):
    for i, ref in enumerate(cell.refmoleclist):
        ref_info = f"Reference Molecule {i}: {ref.formula} "
        if ref.iscomplex:
            ref_info += "(TM Complex) "
        if ref.has_IA_IIA:
            ref_info += "(Complex with Alkali or Alkaline metals) "
        if ref.has_post_transition_metal:
            ref_info += "(Complex with Post-Transition metals) "
        if (
            not ref.iscomplex
            and not ref.has_IA_IIA
            and not ref.has_post_transition_metal
        ):
            ref_info += "(Non-complex)"
        ref_info += "\n"
        if ref.totcharge is not None:
            ref_info += f"totcharge={ref.totcharge} "
        if ref.totcharge_cif is not None:
            ref_info += f"totcharge_cif={ref.totcharge_cif} "
        if ref.smiles is not None:
            ref_info += f"smiles={ref.smiles}"
        print(ref_info)

        if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
            for met in ref.metals:
                met_info = f"\t{met.formula} ({met.subtype}) atom_site_label={met.atom_site_label}"
                if met.charge is not None:
                    met_info += f" metal_OS={met.charge}"
                elif met.possible_cs is not None:
                    met_info += f" metal_possible_OS={met.possible_cs}"
                print(met_info)

                if met.coord_sphere_formula is not None:
                    print(
                        f"\t|--Coordination information coord_sphere_formula={met.coord_sphere_formula}"
                    )

                if all(
                    hasattr(met, attr)
                    for attr in ["coord_nr", "coord_geometry", "geom_deviation"]
                ):
                    print(
                        f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}"
                    )

                if all(
                    hasattr(met, attr)
                    for attr in [
                        "coord_nr_with_metal_bonds",
                        "coord_geometry_with_metal_bonds",
                        "geom_deviation_with_metal_bonds",
                        "metals",
                    ]
                ):
                    bonded_metals = [m.label for m in met.metals]
                    if len(bonded_metals) > 0:
                        print(
                            f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}"
                        )

            for lig in ref.ligands:
                lig_info = f"\t{lig.formula} ({lig.subtype})"
                for attr in ["smiles", "denticity", "totcharge"]:
                    if hasattr(lig, attr):
                        lig_info += f" {attr}={getattr(lig, attr)}"

                if lig.possible_cs is not None and lig.totcharge is None:
                    if len(lig.possible_cs) > 0:
                        lig_info += " lig.possible_cs Exists"
                    else:
                        lig_info += " lig.possible_cs Does not exist"

                print(lig_info)
                if lig.groups is not None:
                    for group in lig.groups:
                        group_info = f"\t|--(group) {group.labels}"
                        for attr in ["denticity"]:
                            if hasattr(group, attr):
                                group_info += f" {attr}={getattr(group, attr)}"
                        for attr in ["is_haptic", "haptic_type"]:
                            if hasattr(group, attr):
                                if group.is_haptic:
                                    group_info += f" {attr}={getattr(group, attr)}"
                        if group.metals is not None:
                            group_info += f" connected_metals={[m.atom_site_label for m in group.metals]} "
                        # if group.closest_metal is not None:
                        #     group_info += f" closest_metal.label={group.closest_metal.label}"
                        print(group_info)


######################################################
def print_unique_species(cell):
    unique_species = getattr(cell, "unique_species", None)
    if not unique_species:
        print("\nNo unique species found in the cell object.")
        return

    print(f"\nUnique Species in {cell.subtype}:")
    for specie in unique_species:
        parts = [f"{specie.unique_index=}", f"{specie.formula}", f"({specie.subtype})"]

        if specie.subtype == "metal":
            parts.append(f"{specie.coord_sphere_formula=}")
            if getattr(specie, "charge", None) is not None:
                parts.append(f"{specie.charge=}")
        elif specie.subtype == "ligand":
            parts.append(f"{specie.denticity=}")
            if specie.is_haptic:
                parts.append(f"{specie.haptic_type=}")
            if getattr(specie, "smiles", None) is not None:
                parts.append(f"{specie.smiles=}")
            if getattr(specie, "totcharge", None) is not None:
                parts.append(f"{specie.totcharge=}")
            if getattr(specie, "groups", None) is not None:
                parts.append(f"groups={[group.formula for group in specie.groups]}")
        else:
            if getattr(specie, "smiles", None) is not None:
                parts.append(f"{specie.smiles=}")
            if getattr(specie, "totcharge", None) is not None:
                parts.append(f"{specie.totcharge=}")

        print("\t" + " ".join(parts))


######################################################
def print_possible_charges(cell, debug=0):
    """
    Print the possible charges for each species in the cell object.
    """
    if cell.species_list is not None:
        print(f"\nPossible charges of species in {cell.subtype}:")
        for specie in cell.species_list:
            if specie.possible_cs is not None:
                if specie.subtype == "metal":
                    print(
                        f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} {specie.possible_cs=}"
                    )
                else:
                    print(
                        f"\t{specie.unique_index=} {specie.formula} ({specie.subtype})\n\t{specie.possible_cs=}"
                    )
            else:
                if specie.subtype == "metal":
                    print(
                        f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} No possible cs"
                    )
                else:
                    print(
                        f"\t{specie.unique_index=} {specie.formula}, {specie.subtype}  No possible cs"
                    )  # [p.subtype for p in specie.parents])
    else:
        print("\nNo species list found in the cell object.")
    print("")


######################################################
def print_moleclist(cell):
    if cell.moleclist is not None:
        print(f"\nMolecules in {cell.subtype}:")
        for i, mol in enumerate(cell.moleclist):
            if mol.totcharge is not None:
                if mol.iscomplex:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (TM complex)"
                    )
                elif mol.has_IA_IIA:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex with Alkali or Alkaline metals)"
                    )
                elif mol.has_post_transition_metal:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex with Post-Transition metals)"
                    )
                else:
                    if mol.smiles is not None:
                        print(
                            f"Unitcell Molecule {i} : {mol.formula} {mol.totcharge=} (Non-complex) {mol.smiles=}"
                        )
                    else:
                        print(
                            f"Unitcell Molecule {i} : {mol.formula} {mol.totcharge=} (Non-complex)"
                        )
            else:
                if mol.iscomplex:
                    print(f"Unitcell Molecule {i}: {mol.formula} (TM complex)")
                elif mol.has_IA_IIA:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} (Complex with Alkali or Alkaline metals)"
                    )
                elif mol.has_post_transition_metal:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} (Complex with Post-Transition metals)"
                    )
                else:
                    print(f"Unitcell Molecule {i} : {mol.formula} (Non-complex)")

            if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
                for met in mol.metals:
                    met_info = f"\t{met.formula} ({met.subtype}) atom_site_label={met.atom_site_label}"

                    if met.charge is not None:
                        met_info += f" metal_OS={met.charge}"
                    elif met.possible_cs is not None:
                        met_info += f" metal_possible_OS={met.possible_cs}"
                    print(met_info)

                    for attr in ["coord_sphere_formula"]:
                        if hasattr(met, attr):
                            print(
                                f"\t|--Coordination information {attr}={getattr(met, attr)}"
                            )

                    if all(
                        hasattr(met, attr)
                        for attr in ["coord_nr", "coord_geometry", "geom_deviation"]
                    ):
                        print(
                            f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}"
                        )

                    if all(
                        hasattr(met, attr)
                        for attr in [
                            "coord_nr_with_metal_bonds",
                            "coord_geometry_with_metal_bonds",
                            "geom_deviation_with_metal_bonds",
                            "metals",
                        ]
                    ):
                        bonded_metals = [m.label for m in met.metals]
                        if len(bonded_metals) > 0:
                            print(
                                f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}"
                            )
                print("")

                for lig in mol.ligands:
                    lig_info = f"\t{lig.formula} ({lig.subtype})"
                    for attr in ["smiles", "denticity", "totcharge"]:
                        if hasattr(lig, attr):
                            lig_info += f" {attr}={getattr(lig, attr)}"
                    print(lig_info)
                    if lig.groups is not None:
                        for group in lig.groups:
                            group_info = f"\t|--(group){group.labels}"
                            for attr in ["denticity"]:
                                if hasattr(group, attr):
                                    group_info += f" {attr}={getattr(group, attr)}"
                            for attr in ["is_haptic", "haptic_type"]:
                                if hasattr(group, attr):
                                    if group.is_haptic:
                                        group_info += f" {attr}={getattr(group, attr)}"
                            if group.metals is not None:
                                group_info += f" connected_metals={[m.atom_site_label for m in group.metals]}"
                            # if group.closest_metal is not None:
                            #     group_info += f" closest_metal.label={group.closest_metal.label}"
                            print(group_info)

                            # Optional: print group-metals connectivity
                            # for met in group.metals:
                            #     print(f"\t|--(group.metals){met.label} {met.mconnec=}")
    else:
        print("\nNo molecules found in the cell object.")


######################################################
def print_molecule(mol):
    if mol is not None:
        print("\nMolecule:")
        if mol.totcharge is not None:
            if mol.iscomplex:
                print(f"{mol.formula} {mol.totcharge=} (TM complex)")
            elif mol.has_IA_IIA:
                print(
                    f"{mol.formula} {mol.totcharge=} (Complex with Alkali or Alkaline metals)"
                )
            elif mol.has_post_transition_metal:
                print(
                    f"{mol.formula} {mol.totcharge=} (Complex with Post-Transition metals)"
                )
            else:
                if mol.smiles is not None:
                    print(f"{mol.formula} {mol.totcharge=} (Non-complex) {mol.smiles=}")
                else:
                    print(f"{mol.formula} {mol.totcharge=} (Non-complex)")
        else:
            if mol.iscomplex:
                print(f"{mol.formula} (TM complex)")
            elif mol.has_IA_IIA:
                print(f"{mol.formula} (Complex with Alkali or Alkaline metals)")
            elif mol.has_post_transition_metal:
                print(f"{mol.formula} (Complex with Post-Transition metals)")
            else:
                print(f"{mol.formula} (Non-complex)")

        if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
            for met in mol.metals:
                met_info = f"\t{met.formula} ({met.subtype}) atom_site_label={met.atom_site_label}"

                if met.charge is not None:
                    met_info += f" metal_OS={met.charge}"
                elif met.possible_cs is not None:
                    met_info += f" metal_possible_OS={met.possible_cs}"
                print(met_info)

                for attr in ["coord_sphere_formula"]:
                    if hasattr(met, attr):
                        print(
                            f"\t|--Coordination information {attr}={getattr(met, attr)}"
                        )

                if all(
                    hasattr(met, attr)
                    for attr in ["coord_nr", "coord_geometry", "geom_deviation"]
                ):
                    print(
                        f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}"
                    )

                if all(
                    hasattr(met, attr)
                    for attr in [
                        "coord_nr_with_metal_bonds",
                        "coord_geometry_with_metal_bonds",
                        "geom_deviation_with_metal_bonds",
                        "metals",
                    ]
                ):
                    bonded_metals = [m.label for m in met.metals]
                    if len(bonded_metals) > 0:
                        print(
                            f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}"
                        )
            print("")

            for lig in mol.ligands:
                lig_info = f"\t{lig.formula} ({lig.subtype})"
                for attr in ["smiles", "denticity", "totcharge"]:
                    if hasattr(lig, attr):
                        lig_info += f" {attr}={getattr(lig, attr)}"
                print(lig_info)
                if lig.groups is not None:
                    for group in lig.groups:
                        group_info = f"\t|--(group){group.labels}"
                        for attr in ["denticity"]:
                            if hasattr(group, attr):
                                group_info += f" {attr}={getattr(group, attr)}"
                        for attr in ["is_haptic", "haptic_type"]:
                            if hasattr(group, attr):
                                if group.is_haptic:
                                    group_info += f" {attr}={getattr(group, attr)}"
                        if group.metals is not None:
                            group_info += f" connected_metals={[m.atom_site_label for m in group.metals]}"

                        print(group_info)
    else:
        print("\nNo molecule object.")


def handle_error(case: int):
    print(f"Cell2mol terminated with error number {case}. Message:")
    if case == 1:
        print(
            "The cell object has isolated H atoms in the reference molecules list. This typically indicates an error. STOPPING"
        )
    if case == 2:
        print(
            "We detected that H atoms are likely missing. This will cause errors in the charge prediction, so STOPPING pre-emptively."
        )
    if case == 3:
        print("We failed to get fragments. STOPPING pre-emptively.")
    if case == 4:
        print(
            "After reconstruction of the unit cell, we still detected some fragments. STOPPING pre-emptively."
        )
    if case == 5:
        print("Error in list of possible charges received for molecule or ligand")
    if case == 6:
        print("More than one valid possible charge distribution found")
    if case == 7:
        print("No valid possible charge distribution found")
    # if case == 8: print("Error while preparing molecules")
    if case == 8:
        print("Error while creating bonds for molecule or ligand")
    # if case == 9: print("The charge neutralization failed.")
    if case == 9:
        print(
            "Discrepancies found between refcell and CIF. This will cause errors in the charge prediction, so STOPPING pre-emptively."
        )

    if case == 0:
        print("No errors Found")
    # sys.exit(1)


def setup_logger(log_file: str | None = None):
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
