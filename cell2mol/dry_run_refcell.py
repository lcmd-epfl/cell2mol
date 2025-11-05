import os
import sys
from ase.io import read
from contextlib import redirect_stdout
from cell2mol.classes import Cell
from cell2mol.cell_operations import frac2cart_fromparam
from cell2mol.other import handle_error
from cell2mol.connectivity import labels2formula, get_alkali_alkaline_earth_metal_idxs
import time
from cell2mol.elementdata import ElementData
from cell2mol.read_write import (
    writexyz,
    get_wyckoff_positions,
    get_geom_bond,
    extract_chemical_name,
    extract_metal_oxidation_state,
    extract_moiety,
    cifformula_to_list,
    find_closest_matches,
    sum_formulas,
    compare_totals,
    get_cell_parameters,
    prefilter_cif,
)
from ase import io

elemdatabase = ElementData()

VERSION = "2.0"
COV_FACTOR = 1.0
METAL_FACTOR = 1.0


def extract_xyz_from_reference(
    input_path, name, current_dir, cif_bond_info=True, debug=0
):
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    cif_okay, error_message = prefilter_cif(input_path)
    if not cif_okay:
        print(f"[INFO] {name} CIF file is not suitable for processing {error_message}")
        sys.exit(0)
    structure = read(input_path)
    cell_labels, cell_pos, cell_fracs, cell_vector, cell_param, sym_ops = (
        get_cell_parameters(structure)
    )

    # structure.info["refcode"] = name
    # structure.info["cell_param"] = cell_param
    # io.write(f"{name}.xyz", structure)

    # Create the reference cell
    refcell = create_reference(
        input_path, name, cell_vector, cell_param, cif_bond_info, debug
    )

    for i, ref in enumerate(refcell.refmoleclist):
        if ref.totcharge_cif is not None:
            N = 0
            for atom in ref.labels:
                N += elemdatabase.elementnr[atom]
            N -= ref.totcharge_cif
            if N % 2 == 0:
                spin = 1
            else:
                spin = 2
            if ref.iscomplex:
                filename = f"{name}_Ref_{i}_Complex_{ref.formula}_charge_{ref.totcharge_cif}_lowspin_{spin}.xyz"
            else:
                filename = f"{name}_Ref_{i}_Molecule_{ref.formula}_charge_{ref.totcharge_cif}_lowspin_{spin}.xyz"
            writexyz(
                current_dir,
                filename,
                ref.labels,
                ref.coord,
                charge=ref.totcharge_cif,
                spin=spin,
            )
            print(
                f"Ref molecule {i} {ref.formula} total charge {ref.totcharge_cif} lowest spin multiplicity {spin}"
            )
        # if ref.iscomplex:
        #     for j, lig in enumerate(ref.ligands):
        #         coord_indices = []
        #         for gr in lig.groups:
        #             for atom in gr.atoms:
        #                 coord_indices.append(atom.get_parent_index("ligand"))
        #         writexyz(
        #             current_dir,
        #             f"{name}_Ref_{i}_Ligand_{j}_{lig.formula}.xyz",
        #             lig.labels,
        #             lig.coord,
        #             charge=None,
        #             spin=None,
        #             info=f"{coord_indices=}",
        #         )
        #         print(
        #             f"Ref molecule {i} {ref.formula} ligand {lig.formula} written to xyz file"
        #         )
    if refcell.error_case == 0:
        get_unique_species_in_reference(refcell, debug)
    else:
        print(
            f"Error occurred in processing reference cell: error case {refcell.error_case}"
        )
    refcell.save(ref_cell_fname)


def remove_disorder_atoms(atom_site_labels, ref_labels, ref_fracs, debug=0):
    substring_to_remove = "?"

    # Build new filtered lists
    new_atom_site_labels = []
    new_ref_labels = []
    new_ref_fracs = []

    for atom_site_label, ref_label, ref_frac in zip(
        atom_site_labels, ref_labels, ref_fracs
    ):
        if substring_to_remove not in atom_site_label:
            new_atom_site_labels.append(atom_site_label)
            new_ref_labels.append(ref_label)
            new_ref_fracs.append(ref_frac)
        else:
            if debug >= 1:
                print(f"Removing disorder atom: {atom_site_label}")

    # Optionally overwrite originals
    atom_site_labels = new_atom_site_labels
    ref_labels = new_ref_labels
    ref_fracs = new_ref_fracs

    return atom_site_labels, ref_labels, ref_fracs


def create_reference(input_path, name, cell_vector, cell_param, cif_bond_info, debug):
    """Create the reference cell object."""

    tini = time.time()

    # Read the CIF file and extract the Wyckoff positions
    atom_site_labels, ref_labels, ref_fracs = get_wyckoff_positions(input_path)
    atom_site_labels, ref_labels, ref_fracs = remove_disorder_atoms(
        atom_site_labels, ref_labels, ref_fracs, debug=debug
    )
    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)

    # Generate the reference cell object
    refcell = Cell.from_positional(
        name, ref_labels, ref_pos, ref_fracs, cell_vector, cell_param
    )
    refcell.set_atom_site_labels(atom_site_labels)
    refcell.set_subtype("reference")

    # Read the CIF file and extract bond information if exists
    geom_bond_cif, moiety_list_cif = get_geom_bond(input_path)
    refcell.get_cif_bond_moiety(cif_bond_info, geom_bond_cif, moiety_list_cif)
    print(f"refcell.exist_cif_bond_moiety: {refcell.exist_cif_bond_moiety}")

    if cif_bond_info:
        refcell.get_reference_molecules_from_moiety(
            ref_labels,
            ref_fracs,
            cov_factor=COV_FACTOR,
            metal_factor=METAL_FACTOR,
            debug=debug,
        )
    else:
        refcell.get_reference_molecules(
            ref_labels,
            ref_fracs,
            cov_factor=COV_FACTOR,
            metal_factor=METAL_FACTOR,
            debug=debug,
        )

    if refcell.refmoleclist == []:
        print("No reference molecules found in the CIF file.")
        return refcell

    compare_with_CIF(input_path, refcell, debug=debug)
    refcell.check_missing_H(debug=debug)
    refcell.assess_errors(mode="hydrogens")

    tend = time.time()

    if debug >= 1:
        print(
            f"\nReference molecules are generated. Total execution time: {tend - tini:.2f} seconds"
        )

    return refcell


def compare_with_CIF(input_path, refcell: Cell, debug=0):
    """Extract chemical name, metal oxidation state, and moiety information from the CIF file."""

    chemical_name = extract_chemical_name(input_path)
    reported_metal_os = extract_metal_oxidation_state(chemical_name)
    moiety_dicts = extract_moiety(input_path)

    refcell.chemical_name = chemical_name
    refcell.reported_metal_os = reported_metal_os
    refcell.moiety_dicts = moiety_dicts

    print(f"_chemical_name_systematic in CIF: {refcell.chemical_name}")
    print(f"Reported oxidation states in CIF: {refcell.reported_metal_os}")
    print(f"Moiety dictionaries: {refcell.moiety_dicts}")

    formulas_from_refcell = [ref.formula for ref in refcell.refmoleclist]

    if len(moiety_dicts) == 0:
        print("No _chemical_formula_moiety information found in the CIF file.")
        return

    formulas_from_cif = [
        labels2formula(cifformula_to_list(moiety["formula"])) for moiety in moiety_dicts
    ]
    ratios_from_cif = [moiety["ratio"] for moiety in moiety_dicts]
    charges_from_cif = [moiety["charge"] for moiety in moiety_dicts]
    matches = find_closest_matches(formulas_from_refcell, formulas_from_cif)

    print(f"Formulas from CIF: {formulas_from_cif}")
    print(f"Ratios from CIF: {ratios_from_cif}")
    print(f"Charges from CIF: {charges_from_cif}")
    print(f"Formulas from refcell: {formulas_from_refcell}")

    disagree_with_cif = []
    for ref_idx, info in matches.items():
        ref = refcell.refmoleclist[ref_idx]

        if len(info["diff_dict"]) == 0:
            print(f"{ref_idx=} {info['ref']}: Exact match found. {info['match']}")

            # Find the index of the matched formula in the CIF formulas list
            try:
                cif_idx = formulas_from_cif.index(info["match"])
                ref.totcharge_cif = charges_from_cif[cif_idx]

            except ValueError:
                print(
                    f"\tFailed to Assign charge to refcell molecule {info['ref']} matched with CIF {info['match']}"
                )

        else:
            # If there are differences, print them
            print(
                f"{ref_idx=} {info['ref']}: Closest match found. {info['match']} with difference of {info['diff_dict']}"
            )
            if (
                len(
                    get_alkali_alkaline_earth_metal_idxs(list(info["diff_dict"].keys()))
                )
                > 0
            ):
                print(
                    f"{ref_idx=} {info['ref']}: Discrepancy found due to covalent radius of alkali/alkaline earth metals."
                )
                print(
                    "This will cause errors in unit cell reconstruction. Set --cif_bond_info as True and re-run."
                )
            disagree_with_cif.append(ref_idx)

    if len(disagree_with_cif) > 0:
        print("Discrepancies found between refcell and CIF")
        refcell.disagree_with_cif_formula = True
        try:
            cif_totals = sum_formulas(formulas_from_cif, ratios_from_cif)
            if cif_totals is not None:
                print(f"Element totals from CIF: {cif_totals}")
                ref_totals = sum_formulas(formulas_from_refcell)
                df_compare, all_match = compare_totals(
                    cif_totals, ref_totals, atol=1e-9
                )
                if not all_match:
                    print(
                        f"Element totals differ between refcell and CIF:\n{df_compare}"
                    )
                    print("Possible causes:")
                    print(
                        "- Missing atoms in the crystal structure (mismatch with CIF moiety)."
                    )
                    print(
                        "- Different adjacency matrix in refcell changed connectivity."
                    )
        except:
            print(
                "Can not calculate element totals from CIF. This may be due to non-float ratios in moieties in CIF."
            )

        # if not all_match:
        #     print(f"Element totals differ between refcell and CIF:\n{df_compare}")
        #     print("Possible causes:")
        #     print("- Missing atoms in the crystal structure (mismatch with CIF moiety).")
        #     print("- Different adjacency matrix in refcell changed connectivity.")
        #     refcell.disagree_with_cif_formula = True
        # else:
        #     print("Element totals in refcell and CIF match.")
        #     refcell.disagree_with_cif_formula = False
    else:
        print("No discrepancies found between formulas from refcell and CIF.")
        refcell.disagree_with_cif_formula = False

    return


def get_unique_species_in_reference(refcell, debug):
    """Processes the reference cell to obtain unique species and species_list"""
    # tini = time.time()

    refcell.get_unique_species(debug=debug)

    if debug >= 1:
        print(
            f"Unique species: {[specie.formula for specie in refcell.unique_species]}"
        )
        print(f"Species list: {[specie.formula for specie in refcell.species_list]}\n")

    # refcell.get_selected_cs(debug=debug)
    # refcell.assess_errors(mode="possible_charges")

    # tend = time.time()

    # if debug >= 1: print(f"\nAssign possible charges of Reference molecules. Total execution time: {tend - tini:.2f} seconds")

    return


def print_error_case(error_case, error_fname):
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


if __name__ == "__main__":
    input = sys.argv[1]
    cif_bond_info = (
        sys.argv[2].strip().lower() == "true" if len(sys.argv) > 2 else False
    )
    cif_bond_info = True
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)
    print(f"Input file: {input_path}")
    print("CIF bond information:", cif_bond_info)
    extract_xyz_from_reference(input_path, name, current_dir, cif_bond_info, debug=1)
