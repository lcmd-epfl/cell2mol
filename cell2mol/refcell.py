import os
import sys
from ase.io import read
from contextlib import redirect_stdout
from cell2mol.classes import cell
from cell2mol.read_write import get_wyckoff_positions, print_refmoleclist, print_unique_species, extract_chemical_name, extract_metal_oxidation_state, extract_moiety, cifformula_to_list
from cell2mol.cell_operations import frac2cart_fromparam
from cell2mol.new_cell_reconstruction import modify_cov_factor_due_to_H, modify_cov_factor_due_to_possible_charges
from cell2mol.other import handle_error
from cell2mol.connectivity import labels2formula
import time
import numpy as np
from collections import Counter
import re

# Helper to convert string like "H13-C5-N2" to Counter {'H':13, 'C':5, 'N':2}
def parse_formula_string(formula_str):
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    return Counter({el: int(cnt) if cnt else 1 for el, cnt in tokens})

# Function to compute atom-wise difference between two formula strings
# def formula_diff(f1, f2):
#     c1 = parse_formula_string(f1)
#     c2 = parse_formula_string(f2)
#     all_elements = set(c1) | set(c2)
#     diff = sum(abs(c1[el] - c2[el]) for el in all_elements)
#     return diff

def formula_diff_dict(f1, f2):
    c1 = parse_formula_string(f1)
    c2 = parse_formula_string(f2)
    all_elements = set(c1) | set(c2)
    diff = {el: abs(c1[el] - c2[el]) for el in all_elements if c1[el] != c2[el]}
    return diff

# Main comparison logic
# def find_closest_matches(reference, target):
#     matches = {}
#     for ref in reference:
#         if ref in target:
#             matches[ref] = {'match': ref, 'diff': 0}
#         else:
#             # Find the closest match in target
#             diffs = [(tgt, formula_diff(ref, tgt)) for tgt in target]
#             best_match, min_diff = min(diffs, key=lambda x: x[1])
#             matches[ref] = {'match': best_match, 'diff': min_diff}
#     return matches


def find_closest_matches(reference, target):
    matches = {}
    for i, ref in enumerate(reference):
        if ref in target:
            matches[i] = {'ref': ref, 'match': ref, 'diff_dict': {}}
        else:
            diffs = [(tgt, formula_diff_dict(ref, tgt)) for tgt in target]
            # Select the one with the smallest total difference
            best_match, best_diff = min(diffs, key=lambda x: sum(x[1].values()))
            matches[i] = {'ref': ref, 'match': best_match, 'diff_dict': best_diff}
    return matches

# Constants
VERSION = "2.0"
COV_FACTOR = 1.3
METAL_FACTOR = 1.0

def process_refcell(input_path, name, current_dir, debug=0):
    # Set up filenames
    cell_fname = os.path.join(current_dir, f"Cell_{name}.cell")
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    output_fname = os.path.join(current_dir, "cell2mol.out")
    summary_fname = os.path.join(current_dir, "summary.out")
    with open(output_fname, "w") as output:
        with redirect_stdout(output):
            print(f"cell2mol version {VERSION}")
            print(f"INITIATING cell object from input path: {input_path}")
            print(f"Debug level: {debug}")

            # # Read .cif file
            structure = read(input_path)
            cell_vector = structure.cell.array
            cell_param = structure.cell.cellpar()      

            # Create the reference cell
            refcell = create_reference(input_path, name, cell_vector, cell_param, debug)

            # Finalize and save the reference cell object if no errors
            if refcell.error_case == 0:
                get_unique_species_in_reference(refcell, debug) 
                print_refmoleclist(refcell)
            else:
                print(f"Error occurred in processing reference cell: error case {refcell.error_case}")
            
            if hasattr(refcell, "unique_species"):
                print_unique_species(refcell)
            refcell.save(ref_cell_fname)   
    # if os.path.exists(ref_cell_fname):
    #     with open(output_fname, "a") as output:
    #         with redirect_stdout(output):
    #             print("=====================================================")
    #             print(f"cell2mol version {VERSION}")
    #             print(f"Reference cell file {ref_cell_fname} already exists. Skipping reference cell generation.")
    #             print(f"Debug level: {debug}")
    #             refcell = np.load(ref_cell_fname, allow_pickle=True)
    #             if refcell.error_case == 0:
    #                 get_unique_species_in_reference(refcell, debug) 
    #             else:
    #                 print(f"Error occurred in processing reference cell: error case {refcell.error_case}")   
    #             refcell.save(ref_cell_fname)
    # else:
    #     with open(output_fname, "w") as output:
    #         with redirect_stdout(output):
    #             print(f"cell2mol version {VERSION}")
    #             print(f"INITIATING cell object from input path: {input_path}")
    #             print(f"Debug level: {debug}")

    #             # # Read .cif file
    #             structure = read(input_path)
    #             cell_vector = structure.cell.array
    #             cell_param = structure.cell.cellpar()      

    #             # Create the reference cell
    #             refcell = create_reference(input_path, name, cell_vector, cell_param, debug)

    #             # Finalize and save the reference cell object if no errors
    #             if refcell.error_case == 0:
    #                 pass
    #             else:
    #                 print(f"Error occurred in processing reference cell: error case {refcell.error_case}")
    #             refcell.save(ref_cell_fname)
    
    error_fname = os.path.join(current_dir, f"reference_error_{refcell.error_case}.out")
    with open(error_fname, "w") as error_output:
        with redirect_stdout(error_output):
            if refcell.error_case == 2 or refcell.error_case == 3 or refcell.error_case == 4 :
                handle_error(2)
                if refcell.error_case == 2:
                    print("    - Missing Hydrogens in Water Molecules")
                elif refcell.error_case == 3:
                    print("    - Missing Hydrogens in Coordinated Water Molecules")
                elif refcell.error_case == 4:
                    print("    - Missing Hydrogens in Carbon Atoms")
            elif refcell.error_case == 9:
                handle_error(9) 
                print("    - Missing elements in Reference Molecules compared to CIF")
            else :
                handle_error(refcell.error_case)
    return refcell

def create_reference (input_path, name, cell_vector, cell_param, debug):
    """Create the reference cell object."""
    tini = time.time()
    ref_labels, ref_fracs = get_wyckoff_positions(input_path)
    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)

    refcell = cell(name, ref_labels, ref_pos, ref_fracs, cell_vector, cell_param)
    refcell.get_subtype("reference")
    refcell.get_reference_molecules(ref_labels, ref_fracs, cov_factor=COV_FACTOR, debug=debug)
    #refcell = modify_cov_factor_due_to_H(refcell, debug=debug)
    
    if not refcell.has_isolated_H:
        compare_with_CIF(input_path, refcell)
    refcell.assess_errors(mode="cif_formula") 

    if not refcell.disagree_with_cif_formula:  
        refcell.check_missing_H(debug=debug)  
        refcell.assess_errors(mode="hydrogens") 
       
    tend = time.time()    
    if debug >= 1: print(f"\nReference molecules are generated. Total execution time: {tend - tini:.2f} seconds")
    return refcell

def compare_with_CIF (input_path, refcell):
    """Extract chemical name, oxidation state, and moiety information from the CIF file."""

    chemical_name = extract_chemical_name(input_path)
    refcell.chemical_name = chemical_name
    reported_metal_os = extract_metal_oxidation_state(chemical_name)
    print(f"_chemical_name_systematic in CIF: {refcell.chemical_name}")
    refcell.reported_metal_os = reported_metal_os
    print(f"Reported oxidation states: {refcell.reported_metal_os}")
    moiety_dicts = extract_moiety(input_path)
    refcell.moiety_dicts = moiety_dicts
    print(f"Moiety dictionaries: {refcell.moiety_dicts}")
    
    formulas_from_cif = [labels2formula(cifformula_to_list(moiety['formula'])) for moiety in moiety_dicts]
    ratios_from_cif = [moiety['ratio'] for moiety in moiety_dicts]
    print(f"Formulas from CIF: {formulas_from_cif}")
    print(f"Ratios from CIF: {ratios_from_cif}")
    formulas_from_refcell = [ref.formula for ref in refcell.refmoleclist]
    print(f"Formulas from refcell: {formulas_from_refcell}")
    matches = find_closest_matches(formulas_from_refcell, formulas_from_cif)

    disagree_with_cif = []
    for ref_idx, info in matches.items():
        if len(info['diff_dict']) == 0:
            print(f"{ref_idx=} {info['ref']}: Exact match found. {info['match']}")
        else:
            print(f"{ref_idx=} {info['ref']}: Closest match found. {info['match']} with difference of {info['diff_dict']}")
            disagree_with_cif.append(ref_idx)
    
    if len(disagree_with_cif) > 0:
        print("Discrepancies found between refcell and CIF") #: {disagree_with_cif}")
        refcell.disagree_with_cif_formula = True
    else:
        print("No discrepancies found between formulas from refcell and CIF.")
        refcell.disagree_with_cif_formula = False


def get_unique_species_in_reference (refcell, debug):
    """Processes the reference cell to obtain unique species and handle any errors."""
    tini = time.time()
    refcell.get_unique_species(debug=debug)
    if debug >= 1:
        print(f"Unique species: {[specie.formula for specie in refcell.unique_species]}")
        print(f"Species list: {[specie.formula for specie in refcell.species_list]}\n")
    #refcell = modify_cov_factor_due_to_possible_charges(refcell, debug=debug)
    refcell.get_selected_cs(debug=debug)
    refcell.assess_errors(mode="possible_charges")
    if debug >= 2:
        print("Results of possible charges")
        for specie in refcell.species_list:
            # for specie in refcell.unique_species:
            if hasattr(specie, "possible_cs"):
                if specie.subtype == "metal":
                    print(f"{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} {specie.possible_cs=}") 
                else:
                    print(f"{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.possible_cs=}")
            else:
                if specie.subtype == "metal":
                    print(f"{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} No possible cs")
                else:
                    print(f"{specie.unique_index=} {specie.formula}, {specie.subtype}  No possible cs") #[p.subtype for p in specie.parents])    
    tend = time.time()    
    if debug >= 1: print(f"\nAssign possible charges of Reference molecules. Total execution time: {tend - tini:.2f} seconds")

# Run the main function
if __name__ == "__main__":

    input = sys.argv[1]
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    # Example usage, replace with actual arguments
    process_refcell(input_path, name, current_dir, debug=1)
