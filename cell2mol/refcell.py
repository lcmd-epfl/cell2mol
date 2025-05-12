import os
import sys
from ase.io import read
from contextlib import redirect_stdout
from cell2mol.classes import cell
from cell2mol.read_write import *
from cell2mol.cell_operations import frac2cart_fromparam
from cell2mol.other import handle_error
from cell2mol.connectivity import labels2formula
import time
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

VERSION = "2.0"
COV_FACTOR = 1.3
METAL_FACTOR = 1.0

def process_refcell(input_path, name, current_dir, debug=0):
    """
    Process the reference molecules from a CIF file and generate a reference cell object.
    Args:
        input_path (str): Path to the CIF file (downloaded from CSD).
        name (str): CSD refcode.
        current_dir (str): Current working directory.
        debug (int, optional): Debug level (default is 0).
    Returns:
        refcell (object): Reference cell object containing the reference molecules and their information.
    """
    
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    output_fname = os.path.join(current_dir, "cell2mol.out")

    with open(output_fname, "w") as output:
        with redirect_stdout(output):
            print(f"cell2mol version {VERSION}")
            print(f"INITIATING cell object from input path: {input_path}")
            print(f"Debug level: {debug}")

            structure = read(input_path)
            cell_vector = structure.cell.array
            cell_param = structure.cell.cellpar()      

            # Create the reference cell
            refcell = create_reference(input_path, name, cell_vector, cell_param, debug)
        
            for i, ref in enumerate(refcell.refmoleclist):
                if hasattr(ref, "totcharge_cif"):   
                    N = 0
                    for atom in ref.labels:
                        N += elemdatabase.elementnr[atom]
                    N -= ref.totcharge_cif
                    if N % 2 == 0:
                        spin = 1
                    else:
                        spin = 2
                    writexyz(current_dir, f"{name}_Ref_{i}_{ref.formula}_charge_{ref.totcharge_cif}_lowspin_{spin}.xyz", ref.labels, ref.coord, charge=ref.totcharge_cif, spin=spin)
                    print(f"Ref molecule {i} {ref.formula} total charge {ref.totcharge_cif} lowest spin multiplicity {spin}")
                else:
                    writexyz(current_dir, f"{name}_Ref_{i}_{ref.formula}.xyz", ref.labels, ref.coord, charge="", spin="")
                    print(f"Ref molecule {i} {ref.formula} without charge and spin information")

            if refcell.error_case == 0:
                get_unique_species_in_reference(refcell, debug) 
            else:
                print(f"Error occurred in processing reference cell: error case {refcell.error_case}")
            refcell.save(ref_cell_fname)
    
        # Print summary information
        summary_fname = os.path.join(current_dir, "reference_summary.out")
        with open(summary_fname, "w") as summary:
            with redirect_stdout(summary):
                print_refmoleclist(refcell)
                print_unique_species(refcell)
                print_possible_charges(refcell)

    # Print error case information
    error_fname = os.path.join(current_dir, f"reference_error_{refcell.error_case}.out")
    print_error_case(refcell, error_fname)

    return refcell

def create_reference (input_path, name, cell_vector, cell_param, debug):
    """Create the reference cell object."""
    
    tini = time.time()
    
    # Read the CIF file and extract the Wyckoff positions
    atom_site_labels, ref_labels, ref_fracs = get_wyckoff_positions (input_path)
    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)

    # Generate the reference cell object
    refcell = cell(name, ref_labels, ref_pos, ref_fracs, cell_vector, cell_param)
    refcell.set_atom_site_labels(atom_site_labels)
    refcell.set_subtype("reference")

    # Read the CIF file and extract bond information if exists
    geom_bond_cif, moiety_list_cif  = get_geom_bond (input_path)
    refcell.get_cif_bond_moiety(geom_bond_cif, moiety_list_cif)
    print(f"refcell.exist_cif_bond_moiety: {refcell.exist_cif_bond_moiety}")

    if refcell.exist_cif_bond_moiety:
        refcell.get_reference_molecules_from_moiety (ref_labels, ref_fracs, cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR, debug=debug)
        compare_with_CIF(input_path, refcell, debug=debug)
        if refcell.disagree_with_cif_formula:
            refcell.error_case = 9
        else :
            refcell.error_case = 0
    else:
        refcell.get_reference_molecules(ref_labels, ref_fracs, cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR, debug=debug)
        compare_with_CIF(input_path, refcell, debug=debug)
        if refcell.has_isolated_H :
            refcell.error_case = 1
        else : 
            refcell.error_case = 0

    if refcell.error_case == 0:
        refcell.check_missing_H(debug=debug)  
        refcell.assess_errors(mode="hydrogens") 
    
    tend = time.time()    

    if debug >= 1: print(f"\nReference molecules are generated. Total execution time: {tend - tini:.2f} seconds")
    
    return refcell

def compare_with_CIF (input_path, refcell, debug=0):
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
    formulas_from_cif = [labels2formula(cifformula_to_list(moiety['formula'])) for moiety in moiety_dicts]
    ratios_from_cif = [moiety['ratio'] for moiety in moiety_dicts]
    charges_from_cif = [moiety['charge'] for moiety in moiety_dicts]
    matches = find_closest_matches(formulas_from_refcell, formulas_from_cif)
    
    print(f"Formulas from CIF: {formulas_from_cif}")
    print(f"Ratios from CIF: {ratios_from_cif}")
    print(f"Charges from CIF: {charges_from_cif}")
    print(f"Formulas from refcell: {formulas_from_refcell}")

    disagree_with_cif = []
    for ref_idx, info in matches.items():
        ref = refcell.refmoleclist[ref_idx]

        if len(info['diff_dict']) == 0:
            print(f"{ref_idx=} {info['ref']}: Exact match found. {info['match']}")
            
            # Find the index of the matched formula in the CIF formulas list
            try:
                cif_idx = formulas_from_cif.index(info['match'])
                ref.totcharge_cif = charges_from_cif[cif_idx]

            except ValueError:
                print(f"\tFailed to Assign charge to refcell molecule {info['ref']} matched with CIF {info['match']}")

        else:
            print(f"{ref_idx=} {info['ref']}: Closest match found. {info['match']} with difference of {info['diff_dict']}")
            disagree_with_cif.append(ref_idx)
    
    if len(disagree_with_cif) > 0:
        print("Discrepancies found between refcell and CIF") 
        refcell.disagree_with_cif_formula = True
    else:
        print("No discrepancies found between formulas from refcell and CIF.")
        refcell.disagree_with_cif_formula = False

    return

def get_unique_species_in_reference (refcell, debug):
    """Processes the reference cell to obtain unique species and handle any errors."""
    
    tini = time.time()
    
    refcell.get_unique_species(debug=debug)
    
    if debug >= 1:
        print(f"Unique species: {[specie.formula for specie in refcell.unique_species]}")
        print(f"Species list: {[specie.formula for specie in refcell.species_list]}\n")

    refcell.get_selected_cs(debug=debug)
    refcell.assess_errors(mode="possible_charges")
    
    tend = time.time()    
    
    if debug >= 1: print(f"\nAssign possible charges of Reference molecules. Total execution time: {tend - tini:.2f} seconds")
    
    return

def print_error_case(refcell, error_fname):
    """Prints the error case to a file."""

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
                print("    - Missing elements in Reference Molecules compared to moieties reported in CIF")
            else :
                handle_error(refcell.error_case)
    return

if __name__ == "__main__":

    input = sys.argv[1]
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    process_refcell(input_path, name, current_dir, debug=1)
