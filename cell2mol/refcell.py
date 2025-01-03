import os
import sys
from ase.io import read
from contextlib import redirect_stdout
from cell2mol.classes import cell
from cell2mol.read_write import get_wyckoff_positions
from cell2mol.cell_operations import frac2cart_fromparam
from cell2mol.new_cell_reconstruction import modify_cov_factor_due_to_H, modify_cov_factor_due_to_possible_charges
from cell2mol.other import handle_error
import time

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
    

    # Redirect stdout to the output file for logging
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
                pass
                # get_unique_species_in_reference(refcell, debug)
            else:
                print(f"Error occurred in processing reference cell: error case {refcell.error_case}")
            refcell.save(ref_cell_fname)
    
    error_fname = os.path.join(current_dir, f"reference_error_{refcell.error_case}.out")
    with open(error_fname, "w") as error_output:
        with redirect_stdout(error_output):
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
        refcell.check_missing_H(debug=debug)  
    refcell.assess_errors(mode="hydrogens")    
    tend = time.time()    
    if debug >= 1: print(f"\nReference molecules are generated. Total execution time: {tend - tini:.2f} seconds")
    return refcell

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
