import os
import sys
import logging
from cell2mol.refcell import process_refcell
from contextlib import redirect_stdout
from ase.io import read
from cell2mol.classes import cell
from cell2mol.final_c2m_module import cell2mol_mode
from cell2mol.other import handle_error
from cell2mol.read_write import print_refmoleclist, print_unique_species, print_moleclist
import copy

VERSION = "2.0"
COV_FACTOR = 1.3
METAL_FACTOR = 1.0

# Set up logging for debug information
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def process_unitcell(input_path, name, current_dir, cif_bond_info, debug=0):
    # Set up file paths
    cell_fname = os.path.join(current_dir, f"Cell_{name}.cell")
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    output_fname = os.path.join(current_dir, "cell2mol.out")
    
    # Process reference cell 
    logging.info("Starting the cell2mol process for the reference cell")
    refcell = process_refcell(input_path, name, current_dir, cif_bond_info, debug=debug)

    if refcell.error_case != 0:
        logging.error("Error encountered while processing the reference cell")
        return refcell
    elif refcell.disagree_with_cif_formula is not None and refcell.disagree_with_cif_formula == True:
        logging.error("Discrepancies found between refcell and CIF. This will cause errors in the charge prediction, so STOPPING pre-emptively.")
        return refcell        
    else:
        # Redirect stdout to file for logging
        with open(output_fname, "a") as output, redirect_stdout(output):
            logging.info(f"cell2mol version {VERSION}")
            logging.info(f"Initializing cell object from input path: {input_path}")
            logging.info(f"Debug level: {debug}")

            # Read CIF file and initialize unit cell parameters
            structure = read(input_path)
            cell_labels, cell_pos, cell_fracs, cell_vector, cell_param, sym_ops = get_cell_parameters(structure) 

            # Create and process unit cell
            newcell = cell.from_positional(name, cell_labels, cell_pos, cell_fracs, cell_vector, cell_param)
            newcell.set_subtype("unitcell")
            perform_cell2mol(newcell, refcell, sym_ops, cell_fname, ref_cell_fname, debug)
            refcell.save(ref_cell_fname)
            newcell.save(cell_fname)

            # Print summary information for the unit cell
            summary_fname = os.path.join(current_dir, "unitcell_summary.out")
            with open(summary_fname, "w") as summary:
                with redirect_stdout(summary):
                    print(name)
                    print_refmoleclist(newcell)
                    print_unique_species(newcell)
                    print_moleclist(newcell)

            summary_fname_ref = os.path.join(current_dir, "reference_summary.out")
            with open(summary_fname_ref, "a") as summary_ref:
                with redirect_stdout(summary_ref):
                    print("************ After charge assignment of unit cell ************")
                    print_refmoleclist(refcell)   

            # Handle error cases for the unit cell
            if hasattr(newcell, 'error_case'):
                error_fname = os.path.join(current_dir, f"unitcell_error_{newcell.error_case}.out")
                with open(error_fname, "w") as error_output:
                    with redirect_stdout(error_output):
                        handle_error(newcell.error_case)
                        
            if hasattr(refcell, 'error_case') and refcell.error_case != 0:
                error_fname_ref = os.path.join(current_dir, f"reference_error_{refcell.error_case}.out")
                with open(error_fname_ref, "w") as error_output_ref:
                    with redirect_stdout(error_output_ref):
                        handle_error(refcell.error_case)

            return newcell

def get_cell_parameters(structure):
    """Extracts cell parameters and symmetry operations from structure."""
    wrap_keywords = {'pbc': True, 'center': (0.5, 0.5, 0.5)}
    cell_labels = []
    for l, n, m in zip(structure.get_chemical_symbols(), structure.get_atomic_numbers(), structure.get_masses()):
        if n == 1 and (m > 2 or  m== 2.01355) : # Deuterium
            cell_labels.append("D")
        else:
            cell_labels.append(l)

    cell_pos = structure.get_positions(wrap=True, **wrap_keywords)
    cell_fracs = structure.get_scaled_positions()
    cell_vector = structure.cell.array
    cell_param = structure.cell.cellpar()
    space_group = structure.info.get('spacegroup')
    sym_ops = space_group.get_op() if space_group else None
    
    return cell_labels, cell_pos, cell_fracs, cell_vector, cell_param, sym_ops


def perform_cell2mol(newcell, refcell, sym_ops, cell_fname, ref_cell_fname, debug):
    """Handles the reconstruction, charge assignment, and spin assignment for molecules."""
    cov_factor = refcell.refmoleclist[0].cov_factor if refcell.refmoleclist else COV_FACTOR

    # Copy reference molecules from refcell
    newcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)

    newcell.has_isolated_H = refcell.has_isolated_H
    newcell.has_missing_H = refcell.has_missing_H
    newcell.error_get_poscharges = refcell.error_get_poscharges
    logging.info("Starting the cell2mol process for the unit cell")

    # Step-by-step molecule reconstruction and error assessment
    mode = "reconstruction"
    cell2mol_mode(newcell, refcell, sym_ops, mode, debug)
    newcell.assess_errors(mode=mode)

    if newcell.error_case == 0 :
        mode = "charge_assignment"
        cell2mol_mode(newcell, refcell, sym_ops, mode, debug)
        newcell.assess_errors(mode=mode)
        
if __name__ == "__main__":
    input = sys.argv[1]
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    process_unitcell(input_path, name, current_dir, debug=1)
