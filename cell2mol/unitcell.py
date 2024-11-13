import os
import sys
import logging
from cell2mol.refcell import process_refcell
from contextlib import redirect_stdout
from ase.io import read
from cell2mol.classes import cell
from cell2mol.new_c2m_module import cell2mol
from cell2mol.new_charge_assignment import assign_charge_state_for_unique_species, balance_charge
from cell2mol.other import handle_error
import copy
# Constants
VERSION = "2.0"
COV_FACTOR = 1.3
METAL_FACTOR = 1.0

# Set up logging for debug information
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def process_unitcell(input_path, name, current_dir, debug=0):
    # Set up file paths
    cell_fname = os.path.join(current_dir, f"Cell_{name}.cell")
    ref_cell_fname = os.path.join(current_dir, f"Ref_Cell_{name}.cell")
    output_fname = os.path.join(current_dir, "cell2mol.out")
    
    # Process reference cell and update new cell with molecules and properties
    refcell = process_refcell(input_path, name, current_dir, debug=debug)
    
    if refcell.error_case == 0:
        # Redirect stdout to file for logging
        with open(output_fname, "a") as output, redirect_stdout(output):
            logging.info(f"cell2mol version {VERSION}")
            logging.info(f"Initializing cell object from input path: {input_path}")
            logging.info(f"Debug level: {debug}")
            # Read CIF file and initialize unit cell parameters
            structure = read(input_path)
            cell_labels, cell_pos, cell_fracs, cell_vector, cell_param, sym_ops = get_cell_parameters(structure) 

            # Create and process unit cell
            newcell = create_unitcell_object(name, cell_labels, cell_pos, cell_fracs, cell_vector, cell_param, "unitcell")

            perform_cell2mol(newcell, refcell, sym_ops, cell_fname, ref_cell_fname, debug)

            # Handle error cases for the unit cell
            if hasattr(newcell, 'error_case'):
                error_fname = os.path.join(current_dir, f"unitcell_error_{newcell.error_case}.out")
                with open(error_fname, "w") as error_output:
                    with redirect_stdout(error_output):
                        handle_error(newcell.error_case)
    else:
        logging.error("Error encountered while processing the reference cell")

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


def create_unitcell_object(name, labels, pos, fracs, vector, param, subtype):
    """Creates a cell object and sets its subtype."""
    newcell = cell(name, labels, pos, fracs, vector, param)
    newcell.get_subtype(subtype)
    return newcell


def perform_cell2mol(newcell, refcell, sym_ops, cell_fname, ref_cell_fname, debug):
    """Handles the reconstruction, charge assignment, and spin assignment for molecules."""
    cov_factor = refcell.refmoleclist[0].cov_factor if refcell.refmoleclist else COV_FACTOR

    # Get reference molecules for the new cell
    # newcell.get_reference_molecules(refcell.labels, refcell.frac_coord, cov_factor=cov_factor, debug=-1)
    # if not newcell.has_isolated_H:
    #     newcell.check_missing_H(debug=-1)
    newcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)

    newcell.has_isolated_H = refcell.has_isolated_H
    newcell.has_missing_H = refcell.has_missing_H
    newcell.error_get_poscharges = refcell.error_get_poscharges
    logging.info("Starting molecule reconstruction with cell2mol")
    
    # Step-by-step molecule reconstruction and error assessment
    mode = "reconstruction"
    cell2mol_mode(newcell, refcell, sym_ops, mode, debug)
    newcell.assess_errors(mode=mode)

    if newcell.error_case == 0 :
        mode = "charge_assignment"
        cell2mol_mode(newcell, refcell, sym_ops, mode, debug)
        newcell.assess_errors(mode=mode)

        if newcell.error_case == 0:        
            mode = "spin_assignment"
            cell2mol_mode(newcell, refcell, sym_ops, mode, debug)

            # Assign and balance charges
            final_charge_distribution, final_charges = balance_charge(newcell.unique_indices, refcell.unique_species, debug=debug)
            print(f"{final_charges=}")
            refcell.unique_species = assign_charge_state_for_unique_species(newcell.unique_species, final_charges[0], debug=2)
            for specie in refcell.unique_species:
                if specie.subtype == "metal":
                    print("refcell.unique_species", specie.formula, specie.charge)
                else:
                    print("refcell.unique_species", specie.formula, specie.totcharge)
            # Finalize refcell properties and save both cell objects
            refcell.assign_charges_for_refcell(debug=debug)
            refcell.assign_spin(debug=debug)
            refcell.create_bonds(debug=debug)
            refcell.save(ref_cell_fname)
    newcell.save(cell_fname)


def cell2mol_mode (newcell, refcell, sym_ops, mode, debug):
    """Applies cell2mol with specific reconstruction or assignment mode."""
    reconstruction = mode == "reconstruction"
    charge_assignment = mode == "charge_assignment"
    spin_assignment = mode == "spin_assignment"
    newcell = cell2mol(newcell, refcell, sym_ops, reconstruction, charge_assignment, spin_assignment, debug=debug)
    logging.info(f"Completed {mode} mode in cell2mol")


if __name__ == "__main__":
    input = sys.argv[1]
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    process_unitcell(input_path, name, current_dir, debug=1)
