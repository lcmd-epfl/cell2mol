import os
import sys
from cell2mol.helper import parsing_arguments
from cell2mol.cif2info import cif_2_info
from cell2mol.classes import cell
from cell2mol.read_write import readinfo, prefiter_cif, writexyz
from cell2mol.other import handle_error
from cell2mol.cell_operations import frac2cart_fromparam
from ase.io import read
from cell2mol.charge_assignment import balance_charge
from cell2mol.new_charge_assignment import print_output, compare_molecules, prepare_mol, get_unique_indices, assign_charge_state_for_unique_species
from ase import Atoms
from cell2mol.new_cell_reconstruction import *
import copy


if __name__ == "__main__" or __name__ == "cell2mol.new_c2m_driver":
    
    input, isverbose, isquiet = parsing_arguments()
    current_dir     = os.getcwd()
    input_path      = os.path.normpath(input)
    dir, file       = os.path.split(input_path)
    root, extension = os.path.splitext(file)
    root = root.split(".")
    name = root[0]

    stdout = sys.stdout
    stderr = sys.stderr

    # Filenames for output and cell object
    cell_fname   = os.path.join(current_dir, "Cell_{}.cell".format(name))
    ref_cell_fname = os.path.join(current_dir, "Ref_Cell_{}.cell".format(name))
    output_fname = os.path.join(current_dir, "cell2mol.out")
    surmmary_fname = os.path.join(current_dir, "surmmary.out")

    ##### Deals with the parsed arguments for verbosity ######
    if isverbose and not isquiet:       debug = 2
    elif isverbose and isquiet:         debug = 0
    elif not isverbose and isquiet:     debug = 0
    elif not isverbose and not isquiet: debug = 1

    ##### Deals with files ######
    if os.path.exists(input_path):    
        ## If the input is a .cif file, then it is converted to a .info file using cif_2_info from cif2cell
        if extension == ".cif":
            # Pre-filtering of the .cif file
            prefiter_cif(input_path)
            errorpath    = os.path.join(current_dir, "cif2cell.err")
            infopath     = os.path.join(current_dir, "{}.info".format(name))
            # if error exist : sys.exit(1)
            # Create .info file 
            cif_2_info(input_path, infopath, errorpath)
            # Checks errors in cif_2_info
            with open(errorpath, 'r') as err:
                for line in err.readlines():
                    if "Error" in line: sys.exit(1)

        ## If the input is an .info file, then is used directly
        elif extension == ".info": infopath = input_path
        else:                      sys.exit(1)

    output = open(output_fname, "w")
    sys.stdout = output 

    version = "2.0"
    print(f"cell2mol version {version}")
    print(f"INITIATING cell object from input path: {input_path}") 
    print(f"Debug level: {debug}")  

    # Read cif file
    atoms = read(input_path)
    cell_labels = atoms.get_chemical_symbols()
    cell_pos = atoms.positions
    cell_fracs = atoms.get_scaled_positions()
    cell_vector = atoms.cell.array
    # cell_parameters = atoms.cell.cellpar()
    space_group = atoms.info['spacegroup']
    sym_ops = space_group.get_op()

    ##########################################
    ### PREPARES THE REFERENCE CELL OBJECT ###
    ##########################################

    # Get reference molecules
    labels, pos, ref_labels, ref_fracs, cellvec, cell_param = readinfo(infopath)
    # labels, pos, and cellvec will not be used
    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)
    
    # Create reference cell object
    refcell = cell(name, ref_labels, ref_pos, ref_fracs, cell_vector, cell_param)
    refcell.get_subtype("reference")
    refcell.get_reference_molecules(ref_labels, ref_fracs, debug=debug)

    if not refcell.has_isolated_H:  
        refcell.check_missing_H(debug=debug)                                     
    refcell.assess_errors(mode="hydrogens")

    if refcell.error_case == 0:
        # Define new cell object for the unit cell
        newcell = cell(name, cell_labels, cell_pos, cell_fracs, cell_vector, cell_param)
        newcell.get_subtype("unit_cell")
        
        # Get reference molecules
        newcell.get_reference_molecules(refcell.labels, refcell.frac_coord, debug=debug)
        if not newcell.has_isolated_H:  
            newcell.check_missing_H(debug=debug)                                     
        newcell.assess_errors(mode="hydrogens")
        
        # if newcell.error_case == 0: # Omit this condition 
        #since newcell.error_case with mode="hydrogens" is same as refcell.error_case with mode="hydrogens"

        # Reconstruction of the unit cell
        reference = Atoms(symbols=refcell.labels, scaled_positions=refcell.frac_coord, cell=cell_vector, pbc=True)
        all_molecules, reconstructed_molecules = reconstuct(reference, newcell, sym_ops, debug=debug)    
        all_molecules.extend(reconstructed_molecules)

        if not newcell.error_reconstruction :
            # Get moleclist for the unit cell
            newcell = get_moleclist(newcell, refcell, all_molecules, debug=debug)
            refcell.get_unique_species(debug=debug)
            print("refcell.unique_species", [specie.formula for specie in refcell.unique_species], refcell.unique_indices)
            selected_cs = refcell.get_selected_cs(debug=debug)
            if selected_cs is None:
                newcell.error_empty_poscharges = True
            else:
                newcell.error_empty_poscharges = False
            print("selected_cs for reference cell")
            for specie, select in zip(refcell.unique_species, selected_cs):
                print(specie.possible_cs, select)
            for specie, idx in zip(refcell.species_list, refcell.unique_indices):
                print(specie.formula, specie.unique_index, idx)

            newcell = get_unique_indices(newcell, refcell.species_list, debug=debug)
            print("newcell.unique_indices", newcell.unique_indices)
            for specie, idx in zip(newcell.species_list, newcell.unique_indices):
                print(specie.formula, specie.unique_index, idx)
            
            newcell.unique_species = copy.deepcopy(refcell.unique_species)

            final_charge_distribution, final_charges = balance_charge(newcell.unique_indices, refcell.unique_species, debug=debug)
            # print("final_charge_distribution", final_charge_distribution)
            # print("final_charges", final_charges)
            newcell.assign_charges(debug=debug)
            newcell.assess_errors(mode="unit_cell")
            newcell.check_charge_neutrality(debug=debug)
            if newcell.error_case == 0 and newcell.is_neutral:
                newcell.assign_spin(debug=debug)
                newcell.create_bonds(debug=debug)
                
                refcell.unique_species = assign_charge_state_for_unique_species(refcell.unique_species, final_charges[0], debug=debug)
                refcell.assign_charges_for_refcell(debug=debug)
                refcell.assign_spin(debug=debug)
                refcell.create_bonds(debug=debug)
        # Save cell object
        
        newcell.save(cell_fname)
    
    # Save reference cell object
    refcell.save(ref_cell_fname)
    
    output.close()
    sys.stdout = stdout

    # Error handling
    case = refcell.error_case
    error_fname = os.path.join(current_dir, f"refcell_error_{case}.out")
    error = open(error_fname, "w")
    sys.stdout = error
    handle_error(case)
    error.close()
    sys.stdout = stdout
    
    # Summary
    surmmary = open(surmmary_fname, "w")
    sys.stdout = surmmary
    print("*** Reference molecules ***")
    print(refcell)
    print_output(refcell.refmoleclist)
    surmmary.close()
    sys.stdout = stdout


    if newcell.error_case == 0:
        # Error handling
        case = newcell.error_case
        error_fname = os.path.join(current_dir, f"unitcell_error_{case}.out")
        error = open(error_fname, "w")
        sys.stdout = error
        handle_error(case)
        error.close()
        sys.stdout = stdout

        # Summary
        surmmary = open(surmmary_fname, "a")
        sys.stdout = surmmary
        print("***Unit cell molecules ***")
        print(newcell)
        print_output(newcell.moleclist)
        surmmary.close()
        sys.stdout = stdout