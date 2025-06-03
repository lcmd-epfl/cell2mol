import os
import sys
from ase.io import read
from cell2mol.helper import parsing_arguments
from cell2mol.cif2info import cif_2_info
from cell2mol.classes import cell
from cell2mol.read_write import readinfo, prefiter_cif, print_output, writexyz, get_wyckoff_positions
from cell2mol.new_c2m_module import cell2mol
from cell2mol.other import handle_error
from cell2mol.cell_operations import frac2cart_fromparam
from cell2mol.new_charge_assignment import assign_charge_state_for_unique_species, balance_charge
from cell2mol.new_cell_reconstruction import modify_cov_factor_due_to_H, modify_cov_factor_due_to_possible_charges

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
            try:
                errorpath    = os.path.join(current_dir, "cif2cell.err")
                infopath     = os.path.join(current_dir, "{}.info".format(name))
                # if error exist : sys.exit(1)
                # Create .info file 
                cif_2_info(input_path, infopath, errorpath)
                # Checks errors in cif_2_info
                with open(errorpath, 'r') as err:
                    for line in err.readlines():
                        if "Error" in line: sys.exit(1)
            except:
                pass
        ## If the input is an .info file, then is used directly
        elif extension == ".info": infopath = input_path
        else:                      sys.exit(1)


    output = open(output_fname, "w")
    sys.stdout = output 

    version = "2.0"
    print(f"cell2mol version {version}")
    print(f"INITIATING cell object from input path: {input_path}") 
    print(f"Debug level: {debug}")  
    
    # Pre-filtering of the .cif file
    # if not prefiter_cif(input_path):
    #     output.close()
    #     sys.stdout = stdout        
    #     sys.exit(1)

    # Read cif file
    atoms = read(input_path)
    
    wrap_keywords = {
        'pbc': True,                  # Periodic boundary conditions
        'center': (0.5, 0.5, 0.5),    # Center the positions in the unit cell
        }
    
    cell_labels = atoms.get_chemical_symbols()
    cell_pos = atoms.get_positions(wrap=True, **wrap_keywords)
    cell_fracs = atoms.get_scaled_positions()
    cell_vector = atoms.cell.array
    cell_param = atoms.cell.cellpar()
    space_group = atoms.info['spacegroup']
    sym_ops = space_group.get_op()

    ##########################################
    ### PREPARES THE REFERENCE CELL OBJECT ###
    ##########################################
    cov_factor = 1.3
    # cov_factor -= 0.1
    print(f"{cov_factor=}")
    metal_factor = 1.0
    
    # Get reference molecules
    # labels, pos, ref_labels, ref_fracs, cellvec, cell_param = readinfo(infopath)
    # labels, pos, and cellvec will not be used
    ref_labels, ref_fracs = get_wyckoff_positions(input_path)
    ref_pos = frac2cart_fromparam(ref_fracs, cell_param)
    
    # Create reference cell object
    refcell = cell.from_positional(name, ref_labels, ref_pos, ref_fracs, cell_vector, cell_param)
    refcell.set_subtype("reference")
    refcell.get_reference_molecules(ref_labels, ref_fracs, cov_factor=cov_factor, debug=debug)

    refcell = modify_cov_factor_due_to_H(refcell, debug=debug)
    refcell.save(ref_cell_fname)
    
    if refcell.error_case == 0:
        refcell.get_unique_species(debug=debug) # Get unique_species, unique_indices, and species_list of the reference cell
        if debug >= 1:
            print(f"refcell.unique_species {[specie.formula for specie in refcell.unique_species]} {refcell.unique_indices=}")
            print(f"refcell.species_list {[specie.formula for specie in refcell.species_list]}\n")
        
        refcell = modify_cov_factor_due_to_possible_charges(refcell, debug=debug)
        refcell.get_selected_cs(debug=debug) # for the last time for unique species
        refcell.assess_errors(mode="possible_charges")
    # Save reference cell object
    refcell.save(ref_cell_fname)

    ##########################################
    # Define new cell object for the unit cell
    newcell = cell.from_positional(name, cell_labels, cell_pos, cell_fracs, cell_vector, cell_param)
    newcell.set_subtype("unitcell")
    
    if refcell.error_case != 0:
        pass
    else:
        reconstruction = True
        charge_assignment = False
        spin_assignment = False
        cov_factor = refcell.refmoleclist[0].cov_factor
        # Get reference molecules
        newcell.get_reference_molecules(refcell.labels, refcell.frac_coord, cov_factor=cov_factor, debug=-1)
        if not newcell.has_isolated_H:  
            newcell.check_missing_H(debug=-1)                                     

        print(f"ENTERING cell2mol with {debug=}")

        newcell = cell2mol(newcell, refcell, sym_ops, reconstruction, charge_assignment, spin_assignment, debug=debug)        
        newcell.assess_errors(mode="reconstruction")
    
        if newcell.error_case == 0 and reconstruction :
            reconstruction = False
            charge_assignment = True
            newcell = cell2mol(newcell, refcell, sym_ops, reconstruction, charge_assignment, spin_assignment, debug=debug)        
            newcell.assess_errors(mode="charge_assignment")

            if newcell.error_case == 0 and charge_assignment : 
                reconstruction = False
                charge_assignment = False
                spin_assignment = True
                newcell = cell2mol(newcell, refcell, sym_ops, reconstruction, charge_assignment, spin_assignment, debug=debug)

                final_charge_distribution, final_charges = balance_charge(newcell.unique_indices, refcell.unique_species, debug=debug)
                refcell.unique_species = assign_charge_state_for_unique_species(refcell.unique_species, final_charges[0], debug=debug)
                refcell.assign_charges_for_refcell(debug=debug)
                refcell.assign_spin(debug=debug)
                refcell.create_bonds(debug=debug)
            
                # Update reference cell object
                refcell.save(ref_cell_fname)
    
        # Save unit cell object
        newcell.save(cell_fname)

    output.close()
    sys.stdout = stdout
    

    # Summary
    surmmary = open(surmmary_fname, "w")
    sys.stdout = surmmary
    print(name)
    print("*** Reference molecules ***")
    print(refcell)
    print_output(refcell.refmoleclist)

    print("***Unit cell molecules ***")
    print(newcell)
    if newcell.moleclist is not None:
        print_output(newcell.moleclist)

    surmmary.close()
    sys.stdout = stdout

    # Error handling
    case = refcell.error_case
    error_fname = os.path.join(current_dir, f"refcell_error_{case}.out")
    error = open(error_fname, "w")
    sys.stdout = error
    handle_error(case)
    error.close()
    sys.stdout = stdout

    # Error handling
    if newcell.error_case is not None:
        case = newcell.error_case
        error_fname = os.path.join(current_dir, f"unitcell_error_{case}.out")
        error = open(error_fname, "w")
        sys.stdout = error
        handle_error(case)
        error.close()
        sys.stdout = stdout

