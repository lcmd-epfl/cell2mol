#!/usr/bin/env python

import time
from cell2mol.other import handle_error
from cell2mol.new_cell_reconstruction import *
from cell2mol.new_charge_assignment import *

##################################################################################
################################## MAIN ##########################################
##################################################################################
def cell2mol(newcell: object, refcell: object, sym_ops, reconstruction: bool=True, 
             charge_assignment: bool=True, spin_assignment=True, debug: int=1) -> object:

    if reconstruction:
        if debug >= 1:
            print("#########################################")
            print("           Cell Reconstruction           ")
            print("#########################################")         
        tini = time.time()
        if not newcell.has_isolated_H and not newcell.has_missing_H:
            # Cell Reconstruction
            all_molecules, reconstructed_molecules = reconstuct(refcell, newcell, sym_ops, debug=debug)    
            all_molecules.extend(reconstructed_molecules)                                       
            tend = time.time()
            if newcell.error_get_fragments:     
                if debug >= 1: print(f"\nCell Reconstruction Failed. Total execution time: {tend - tini:.2f} seconds")
                return newcell
            elif newcell.error_reconstruction:  
                if debug >= 1: print(f"\nCell Reconstruction Failed. Total execution time: {tend - tini:.2f} seconds")
                return newcell
            else:
                if debug >= 1: print(f"\nCell Reconstruction Finished Normally. Total execution time: {tend - tini:.2f} seconds")
                
                # Get moleclist for the unit cell
                newcell = get_moleclist(newcell, refcell, all_molecules, debug=0)

                # Get unique indices for moleclist and species list in the unit cell using reference cell
                newcell.unique_species = copy.deepcopy(refcell.unique_species)
                newcell = get_unique_indices(newcell, refcell.species_list, debug=debug) 
                if debug >= 1: 
                    print(f"newcell.unique_species formula={[specie.formula for specie in newcell.unique_species]}")
                    print(f"newcell.species_list formula={[specie.formula for specie in newcell.species_list]}")
                    print(f"{newcell.unique_indices=}")
        else:
            return newcell

    if charge_assignment:
        tini = time.time()
        if debug >= 1:
            print("#########################################")
            print(" Get Unique Species and Balance Charges  ")
            print("#########################################")         
        if not newcell.error_reconstruction:

            if None in refcell.selected_cs :
                newcell.error_get_poscharges = True
            else:
                newcell.error_get_poscharges = False
                print_possible_and_selected_cs(newcell, refcell, debug=debug)

                # Find charge distribution for the unit cell                
                if debug >= 1: print(f"\nFind charge distribution for the unit cell")
                final_charge_distribution, final_charges = balance_charge(newcell.unique_indices, refcell.unique_species, debug=debug)
                if debug >= 1: print(f"{final_charge_distribution=}")
                if debug >= 1: print(f"{final_charges=}")

                # Assign charge for the unit cell and check charge neutrality
                newcell.assign_charges(debug=debug)
            tend = time.time()
            if   newcell.error_get_poscharges :   
                if debug >= 1: print(f"Charge Assignment Failed. Total execution time: {tend - tini:.2f} seconds")
                return newcell
            elif newcell.error_multiple_distrib :   
                if debug >= 1: print(f"Charge Assignment Failed. Total execution time: {tend - tini:.2f} seconds")
                return newcell
            elif newcell.error_empty_distrib :      
                if debug >= 1: print(f"Charge Assignment Failed. Total execution time: {tend - tini:.2f} seconds")
                return newcell
            else :
                
                if debug >= 1: print(f"Charge Assignment Finished Normally. Total execution time: {tend - tini:.2f} seconds")

                newcell.check_charge_neutrality(debug=debug)
                newcell.create_bonds(debug=debug)

                if newcell.error_create_bonds:      
                    if debug >= 1: print(f"Creating bonds Failed")
                    return newcell
                else:
                    if debug >= 1: print("Creating bonds Finished Normally")
        else:
            return newcell  

    if spin_assignment:
        if debug >= 1:
            print("#########################################")
            print("              Spin Assignment            ")
            print("#########################################")  
        tini = time.time()
        if not newcell.error_get_poscharges and not newcell.error_multiple_distrib and not newcell.error_empty_distrib:
            newcell.assign_spin(debug=debug)
            tend = time.time()
            if debug >= 1: print(f"\nTotal execution time for Spin Assignment: {tend - tini:.2f} seconds")
        else:
            return newcell

    return newcell




