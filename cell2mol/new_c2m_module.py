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
        tini = time.time()
        if not newcell.has_isolated_H and not newcell.has_missing_H:
            # Cell Reconstruction
            all_molecules, reconstructed_molecules = reconstuct(refcell, newcell, sym_ops, debug=debug)    
            all_molecules.extend(reconstructed_molecules)                                       

            if newcell.error_get_fragments:     return newcell
            elif newcell.error_reconstruction:  return newcell
            else:
                tend = time.time()
                if debug >= 1: print(f"\nCell Reconstruction Finished Normally. Total execution time: {tend - tini:.2f} seconds")
        else:
            return newcell

    if charge_assignment:
        tini = time.time()

        if not newcell.error_reconstruction:
            # Get moleclist for the unit cell
            newcell = get_moleclist(newcell, refcell, all_molecules, debug=debug)
            
            # Get unique species for the reference cell
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

                # Get unique indices for moleclist in the unit cell using reference cell
                newcell = get_unique_indices(newcell, refcell.species_list, debug=debug)
                print("newcell.unique_indices", newcell.unique_indices)
                for specie, idx in zip(newcell.species_list, newcell.unique_indices):
                    print(specie.formula, specie.unique_index, idx)
                newcell.unique_species = copy.deepcopy(refcell.unique_species)

                # Find charge distribution for the unit cell
                final_charge_distribution, final_charges = balance_charge(newcell.unique_indices, refcell.unique_species, debug=debug)
                print("final_charge_distribution", final_charge_distribution)
                print("final_charges", final_charges)

                # Assign charge for the unit cell and check charge neutrality
                newcell.assign_charges(debug=debug)
                newcell.assess_errors(mode="unit_cell")
                newcell.check_charge_neutrality(debug=debug)

            if   newcell.error_empty_poscharges :   return newcell
            elif newcell.error_multiple_distrib :   return newcell
            elif newcell.error_empty_distrib :      return newcell
            else :
                newcell.create_bonds(debug=debug)
                tend = time.time()
                if debug >= 1: print(f"Charge Assignment Finished Normally. Total execution time: {tend - tini:.2f} seconds")
        else:
            return newcell  

    if spin_assignment:
        tini = time.time()
        if not newcell.error_empty_poscharges and not newcell.error_multiple_distrib and not newcell.error_empty_distrib:
            newcell.assign_spin(debug=debug)
            tend = time.time()
            if debug >= 1: print(f"\nTotal execution time for Spin Assignment: {tend - tini:.2f} seconds")
        else:
            return newcell

    return newcell




