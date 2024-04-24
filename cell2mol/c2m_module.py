#!/usr/bin/env python

import time
from cell2mol.other import handle_error


##################################################################################
################################## MAIN ##########################################
##################################################################################
def cell2mol(newcell: object, reconstruction: bool=True, charge_assignment: bool=True, spin_assignment: bool=True, debug: int=1) -> object:

    if reconstruction: 
        tini = time.time()
        if not newcell.has_isolated_H and not newcell.has_missing_H:
            # Cell Reconstruction
            newcell.reconstruct(debug=debug)                                         

            if newcell.error_get_fragments:     return newcell
            elif newcell.error_reconstruction:  return newcell
            else:
                tend = time.time()
                if debug >= 1: print(f"\nCell Reconstruction Finished Normally. Total execution time: {tend - tini:.2f} seconds")

    if charge_assignment:
        tini = time.time()

        if not newcell.error_reconstruction:
            newcell.assign_charges(debug=debug)

            if   newcell.error_empty_poscharges :   return newcell
            elif newcell.error_multiple_distrib :   return newcell
            elif newcell.error_empty_distrib :      return newcell
            elif newcell.error_prepare_mols :       return newcell
            else :
                newcell.create_bonds(debug=debug) 
                tend = time.time()
                if debug >= 1: print(f"Charge Assignment Finished Normally. Total execution time: {tend - tini:.2f} seconds")
                # newcell.predict_metal_ox(debug=debug) # predict metal oxidation state using Random Forest model
        
    if spin_assignment:
        if not newcell.error_prepare_mols:
            tini = time.time()
            newcell.assign_spin(debug=debug)
            tend = time.time()
            if debug >= 1: print(f"\nTotal execution time for Spin Assignment: {tend - tini:.2f} seconds")

    return newcell
