#!/usr/bin/env python

import time
from cell2mol.other import handle_error


##################################################################################
################################## MAIN ##########################################
##################################################################################
def cell2mol(newcell: object, reconstruction: bool=True, charge_assignment: bool=True, spin_assignment: bool=True, debug: int=1) -> object:

    if reconstruction: 
        tini = time.time()
     
        ## Evaluates boolean variable self.has_isolated_H. If true, indicates a problem with the cif
        if newcell.has_isolated_H:        handle_error(1)
        newcell.check_missing_H(debug=debug)                                     
        ## Evaluates boolean variable self.has_missing_H. If true, indicates a problem with the cif
        if newcell.has_missing_H:         handle_error(2)
  
        # Cell Reconstruction
        newcell.reconstruct(debug=debug)                                         
        ## Evaluates boolean variable self.error_reconstruction. If true, fragments remain
        if newcell.error_reconstruction:  handle_error(3)

        tend = time.time()
        if debug >= 1: print(f"\nCell Reconstruction Finished Normally. Total execution time: {tend - tini:.2f} seconds")

    if charge_assignment:
        #if not newcell.has_missing_H and not newcell.has_isolated_H and not newcell.is_fragmented:
        tini = time.time()
        if not newcell.is_fragmented:
            newcell.reset_charges() 
            newcell.assign_charges(debug=debug)
            newcell.create_bonds(debug=debug)

            tend = time.time()
            if debug >= 1: print(f"\nTotal execution time for Charge Assignment: {tend - tini:.2f} seconds")

            if   newcell.error_empty_poscharges :  handle_error(4)
            elif newcell.error_multiple_distrib :  handle_error(5)
            elif newcell.error_empty_distrib :     handle_error(6)
            elif newcell.error_prepare_mols :        handle_error(7)
            else :
                if debug >= 1: print("Charge Assignment successfully finished.\n")
                # TODO : Compare assigned charges with ML predicted charges
                
    if spin_assignment:
        tini = time.time()
        newcell.assign_spin(debug=debug)
        tend = time.time()
        if debug >= 1: print(f"\nTotal execution time for Spin Assignment: {tend - tini:.2f} seconds")

    return newcell
