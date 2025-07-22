import copy
import logging
import time

from cell2mol.new_cell_reconstruction import (
    get_moleclist,
    get_unique_indices,
    reconstruct,
)
from cell2mol.new_charge_assignment import (
    assign_charge_to_specie,
    balance_charge,
    print_possible_and_selected_cs,
)


def cell2mol_mode(newcell, refcell, sym_ops, mode, debug):
    """
    Wrapper to run cell2mol with a specific mode.

    Parameters
    ----------
    newcell : object
        The unit cell object to be processed.
    refcell : object
        The reference unit cell.
    sym_ops : object
        Symmetry operations used for reconstruction.
    mode : str
        Mode of operation: 'reconstruction' or 'charge_assignment'.
    debug : int
        Debug level for verbosity.
    """
    if mode == "reconstruction":
        newcell = unitcell_reconstruction(newcell, refcell, sym_ops, debug=debug)
    elif mode == "charge_assignment":
        newcell, refcell = charge_assignment(newcell, refcell, debug=debug)

    logging.info("Completed '%s' mode in cell2mol", mode)


def unitcell_reconstruction(newcell, refcell, sym_ops, debug):
    """Reconstruct the unit cell based on the reference cell and identify all molecules."""
    start_time = time.time()

    if debug :
        print("#########################################")
        print("        Unit Cell Reconstruction         ")
        print("#########################################")

    if newcell.has_isolated_H or newcell.has_missing_H:
        return newcell

    all_molecules, reconstructed_molecules = reconstruct(
        refcell, newcell, sym_ops, debug=debug
    )
    all_molecules.extend(reconstructed_molecules)

    if newcell.error_get_fragments or newcell.error_reconstruction:
        print_elapsed("Unit Cell Reconstruction Failed.", start_time)
        return newcell

    print_elapsed("Unit Cell Reconstruction Finished Normally.", start_time)

    newcell = get_moleclist(newcell, refcell, all_molecules, debug=debug)
    newcell.unique_species = copy.deepcopy(refcell.unique_species)
    newcell = get_unique_indices(newcell, refcell.species_list, debug=debug)

    return newcell

def charge_assignment(newcell, refcell, debug):
    """Assign charges to unique species to satisfy charge neutrality."""
    start_time = time.time()

    if debug:
        print("#########################################")
        print(" Get Unique Species and Balance Charges  ")
        print("#########################################")

    # Skip charge assignment if reconstruction failed
    if newcell.error_reconstruction:
        return newcell, refcell

    # Check for missing charge states
    if None in refcell.selected_cs:
        newcell.error_get_poscharges = True
    else:
        newcell.error_get_poscharges = False

    # Print possible and selected charge states
    print_possible_and_selected_cs(newcell, refcell, debug=debug)

    # Balance charges for the unit cell
    final_charge_distribution, final_charges = balance_charge(
        newcell.unique_indices,
        refcell.unique_species,
        debug=debug,
    )

    # Handle multiple or no charge distributions
    dist_count = len(final_charge_distribution)
    newcell.error_multiple_distrib = dist_count > 1
    newcell.error_empty_distrib = dist_count == 0
    
    if dist_count != 1:
        # Attempt to balance charges again with more specific conditions
        if newcell.error_multiple_distrib :
            print("More than one possible distribution found.")
            second_final_charge_distribution, second_final_charges = balance_charge(
                newcell.unique_indices,
                refcell.unique_species,
                aromatic=True, debug=debug,
            )    
        elif newcell.error_empty_distrib :
            print("No valid distribution found.")
            second_final_charge_distribution, second_final_charges = balance_charge(
                newcell.unique_indices,
                refcell.unique_species,
                rare=True, debug=debug,
            )           
        second_dist_count = len(second_final_charge_distribution)
        newcell.error_multiple_distrib = second_dist_count > 1
        newcell.error_empty_distrib = second_dist_count == 0    

        if second_dist_count == 1:
            final_charge_distribution = second_final_charge_distribution
            final_charges = second_final_charges
            print("Using the second distribution found.")
            
    # If any error was flagged, report failure
    if any([
        newcell.error_get_poscharges,
        newcell.error_multiple_distrib,
        newcell.error_empty_distrib,
    ]):
        print_elapsed("Charge Assignment Failed.", start_time)
        return newcell, refcell

    # Assign charges to unique species in the reference cell
    for specie, charge in zip(newcell.unique_species, final_charges[0]):
        assign_charge_to_specie(specie, charge, debug=debug)
        for refspecie in refcell.unique_species:
            if specie.unique_index == refspecie.unique_index:
                assign_charge_to_specie(refspecie, charge, debug=debug)

    # Assign charges to reference molecules in the reference cell
    refcell.assign_charges_for_refcell(debug=debug)
    refcell.create_bonds(debug=debug)

    if refcell.error_create_bonds:
        refcell.error_case = 8
        print_elapsed("Creating bonds Failed for reference cell.", start_time)
        return newcell, refcell
    
    # Assign spin multiplicity to reference molecules with TMs
    # for ref in refcell.refmoleclist:
    #     if ref.iscomplex:
    #         for metal in ref.metals:
    #             metal.get_spin(debug=debug)
    #         ref.get_spin(debug=debug)

    newcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
    newcell.unique_species = copy.deepcopy(refcell.unique_species)

    # Assign charges to molecules in the unit cell
    newcell.assign_charges_for_unitcell(debug=debug)
    newcell.create_bonds(debug=debug)
    newcell.check_charge_neutrality(debug=debug)

    if newcell.error_create_bonds:
        print_elapsed("Creating bonds Failed for unit cell.", start_time)
        return newcell, refcell

    print_elapsed("Charge Assignment Finished Normally.", start_time)
    
    # for mol in newcell.moleclist:
    #     if mol.iscomplex:
    #         for metal in mol.metals:
    #             metal.get_spin(debug=debug)
    #         mol.get_spin(debug=debug)
            
    return newcell, refcell





def print_elapsed(message: str, start_time: float):
    """Print the elapsed time since start_time."""
    elapsed = time.time() - start_time
    print(f"{message} Total execution time: {elapsed:.2f} seconds")
