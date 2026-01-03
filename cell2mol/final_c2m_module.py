import copy
import logging
import time

from cell2mol.new_charge_assignment import (
    assign_charge_to_specie,
    balance_charge,
    print_possible_and_selected_cs,
)

logger = logging.getLogger(__name__)


def charge_assignment(newcell, refcell, debug):
    """Assign charges to unique species to satisfy charge neutrality."""
    start_time = time.time()

    logger.info("#### Get Unique Species and Balance Charges ####")

    # Skip charge assignment if reconstruction failed
    if newcell.error_reconstruction:
        return newcell, refcell

    # Check for missing charge states
    refcell.get_selected_cs()
    # refcell.assess_errors(mode="possible_charges")
    if None in refcell.selected_cs:
        newcell.error_get_poscharges = True
    elif refcell.error_get_poscharges:
        newcell.error_get_poscharges = True
    else:
        newcell.error_get_poscharges = False

    newcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
    newcell.unique_species = copy.deepcopy(refcell.unique_species)

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

    # second_try = True
    second_try = False
    if dist_count != 1 and second_try:
        # Attempt to balance charges again with more specific conditions
        if newcell.error_multiple_distrib:
            logger.info("More than one possible distribution found.")
            second_final_charge_distribution, second_final_charges = balance_charge(
                newcell.unique_indices,
                refcell.unique_species,
                aromatic=True,
                debug=debug,
            )
        elif newcell.error_empty_distrib:
            logger.info("No valid distribution found.")
            second_final_charge_distribution, second_final_charges = balance_charge(
                newcell.unique_indices,
                refcell.unique_species,
                rare=True,
                debug=debug,
            )
        second_dist_count = len(second_final_charge_distribution)
        newcell.error_multiple_distrib = second_dist_count > 1
        newcell.error_empty_distrib = second_dist_count == 0

        if second_dist_count == 1:
            final_charge_distribution = second_final_charge_distribution
            final_charges = second_final_charges
            logger.info("Using the second distribution found.")

    # If any error was flagged, report failure
    if any(
        [
            newcell.error_get_poscharges,
            newcell.error_multiple_distrib,
            newcell.error_empty_distrib,
        ]
    ):
        get_elapsed_time("Charge Assignment Failed.", start_time)
        return newcell, refcell

    # Assign charges to unique species in the reference cell
    for specie, charge in zip(newcell.unique_species, final_charges[0]):
        assign_charge_to_specie(specie, charge, debug=debug)
        for refspecie in refcell.unique_species:
            if specie.unique_index == refspecie.unique_index:
                assign_charge_to_specie(refspecie, charge, debug=debug)

    # Assign charges to reference molecules in the reference cell
    refcell.assign_charges_for_refcell(debug=debug)

    if refcell.error_create_bonds:
        refcell.error_case = 8
        get_elapsed_time("Creating bonds Failed for reference cell.", start_time)
        return newcell, refcell

    newcell.refmoleclist = copy.deepcopy(refcell.refmoleclist)
    newcell.unique_species = copy.deepcopy(refcell.unique_species)

    # Assign charges to molecules in the unit cell
    newcell.assign_charges_for_unitcell(debug=debug)
    newcell.check_charge_neutrality(debug=debug)

    if newcell.error_create_bonds:
        get_elapsed_time("Creating bonds Failed for unit cell.", start_time)
        return newcell, refcell

    get_elapsed_time("Charge Assignment Finished Normally.", start_time)

    return newcell, refcell


def get_elapsed_time(message: str, start_time: float):
    """Print the elapsed time since start_time."""
    elapsed = time.time() - start_time
    logger.info(f"{message} Total execution time: {elapsed:.2f} seconds")
