import logging
import itertools
from typing import List
from cell2mol.charge.utils import aromatic_info
from cell2mol.write_results import log_charge_state_details
from cell2mol.charge.specie_assigner import assign_charge_to_specie

logger = logging.getLogger(__name__)


def balance_unitcell_charge(refcell, unitcell):
    """
    Resolves and assigns charges to unique species to ensure unit cell neutrality.
    """
    logger.info("=" * 40)
    logger.info(" Get Unique Species and Balance Charges ")
    logger.info("=" * 40)

    # check error flags
    if unitcell.error_reconstruction:
        logger.warning(" Not proceed due to reconstruction error.")
        return refcell, unitcell

    if refcell.error_plausible_charges:
        logger.error(" Not proceed due to no charge states for some unique species.")
        return refcell, unitcell
    log_charge_state_details(unitcell, refcell)

    # Charge Balancing
    expanded_species_charges, unique_species_charges = resolve_charge_distributions(
        unitcell.unique_indices, refcell.unique_species
    )

    dist_count = len(expanded_species_charges)
    unitcell.error_multiple_distrib = dist_count > 1
    unitcell.error_empty_distrib = dist_count == 0

    # If the primary attempt failed (too many or empty results), try heuristics.
    # if dist_count != 1:
    #     retry_expanded, retry_charges = [], []
    #     if unitcell.error_multiple_distrib:
    #         logger.info(
    #             "Ambiguity found (%d distributions). Retrying with Aromaticity preference ...",
    #             dist_count,
    #         )
    #         retry_expanded, retry_charges = resolve_charge_distributions(
    #             unitcell.unique_indices,
    #             refcell.unique_species,
    #             aromatic=True,
    #         )
    #     elif unitcell.error_empty_distrib:
    #         logger.info(
    #             "No valid distribution found. Retrying with Rare Metal Oxidation States...",
    #             dist_count,
    #         )
    #         retry_expanded, retry_charges = resolve_charge_distributions(
    #             unitcell.unique_indices,
    #             refcell.unique_species,
    #             rare=True,
    #         )

    #     # Evaluate retry results
    #     retry_count = len(retry_expanded)
    #     if len(retry_expanded) == 1:
    #         logger.info("Retry success. Valid distribution found.")
    #         expanded_species_charges = retry_expanded
    #         unique_species_charges = retry_charges
    #         unitcell.error_multiple_distrib = False
    #         unitcell.error_empty_distrib = False
    #     else:
    #         # Update flags based on retry failure (e.g., still multiple or still empty)
    #         unitcell.error_multiple_distrib = retry_count > 1
    #         unitcell.error_empty_distrib = retry_count == 0

    # If nothing summed to neutrality, the usual reason is that the metal's
    # true oxidation state simply isn't in its candidate list (an unusual
    # state METAL_OXIDATION_STATES doesn't cover) -- but only when every
    # other (non-metal) unique specie is already unambiguous, since
    # otherwise there's no way to know how much of the "missing" charge is
    # really the metal's vs. a ligand option we didn't consider.
    if unitcell.error_empty_distrib:
        inferred_charges = _infer_metal_charge_from_fixed_ligands(
            unitcell.unique_indices, refcell.unique_species
        )
        if inferred_charges is not None:
            logger.warning(
                "No standard charge distribution summed to neutrality; "
                "inferred a non-standard metal oxidation state from the "
                "unambiguous ligand charges instead: %s",
                inferred_charges,
            )
            unique_species_charges = [inferred_charges]
            unitcell.error_empty_distrib = False

    # Final Error Check
    if unitcell.error_multiple_distrib or unitcell.error_empty_distrib:
        logger.error(
            "Balance charges failed. Flags in distribution: Multiple=%s, Empty=%s",
            unitcell.error_multiple_distrib,
            unitcell.error_empty_distrib,
        )
        return refcell, unitcell

    # Assign charges to the unique species only.
    # Then propagate to refcell.refmoleclist later by refcell.assign_charges()
    final_charges = unique_species_charges[0]
    for specie, charge in zip(refcell.unique_species, final_charges):
        assign_charge_to_specie(specie, charge)

    return refcell, unitcell


def _infer_metal_charge_from_fixed_ligands(
    unique_indices, unique_species, input_charge: int = 0
) -> List[int] | None:
    """
    Fallback for when no combination of standard charge options summed to
    neutrality (unitcell.error_empty_distrib): if every non-metal unique
    specie already has exactly one feasible charge, the metal's true
    oxidation state is the one value that makes the total balance out --
    regardless of whether it's in the metal's usual candidate list
    (spec.plausible_os, drawn from METAL_OXIDATION_STATES) -- since an
    unusual/rare oxidation state not in that list is the most likely
    reason the normal search came up empty in the first place.

    Only handles a single unique metal specie (though it may appear
    multiple times in the cell via symmetry): with more than one distinct
    metal type, there's no way to know how to split the "missing" charge
    between them, so this declines rather than guess.

    Returns a list of charges (one per entry in `unique_species`, in
    order) if a unique, integer-valued metal charge can be solved for,
    else None.
    """
    metal_unique_indices: set[int] = set()
    non_metal_options: List[tuple[int, List[int]]] = []
    for idx, spec in enumerate(unique_species):
        if spec.subtype == "metal":
            metal_unique_indices.add(idx)
        else:
            options = _get_ligand_options(spec, aromatic=False)
            if len(set(options)) != 1:
                logger.debug(
                    "Cannot infer metal charge: non-metal specie %s (index %d) "
                    "is still ambiguous (%s).",
                    spec.formula,
                    idx,
                    options,
                )
                return None
            non_metal_options.append((idx, options))

    if len(metal_unique_indices) != 1:
        logger.debug(
            "Cannot infer metal charge: found %d distinct metal species "
            "(need exactly 1).",
            len(metal_unique_indices),
        )
        return None

    fixed_charge_by_index = {idx: options[0] for idx, options in non_metal_options}

    non_metal_sum = 0
    unique_metal_count = 0
    for u_idx in unique_indices:
        if u_idx in fixed_charge_by_index:
            non_metal_sum += fixed_charge_by_index[u_idx]
        else:
            unique_metal_count += 1

    if unique_metal_count == 0:
        return None

    remainder = input_charge - non_metal_sum
    if remainder % unique_metal_count != 0:
        logger.debug(
            "Cannot infer metal charge: needed total charge %d does not "
            "split evenly across %d metal occurrence(s).",
            remainder,
            unique_metal_count,
        )
        return None

    inferred_metal_charge = remainder // unique_metal_count
    (metal_idx,) = metal_unique_indices

    return [
        inferred_metal_charge if idx == metal_idx else fixed_charge_by_index[idx]
        for idx in range(len(unique_species))
    ]


def balance_molecule_charge(molecule, input_charge: int = 0, second_try: bool = True):
    """
    Resolves and assigns charges to unique species in the single molecule
    to achieve the target input_charge.
    """
    if molecule.unique_species is None:
        molecule.get_unique_species()
    if molecule.plausible_charges is None:
        molecule.get_plausible_charges()

    # Flag error if plausible_charges is missing or contains None entries
    molecule.error_plausible_charges = (molecule.plausible_charges is None) or (
        None in molecule.plausible_charges
    )
    if molecule.error_plausible_charges:
        logger.error("No charge states available for some species.")
        return molecule

    unique_indices = [spec.unique_index for spec in molecule.species_list]

    # Primary search
    expanded_species_charges, unique_species_charges = resolve_charge_distributions(
        unique_indices, molecule.unique_species, input_charge=input_charge
    )

    # Refinement for the retry logic inside balance_molecule_charge
    # if len(expanded_species_charges) != 1 and second_try:
    #     logger.info("Retrying with fallbacks...")

    #     # Capture the retry results
    #     res_expanded, res_unique = [], []
    #     if len(expanded_species_charges) > 1:
    #         res_expanded, res_unique = resolve_charge_distributions(
    #             unique_indices,
    #             molecule.unique_species,
    #             input_charge=input_charge,
    #             aromatic=True,
    #         )
    #     elif len(expanded_species_charges) == 0:
    #         res_expanded, res_unique = resolve_charge_distributions(
    #             unique_indices,
    #             molecule.unique_species,
    #             input_charge=input_charge,
    #             rare=True,
    #         )

    #     # If the retry found a unique solution, update the main variables
    #     if len(res_expanded) == 1:
    #         expanded_species_charges, unique_species_charges = res_expanded, res_unique

    # Update state flags based on final results
    dist_count = len(expanded_species_charges)
    molecule.error_multiple_distrib = dist_count > 1
    molecule.error_empty_distrib = dist_count == 0

    if molecule.error_multiple_distrib or molecule.error_empty_distrib:
        logger.error(
            "Balance charges failed. Error in distribution: Multiple=%s, Empty=%s",
            molecule.error_multiple_distrib,
            molecule.error_empty_distrib,
        )
        return molecule

    final_charges = unique_species_charges[0]
    if molecule.unique_species is not None:
        for specie, charge in zip(molecule.unique_species, final_charges):
            assign_charge_to_specie(specie, charge)
            for ref_specie in molecule.species_list:
                if specie.unique_index == ref_specie.unique_index:
                    assign_charge_to_specie(ref_specie, charge)
    return molecule


def resolve_charge_distributions(
    unique_indices,
    unique_species,
    input_charge: int = 0,
    rare: bool = False,
    predict: bool = False,
    aromatic: bool = False,
):
    """
    Computes all valid charge distributions that sum to the target input_charge.
    """

    species_charge_options: List[List[int]] = []

    # Gather Charge options per unique specie
    for idx, spec in enumerate(unique_species):
        options = []

        if spec.subtype == "metal":
            options = _get_metal_options(spec, rare, predict)
        else:
            options = _get_ligand_options(spec, aromatic)

        if not options:
            logger.error(
                "Unique Species %s (Index %d) has no valid charge options.",
                spec.formula,
                idx,
            )
            return [], []

        species_charge_options.append(options)

    logger.debug("Charge options per unique specie: %s", species_charge_options)
    logger.debug("Unique indices mapping: %s", unique_indices)

    # Generate Combinations and Filter
    expanded_species_charges = []
    unique_species_charges = []

    combinations = list(itertools.product(*species_charge_options))
    logger.debug("Generated %d charge combinations.", len(combinations))

    for combo in combinations:
        expanded_dist = [combo[u_idx] for u_idx in unique_indices]

        # Check if total charge matches input
        if sum(expanded_dist) == input_charge:
            expanded_species_charges.append(expanded_dist)
            unique_species_charges.append(combo)

            logger.debug(
                "Found valid distribution (charge %d): %s", input_charge, expanded_dist
            )

    logger.debug("expanded_species_charges: %s", expanded_species_charges)
    logger.debug("unique_species_charges: %s", unique_species_charges)
    logger.debug(
        "Final Count: %d valid distributions found.",
        len(expanded_species_charges),
    )

    return expanded_species_charges, unique_species_charges


def _get_metal_options(spec, rare: bool, predict: bool) -> List[int]:
    """Determines valid oxidation states for metal species."""
    all_possible_m_ox = [0, 1, 2, 3, 4, 5, 6, 7]

    if rare:
        rare_states = [x for x in all_possible_m_ox if x not in spec.plausible_os]
        logger.debug("RARE METAL OXIDATION STATES: %s %s", spec.formula, rare_states)
        return rare_states

    # if predict:
    #     predicted_charge = predict_metal_ox(spec)
    #     return [predicted_charge]

    return list(spec.plausible_os)


def _get_ligand_options(spec, aromatic: bool) -> List[int]:
    """Determines valid charges for ligand species, optionally filtering by aromaticity."""
    plausible = spec.plausible_charge_states

    if not plausible:
        return []

    if len(plausible) == 1:
        return [plausible[0].specie_total_charge]

    if aromatic:
        return _filter_by_aromaticity(spec, plausible)

    return [cs.specie_total_charge for cs in plausible]


def _filter_by_aromaticity(spec, plausible) -> List[int]:
    """Selects charge states that maximize aromatic atoms, then aromatic rings."""
    aromatic_stats = []
    for cs in plausible:
        # Get indices of added protons
        added_indices = [
            i
            for i, n_added in enumerate(cs.protonation.site_proton_counts)
            if n_added > 0
        ]

        info = aromatic_info(cs.rdkit_obj, added_indices)
        aromatic_stats.append(
            {
                "charge": cs.specie_total_charge,
                "smiles": cs.smiles,
                "atoms": info["Aromatic atoms"],
                "rings": info["Number of aromatic rings"],
                "obj": cs,
            }
        )

    atoms_counts = [x["atoms"] for x in aromatic_stats]
    rings_counts = [x["rings"] for x in aromatic_stats]

    logger.debug("  aromatic_counts: %s %s", spec.formula, atoms_counts)
    logger.debug("  aromatic_ring: %s %s", spec.formula, rings_counts)

    # If no aromaticity anywhere, return all options
    if all(count == 0 for count in atoms_counts):
        return [x["charge"] for x in aromatic_stats]

    # 1. Filter by Max Aromatic Atoms
    max_atoms = max(atoms_counts)
    candidates = [x for x in aromatic_stats if x["atoms"] == max_atoms]

    candidate_indices = [plausible.index(c["obj"]) for c in candidates]
    logger.debug(
        "  Primary indices with max aromatic atoms (%d): %s",
        max_atoms,
        candidate_indices,
    )

    # 2. Filter by Max Aromatic Rings (among those with max atoms)
    max_rings = max(c["rings"] for c in candidates)
    best_candidates = [c for c in candidates if c["rings"] == max_rings]

    # Logging details for best candidates
    if logger.isEnabledFor(logging.DEBUG):
        indices = [plausible.index(c["obj"]) for c in best_candidates]
        logger.debug("  Max aromatic rings (%d) at indices: %s", max_rings, indices)
        for c in best_candidates:
            logger.debug("  - Selected: %s (Charge: %s)", c["smiles"], c["charge"])

    return [c["charge"] for c in best_candidates]
