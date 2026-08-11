import logging
import itertools
from typing import List
from cell2mol.charge.utils import (
    aromatic_info,
    METAL_OS_OBSERVED,
    METAL_OXIDATION_STATES,
)
from cell2mol.write_results import log_charge_state_details
from cell2mol.charge.specie_assigner import assign_charge_to_specie

logger = logging.getLogger(__name__)

# Ceiling on the ligand charge combinations the metal-OS inference will enumerate.
_MAX_INFERENCE_COMBINATIONS = 100_000

# How much likelier one metal oxidation state must be than its rival before the
# observed frequencies are allowed to settle an otherwise ambiguous cell.
_PRIOR_DECISIVE_RATIO = 10


def _observed_os_histogram(label: str) -> dict[int, int]:
    """Observed {oxidation state: count} for an element, empty if unknown.

    Elements outside the CSD tabulation -- main group, lanthanides, actinides --
    are given the old rule instead, every state up to one past the curated table,
    weighted equally so that the frequency comparisons below never fire on them.
    """
    observed = METAL_OS_OBSERVED.get(label)
    if observed is not None:
        return observed
    known_os = METAL_OXIDATION_STATES.get(label)
    if not known_os:
        return {}
    return {os: 1 for os in range(0, max(known_os) + 2)}


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
    _, unique_species_charges = resolve_charge_distributions(
        unitcell.unique_indices, refcell.unique_species
    )

    # _preferred_distributions() would narrow several balancing distributions to
    # one here. Left off: a data-driven model is to take over that decision.

    dist_count = len(unique_species_charges)
    unitcell.error_multiple_distrib = dist_count > 1
    unitcell.error_empty_distrib = dist_count == 0

    # Nothing balanced: usually the metal's true oxidation state is missing from
    # its plausible_os set.
    if unitcell.error_empty_distrib:
        inferred_charges = _infer_metal_charge_from_fixed_ligands(
            unitcell.unique_indices, refcell.unique_species
        )
        if inferred_charges is not None:
            logger.warning(
                "No standard charge distribution summed to neutrality; "
                "inferred a non-standard metal oxidation state from the "
                "ligand charges instead: %s",
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


def _preferred_distributions(
    unique_indices, unique_species, distributions: List[List[int]]
) -> List[List[int]]:
    """Narrow several neutral distributions, on two criteria in order: drop any
    whose metal oxidation states are far rarer than the best on offer, then keep
    those separating the least charge -- summed |ligand charge| per occurrence.
    """
    occurrences: dict[int, int] = {}
    for u_idx in unique_indices:
        occurrences[u_idx] = occurrences.get(u_idx, 0) + 1

    metal_indices = [
        idx for idx, spec in enumerate(unique_species) if spec.subtype == "metal"
    ]

    def separated_charge(distribution: List[int]) -> int:
        return sum(
            abs(charge) * occurrences.get(idx, 0)
            for idx, (charge, spec) in enumerate(zip(distribution, unique_species))
            if spec.subtype != "metal"
        )

    histograms = {
        idx: _observed_os_histogram(unique_species[idx].label) for idx in metal_indices
    }

    def metal_likelihood(distribution: List[int]) -> float:
        """P(oxidation state) across the cell's metals; 1.0 when none is tabulated."""
        likelihood = 1.0
        for idx in metal_indices:
            histogram = histograms[idx]
            if histogram:
                likelihood *= histogram.get(distribution[idx], 0) / sum(
                    histogram.values()
                )
        return likelihood

    likelihoods = [metal_likelihood(dist) for dist in distributions]
    likeliest = max(likelihoods)
    if likeliest > 0:
        alive = [
            dist
            for dist, likelihood in zip(distributions, likelihoods)
            if likelihood * _PRIOR_DECISIVE_RATIO >= likeliest
        ]
        if len(alive) < len(distributions):
            logger.info(
                "Discarded %d of %d distribution(s) whose metal oxidation states "
                "are over %dx rarer than the likeliest reading.",
                len(distributions) - len(alive),
                len(distributions),
                _PRIOR_DECISIVE_RATIO,
            )
            distributions = alive

    burdens = [separated_charge(dist) for dist in distributions]
    lowest = min(burdens)
    best = [dist for dist, burden in zip(distributions, burdens) if burden == lowest]

    if len(best) < len(distributions):
        logger.info(
            "%d distributions balanced the cell; kept the %d separating the "
            "least ligand charge (%d, down from %d).",
            len(distributions),
            len(best),
            lowest,
            max(burdens),
        )
    return best


def _infer_metal_charge_from_fixed_ligands(
    unique_indices, unique_species, input_charge: int = 0
) -> List[int] | None:
    """Last resort when nothing balanced: try every combination of the ligands'
    candidate charges, keeping those that leave the metal at a whole-number
    oxidation state attested for that element. A lone survivor is taken; among
    several the observed frequencies decide, and only by _PRIOR_DECISIVE_RATIO
    or more. Declines otherwise, and when the cell holds more than one distinct
    metal. Returns one charge per ``unique_species`` entry, or None.
    """
    metal_unique_indices: set[int] = set()
    non_metal_options: List[tuple[int, List[int]]] = []
    for idx, spec in enumerate(unique_species):
        if spec.subtype == "metal":
            metal_unique_indices.add(idx)
        else:
            options = sorted(set(_get_ligand_options(spec, aromatic=False)))
            if not options:
                logger.debug(
                    "Cannot infer metal charge: non-metal specie %s (index %d) "
                    "has no charge options.",
                    spec.formula,
                    idx,
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

    (metal_idx,) = metal_unique_indices
    metal_label = unique_species[metal_idx].label
    observed_os = _observed_os_histogram(metal_label)
    if not observed_os:
        logger.debug(
            "Cannot infer metal charge: no observed oxidation states for %s.",
            metal_label,
        )
        return None

    # How many times each unique specie occurs in the cell.
    occurrences = {idx: 0 for idx, _ in non_metal_options}
    metal_occurrences = 0
    for u_idx in unique_indices:
        if u_idx in occurrences:
            occurrences[u_idx] += 1
        else:
            metal_occurrences += 1

    if metal_occurrences == 0:
        return None

    combination_count = 1
    for _, options in non_metal_options:
        combination_count *= len(options)
    if combination_count > _MAX_INFERENCE_COMBINATIONS:
        logger.debug(
            "Cannot infer metal charge: %d ligand charge combinations exceed "
            "the search limit of %d.",
            combination_count,
            _MAX_INFERENCE_COMBINATIONS,
        )
        return None

    solutions: List[List[int]] = []
    for combo in itertools.product(*[options for _, options in non_metal_options]):
        charge_by_index = {
            idx: charge for (idx, _), charge in zip(non_metal_options, combo)
        }
        non_metal_sum = sum(
            charge_by_index[idx] * count for idx, count in occurrences.items()
        )
        remainder = input_charge - non_metal_sum
        if remainder % metal_occurrences != 0:
            continue
        metal_charge = remainder // metal_occurrences
        if metal_charge not in observed_os:
            continue
        solutions.append(
            [
                metal_charge if idx == metal_idx else charge_by_index[idx]
                for idx in range(len(unique_species))
            ]
        )

    if not solutions:
        logger.debug(
            "Cannot infer metal charge: no ligand charge combination leaves %s "
            "with a whole-number oxidation state among the observed %s.",
            metal_label,
            sorted(observed_os),
        )
        return None

    if len(solutions) == 1:
        return solutions[0]

    # Several balance. Let the prior decide, but only when it is emphatic: two
    # solutions sharing a metal oxidation state tie at ratio 1 and are declined,
    # as are near-neighbours like V(V) over V(IV).
    solutions.sort(key=lambda sol: observed_os.get(sol[metal_idx], 0), reverse=True)
    best, runner_up = (observed_os.get(sol[metal_idx], 0) for sol in solutions[:2])
    if best < _PRIOR_DECISIVE_RATIO * runner_up:
        logger.debug(
            "Cannot infer metal charge: %d combinations balance the cell with "
            "%s oxidation states %s; the prior does not separate them.",
            len(solutions),
            metal_label,
            sorted({sol[metal_idx] for sol in solutions}),
        )
        return None

    logger.info(
        "%d combinations balanced the cell; took %s(%d), seen in %d of %d "
        "surveyed structures, over the next best at %d.",
        len(solutions),
        metal_label,
        solutions[0][metal_idx],
        best,
        sum(observed_os.values()),
        runner_up,
    )
    return solutions[0]


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
