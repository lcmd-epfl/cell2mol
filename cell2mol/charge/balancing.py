import logging
import itertools
import math
from typing import List
from cell2mol.charge.aromaticity import aromatic_info
from cell2mol.charge.oxidation_states import METAL_OS_OBSERVED, METAL_OXIDATION_STATES
from cell2mol.write_results import log_charge_diagnostics
from cell2mol.charge.charge_assignment import assign_charge_to_species

logger = logging.getLogger(__name__)

# Ceiling on the ligand charge combinations the metal-OS inference will enumerate.
_MAX_INFERENCE_COMBINATIONS = 100_000

# Ceiling on the full charge-distribution search.
_MAX_DISTRIBUTION_COMBINATIONS = 10_000_000

# Frequency ratio required to resolve an ambiguous metal oxidation state.
_PRIOR_DECISIVE_RATIO = 10


############################################################
# Charge balancing entry points
############################################################


def balance_unitcell_charge(refcell, unitcell):
    """
    Resolves and assigns charges to unique species to ensure unit cell neutrality.
    """
    logger.info("=" * 40)
    logger.info(" Get Unique Species and Balance Charges ")
    logger.info("=" * 40)

    if unitcell.error_reconstruction:
        logger.warning(" Not proceed due to reconstruction error.")
        return refcell, unitcell

    if refcell.error_plausible_charges:
        logger.error(" Not proceed due to no charge states for some unique species.")
        return refcell, unitcell
    log_charge_diagnostics(unitcell, refcell)

    _, unique_species_charges = resolve_charge_distributions(
        unitcell.unique_indices, refcell.unique_species
    )

    dist_count = len(unique_species_charges)
    unitcell.error_multiple_distrib = dist_count > 1
    unitcell.error_empty_distrib = dist_count == 0

    # Try metal states absent from the initial oxidation-state list.
    # if unitcell.error_empty_distrib:
    #     inferred_charges = _balance_with_observed_metal_os(
    #         unitcell.unique_indices, refcell.unique_species
    #     )
    #     if inferred_charges is not None:
    #         logger.info(
    #             "No neutral charge distribution found; "
    #             "inferred a less common metal oxidation state from the "
    #             "ligand charges instead: %s",
    #             inferred_charges,
    #         )
    #         unique_species_charges = [inferred_charges]
    #         unitcell.error_empty_distrib = False

    if unitcell.error_multiple_distrib or unitcell.error_empty_distrib:
        logger.error(
            "Balance charges failed. Flags in distribution: Multiple=%s, Empty=%s",
            unitcell.error_multiple_distrib,
            unitcell.error_empty_distrib,
        )
        return refcell, unitcell

    # Assign unique species now; propagate to copies in refcell.assign_charges().
    final_charges = unique_species_charges[0]
    for species, charge in zip(refcell.unique_species, final_charges):
        assign_charge_to_species(species, charge)

    return refcell, unitcell


def balance_molecule_charge(molset, input_charge: int = 0):
    """
    Resolves and assigns charges to the unique species of a MoleculeSet so
    that they add up to the target input_charge.
    """
    if molset.unique_species is None:
        molset.collect_unique_species()
    if molset.plausible_charges is None:
        molset.get_plausible_charges()

    molset.error_plausible_charges = (molset.plausible_charges is None) or (
        None in molset.plausible_charges
    )
    if molset.error_plausible_charges:
        logger.error("No charge states available for some species.")
        return molset

    unique_indices = [species.unique_index for species in molset.species_list]

    expanded_species_charges, unique_species_charges = resolve_charge_distributions(
        unique_indices, molset.unique_species, input_charge=input_charge
    )

    dist_count = len(expanded_species_charges)
    molset.error_multiple_distrib = dist_count > 1
    molset.error_empty_distrib = dist_count == 0

    # if molset.error_empty_distrib:
    #     inferred_charges = _balance_with_observed_metal_os(
    #         unique_indices, molset.unique_species, input_charge=input_charge
    #     )
    #     if inferred_charges is not None:
    #         unique_species_charges = [inferred_charges]
    #         molset.error_empty_distrib = False

    if molset.error_multiple_distrib or molset.error_empty_distrib:
        logger.error(
            "Balance charges failed. Error in distribution: Multiple=%s, Empty=%s",
            molset.error_multiple_distrib,
            molset.error_empty_distrib,
        )
        return molset

    final_charges = unique_species_charges[0]
    if molset.unique_species is not None:
        for species, charge in zip(molset.unique_species, final_charges):
            assign_charge_to_species(species, charge)
            for ref_species in molset.species_list:
                if species.unique_index == ref_species.unique_index:
                    assign_charge_to_species(ref_species, charge)
    return molset


############################################################
# Charge-distribution search
############################################################


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

    for idx, species in enumerate(unique_species):
        options = []

        if species.subtype == "metal":
            options = _get_metal_options(species, rare, predict)
        else:
            options = _get_ligand_options(species, aromatic)

        if not options:
            logger.error(
                "Unique Species %s (Index %d) has no valid charge options.",
                species.formula,
                idx,
            )
            return [], []

        species_charge_options.append(options)

    logger.debug("Charge options per unique species: %s", species_charge_options)
    logger.debug("Unique indices mapping: %s", unique_indices)

    expanded_species_charges = []
    unique_species_charges = []

    combination_count = math.prod(len(options) for options in species_charge_options)
    logger.debug("Generating %d charge combinations.", combination_count)

    if combination_count > _MAX_DISTRIBUTION_COMBINATIONS:
        logger.error(
            "Charge distribution search space (%d combinations over %d unique "
            "species) exceeds the limit of %d; cannot balance charges.",
            combination_count,
            len(species_charge_options),
            _MAX_DISTRIBUTION_COMBINATIONS,
        )
        return [], []

    for combo in itertools.product(*species_charge_options):
        expanded_dist = [combo[u_idx] for u_idx in unique_indices]

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


############################################################
# Species charge options
############################################################


def _get_metal_options(species, rare: bool, predict: bool) -> List[int]:
    """Determines valid oxidation states for metal species."""
    all_possible_m_ox = [0, 1, 2, 3, 4, 5, 6, 7]

    if rare:
        rare_states = [x for x in all_possible_m_ox if x not in species.plausible_os]
        logger.debug("RARE METAL OXIDATION STATES: %s %s", species.formula, rare_states)
        return rare_states

    return list(species.plausible_os)


def _get_ligand_options(species, aromatic: bool) -> List[int]:
    """Determines valid charges for ligand species, optionally filtering by aromaticity."""
    plausible = species.plausible_charge_states

    if not plausible:
        return []

    if len(plausible) == 1:
        return [plausible[0].species_total_charge]

    if aromatic:
        return _filter_by_aromaticity(species, plausible)

    return [cs.species_total_charge for cs in plausible]


def _filter_by_aromaticity(species, plausible) -> List[int]:
    """Selects charge states that maximize aromatic atoms, then aromatic rings."""
    aromatic_stats = []
    for cs in plausible:
        added_indices = [
            i
            for i, n_added in enumerate(cs.protonation.site_proton_counts)
            if n_added > 0
        ]

        info = aromatic_info(cs.protonated_rdkit_obj, added_indices)
        aromatic_stats.append(
            {
                "charge": cs.species_total_charge,
                "smiles": cs.protonated_smiles,
                "atoms": info["Aromatic atoms"],
                "rings": info["Number of aromatic rings"],
                "obj": cs,
            }
        )

    atoms_counts = [x["atoms"] for x in aromatic_stats]
    rings_counts = [x["rings"] for x in aromatic_stats]

    logger.debug("  aromatic_counts: %s %s", species.formula, atoms_counts)
    logger.debug("  aromatic_ring: %s %s", species.formula, rings_counts)

    if all(count == 0 for count in atoms_counts):
        return [x["charge"] for x in aromatic_stats]

    max_atoms = max(atoms_counts)
    candidates = [x for x in aromatic_stats if x["atoms"] == max_atoms]

    candidate_indices = [plausible.index(c["obj"]) for c in candidates]
    logger.debug(
        "  Primary indices with max aromatic atoms (%d): %s",
        max_atoms,
        candidate_indices,
    )

    max_rings = max(c["rings"] for c in candidates)
    best_candidates = [c for c in candidates if c["rings"] == max_rings]

    if logger.isEnabledFor(logging.DEBUG):
        indices = [plausible.index(c["obj"]) for c in best_candidates]
        logger.debug("  Max aromatic rings (%d) at indices: %s", max_rings, indices)
        for c in best_candidates:
            logger.debug("  - Selected: %s (Charge: %s)", c["smiles"], c["charge"])

    return [c["charge"] for c in best_candidates]


############################################################
# Distribution preferences and metal-charge inference
############################################################


def _preferred_distributions(
    unique_indices, unique_species, distributions: List[List[int]]
) -> List[List[int]]:
    """Prefer common metal oxidation states, then minimize total absolute ligand charge."""
    occurrences: dict[int, int] = {}
    for u_idx in unique_indices:
        occurrences[u_idx] = occurrences.get(u_idx, 0) + 1

    metal_indices = [
        idx for idx, species in enumerate(unique_species) if species.subtype == "metal"
    ]

    def separated_charge(distribution: List[int]) -> int:
        return sum(
            abs(charge) * occurrences.get(idx, 0)
            for idx, (charge, species) in enumerate(zip(distribution, unique_species))
            if species.subtype != "metal"
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


def _balance_with_observed_metal_os(
    unique_indices, unique_species, input_charge: int = 0
) -> List[int] | None:
    """Balance fixed non-metal charges with one unique metal species.

    Accept an observed integer metal OS outside the common oxidation states.
    Return charges in unique_species order, or None if no solution exists.
    """
    metal_indices = [
        idx for idx, species in enumerate(unique_species) if species.subtype == "metal"
    ]
    if len(metal_indices) != 1:
        return None
    metal_idx = metal_indices[0]
    charges = [0] * len(unique_species)
    for idx, species in enumerate(unique_species):
        if idx == metal_idx:
            continue
        options = set(_get_ligand_options(species, aromatic=False))
        if len(options) != 1:
            return None
        charges[idx] = options.pop()

    metal_count = sum(idx == metal_idx for idx in unique_indices)
    if metal_count == 0:
        return None
    non_metal_sum = sum(charges[idx] for idx in unique_indices if idx != metal_idx)
    remainder = input_charge - non_metal_sum
    if remainder % metal_count:
        return None
    metal_charge = remainder // metal_count
    metal_label = unique_species[metal_idx].label
    known_os = METAL_OXIDATION_STATES.get(metal_label, [])
    observed_os = METAL_OS_OBSERVED.get(metal_label, {})
    if metal_charge in known_os or observed_os.get(metal_charge, 0) <= 0:
        return None

    charges[metal_idx] = metal_charge
    logger.warning(
        "Fixed non-metal charges require %s(%d), observed outside the curated "
        "oxidation states %s, to balance total charge %d.",
        metal_label,
        metal_charge,
        known_os,
        input_charge,
    )
    return charges


def _infer_metal_charge_from_fixed_ligands(
    unique_indices, unique_species, input_charge: int = 0
) -> List[int] | None:
    """Infer one metal species charge from ligand combinations and observed states.

    Accept a unique solution or a decisive frequency winner. Return charges in
    unique_species order, or None for ambiguity or multiple metal species.
    """
    metal_unique_indices: set[int] = set()
    non_metal_options: List[tuple[int, List[int]]] = []
    for idx, species in enumerate(unique_species):
        if species.subtype == "metal":
            metal_unique_indices.add(idx)
        else:
            options = sorted(set(_get_ligand_options(species, aromatic=False)))
            if not options:
                logger.debug(
                    "Cannot infer metal charge: non-metal species %s (index %d) "
                    "has no charge options.",
                    species.formula,
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

    # How many times each unique species occurs in the cell.
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

    # Resolve multiple solutions only with a decisive frequency ratio.
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


def _observed_os_histogram(label: str) -> dict[int, int]:
    """Return observed oxidation-state counts; use equally weighted curated states
    plus one higher state for elements absent from the CSD table.
    """
    observed = METAL_OS_OBSERVED.get(label)
    if observed is not None:
        return observed
    known_os = METAL_OXIDATION_STATES.get(label)
    if not known_os:
        return {}
    return {os: 1 for os in range(0, max(known_os) + 2)}
