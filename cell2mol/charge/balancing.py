import logging
import itertools
import math
from typing import List
from cell2mol.charge.aromaticity import aromatic_info
from cell2mol.charge.oxidation_states import METAL_OS_OBSERVED, METAL_OXIDATION_STATES
from cell2mol.write_results import log_charge_diagnostics
from cell2mol.charge.charge_assignment import assign_charge_to_species
from cell2mol.standardize_metal_os import is_metallocene

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

    warnings = dict(getattr(refcell, "potential_warnings", None) or {})
    warnings.pop("metal_os_ambiguity", None)
    _, unique_species_charges = resolve_charge_distributions(
        unitcell.unique_indices, refcell.unique_species, warnings=warnings
    )
    refcell.potential_warnings = warnings

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
    warnings: dict | None = None,
):
    """
    Find charge-balanced distributions, apply preferences, and expand survivors.
    """

    species_charge_options: List[List[int]] = []

    for idx, species in enumerate(unique_species):
        options = []

        if species.subtype == "metal":
            options = _get_metal_options(species, rare, predict)
        else:
            options = _get_ligand_options(species, aromatic)

        if not options:
            if species.subtype == "metal" and not rare:
                inferred = _distribute_fixed_ligand_metal_charges(
                    unique_indices, unique_species, input_charge, warnings=warnings
                )
                if len(inferred) > 1:
                    inferred = _preferred_distributions(
                        unique_indices, unique_species, inferred
                    )
                return [
                    [charges[i] for i in unique_indices] for charges in inferred
                ], inferred
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

    # Apply structural preferences before donor-based metal inference can
    # collapse multiple candidates to one distribution.
    if len(unique_species_charges) > 1:
        unique_species_charges = _preferred_distributions(
            unique_indices, unique_species, unique_species_charges
        )

    if len(unique_species_charges) > 1 and not rare:
        ranked = _distribute_fixed_ligand_metal_charges(
            unique_indices,
            unique_species,
            input_charge,
            distributions=unique_species_charges,
            warnings=warnings,
        )
        if ranked:
            unique_species_charges = ranked
            expanded_species_charges = [
                [charges[idx] for idx in unique_indices] for charges in ranked
            ]

    if not unique_species_charges and not rare:
        unique_species_charges = _distribute_fixed_ligand_metal_charges(
            unique_indices, unique_species, input_charge, warnings=warnings
        )
        expanded_species_charges = [
            [charges[idx] for idx in unique_indices]
            for charges in unique_species_charges
        ]

    # Keep expanded and unique results aligned after every filtering path.
    expanded_species_charges = [
        [charges[idx] for idx in unique_indices] for charges in unique_species_charges
    ]
    logger.info("Charge distributions remaining: %d", len(unique_species_charges))
    return expanded_species_charges, unique_species_charges


############################################################
# Species charge options
############################################################


def _get_metal_options(species, rare: bool, predict: bool) -> List[int]:
    """Determines valid oxidation states for metal species."""
    all_possible_m_ox = [0, 1, 2, 3, 4, 5, 6, 7]

    if rare:
        rare_states = [
            x for x in all_possible_m_ox if x not in (species.plausible_os or [])
        ]
        logger.debug("RARE METAL OXIDATION STATES: %s %s", species.formula, rare_states)
        return rare_states

    return list(species.plausible_os or [])


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


def _distribute_fixed_ligand_metal_charges(
    unique_indices, unique_species, input_charge=0, *, distributions=None, warnings=None
):
    """Choose metal oxidation states while keeping ligand charges fixed.

    Use donor formal charges to divide ligand charge among metals; shared donors
    split their contribution equally, and neutral ligands contribute nothing.
    Choose integer metal charges closest to these targets that balance the total
    charge, using supplied candidates when available. For ties, select the first
    in unique_index order and report metal_os_ambiguity.

    Return [chosen_distribution], or [] if data is missing or no solution exists.
    """
    metal_indices = [
        idx for idx, species in enumerate(unique_species) if species.subtype == "metal"
    ]
    if not metal_indices:
        return []
    counts = [unique_indices.count(idx) for idx in range(len(unique_species))]
    if any(counts[idx] == 0 for idx in metal_indices):
        return []
    # Fix non-metal charges before calculating the total charge metals must supply.
    charges = [0] * len(unique_species)
    for idx, species in enumerate(unique_species):
        if idx in metal_indices:
            continue
        states = species.plausible_charge_states or []
        if len(states) != 1:
            return []
        charges[idx] = states[0].species_total_charge
    remainder = input_charge - sum(
        charges[idx] for idx in unique_indices if idx not in metal_indices
    )

    # Use each molecule's own ligand copies so atom charges and coordination
    # indices refer to the same atom order. Visit each parent molecule once.
    molecules = {}
    for idx in metal_indices:
        metal = unique_species[idx]
        mol = metal.get_parent("molecule")
        if mol is not None:
            molecules[id(mol)] = mol
    burdens = {idx: [] for idx in metal_indices}
    for mol in molecules.values():
        metals = {
            metal.get_parent_index("molecule"): metal for metal in mol.metals or []
        }
        local = {index: 0.0 for index in metals}
        for ligand in mol.ligands or []:
            states = ligand.plausible_charge_states or []
            if len(states) != 1:
                return []
            state = states[0]
            net = state.species_total_charge
            if net == 0:
                continue
            # Map each coordinating ligand atom to the metals it contacts.
            contacts = {}
            for index, mode in ligand.coordination.items():
                if index not in metals:
                    return []
                atoms = set(mode["kappa_atoms"])
                atoms.update(atom for site in mode["eta_groups"] for atom in site)
                for atom in atoms:
                    contacts.setdefault(atom, set()).add(index)
            if not contacts:
                continue  # Unbound ions affect the total charge, not local targets.
            atom_charges = state.species_atom_charges
            if len(atom_charges) != ligand.natoms:
                return []
            # Normalize donor weights to the ligand's net charge. This avoids
            # counting internal positive/negative charge separation as extra charge.
            weights = {
                atom: abs(atom_charges[atom])
                for atom in contacts
                if atom_charges[atom] * net > 0
            }
            weight_sum = sum(weights.values())
            if not weight_sum:
                return []  # Cannot divide this ligand's charge using its donors.
            for atom, weight in weights.items():
                for index in contacts[atom]:
                    local[index] -= net * weight / weight_sum / len(contacts[atom])
        for index, metal in metals.items():
            unique_idx = metal.unique_index
            if unique_idx in burdens:
                burdens[unique_idx].append(local[index])

    if len(metal_indices) > 1 and any(not burdens[idx] for idx in metal_indices):
        return []
    # Copies sharing a unique_index must receive the same oxidation state.
    targets = {
        idx: sum(burdens[idx]) / len(burdens[idx]) if burdens[idx] else 0.0
        for idx in metal_indices
    }
    n_metals = sum(counts[idx] for idx in metal_indices)
    # Account for charge not allocated locally, such as unbound counterions.
    offset = (
        remainder - sum(counts[idx] * targets[idx] for idx in metal_indices)
    ) / n_metals
    targets = {idx: target + offset for idx, target in targets.items()}
    # For fallback inference, use integer charges spanning zero, the required
    # metal charge, and the local targets. Supplied candidates bypass this range.
    lower = math.floor(min(0, remainder, *targets.values()))
    upper = math.ceil(max(0, remainder, *targets.values()))
    options = range(lower, upper + 1)
    if (
        distributions is None
        and len(options) ** len(metal_indices) > _MAX_INFERENCE_COMBINATIONS
    ):
        return []
    combinations = (
        (tuple(dist[idx] for idx in metal_indices) for dist in distributions)
        if distributions is not None
        else itertools.product(options, repeat=len(metal_indices))
    )
    best_score = math.inf
    best = []
    for combo in combinations:
        if (
            sum(counts[idx] * charge for idx, charge in zip(metal_indices, combo))
            != remainder
        ):
            continue
        # Prefer charges close to the targets, counting each actual metal copy.
        score = sum(
            counts[idx] * (charge - targets[idx]) ** 2
            for idx, charge in zip(metal_indices, combo)
        )
        if score < best_score and not math.isclose(score, best_score, abs_tol=1e-9):
            best_score = score
            best = []
        if math.isclose(score, best_score, abs_tol=1e-9):
            candidate = charges.copy()
            for idx, charge in zip(metal_indices, combo):
                candidate[idx] = charge
            best.append(candidate)
    if best:
        logger.info(
            "Metal OS targets: %s; %d best charge distribution(s): %s",
            targets,
            len(best),
            best,
        )
    if len(best) > 1:
        # Make the representative reproducible without claiming the tie is resolved.
        best.sort()
        if warnings is not None:
            warnings["metal_os_ambiguity"] = True
        logger.warning(
            "Metal oxidation-state ambiguity: tied distributions %s. "
            "Selecting %s (unique_index order).",
            best,
            best[0],
        )
        return best[:1]
    return best


def _preferred_distributions(
    unique_indices, unique_species, distributions: List[List[int]]
) -> List[List[int]]:
    """Prefer metallocene Fe(II/III) and avoidable oxygen cations."""
    if not distributions:
        return []
    occurrences: dict[int, int] = {}
    for u_idx in unique_indices:
        occurrences[u_idx] = occurrences.get(u_idx, 0) + 1

    metal_indices = [
        idx for idx, species in enumerate(unique_species) if species.subtype == "metal"
    ]

    ferrocene_indices = [
        idx
        for idx in metal_indices
        if unique_species[idx].label == "Fe"
        and getattr(unique_species[idx], "groups", None)
        and is_metallocene(unique_species[idx])
    ]
    if ferrocene_indices:
        penalties = [
            sum(
                occurrences.get(idx, 0)
                for idx in ferrocene_indices
                if dist[idx] not in (2, 3)
            )
            for dist in distributions
        ]
        lowest_penalty = min(penalties)
        preferred = [
            dist
            for dist, penalty in zip(distributions, penalties)
            if penalty == lowest_penalty
        ]
        if len(preferred) < len(distributions):
            logger.info(
                "Metallocene Fe(II/III) preference kept %d of %d distributions.",
                len(preferred),
                len(distributions),
            )
        distributions = preferred

    # Metal frequency and total absolute ligand charge preferences are disabled.
    best = distributions
    # A lower oxygen penalty must not increase internal formal-charge pairing.
    metrics = [
        _oxygen_distribution_metrics(unique_indices, unique_species, dist)
        for dist in best
    ]
    survivors = [
        dist
        for i, dist in enumerate(best)
        if metrics[i] is None
        or not any(
            other is not None and other[0] < metrics[i][0] and other[1] <= metrics[i][1]
            for other in metrics
        )
    ]
    if len(survivors) < len(best):
        logger.info(
            "Oxygen cation preference kept %d of %d distributions without "
            "increasing internal ligand charge separation.",
            len(survivors),
            len(best),
        )
    return survivors


def _oxygen_distribution_metrics(unique_indices, unique_species, distribution):
    """Return occurrence-weighted (O positive charge, internal charge pairing).

    Use original species structures, never temporarily protonated structures.
    Missing or nonunique structures leave the distribution unranked.
    """
    oxygen = separation = 0
    for idx, (species, charge) in enumerate(zip(unique_species, distribution)):
        count = unique_indices.count(idx)
        if species.subtype == "metal" or not count:
            continue
        values = set()
        for state in getattr(species, "plausible_charge_states", None) or []:
            if state.species_total_charge != charge:
                continue
            mol = getattr(state, "species_rdkit_obj", None)
            if mol is None:
                return None
            atoms = list(mol.GetAtoms())
            charges = [atom.GetFormalCharge() for atom in atoms]
            if sum(charges) != charge:
                return None
            values.add(
                (
                    sum(
                        max(0, atom.GetFormalCharge())
                        for atom in atoms
                        if atom.GetAtomicNum() == 8
                    ),
                    (sum(abs(q) for q in charges) - abs(charge)) // 2,
                )
            )
        if len(values) != 1:
            return None
        o, pairing = values.pop()
        oxygen += count * o
        separation += count * pairing
    return oxygen, separation


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
