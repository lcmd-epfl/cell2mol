"""Rank charge-state structures and prune bond-perception artifacts."""

from __future__ import annotations

import logging
from collections import defaultdict

from rdkit import Chem

from cell2mol.classes.charge_state import ChargeState
from cell2mol.charge.aromaticity import aromatic_info, _unearned_aromatic_atoms
from cell2mol.charge.molecule_builder import obligate_charge_separation_atoms

logger = logging.getLogger(__name__)


def select_and_prune_charge_states(
    charge_states: list[ChargeState],
) -> list[ChargeState]:
    """Select a representative structure for each charge, then prune artifact charges.

    Group valid candidates by species total charge and call select_best_structure_for_charge
    for each group. Return the surviving states as a list, or [] if none remain.
    This retains multiple possible charges; final charge selection is handled
    by the charge balancer.
    """
    valid = [s for s in charge_states if s is not None and s.is_valid]
    if not valid:
        return []

    by_charge: dict[int, list[ChargeState]] = defaultdict(list)
    for state in valid:
        by_charge[state.species_total_charge].append(state)

    best_per_charge = [
        select_best_structure_for_charge(by_charge[charge], charge)
        for charge in sorted(by_charge)
    ]
    return prune_charges(best_per_charge)


def select_best_structure_for_charge(
    candidates: list[ChargeState], charge: int
) -> ChargeState:
    """Select one Lewis structure from nonempty candidates sharing the given charge.

    Compare alternative structures within one charge and return a single
    ChargeState.
    """
    # Drop drawings that park charge on a carbon bonded to no metal, unless
    # every drawing does.
    candidates = _preferring_uncharged_carbons(candidates, charge)

    ranked = sorted(candidates, key=ranking_key)
    best = ranked[0]
    if len(ranked) > 1:
        logger.debug(
            "   Charge %+d: %d candidates, best key %s (%s), runner-up key %s",
            charge,
            len(ranked),
            ranking_key(best),
            best.species_smiles,
            ranking_key(ranked[1]),
        )

    return _better_drawn_sibling(best, candidates)


def ranking_key(state: ChargeState) -> tuple:
    """Sort key for structures of one charge; smaller is better."""
    earned_aromatic_atoms, spurious_separation = _structural_quality(state)
    return (
        state.protonated_sum_abs_atom_charges,  # fewest and smallest formal charges
        -earned_aromatic_atoms,  # most aromatic atoms
        _protons_on_aromatic_atoms(state),  # protons kept off aromatic atoms
        spurious_separation,  # least spurious +/- pairing
        state.species_smiles
        or state.protonated_smiles
        or "",  # deterministic tie-break
    )


def prune_charges(states: list[ChargeState]) -> list[ChargeState]:
    """Drop charges that are artifacts of bond order assignment."""
    states = _drop_lone_pair_stripped_charges(states)
    states = _drop_cationic_donors(states)
    states = _drop_charged_uncoordinated_carbons(states)
    return _drop_dominated_charges(states)


def _protons_on_aromatic_atoms(state: ChargeState) -> bool:
    """True when a temporary proton sits on an aromatic atom."""
    mol = state.protonated_rdkit_obj
    if mol is None:
        return False
    return any(
        n_added > 0 and mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx, n_added in enumerate(state.site_proton_counts or [])
    )


def _better_drawn_sibling(winner: ChargeState, pool: list[ChargeState]) -> ChargeState:
    """Replace an artifact zwitterion with a same-charge candidate that improves
    one structural metric without worsening the other.
    """
    if len(pool) < 2 or not _is_artifact_zwitterion(winner):
        return winner

    best = winner
    for candidate in pool:
        if candidate is best:
            continue
        q_new, q_best = _structural_quality(candidate), _structural_quality(best)
        if q_new[0] >= q_best[0] and q_new[1] <= q_best[1] and q_new != q_best:
            best = candidate

    if best is not winner:
        logger.debug(
            "   Better-drawn sibling at charge %+d: %s -> %s (%s)",
            winner.species_total_charge,
            _structural_quality(winner),
            _structural_quality(best),
            best.species_smiles,
        )
    return best


def _preferring_uncharged_carbons(
    candidates: list[ChargeState], charge: int
) -> list[ChargeState]:
    """Prefer drawings without charged uncoordinated carbons; keep all if none qualify."""
    clean = [c for c in candidates if not _has_charged_uncoordinated_carbon(c)]
    if not clean or len(clean) == len(candidates):
        return candidates

    logger.debug(
        "   Charge %+d: preferring %d of %d drawing(s) that leave uncoordinated "
        "carbons neutral",
        charge,
        len(clean),
        len(candidates),
    )
    return clean


def _protonation_pattern(state: ChargeState) -> tuple:
    """Protons added per site -- which states are drawings of the same structure."""
    return tuple(state.site_proton_counts or ())


def _bond_order_signature(state: ChargeState):
    """Return protonation, atom count and bonds without charges; None if no molecule."""
    mol = state.protonated_rdkit_obj
    if mol is None:
        return None
    bonds = tuple(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                b.GetBondTypeAsDouble(),
            )
            for b in mol.GetBonds()
        )
    )
    return tuple(state.site_proton_counts or ()), mol.GetNumAtoms(), bonds


def _drop_lone_pair_stripped_charges(states: list[ChargeState]) -> list[ChargeState]:
    """Drop states with identical bonds but larger absolute net charge.

    Equal absolute charges remain ambiguous; genuine redox states change bonds.
    """
    if len(states) < 2:
        return states

    signatures = [_bond_order_signature(s) for s in states]
    kept = []
    for idx, signature in enumerate(signatures):
        beaten_by = (
            None
            if signature is None
            else next(
                (
                    other.species_total_charge
                    for jdx, other in enumerate(states)
                    if jdx != idx
                    and signatures[jdx] == signature
                    and abs(other.species_total_charge)
                    < abs(states[idx].species_total_charge)
                ),
                None,
            )
        )
        if beaten_by is None:
            kept.append(states[idx])
        else:
            logger.debug(
                "   Dropping charge %+d: same bonds as charge %+d, the extra "
                "charge is a stripped lone pair (%s)",
                states[idx].species_total_charge,
                beaten_by,
                states[idx].species_smiles,
            )

    return kept or states


def _has_cationic_donor(state: ChargeState) -> bool:
    """True when an atom coordinating a metal carries a positive formal charge."""
    parent = getattr(state.protonation, "parent", None)
    atoms = getattr(parent, "atoms", None) or []
    charges = state.species_atom_charges or []
    if not atoms or len(atoms) != len(charges):
        return False
    return any(
        (a.metal_contact_count or 0) > 0 and q > 0 for a, q in zip(atoms, charges)
    )


def _drop_cationic_donors(states: list[ChargeState]) -> list[ChargeState]:
    """Remove states with positively charged metal donors if any alternative remains."""
    if len(states) < 2:
        return states

    clean = [s for s in states if not _has_cationic_donor(s)]
    if not clean or len(clean) == len(states):
        return states

    for state in states:
        if _has_cationic_donor(state):
            logger.debug(
                "   Dropping charge %+d: positive formal charge on a metal donor (%s)",
                state.species_total_charge,
                state.species_smiles,
            )
    return clean


def _has_charged_uncoordinated_carbon(state: ChargeState) -> bool:
    """True when a carbon that coordinates nothing carries a formal charge."""
    parent = getattr(state.protonation, "parent", None)
    atoms = getattr(parent, "atoms", None) or []
    charges = state.species_atom_charges or []
    if not atoms or len(atoms) != len(charges):
        return False
    return any(
        a.label == "C" and (a.metal_contact_count or 0) == 0 and q != 0
        for a, q in zip(atoms, charges)
    )


def _drop_charged_uncoordinated_carbons(
    states: list[ChargeState],
) -> list[ChargeState]:
    """Drop charges that park formal charge on a carbon bonded to no metal, as long
    as something is left.
    """
    if len(states) < 2:
        return states

    clean = [s for s in states if not _has_charged_uncoordinated_carbon(s)]
    if not clean or len(clean) == len(states):
        return states

    for state in states:
        if _has_charged_uncoordinated_carbon(state):
            logger.debug(
                "   Dropping charge %+d: formal charge on a carbon that "
                "coordinates no metal (%s)",
                state.species_total_charge,
                state.species_smiles,
            )
    return clean


def _drop_dominated_charges(states: list[ChargeState]) -> list[ChargeState]:
    """Remove states dominated in aromaticity and artifact separation.

    Within one protonation pattern, break structural ties by absolute net charge.
    """
    if len(states) < 2:
        return states

    quality = [_structural_quality(s) for s in states]
    kept = []
    for i, (arom_i, sep_i) in enumerate(quality):
        dominated_by = next(
            (
                (states[j].species_total_charge, arom_j, sep_j)
                for j, (arom_j, sep_j) in enumerate(quality)
                if j != i
                and (
                    (arom_j >= arom_i and sep_j < sep_i)
                    or (
                        arom_j > arom_i
                        and sep_j <= sep_i
                        and abs(states[j].species_total_charge)
                        <= abs(states[i].species_total_charge)
                    )
                    or (
                        arom_j == arom_i
                        and sep_j == sep_i
                        and _protonation_pattern(states[j])
                        == _protonation_pattern(states[i])
                        and abs(states[j].species_total_charge)
                        < abs(states[i].species_total_charge)
                    )
                )
            ),
            None,
        )
        if dominated_by is None:
            kept.append(states[i])
        else:
            winner_charge, winner_arom, winner_sep = dominated_by
            logger.debug(
                "   Dropping charge %+d (aromatic %d, separation %d): dominated "
                "by charge %+d (aromatic %d, separation %d)",
                states[i].species_total_charge,
                arom_i,
                sep_i,
                winner_charge,
                winner_arom,
                winner_sep,
            )

    # Strict domination cannot remove every state; retain a defensive fallback.
    return kept or states


def _artifact_separation(state: ChargeState) -> int:
    """Return min(positive charge, negative charge) in the species frame,
    excluding obligate pairs. Added protons can hide separation in the protonated frame.
    """
    charges = state.species_atom_charges or state.protonated_atom_charges or []
    mol = (
        state.species_rdkit_obj
        if state.species_atom_charges
        else state.protonated_rdkit_obj
    )
    obligate: set[int] = set()
    if mol is not None and mol.GetNumAtoms() == len(charges):
        obligate = obligate_charge_separation_atoms(mol)

    free = [q for idx, q in enumerate(charges) if idx not in obligate]
    return min(sum(q for q in free if q > 0), -sum(q for q in free if q < 0))


def _is_artifact_zwitterion(charge_state: ChargeState) -> bool:
    """Detect zwitterions after excluding obligate charge separation."""
    mol = charge_state.species_rdkit_obj
    charges = charge_state.species_atom_charges or []
    if mol is None or mol.GetNumAtoms() != len(charges):
        return bool(charge_state.species_is_zwitterionic)

    obligate = obligate_charge_separation_atoms(mol)
    if not obligate:
        return bool(charge_state.species_is_zwitterionic)

    remaining = [q for idx, q in enumerate(charges) if idx not in obligate]
    return any(q > 0 for q in remaining) and any(q < 0 for q in remaining)


def _drawing_looks_forced(state: ChargeState) -> bool:
    """Flag charged uncoordinated carbons or avoidable separation for a second opinion."""
    return _has_charged_uncoordinated_carbon(state) or _is_artifact_zwitterion(state)


def _structural_quality(state: ChargeState) -> tuple[int, int]:
    """Return earned aromatic atoms in the protonated frame and artifact
    separation in the species frame; prefer more of the former and less of the latter.
    """
    added_indices = [
        idx for idx, n in enumerate(state.protonation.site_proton_counts or []) if n > 0
    ]
    aromatic = aromatic_info(state.protonated_rdkit_obj, added_indices=added_indices)
    unearned = _unearned_aromatic_atoms(state.protonated_rdkit_obj)
    if unearned:
        logger.debug(
            "   Discounting %d unearned aromatic atom(s) %s for charge %+d (%s)",
            len(unearned),
            sorted(unearned),
            state.species_total_charge,
            state.species_smiles,
        )
    return int(aromatic["Aromatic atoms"]) - len(unearned), _artifact_separation(state)
