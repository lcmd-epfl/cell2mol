"""Enumerate Lewis structures, select one per charge and prune perception artifacts.

The charge balancer chooses the final charge using cell neutrality.
See README.md for chemistry rules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit import Chem

from cell2mol.charge.bond_orders import generate_valid_charge_states
from cell2mol.charge.state_ranking import select_and_prune_charge_states

from cell2mol.charge.special_cases import (
    _find_charged_moiety,
    generate_oxidised_donor_charge_states,
    generate_porphyrin_charge_state,
    generate_special_charge_states,
)
from cell2mol.charge.special_cases.manual import (
    MANUAL_CHARGE_ASSIGN_SPECIES,
    generate_manual_charge_state,
)
from cell2mol.element_utils import HALOGENS, NOBLE_GASES
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation

if TYPE_CHECKING:
    from cell2mol.classes.species import Species

logger = logging.getLogger(__name__)


############################################################
# Charge state generation and selection
############################################################


def generate_charge_states(species: Species) -> list[ChargeState] | None:
    """Best Lewis structure for each charge a ligand or metal-free molecule can
    carry, or None if bond order perception found nothing.
    """
    if species.protonation_states is None:
        species.get_protonation_states()
    if not species.protonation_states:
        logger.warning("No protonation states available for %s", species.formula)
        return None

    if species.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        charge_state = generate_manual_charge_state(species)
        return [charge_state] if charge_state is not None else None

    special_charge_states = generate_special_charge_states(species)
    if special_charge_states is not None:
        return special_charge_states

    diagnostics: dict[str, bool] = {}
    valid_states = _enumerate_valid_states(species, diagnostics)

    best = select_and_prune_charge_states(valid_states)

    # TTF-type donors only. Their +1/+2 states are distinct redox states,
    # so they are appended after selection, not ranked.
    best = _with_ttf_oxidised_states(best)

    # Report why bond order assignment gave up only when it found nothing at all.
    found_nothing = not best
    species.valence_search_too_large = bool(
        diagnostics.get("valence_search_too_large") and found_nothing
    )
    species.bond_perception_capped = bool(
        diagnostics.get("bond_perception_capped") and found_nothing
    )
    return best or None


def _enumerate_valid_states(
    species: Species, diagnostics: dict[str, bool]
) -> list[ChargeState]:
    """Every valid structure over all protonation states and candidate charges."""
    protonation_states = species.protonation_states
    if not protonation_states:
        return []

    # Porphyrin fast path. It may decline, so fall back on an empty result.
    states = [
        state
        for prot in protonation_states
        if prot.origin == "porphyrin"
        for state in [generate_porphyrin_charge_state(prot)]
        if state is not None and state.is_valid
    ]
    if states:
        return states

    for prot in protonation_states:
        candidate_charges = get_candidate_charges(species, prot)
        logger.debug(
            "Enumerating %s | protonation %s (nH=%d) | candidate charges %s",
            species.formula,
            prot.formula,
            prot.n_protons_added,
            candidate_charges,
        )
        by_charge = generate_valid_charge_states(
            prot, candidate_charges, diagnostics=diagnostics
        )
        states.extend(
            state
            for group in by_charge.values()
            for state in group
            if state is not None and state.is_valid
        )
    return states


def _with_ttf_oxidised_states(states: list[ChargeState]) -> list[ChargeState]:
    """Append the oxidised states of a tetrathiafulvalene-type donor."""
    neutral = next((s for s in states if s.species_total_charge == 0), None)
    if neutral is None:
        return states

    found = {s.species_total_charge for s in states}
    added = [
        state
        for state in generate_oxidised_donor_charge_states(neutral)
        if state.species_total_charge not in found
    ]
    if added:
        logger.debug(
            "Tetrathiafulvalene-type donor %s: adding oxidised states %s",
            neutral.protonation.formula,
            [s.species_total_charge for s in added],
        )
    return states + added


############################################################
# Candidate charge rules
############################################################


def get_candidate_charges(species: Species, prot: Protonation) -> list[int]:
    """Return parity-compatible charges in the protonated frame.

    Candidates stay fixed across protonation states so added protons lower the
    species charge. Use a formula or moiety anchor, otherwise a charge sweep.
    """
    anchor = _anchored_species_charges(species)
    candidates = list(anchor) if anchor is not None else _sweep_charges(species)

    kept = _filter_by_electron_parity(candidates, prot)
    if not kept:
        # If no candidate passes parity, lower charges by one to cover missed anionic centres.
        widened = [c - 1 for c in candidates]
        kept = _filter_by_electron_parity(widened, prot)
        logger.debug(
            "Anchor %s is parity-forbidden for %s; widening downward to %s",
            candidates,
            prot.formula,
            kept,
        )
    if not kept:  # defensive: never hand back an empty candidate list
        kept = candidates
    return kept


def _anchored_species_charges(species: Species) -> list[int] | None:
    """Fixed formula or moiety charges used as protonated-frame candidates."""
    formula = species.formula

    if formula in {"C-O", "H2-O", "C-N", "C-S", "C-Se", "C-Te", "C-P", "C-As", "C-Sb"}:
        return [0]
    if formula in HALOGENS:
        return [-1]
    if formula in NOBLE_GASES:
        return [0]

    # Moiety anchors avoid expensive polycarboxylate charge sweeps.
    charged_moieties = _find_charged_moiety(species)

    # Cation-only detection may miss delocalized anions; use the sweep instead.
    if not any(net_charge < 0 for *_, net_charge in charged_moieties):
        return None

    # Use the moiety sum; protonation covers additional anionic sites.
    anchor = sum(net_charge for *_, net_charge in charged_moieties)
    logger.debug(
        "Charged moieties in %s: %s -> candidate charge %d",
        formula,
        [(kind, nc) for _c, _a, kind, nc in charged_moieties],
        anchor,
    )
    return [anchor]


def _sweep_charges(species: Species) -> list[int]:
    """Fallback protonated-charge sweep for a species with no chemical anchor."""
    if species.is_metal_free_molecule:
        return [0, -1, 1, -2, 2, -3, 3, -4, 4]
    return [0, -1, 1, -2, 2]


def _filter_by_electron_parity(charges: list[int], prot: Protonation) -> list[int]:
    """Keep charges with an even valence-electron count; may return an empty list."""
    atnums = prot.atomic_numbers
    if not atnums:
        return list(charges)

    pt = Chem.GetPeriodicTable()
    n_valence = sum(pt.GetNOuterElecs(int(z)) for z in atnums)
    kept = [c for c in charges if (n_valence - c) % 2 == 0]

    if kept and len(kept) < len(charges):
        logger.debug(
            "Electron-parity screen for %s (%d valence electrons): %s -> %s",
            prot.formula,
            n_valence,
            charges,
            kept,
        )
    return kept
