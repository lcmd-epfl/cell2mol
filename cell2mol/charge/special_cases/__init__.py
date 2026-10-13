"""Special chemical motifs and charge builders, grouped by chemical family.

The package keeps the former special_cases module's function imports available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cell2mol.classes.charge_state import ChargeState
    from cell2mol.classes.specie import Specie


from ._fragments import (
    _charge_capped_fragment,
    _collect_substituent_fragment,
)
from .borane import (
    MANUAL_CONJUNCTO_BORANE_P,
    ConjunctoBoraneSplit,
    _cage_face_sizes,
    _classify_cage,
    find_borane_cage_indices,
    find_borane_cages,
    find_conjuncto_borane_split,
    generate_borane_charge_state,
    generate_conjuncto_borane_charge_state,
    is_borane_cage,
)
from .charged_moieties import (
    _amidinium_rings,
    _classify_charged_moiety,
    _find_cationic_nitrogen,
    _find_charged_moiety,
    _nonmetal_connectivity,
    _rings_by_atom,
)
from .fullerene import (
    FullereneDimerSplit,
    _graph_girth,
    _valid_fullerene_vertex_count,
    check_fullerene_sphericity,
    find_fullerene_cage_indices,
    find_fullerene_dimer_split,
    generate_fullerene_charge_state,
    generate_fullerene_dimer_charge_states,
    has_fullerene,
    has_open_fullerene,
)
from .porphyrin import (
    _find_porphyrin_rings_and_bridges,
    _macrocycle_ring_order_and_bridges,
    find_all_porphyrin_macrocycles,
    find_porphyrin_macrocycle,
    generate_porphyrin_charge_state,
    is_porphyrin_macrocycle,
    porphyrin_reference_protonation_sites,
)
from .sb_halide import (
    generate_sb_halide_charge_state,
    is_sb_halide_only,
)
from .ttf import (
    _alternating_bond_path,
    _oxidised_donor_mol,
    find_dithiolylidene_units,
    generate_oxidised_donor_charge_states,
)

__all__ = [
    "generate_special_charge_states",
    "_charge_capped_fragment",
    "_collect_substituent_fragment",
    "MANUAL_CONJUNCTO_BORANE_P",
    "ConjunctoBoraneSplit",
    "_cage_face_sizes",
    "_classify_cage",
    "find_borane_cage_indices",
    "find_borane_cages",
    "find_conjuncto_borane_split",
    "generate_borane_charge_state",
    "generate_conjuncto_borane_charge_state",
    "is_borane_cage",
    "_amidinium_rings",
    "_classify_charged_moiety",
    "_find_cationic_nitrogen",
    "_find_charged_moiety",
    "_nonmetal_connectivity",
    "_rings_by_atom",
    "FullereneDimerSplit",
    "_graph_girth",
    "_valid_fullerene_vertex_count",
    "check_fullerene_sphericity",
    "find_fullerene_cage_indices",
    "find_fullerene_dimer_split",
    "generate_fullerene_charge_state",
    "generate_fullerene_dimer_charge_states",
    "has_fullerene",
    "has_open_fullerene",
    "_find_porphyrin_rings_and_bridges",
    "_macrocycle_ring_order_and_bridges",
    "find_all_porphyrin_macrocycles",
    "find_porphyrin_macrocycle",
    "generate_porphyrin_charge_state",
    "is_porphyrin_macrocycle",
    "porphyrin_reference_protonation_sites",
    "generate_sb_halide_charge_state",
    "is_sb_halide_only",
    "_alternating_bond_path",
    "_oxidised_donor_mol",
    "find_dithiolylidene_units",
    "generate_oxidised_donor_charge_states",
]


logger = logging.getLogger(__name__)


def generate_special_charge_states(spec: Specie) -> list[ChargeState] | None:
    """Build Sb-halide, fullerene or borane states.

    Return None to use general enumeration, a state list on success, or [] on failure.
    """
    protonation_states = spec.protonation_states
    if not protonation_states:
        return None

    prot0 = protonation_states[0]
    if is_sb_halide_only(spec.labels):
        charge_state = generate_sb_halide_charge_state(prot0)
        if charge_state is None:
            logger.warning(
                "Sb-halide charge builder failed for %s; leaving charge "
                "unassigned rather than falling back to the general search",
                spec.formula,
            )
        return [charge_state] if charge_state is not None else []

    has_fullerene = (
        spec.has_fullerene
        if spec.has_fullerene is not None
        else spec.detect_fullerene()
    )
    # Route dimers separately: their bridge carbons have degree four.
    if has_fullerene:
        if find_fullerene_dimer_split(prot0.atnums, prot0.adjmat) is not None:
            logger.debug("Specie %s is a fullerene dimer", spec.formula)
            dimer_charge_states = generate_fullerene_dimer_charge_states(prot0)
            if not dimer_charge_states:
                logger.warning(
                    "Fullerene dimer charge builder failed for %s; return None",
                    spec.formula,
                )
                return None
            return dimer_charge_states or []

        logger.debug("Specie %s has a fullerene cage", spec.formula)
        charge_state = generate_fullerene_charge_state(prot0)
        if charge_state is None:
            logger.warning(
                "Fullerene charge builder failed for %s; return None",
                spec.formula,
            )
            return None
        return [charge_state] if charge_state is not None else []

    # Fused cages fall through to general enumeration.
    has_borane = (
        spec.has_borane if spec.has_borane is not None else spec.detect_borane()
    )
    if has_borane:
        logger.debug("Specie %s has a borane/carborane cage", spec.formula)
        charge_state = generate_borane_charge_state(prot0)
        if charge_state is None:
            logger.warning(
                "Borane/carborane charge builder failed for %s; return None",
                spec.formula,
            )
            return None
        return [charge_state] if charge_state is not None else []

    return None
