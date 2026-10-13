from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie

logger = logging.getLogger(__name__)


############################################################
# Charged-moiety detection
############################################################


def _find_charged_moiety(
    spec: Specie,
) -> list[tuple[int, list[int], str, int]]:
    """Find net-charged oxo groups and cationic N, ignoring metal coordination.

    Return (centre index, atom indices, kind, net charge) in specie atom order.
    """
    atoms = spec.atoms or []
    parent_molecule = spec.get_parent("molecule")
    neighbor_source = (
        parent_molecule.atoms
        if parent_molecule is not None and parent_molecule.atoms is not None
        else atoms
    )
    local_idx_by_id = {id(a): idx for idx, a in enumerate(atoms)}

    moieties: list[tuple[int, list[int], str, int]] = []
    for i, atom in enumerate(atoms):
        if atom.label not in (
            "C",
            "S",
        ):
            continue

        # Non-metal neighbours only (adjacency minus the metal subset).
        nonmetal_neighbor_idxs = [
            j for j in atom.adjacency if j not in atom.metal_adjacency
        ]
        n_nonmetal = len(nonmetal_neighbor_idxs)
        n_oxygen = 0
        terminal_oxygen_locals: list[int] = []
        for j in nonmetal_neighbor_idxs:
            neighbor = neighbor_source[j]
            if neighbor.label != "O":
                continue
            n_oxygen += 1
            # Terminal O: the centre is its only non-metal neighbour (connec == 1).
            neighbor_nonmetal_neighbor_idxs = [
                j for j in neighbor.adjacency if j not in neighbor.metal_adjacency
            ]

            if len(neighbor_nonmetal_neighbor_idxs) == 1:
                local_o = local_idx_by_id.get(id(neighbor))
                if local_o is not None:
                    terminal_oxygen_locals.append(local_o)

        kind, net_charge = _classify_charged_moiety(
            atom.label, len(terminal_oxygen_locals), n_nonmetal, n_oxygen
        )
        if kind is None:
            continue
        atom_locals = terminal_oxygen_locals if terminal_oxygen_locals else [i]
        moieties.append((i, atom_locals, kind, net_charge))

    moieties.extend(_find_cationic_nitrogen(atoms, neighbor_source, local_idx_by_id))
    return moieties


def _find_cationic_nitrogen(
    atoms, neighbor_source, local_idx_by_id
) -> list[tuple[int, list[int], str, int]]:
    """Find connectivity-pinned +1 charges on quaternary N, pyridinium N or
    five-ring amidinium groups. Exclude other six-ring donors and guanidinium.
    """
    rings_by_atom = _rings_by_atom(atoms, neighbor_source, local_idx_by_id)

    def _is_pyridine_ring(ring: list[int]) -> bool:
        if len(ring) != 6:
            return False
        ring_labels = [atoms[j].label for j in ring]
        return ring_labels.count("N") == 1 and ring_labels.count("C") == 5

    cations: list[tuple[int, list[int], str, int]] = []
    for i, atom in enumerate(atoms):
        # A cation has no lone pair to donate, so it is never a metal donor.
        if atom.label != "N" or atom.metal_adjacency:
            continue

        neighbors = [
            neighbor_source[j] for j in atom.adjacency if j not in atom.metal_adjacency
        ]
        # Terminal O -> N-oxide / nitro / azide: obligate pair, counted elsewhere.
        if any(n.label == "O" and _nonmetal_connectivity(n) == 1 for n in neighbors):
            continue

        if len(neighbors) == 4:
            cations.append((i, [i], "quaternary-N", 1))
            continue

        rings = rings_by_atom.get(i)
        if len(neighbors) != 3 or not rings:
            continue
        # Exclude N in smaller fused rings from pyridinium detection.
        if min(len(ring) for ring in rings) != 6:
            continue
        if not any(_is_pyridine_ring(ring) for ring in rings):
            continue

        sp2_neighbors = sum(1 for n in neighbors if _nonmetal_connectivity(n) == 3)
        if sp2_neighbors >= 2:
            cations.append((i, [i], "pyridinium-N", 1))

    cations.extend(_amidinium_rings(atoms, neighbor_source, rings_by_atom))
    return cations


############################################################
# Classification and ring helpers
############################################################


def _classify_charged_moiety(
    label: str, n_terminal_oxygen: int, n_nonmetal: int, n_oxygen: int
) -> tuple[str | None, int]:
    """Return (kind, net charge) from terminal oxygens and connectivity,
    excluding neutral CO2 and sulfones; return (None, 0) if unmatched.
    """
    if label == "C":
        if (
            n_terminal_oxygen == 2 and n_nonmetal == 3
        ):  # R-COO(-): 2 terminal O + 1 substituent
            return "carboxylate", -1
        if n_terminal_oxygen == 3:
            return "carbonate", -2
    elif label == "S":
        if n_terminal_oxygen == 3:
            return "sulfonate", -1
        if n_terminal_oxygen == 4:
            return "sulfate", -2
        if n_terminal_oxygen == 2 and n_nonmetal == 3:  # 2 O + 1 C -> R-SO2(-)
            return "sulfinate", -1

    # Cationic N needs ring size, which this signature lacks: _find_cationic_nitrogen.
    return None, 0


def _amidinium_rings(
    atoms, neighbor_source, rings_by_atom
) -> list[tuple[int, list[int], str, int]]:
    """Find five-ring amidinium cations with two N and three-connected ring atoms;
    report each delocalized ring charge once.
    """
    seen: set[frozenset[int]] = set()
    cations: list[tuple[int, list[int], str, int]] = []
    for rings in rings_by_atom.values():
        for ring in rings:
            key = frozenset(ring)
            if len(ring) != 5 or key in seen:
                continue
            seen.add(key)

            if any(_nonmetal_connectivity(atoms[j]) != 3 for j in ring):
                continue
            nitrogens = [j for j in ring if atoms[j].label == "N"]
            # A cation has no lone pair to donate, so it is never a metal donor.
            if len(nitrogens) != 2 or any(atoms[j].metal_adjacency for j in nitrogens):
                continue
            # Terminal O/S on the ring -> cyclic urea or thiourea, neutral.
            if any(
                neighbor_source[k].label in ("O", "S")
                and _nonmetal_connectivity(neighbor_source[k]) == 1
                for j in ring
                for k in atoms[j].adjacency
                if k not in atoms[j].metal_adjacency
            ):
                continue
            cations.append((nitrogens[0], nitrogens, "amidinium-N", 1))
    return cations


def _rings_by_atom(
    atoms, neighbor_source, local_idx_by_id
) -> dict[int, list[list[int]]]:
    """The smallest rings each atom belongs to (minimum cycle basis)."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    for i, atom in enumerate(atoms):
        for j in atom.adjacency:
            if j in atom.metal_adjacency:
                continue
            local_j = local_idx_by_id.get(id(neighbor_source[j]))
            if local_j is None or local_j == i:
                continue
            graph.add_edge(i, local_j)

    rings: dict[int, list[list[int]]] = {}
    for cycle in nx.minimum_cycle_basis(graph):
        for idx in cycle:
            rings.setdefault(idx, []).append(list(cycle))
    return rings


def _nonmetal_connectivity(atom) -> int:
    """Sigma bonds to non-metals, the count every rule here is keyed on."""
    return len([j for j in atom.adjacency if j not in atom.metal_adjacency])
