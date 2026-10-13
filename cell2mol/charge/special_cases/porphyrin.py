from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from cell2mol.charge.validation import check_rdkit_obj_connectivity
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation
from cell2mol.connectivity import try_add_ligand_atom

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie

from ._fragments import _charge_capped_fragment, _collect_substituent_fragment

logger = logging.getLogger(__name__)


############################################################
# Macrocycle detection
############################################################


def is_porphyrin_macrocycle(atoms, AC) -> tuple[bool, str]:
    """True if the connectivity contains an N4 macrocycle, regardless of substituents."""
    result = find_porphyrin_macrocycle(atoms, AC)
    if result is None:
        return False, "no_porphyrin_macrocycle"
    _nitrogens, is_contracted, _core_atoms = result
    if is_contracted:
        return True, "corrole_or_corrin_macrocycle_confirmed"
    return True, "porphyrin_macrocycle_confirmed"


def find_porphyrin_macrocycle(atoms, AC) -> tuple[list[int], bool, list[int]] | None:
    """The first entry of find_all_porphyrin_macrocycles, or None."""
    macrocycles = find_all_porphyrin_macrocycles(atoms, AC)
    return macrocycles[0] if macrocycles else None


def find_all_porphyrin_macrocycles(
    atoms, AC
) -> list[tuple[list[int], bool, list[int]]]:
    """Return (cyclic N indices, contracted, core atoms) for each disjoint
    macrocycle; positions i and i+2 are trans for N4 cores.
    """
    atoms = [int(a) for a in atoms]
    macrocycles = []
    for ordered_rings, ordered_bridges in _find_porphyrin_rings_and_bridges(atoms, AC):
        nitrogens = [next(i for i in ring if atoms[i] == 7) for ring in ordered_rings]
        is_contracted = any(bridge is None for bridge in ordered_bridges)
        core_atoms = sorted(
            set().union(*ordered_rings)
            | {bridge for bridge in ordered_bridges if bridge is not None}
        )
        macrocycles.append((nitrogens, is_contracted, core_atoms))
    return macrocycles


############################################################
# Protonation-state generation
############################################################


def porphyrin_reference_protonation_sites(
    macrocycle_nitrogens: list[int], is_contracted: bool
) -> list[int]:
    """Return free-base N-H sites: trans N4 pair, three for a contracted ring,
    or alternating nitrogens for an expanded ring.
    """
    if len(macrocycle_nitrogens) == 4:
        if is_contracted:
            return macrocycle_nitrogens[:3]
        return [macrocycle_nitrogens[0], macrocycle_nitrogens[2]]
    return macrocycle_nitrogens[0::2]


def _generate_porphyrin_protonation_states(
    specie: Specie,
    macrocycles: list[tuple[list[int], bool, list[int]]],
) -> list[Protonation]:
    """Generate free-base N-H patterns: trans N4 pairs, 1/3 H for contracted
    rings, or m0 ± 1 for expanded rings. Fall back to the as-is state.
    """
    asis_state = Protonation.from_positional(
        labels=specie.labels,
        coord=specie.coord,
        cov_factor=specie.cov_factor,
        n_protons_added=0,
        site_proton_counts=[0] * len(specie.labels),
        ligand_donor_electrons=[0] * len(specie.labels),
        mode="none",
        parent=specie,
    )

    # Baseline free-base sites (m0): alternating N-H per ring.
    all_nitrogens: list[int] = []
    base_sites: list[int] = []
    is_expanded = False
    is_contracted = False
    for macrocycle_nitrogens, is_contracted_ring, _core_atoms in macrocycles:
        if macrocycle_nitrogens is None or len(macrocycle_nitrogens) < 4:
            continue
        all_nitrogens.extend(macrocycle_nitrogens)
        base_sites.extend(
            porphyrin_reference_protonation_sites(
                macrocycle_nitrogens, is_contracted_ring
            )
        )
        if len(macrocycle_nitrogens) >= 5:
            is_expanded = True
        elif is_contracted_ring:
            is_contracted = True

    # Preserve insertion order while de-duplicating.
    base_sites = list(dict.fromkeys(base_sites))

    if not base_sites:
        return [asis_state]

    # Only unsubstituted coordinating ring N can accept a free-base proton.
    adjmat = np.asarray(specie.adjmat)
    atoms = specie.atoms or []

    def _can_be_protonated(idx: int) -> bool:
        neighbours = np.nonzero(adjmat[idx])[0]
        if (atoms[idx].mconnec or 0) == 0:
            return False
        if any(specie.labels[j] == "H" for j in neighbours):
            return False
        # A third heavy neighbour is an exocyclic substituent.
        return sum(1 for j in neighbours if specie.labels[j] != "H") <= 2

    base_add = [n for n in base_sites if _can_be_protonated(n)]
    extra_bare = [
        n for n in all_nitrogens if n not in base_sites and _can_be_protonated(n)
    ]

    # Classic N4: m0 only. Expanded: bracket m0 +/- 1. Ring-contracted k=4: emit
    # both corrole (3 N-H) and corrin (1 N-H); connectivity cannot tell them apart.
    site_sets: list[list[int]] = [base_add]
    if is_expanded:
        if extra_bare:
            site_sets.append(base_add + [extra_bare[0]])  # m0 + 1
        if len(base_add) > 1:
            site_sets.append(base_add[:-1])  # m0 - 1
    elif is_contracted and len(base_add) >= 1:
        site_sets.append(base_add[:1])  # corrin: 1 N-H

    states: list[Protonation] = []
    for sites in site_sets:
        newlab = list(specie.labels)
        newcoord = list(specie.coord)
        for site in sites:
            _, newlab, newcoord = try_add_ligand_atom(
                newlab, newcoord, site, specie, element="H", unconditional=True
            )

        site_proton_counts = [0] * len(specie.labels)
        for site in sites:
            site_proton_counts[site] = 1

        states.append(
            Protonation.from_positional(
                labels=newlab,
                coord=np.asarray(newcoord),
                cov_factor=specie.cov_factor,
                n_protons_added=len(sites),
                site_proton_counts=site_proton_counts,
                mode="deterministic",
                parent=specie,
                ligand_donor_electrons=[0] * len(specie.labels),
                origin="porphyrin",
            )
        )

    return states


############################################################
# Charge-state generation
############################################################


def generate_porphyrin_charge_state(prot: Protonation) -> ChargeState | None:
    """Build a neutral Kekulized free-base core; specie charge is minus added
    protons plus substituent charges. Return None to request general enumeration.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    macrocycles = find_all_porphyrin_macrocycles(prot.atnums, prot.adjmat)
    if not macrocycles:
        logger.warning("No porphyrin-family macrocycle found for %s", prot.formula)
        return None
    # Macrocycle cores are disjoint, so several (bis-porphyrin) form one core.
    nitrogens = [n for mac_nitrogens, _c, _core in macrocycles for n in mac_nitrogens]
    core_sets = [set(core) for _n, _c, core in macrocycles]
    core_atoms = sorted(set().union(*core_sets))

    adjmat = np.asarray(prot.adjmat)
    core_set = set(core_atoms)
    exo_indices = [i for i in range(prot.natoms) if i not in core_set]

    def _has_explicit_h(atom_idx: int) -> bool:
        return any(
            prot.labels[j] == "H" and int(np.count_nonzero(adjmat[j])) == 1
            for j in np.nonzero(adjmat[atom_idx])[0]
        )

    # Pyrrolic N-H and imine-type N are both neutral in the free base.
    reference_sites = {n for n in nitrogens if _has_explicit_h(n)}

    atom_charges = [0] * prot.natoms
    for n in nitrogens:
        protonated = _has_explicit_h(n)
        if n in reference_sites:
            atom_charges[n] = 0 if protonated else -1
        else:
            atom_charges[n] = 1 if protonated else 0

    # -(protons added) holds only if every proton landed on a nitrogen.
    misplaced = [
        idx
        for idx, n_added in enumerate(prot.site_proton_counts or [])
        if n_added and prot.atnums[idx] != 7
    ]
    if misplaced:
        logger.warning(
            "Porphyrin-family specie %s protonates %d non-nitrogen site(s) %s; "
            "its charge is not -(protons added)",
            prot.formula,
            len(misplaced),
            misplaced,
        )

    # Non-H substituents are charged separately, as for fullerenes and boranes.
    simple_exo = {
        i
        for i in exo_indices
        if prot.labels[i] == "H" and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [i for i in exo_indices if i not in simple_exo]

    fragment_bond_orders: dict[frozenset[int], Chem.BondType] = {}
    visited_complex: set[int] = set()
    for root in complex_roots:
        if root in visited_complex:
            continue
        fragment = _collect_substituent_fragment(root, core_atoms, adjmat)
        visited_complex.update(fragment)

        attach_bonds = [
            (c, s) for c in core_atoms for s in fragment if adjmat[c, s] != 0
        ]
        if not attach_bonds:
            logger.warning(
                "Porphyrin-family specie %s has a substituent fragment with "
                "no bond back to the macrocycle; cannot split",
                prot.formula,
            )
            return None
        else:
            pass
        attach_atoms = [s for _c, s in attach_bonds]

        frag_charge, frag_atom_charges, frag_bond_orders = _charge_capped_fragment(
            fragment, attach_atoms, adjmat, prot.atnums, prot.formula
        )
        logger.debug(
            "Porphyrin-family specie %s substituent fragment %s has charge %d",
            prot.formula,
            fragment,
            frag_charge,
        )
        if frag_charge is None:
            return None
        for global_i, charge in frag_atom_charges.items():
            atom_charges[global_i] = charge
        fragment_bond_orders.update(frag_bond_orders)

    # sp3 ring atoms (a chlorin) would exceed valence as aromatic; bond them single.
    saturated_core = {
        i
        for i in core_atoms
        if int(np.count_nonzero(adjmat[i])) >= 4 and prot.labels[i] in ("C", "N")
    }
    if saturated_core:
        logger.debug(
            "Porphyrin-family specie %s has %d saturated ring atom(s) %s; "
            "bonding them as single rather than aromatic",
            prot.formula,
            len(saturated_core),
            sorted(saturated_core),
        )

    rwmol = Chem.RWMol()
    for atomic_num in prot.atnums:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    for i in range(prot.natoms):
        for j in range(i + 1, prot.natoms):
            if adjmat[i, j] == 0:
                continue
            pair = frozenset((i, j))
            if pair in fragment_bond_orders:
                bond_type = fragment_bond_orders[pair]
            elif (
                any(i in cs and j in cs for cs in core_sets)
                and i not in saturated_core
                and j not in saturated_core
            ):
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE
            rwmol.AddBond(i, j, bond_type)

    for i in core_atoms:
        if i not in saturated_core:
            rwmol.GetAtomWithIdx(i).SetIsAromatic(True)

    for n in nitrogens:
        rwmol.GetAtomWithIdx(n).SetNoImplicit(True)

    for i, charge in enumerate(atom_charges):
        if charge != 0:
            rwmol.GetAtomWithIdx(i).SetFormalCharge(charge)

    mol = rwmol.GetMol()

    conf = Chem.Conformer(prot.natoms)
    conf.Set3D(True)
    for i, xyz in enumerate(np.asarray(prot.coord, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    try:
        # Charges and NoImplicit already fix the pattern; skip re-perceiving aromaticity.
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
        )
    except Exception as e:
        logger.warning(
            "Porphyrin fast path declined for %s (not a kekulizable aromatic "
            "macrocycle: %s); falling back to general charge enumeration",
            prot.formula,
            e,
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    is_correct = check_rdkit_obj_connectivity(mol, prot.natoms, total_charge)

    return ChargeState.from_positional(
        is_correct,
        total_charge,
        atom_charges,
        mol,
        smiles,
        total_charge,
        True,
        prot,
    )


############################################################
# Metal-bound N4 pocket helpers
############################################################


def _metal_pocket_macrocycles(
    specie: Specie,
) -> list[tuple[list[int], bool, list[int]]]:
    """Build one N4 pseudo-macrocycle per metal; require four unsubstituted N donors each."""
    molecule = specie.get_parent("molecule")
    atoms = specie.atoms or []
    if molecule is None or not atoms:
        return []

    adjmat = np.asarray(specie.adjmat)
    pockets: dict[int, list[int]] = {}
    for idx, atom in enumerate(atoms):
        for metal_idx in atom.metal_adjacency or []:
            heavy = sum(
                1 for j in np.nonzero(adjmat[idx])[0] if specie.labels[j] != "H"
            )
            if atom.label != "N" or heavy > 2:
                return []
            pockets.setdefault(metal_idx, []).append(idx)

    if not pockets or any(len(nitrogens) != 4 for nitrogens in pockets.values()):
        return []

    return [
        (
            _order_pocket_trans(
                molecule.atoms[metal_idx].coord, nitrogens, specie.coord
            ),
            False,
            sorted(nitrogens),
        )
        for metal_idx, nitrogens in pockets.items()
    ]


def _order_pocket_trans(metal_coord, nitrogens: list[int], coords) -> list[int]:
    """Order four donors with trans pairs at 0/2 and 1/3, minimizing paired vector dot products."""
    units = {}
    for idx in nitrogens:
        vec = np.asarray(coords[idx], float) - np.asarray(metal_coord, float)
        norm = np.linalg.norm(vec)
        units[idx] = vec / norm if norm > 0 else vec

    a, b, c, d = nitrogens
    pairings = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    (p0, p1), (p2, p3) = min(
        pairings,
        key=lambda pairing: sum(float(np.dot(units[p], units[q])) for p, q in pairing),
    )
    return [p0, p2, p1, p3]


############################################################
# Macrocycle graph helpers
############################################################


def _find_porphyrin_rings_and_bridges(
    atoms, AC
) -> list[tuple[list[frozenset[int]], list[int | None]]]:
    """Find cyclic pyrrole groups joined by meso atoms or direct bonds.

    Return (rings, bridges) per macrocycle; bridges[i] links rings i and i+1,
    with None denoting a direct bond.
    """
    atoms = [int(a) for a in atoms]
    # Ligand adjmats can be dtype=object, which networkx rejects.
    ac = np.asarray(AC, dtype=int)
    n = len(atoms)

    if ac.shape != (n, n):
        return []

    n_nitrogen = sum(1 for z in atoms if z == 7)
    n_carbon = sum(1 for z in atoms if z == 6)
    # 4 rings x 4 C; meso bridges may be N (phthalocyanine), so no more.
    if n_nitrogen < 4 or n_carbon < 16:
        return []

    graph = nx.from_numpy_array(ac)

    # Pyrrole-type rings: 5-membered, one N and four C.
    pyrrole_rings: set[frozenset[int]] = set()
    for cycle in nx.simple_cycles(graph, length_bound=5):
        if len(cycle) != 5:
            continue
        n_count = sum(1 for i in cycle if atoms[i] == 7)
        c_count = sum(1 for i in cycle if atoms[i] == 6)
        if n_count == 1 and c_count == 4:
            pyrrole_rings.add(frozenset(cycle))

    if len(pyrrole_rings) < 4:
        return []

    # One ring per nitrogen: rings sharing an N are not separate pyrroles.
    rings_by_nitrogen: dict[int, frozenset[int]] = {}
    for ring in pyrrole_rings:
        nitrogen = next(i for i in ring if atoms[i] == 7)
        if nitrogen not in rings_by_nitrogen:
            rings_by_nitrogen[nitrogen] = ring
    remaining_rings = list(rings_by_nitrogen.values())

    # Find small loops first to keep bis-porphyrins as separate N4 cores.
    macrocycles: list[tuple[list[frozenset[int]], list[int | None]]] = []
    while len(remaining_rings) >= 4:
        found_combo = None
        for size in range(4, len(remaining_rings) + 1):
            for combo in itertools.combinations(remaining_rings, size):
                result = _macrocycle_ring_order_and_bridges(combo, graph, atoms)
                if result is not None:
                    found_combo = (combo, result)
                    break
            if found_combo is not None:
                break

        if found_combo is None:
            break

        combo, (ring_order, bridge_by_edge) = found_combo
        k = len(ring_order)
        ordered_rings = [combo[r] for r in ring_order]
        ordered_bridges = [
            bridge_by_edge[frozenset((ring_order[i], ring_order[(i + 1) % k]))]
            for i in range(k)
        ]
        macrocycles.append((ordered_rings, ordered_bridges))
        used_rings = set(combo)
        remaining_rings = [r for r in remaining_rings if r not in used_rings]

    return macrocycles


def _macrocycle_ring_order_and_bridges(
    rings, graph: "nx.Graph[int]", atoms: list[int]
) -> tuple[list[int], dict[frozenset[int], int | None]] | None:
    """Find a closed ring loop and its meso bridges (None for direct bonds).

    Allow at most one direct bond for k=4 to avoid merging separate macrocycles.
    """
    k = len(rings)
    ring_atoms = [set(r) for r in rings]
    all_ring_atoms: set[int] = set().union(*ring_atoms)

    outside_candidates: set[int] = set()
    for atom_set in ring_atoms:
        for i in atom_set:
            for j in graph.neighbors(i):
                if j not in all_ring_atoms:
                    outside_candidates.add(j)

    bridge_graph: "nx.Graph[int]" = nx.Graph()
    bridge_graph.add_nodes_from(range(k))
    bridge_by_edge: dict[frozenset[int], int | None] = {}

    for bridge_atom in outside_candidates:
        if atoms[bridge_atom] not in (6, 7):  # C (porphyrin) or N (phthalocyanine)
            continue
        connected_rings = [
            idx
            for idx, atom_set in enumerate(ring_atoms)
            if any(nb in atom_set for nb in graph.neighbors(bridge_atom))
        ]
        if len(connected_rings) == 2:
            bridge_graph.add_edge(*connected_rings)
            bridge_by_edge[frozenset(connected_rings)] = bridge_atom

    # Direct ring-to-ring bonds: the corrole/corrin contraction.
    for idx_a, idx_b in itertools.combinations(range(k), 2):
        edge = frozenset((idx_a, idx_b))
        if edge in bridge_by_edge:
            continue
        connecting_bonds = sum(
            1
            for i in ring_atoms[idx_a]
            for j in graph.neighbors(i)
            if j in ring_atoms[idx_b]
        )
        if connecting_bonds == 1:
            bridge_graph.add_edge(idx_a, idx_b)
            bridge_by_edge[edge] = None

    if bridge_graph.number_of_edges() != k:
        return None
    if any(degree != 2 for _, degree in bridge_graph.degree()):
        return None
    if not nx.is_connected(bridge_graph):
        return None
    # Expanded rings may have several bipyrrole bonds; only k = 4 is capped.
    if k == 4 and sum(1 for bridge in bridge_by_edge.values() if bridge is None) > 1:
        return None

    order = [0]
    prev, current = None, 0
    for _ in range(k - 1):
        nxt = next(n for n in bridge_graph.neighbors(current) if n != prev)
        order.append(nxt)
        prev, current = current, nxt
    return order, bridge_by_edge
