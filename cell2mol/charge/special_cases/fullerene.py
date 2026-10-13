from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from cell2mol.element_utils import HALOGENS
from cell2mol.charge.validation import check_rdkit_obj_connectivity
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation

if TYPE_CHECKING:
    pass

from ._fragments import _charge_capped_fragment, _collect_substituent_fragment

logger = logging.getLogger(__name__)


class FullereneDimerSplit(TypedDict):
    cages: list[list[int]]
    bridge: tuple[int, int]


############################################################
# Fullerene detection
############################################################


def has_fullerene(atoms, AC) -> tuple[bool, str]:
    """Check for a connected, planar, 3-regular carbon cage with girth ≥5
    and an even vertex count ≥20 except 22. Return (matches, reason).
    """
    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)

    if ac.shape != (n_total, n_total):
        return False, "bad_ac_shape"

    cage_indices = find_fullerene_cage_indices(atoms, ac)
    if cage_indices is None:
        return False, "no_carbon_atoms"
    if not cage_indices:
        return False, "no_fullerene_cage_core"

    n = len(cage_indices)

    # Check vertex count before expensive graph algorithms.
    if not _valid_fullerene_vertex_count(n):
        return False, "invalid_fullerene_vertex_count"

    # Count cage neighbours only; exclude exocyclic substituents.
    sub_ac = ac[np.ix_(cage_indices, cage_indices)]
    degrees = np.count_nonzero(sub_ac, axis=1)
    if not np.all(degrees == 3):
        # Try splitting degree-four bridge carbons into two complete cages.
        if find_fullerene_dimer_split(atoms, ac) is not None:
            return True, "fullerene_dimer"
        return False, "not_all_degree_3"

    graph = nx.from_numpy_array(sub_ac)

    if not nx.is_connected(graph):
        return False, "disconnected_structure"

    is_planar, _ = nx.check_planarity(graph)
    if not is_planar:
        return False, "not_planar"

    # Exclude three/four-membered cycles caused by malformed connectivity.
    girth = _graph_girth(graph)
    if girth < 5:
        return False, f"girth_too_small_{girth}"

    return True, "fullerene_topology_confirmed"


def has_open_fullerene(atoms, AC, coords) -> tuple[bool, str]:
    """Detect an open carbon shell using its 2-core, connectivity and 3D thickness.

    Require ≥40 atoms, ≥60% degree-three vertices and ≥20 independent cycles.
    Return (matches, reason).
    """
    MIN_CAGE = 40
    MIN_FUSED_RINGS = 20
    MIN_SP2_FRACTION = 0.6
    MIN_THICKNESS = 0.25

    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n = len(atoms)
    if ac.shape != (n, n):
        return False, "bad_ac_shape"
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (n, 3):
        return False, "bad_coords_shape"

    carbon_indices = [i for i, z in enumerate(atoms) if z == 6]
    if len(carbon_indices) < MIN_CAGE:
        return False, "too_few_carbons"

    # Carbon-only bond graph; node ids are positions into carbon_indices.
    carbon_graph = nx.from_numpy_array(ac[np.ix_(carbon_indices, carbon_indices)])
    core = nx.k_core(carbon_graph, k=2)
    if core.number_of_nodes() < MIN_CAGE:
        return False, "small_carbon_2core"

    core_nodes = list(core.nodes())
    sp2_fraction = sum(1 for v in core_nodes if carbon_graph.degree(v) >= 3) / len(
        core_nodes
    )
    if sp2_fraction < MIN_SP2_FRACTION:
        return False, f"not_mostly_sp2_{sp2_fraction:.2f}"

    cyclomatic = (
        core.number_of_edges()
        - core.number_of_nodes()
        + nx.number_connected_components(core)
    )
    if cyclomatic < MIN_FUSED_RINGS:
        return False, f"too_few_fused_rings_{cyclomatic}"

    # 3D thickness: map 2-core node ids back to original atom indices for coords.
    cage_atom_indices = [carbon_indices[v] for v in core_nodes]
    cage_xyz = coords[cage_atom_indices]
    cage_xyz = cage_xyz - cage_xyz.mean(axis=0)
    singular = np.linalg.svd(cage_xyz, compute_uv=False)
    thickness = float(singular[2] / singular[0]) if singular[0] > 0 else 0.0
    if thickness < MIN_THICKNESS:
        return False, f"planar_not_a_cage_{thickness:.2f}"

    return True, f"open_fullerene_cage_{core.number_of_nodes()}C_rings{cyclomatic}"


def find_fullerene_cage_indices(atoms, AC) -> list[int] | None:
    """Return the carbon graph’s 3-core, excluding substituents; None if no carbon."""
    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n = len(atoms)
    if ac.shape != (n, n):
        return None

    carbon_indices = [i for i, z in enumerate(atoms) if z == 6]
    if not carbon_indices:
        return None

    carbon_ac = ac[np.ix_(carbon_indices, carbon_indices)]
    carbon_graph = nx.from_numpy_array(carbon_ac)
    core = nx.k_core(carbon_graph, k=3)

    return sorted(carbon_indices[i] for i in core.nodes())


def find_fullerene_dimer_split(atoms, AC) -> FullereneDimerSplit | None:
    """Find two fullerene cages joined by one C-C bridge; return {cages, bridge} or None."""
    atoms_list = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n = len(atoms_list)
    if ac.shape != (n, n):
        return None

    graph = nx.from_numpy_array(ac)
    if not nx.is_connected(graph):
        return None

    for i, j in nx.bridges(graph):
        g2 = graph.copy()
        g2.remove_edge(i, j)
        comps = list(nx.connected_components(g2))
        if len(comps) != 2:
            continue

        comp_a, comp_b = comps
        if i not in comp_a:
            comp_a, comp_b = comp_b, comp_a

        both_valid = True
        for comp in (comp_a, comp_b):
            comp_list = sorted(comp)
            sub_atoms = [atoms_list[k] for k in comp_list]
            sub_ac = ac[np.ix_(comp_list, comp_list)]
            is_cage, _reason = has_fullerene(sub_atoms, sub_ac)
            if not is_cage:
                both_valid = False
                break

        if both_valid:
            return {
                "cages": [sorted(comp_a), sorted(comp_b)],
                "bridge": (i, j),
            }

    return None


def check_fullerene_sphericity(coords, tol: float = 0.35) -> bool:
    """Check cage radii around the centroid for geometric disorder."""
    coords = np.asarray(coords, dtype=float)
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    return (radii.std() / radii.mean()) < tol


############################################################
# Charge-state generation
############################################################


def generate_fullerene_charge_state(prot: Protonation) -> ChargeState | None:
    """Build a neutral cage using perfect matching for double bonds, excluding
    substituted carbons. Add substituent charges separately.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    cage_indices = find_fullerene_cage_indices(prot.atnums, prot.adjmat)
    if not cage_indices:
        logger.warning("No fullerene cage core found for %s", prot.formula)
        return None

    adjmat = np.asarray(prot.adjmat)
    cage_set = set(cage_indices)
    exo_indices = [i for i in range(prot.natoms) if i not in cage_set]

    cage_local_to_global = dict(enumerate(cage_indices))
    global_to_cage_local = {gi: local for local, gi in cage_local_to_global.items()}

    sub_ac = adjmat[np.ix_(cage_indices, cage_indices)]
    cage_graph = nx.from_numpy_array(sub_ac)

    substituted_cage_atoms = {
        gi for gi in cage_indices if any(adjmat[gi, j] != 0 for j in exo_indices)
    }
    unsubstituted_local = [
        global_to_cage_local[gi]
        for gi in cage_indices
        if gi not in substituted_cage_atoms
    ]

    matching = nx.max_weight_matching(
        cage_graph.subgraph(unsubstituted_local), maxcardinality=True
    )

    if len(matching) * 2 != len(unsubstituted_local):
        logger.warning(
            "No perfect matching found for the unsubstituted portion of "
            "fullerene cage %s; cannot build a closed-shell Kekule structure",
            prot.formula,
        )
        return None

    double_bonds = {
        frozenset((cage_local_to_global[a], cage_local_to_global[b]))
        for a, b in matching
    }

    # Preserve substituent bond orders when charging fragments.
    allowed_simple_labels = {"H"} | HALOGENS
    simple_exo = {
        i
        for i in exo_indices
        if prot.labels[i] in allowed_simple_labels
        and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [i for i in exo_indices if i not in simple_exo]

    atom_charges = [0] * prot.natoms
    fragment_bond_orders: dict[frozenset[int], Chem.BondType] = {}
    visited_complex: set[int] = set()
    for root in complex_roots:
        if root in visited_complex:
            continue
        fragment = _collect_substituent_fragment(root, cage_indices, adjmat)
        visited_complex.update(fragment)

        attach_bonds = [
            (c, s) for c in cage_indices for s in fragment if adjmat[c, s] != 0
        ]
        if not attach_bonds:
            logger.warning(
                "Fullerene specie %s has a substituent fragment with no "
                "bond back to the cage; cannot split",
                prot.formula,
            )
            return None
        attach_atoms = [s for _c, s in attach_bonds]

        frag_charge, frag_atom_charges, frag_bond_orders = _charge_capped_fragment(
            fragment, attach_atoms, adjmat, prot.atnums, prot.formula
        )
        if frag_charge is None:
            return None
        for global_i, charge in frag_atom_charges.items():
            atom_charges[global_i] = charge
        fragment_bond_orders.update(frag_bond_orders)

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
            elif pair in double_bonds:
                bond_type = Chem.BondType.DOUBLE
            else:
                bond_type = Chem.BondType.SINGLE
            rwmol.AddBond(i, j, bond_type)

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
        # Keep explicit cage bonds; RDKit aromaticity can misclassify curved cages.
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
        )
    except Exception as e:
        logger.warning(
            "Failed to sanitize fullerene Kekule structure for %s: %s",
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


def generate_fullerene_dimer_charge_states(
    prot: Protonation,
) -> list[ChargeState] | None:
    """Build two bridged cages with one unmatched position each.

    Return all four ±1 charge combinations; topology cannot choose their signs.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    split = find_fullerene_dimer_split(prot.atnums, prot.adjmat)
    if split is None:
        logger.warning("No fullerene dimer split found for %s", prot.formula)
        return None

    adjmat = np.asarray(prot.adjmat)
    bridge_a, bridge_b = split["bridge"]

    double_bonds: set[frozenset[int]] = set()
    leftover_atoms = []

    for cage_atoms, bridge_atom in zip(split["cages"], (bridge_a, bridge_b)):
        matching_atoms = [i for i in cage_atoms if i != bridge_atom]
        sub_ac = adjmat[np.ix_(matching_atoms, matching_atoms)]
        local_to_global = dict(enumerate(matching_atoms))

        cage_graph = nx.from_numpy_array(sub_ac)
        matching = nx.max_weight_matching(cage_graph, maxcardinality=True)

        matched_local = {node for pair in matching for node in pair}
        unmatched_local = [
            i for i in range(len(matching_atoms)) if i not in matched_local
        ]
        if len(unmatched_local) != 1:
            logger.warning(
                "Fullerene dimer specie %s: expected exactly 1 unpaired "
                "atom in cage after excluding the bridge atom, found %d; "
                "cannot build a Kekule structure",
                prot.formula,
                len(unmatched_local),
            )
            return None

        double_bonds.update(
            frozenset((local_to_global[a], local_to_global[b])) for a, b in matching
        )
        leftover_atoms.append(local_to_global[unmatched_local[0]])

    rwmol = Chem.RWMol()
    for atomic_num in prot.atnums:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    for i in range(prot.natoms):
        for j in range(i + 1, prot.natoms):
            if adjmat[i, j] == 0:
                continue
            bond_type = (
                Chem.BondType.DOUBLE
                if frozenset((i, j)) in double_bonds
                else Chem.BondType.SINGLE
            )
            rwmol.AddBond(i, j, bond_type)

    charge_states = []
    for charge_a, charge_b in ((-1, -1), (1, 1), (-1, 1)):
        candidate = Chem.RWMol(rwmol)
        candidate.GetAtomWithIdx(leftover_atoms[0]).SetFormalCharge(charge_a)
        candidate.GetAtomWithIdx(leftover_atoms[1]).SetFormalCharge(charge_b)

        mol = candidate.GetMol()

        conf = Chem.Conformer(prot.natoms)
        conf.Set3D(True)
        for i, xyz in enumerate(np.asarray(prot.coord, dtype=float)):
            conf.SetAtomPosition(
                i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            )
        mol.AddConformer(conf, assignId=True)

        try:
            Chem.SanitizeMol(
                mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
            )
        except Exception as e:
            logger.warning(
                "Failed to sanitize fullerene dimer structure for %s "
                "(candidate charges %d/%d): %s",
                prot.formula,
                charge_a,
                charge_b,
                e,
            )
            continue

        atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        total_charge = sum(atom_charges)
        smiles = Chem.MolToSmiles(mol)
        is_correct = check_rdkit_obj_connectivity(mol, prot.natoms, total_charge)

        charge_states.append(
            ChargeState.from_positional(
                is_correct,
                total_charge,
                atom_charges,
                mol,
                smiles,
                total_charge,
                True,
                prot,
            )
        )

    return charge_states if charge_states else None


############################################################
# Graph helpers
############################################################


def _valid_fullerene_vertex_count(n: int) -> bool:
    return n >= 20 and n % 2 == 0 and n != 22


def _graph_girth(graph: nx.Graph[int]) -> int:
    """Length of the shortest cycle in the graph. networkx>=3.0 has
    nx.girth(); fall back to a BFS-based computation for older versions."""
    if hasattr(nx, "girth"):
        g = nx.girth(graph)
        return int(g) if g != float("inf") else 10**9
    # Fallback: BFS from every node, shortest cycle through it
    best = 10**9
    for src in graph.nodes():
        dist = {src: 0}
        parent: dict[int, int | None] = {src: None}
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in graph.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v:
                    best = min(best, dist[u] + dist[v] + 1)
        if best <= 5:
            break  # can't do better than girth 5 for a fullerene anyway
    return best
