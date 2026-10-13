from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from cell2mol.element_utils import HALOGENS
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation

if TYPE_CHECKING:
    pass

from ._fragments import _charge_capped_fragment, _collect_substituent_fragment

logger = logging.getLogger(__name__)


MANUAL_CONJUNCTO_BORANE_P = {
    # B22: closo-B12 fused to nido-B10, p=1.
    # Source: Volkov, Rath & Barton, J. Organomet. Chem. 2003, 680, 212.
    "B22": 1,
}


class ConjunctoBoraneSplit(TypedDict):
    cage_indices: list[int]
    cut_indices: list[int]
    components: list[list[int]]
    o: int


############################################################
# Borane and carborane detection
############################################################


def is_borane_cage(atoms, AC) -> tuple[bool, str]:
    """Detect closo (≥5 vertices) or nido (≥6 vertices) cages; return (matches, reason)."""
    atoms = [int(a) for a in atoms]
    # Cast object-typed adjacency to numeric values for NetworkX.
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)

    if ac.shape != (n_total, n_total):
        return False, "bad_ac_shape"

    # Require boron; carbon-only cages use the fullerene detector.
    cage_indices = find_borane_cage_indices(atoms, ac)
    if cage_indices is None:
        return False, "no_boron_or_carbon_atoms"
    if not cage_indices:
        return False, "no_borane_cage_core"
    if not any(atoms[i] == 5 for i in cage_indices):
        return False, "no_boron_atoms"

    n = len(cage_indices)

    # Check vertex count before expensive graph algorithms.
    if n < 5:
        return False, "too_few_cage_atoms"

    sub_ac = ac[np.ix_(cage_indices, cage_indices)]
    graph = nx.from_numpy_array(sub_ac)

    # Classify cages separately to support linked cages and sandwich ligands.
    cages = find_borane_cages(atoms, ac)
    if not cages:
        if not nx.is_connected(graph):
            return False, "disconnected_cage"
        return False, "not_deltahedral"

    kinds = {kind for _indices, kind in cages}
    if len(cages) == 1:
        return True, f"{kinds.pop()}_cage_confirmed"
    return True, f"multi_cage_confirmed({'+'.join(sorted(kinds))}, n={len(cages)})"


def find_borane_cage_indices(atoms, AC) -> list[int] | None:
    """Return the B/C graph’s 3-core, excluding substituents; None if no B/C atoms."""
    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n = len(atoms)
    if ac.shape != (n, n):
        return None

    bc_indices = [i for i, z in enumerate(atoms) if z in (5, 6)]
    if not bc_indices:
        return None

    bc_ac = ac[np.ix_(bc_indices, bc_indices)]
    bc_graph = nx.from_numpy_array(bc_ac)
    core = nx.k_core(bc_graph, k=3)

    return sorted(bc_indices[i] for i in core.nodes())


def find_borane_cages(atoms, AC) -> list[tuple[list[int], str]]:
    """Return (atom indices, closo/nido) for each cage, splitting components and bridges."""
    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    core = find_borane_cage_indices(atoms, ac)
    if not core:
        return []

    graph = nx.from_numpy_array(ac[np.ix_(core, core)])
    graph.remove_edges_from(list(nx.bridges(graph)))

    cages: list[tuple[list[int], str]] = []
    for component in nx.connected_components(graph):
        indices = sorted(core[i] for i in component)
        kind = _classify_cage(graph.subgraph(component))
        if kind is None or not any(atoms[i] == 5 for i in indices):
            continue
        cages.append((indices, kind))
    return cages


def find_conjuncto_borane_split(atoms, AC) -> ConjunctoBoraneSplit | None:
    """Detect cages fused at a vertex or edge without classifying the sliced cages.

    Return {cage_indices, cut_indices, components, o}, or None; o is 1 for a vertex, 0 for an edge.
    """
    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)
    if ac.shape != (n_total, n_total):
        return None

    # Strip B/C substituents before testing cage fusion.
    cage_indices = find_borane_cage_indices(atoms, ac)
    if not cage_indices or not any(atoms[i] == 5 for i in cage_indices):
        return None
    # Two B5 cages sharing an edge require at least eight vertices.
    if len(cage_indices) < 8:
        return None

    sub_ac = ac[np.ix_(cage_indices, cage_indices)]
    graph = nx.from_numpy_array(sub_ac)
    if not nx.is_connected(graph):
        return None

    n = len(cage_indices)

    def _valid_side(nodes: set[int]) -> bool:
        if len(nodes) < 4:
            return False
        subg = graph.subgraph(nodes)
        if not nx.is_connected(subg):
            return False
        if any(d < 3 for _, d in subg.degree()):
            return False
        is_planar, _ = nx.check_planarity(subg)
        return is_planar

    for cut_size in (1, 2):
        for cut in itertools.combinations(range(n), cut_size):
            cut_set = set(cut)
            remaining = graph.copy()
            remaining.remove_nodes_from(cut_set)
            comps = list(nx.connected_components(remaining))
            if len(comps) != 2:
                continue
            sides = [comp | cut_set for comp in comps]
            if not all(_valid_side(side) for side in sides):
                continue
            return {
                "cage_indices": cage_indices,
                "cut_indices": sorted(cage_indices[i] for i in cut_set),
                "components": [
                    sorted(cage_indices[i] for i in comps[0]),
                    sorted(cage_indices[i] for i in comps[1]),
                ],
                "o": 1 if cut_size == 1 else 0,
            }
    return None


############################################################
# Charge-state generation
############################################################


def generate_borane_charge_state(prot: Protonation) -> ChargeState | None:
    """Apply Wade’s rules per cage; place the delocalized charge on its first boron.

    Compare vertex electrons v-2+x with 2*(n+1) for closo or 2*(n+2) for nido.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    adjmat = np.asarray(prot.adjmat)

    cage_indices = find_borane_cage_indices(prot.atnums, adjmat) or []
    boron_indices = [i for i in cage_indices if prot.atnums[i] == 5]
    exo_indices = [i for i in range(prot.natoms) if i not in cage_indices]

    if not boron_indices:
        logger.warning(
            "Unexpected structure for borane/carborane specie %s; no boron "
            "cage atoms found",
            prot.formula,
        )
        return None

    # Apply Wade’s rules independently to each cage.
    cages = find_borane_cages(prot.atnums, adjmat)
    if not cages:
        logger.warning(
            "Borane/carborane specie %s failed cage topology re-check", prot.formula
        )
        return None

    cage_charges: list[tuple[int, int]] = []
    for indices, kind in cages:
        n = len(indices)
        required_se = 2 * (n + 1) if kind == "closo" else 2 * (n + 2)
        # A bond to another cage is an exocyclic bond for each cage.
        own = set(indices)
        outside = [i for i in range(prot.natoms) if i not in own]
        contributed_se = 0
        for i in indices:
            valence_electrons = 3 if prot.atnums[i] == 5 else 4
            exo_count = int(np.count_nonzero(adjmat[i][outside])) if outside else 0
            contributed_se += valence_electrons - 2 + exo_count
        cage_boron = [i for i in indices if prot.atnums[i] == 5]
        if not cage_boron:
            logger.warning(
                "Borane/carborane specie %s has a cage with no boron", prot.formula
            )
            return None
        cage_charges.append((cage_boron[0], contributed_se - required_se))
        logger.debug(
            "   %s cage of %d vertices in %s: charge %+d",
            kind,
            n,
            prot.formula,
            contributed_se - required_se,
        )

    allowed_simple_labels = {"H"} | HALOGENS
    simple_exo = {
        i
        for i in exo_indices
        if prot.labels[i] in allowed_simple_labels
        and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [i for i in exo_indices if i not in simple_exo]

    atom_charges = [0] * prot.natoms
    for boron_idx, cage_charge in cage_charges:
        atom_charges[boron_idx] = cage_charge

    # Preserve fragment bond orders to avoid rebuilding aromatic rings as saturated.
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
                "Borane/carborane specie %s has a substituent fragment "
                "with no bond back to the cage; cannot split",
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
            if adjmat[i, j] != 0:
                bond_type = fragment_bond_orders.get(
                    frozenset((i, j)), Chem.BondType.SINGLE
                )
                rwmol.AddBond(i, j, bond_type)

    # Skip valence sanitization and implicit H for nonclassical 3c-2e cage bonds.
    for i in cage_indices:
        rwmol.GetAtomWithIdx(i).SetNoImplicit(True)

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
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception as e:
        logger.warning(
            "Failed to sanitize borane/carborane structure for %s: %s",
            prot.formula,
            e,
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    return ChargeState.from_positional(
        True,
        total_charge,
        atom_charges,
        mol,
        smiles,
        total_charge,
        True,
        prot,
    )


def generate_conjuncto_borane_charge_state(prot: Protonation) -> ChargeState | None:
    """Apply Jemmis’ mno rule with graph-derived m/n/o, tabulated p and q=0.

    Count full valence at shared vertices and one electron per bridging H.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    split = find_conjuncto_borane_split(prot.atnums, prot.adjmat)
    if split is None:
        logger.warning(
            "No conjuncto (fused multi-cage) borane split found for %s",
            prot.formula,
        )
        return None

    cage_indices = split["cage_indices"]
    o = split["o"]
    m = 2
    n = len(cage_indices)

    boron_indices = [i for i in cage_indices if prot.atnums[i] == 5]
    if not boron_indices:
        logger.warning(
            "Unexpected structure for conjuncto borane/carborane specie "
            "%s; no boron cage atoms found",
            prot.formula,
        )
        return None

    n_boron = len(boron_indices)
    n_carbon = n - n_boron
    cage_key = f"B{n_boron}" + (f"C{n_carbon}" if n_carbon else "")
    p = MANUAL_CONJUNCTO_BORANE_P.get(cage_key)
    if p is None:
        logger.warning(
            "No manually-verified missing-vertex count (p) registered for "
            "conjuncto cage composition %s (specie %s); cannot compute charge",
            cage_key,
            prot.formula,
        )
        return None

    required_se = 2 * (m + n + o + p)

    adjmat = np.asarray(prot.adjmat)
    exo_indices = [i for i in range(prot.natoms) if i not in cage_indices]

    bridging_h = {
        i
        for i in exo_indices
        if prot.labels[i] == "H"
        and int(np.count_nonzero(adjmat[i])) == 2
        and all(int(j) in cage_indices for j in np.nonzero(adjmat[i])[0])
    }

    allowed_simple_labels = {"H"} | HALOGENS
    simple_exo = {
        i
        for i in exo_indices
        if i not in bridging_h
        and prot.labels[i] in allowed_simple_labels
        and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [
        i for i in exo_indices if i not in simple_exo and i not in bridging_h
    ]

    contributed_se = len(bridging_h)
    for i in cage_indices:
        valence_electrons = 3 if prot.atnums[i] == 5 else 4
        terminal_exo = [
            j for j in exo_indices if adjmat[i, j] != 0 and j not in bridging_h
        ]
        x = len(terminal_exo)
        contributed_se += valence_electrons if x == 0 else (valence_electrons - 2 + x)

    core_charge = contributed_se - required_se

    atom_charges = [0] * prot.natoms
    atom_charges[boron_indices[0]] = core_charge

    # Preserve fragment bond orders to avoid rebuilding aromatic rings as saturated.
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
                "Conjuncto borane/carborane specie %s has a substituent "
                "fragment with no bond back to the cage; cannot split",
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
            if adjmat[i, j] != 0:
                bond_type = fragment_bond_orders.get(
                    frozenset((i, j)), Chem.BondType.SINGLE
                )
                rwmol.AddBond(i, j, bond_type)

    # Bypass cage valence checks, including degree-two bridging H.
    for i in cage_indices:
        rwmol.GetAtomWithIdx(i).SetNoImplicit(True)
    for i in bridging_h:
        rwmol.GetAtomWithIdx(i).SetNoImplicit(True)

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
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception as e:
        logger.warning(
            "Failed to sanitize conjuncto borane/carborane structure for %s: %s",
            prot.formula,
            e,
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    return ChargeState.from_positional(
        True,
        total_charge,
        atom_charges,
        mol,
        smiles,
        total_charge,
        True,
        prot,
    )


############################################################
# Cage classification helpers
############################################################


def _classify_cage(graph) -> str | None:
    """Classify planar faces as closo (all triangles), nido (one pentagonal opening), or None."""
    if graph.number_of_nodes() < 5:
        return None
    if not nx.is_connected(graph):
        return None
    if any(d < 3 for _, d in graph.degree()):
        return None

    face_sizes = _cage_face_sizes(graph)
    if face_sizes is None:
        return None
    if all(size == 3 for size in face_sizes):
        return "closo"
    if (
        graph.number_of_nodes() >= 6
        and sum(1 for size in face_sizes if size == 5) == 1
        and all(size == 3 for size in face_sizes if size != 5)
    ):
        return "nido"
    return None


def _cage_face_sizes(graph: "nx.Graph[int]") -> list[int] | None:
    """Return planar face sizes, or None if nonplanar."""
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        return None

    visited_half_edges: set[tuple[int, int]] = set()
    sizes = []
    for u, v in embedding.edges:
        if (u, v) in visited_half_edges:
            continue
        face = embedding.traverse_face(u, v, mark_half_edges=visited_half_edges)
        sizes.append(len(face))
    return sizes
