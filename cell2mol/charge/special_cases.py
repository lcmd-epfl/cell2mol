from __future__ import annotations

from typing import TYPE_CHECKING

import itertools
import numpy as np
import networkx as nx
from collections import deque
from cell2mol.charge.utils import (
    check_rdkit_obj_connectivity,
    generate_rdkit_mol_from_AC2mol,
)
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation
import logging
from cell2mol.charge.utils import (
    HALOGENS,
)
from cell2mol.elementdata import ElementData

from rdkit import Chem
from rdkit.Geometry import Point3D

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

# Jemmis mno-rule "p" (vertices missing across all fused sub-polyhedra), keyed
# by cage-only composition since substituents vary. Not derivable from a naive
# graph slice, so each entry is checked against a literature charge.
MANUAL_CONJUNCTO_BORANE_P = {
    # [B22H22]^2- docosaborate: closo-B12 icosahedron (p=0) fused to a
    # nido-B10 cluster (p=1) sharing a common edge. Verified against
    # Volkov, Rath & Barton, J. Organomet. Chem. 2003, 680, 212 (the
    # -OH derivative, [B22H21OH]^2-, charge -2).
    "B22": 1,
}


def generate_special_charge_states(spec: Specie) -> list[ChargeState] | None:
    """Charge states for species built in closed form rather than searched: antimony
    halides, fullerene cages, closo/nido boranes. Three outcomes -- None (not
    special, run the general search), [charge_state] (built), [] (special but failed).
    """
    # Antimony-halide-only species (SbX3/X4/X5/X6-type)
    if is_sb_halide_only(spec.labels):
        charge_state = generate_sb_halide_charge_state(spec.protonation_states[0])
        if charge_state is None:
            logger.warning(
                "Sb-halide charge builder failed for %s; leaving charge "
                "unassigned rather than falling back to the general search",
                spec.formula,
            )
        return [charge_state] if charge_state is not None else []

    # Fullerene cage (possibly substituted)
    has_fullerene = (
        spec.has_fullerene
        if spec.has_fullerene is not None
        else spec.evaluate_has_fullerene()
    )
    # has_fullerene covers cage and dimer alike, so route each to its own builder:
    # the single-cage one would fail on a dimer's degree-4 bridge carbons.
    if has_fullerene:
        prot0 = spec.protonation_states[0]
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

    # Single closo/nido deltahedron only. Fused conjuncto clusters fail
    # is_borane_cage (no dimer-style fallback) and reach the general search.
    has_borane = (
        spec.has_borane if spec.has_borane is not None else spec.evaluate_has_borane()
    )
    if has_borane:
        logger.debug("Specie %s has a borane/carborane cage", spec.formula)
        prot0 = spec.protonation_states[0]
        charge_state = generate_borane_charge_state(prot0)
        if charge_state is None:
            logger.warning(
                "Borane/carborane charge builder failed for %s; return None",
                spec.formula,
            )
            return None
        return [charge_state] if charge_state is not None else []

    # Not a special case -- let the caller run the general search.
    return None


def is_sb_halide_only(labels) -> bool:
    """
    True for species made up of only Sb and halogens (SbX3/X4/X5/X6-type),
    regardless of halogen identity or count.
    """
    return "Sb" in labels and all(
        label in HALOGENS or label == "Sb" for label in labels
    )


def generate_sb_halide_charge_state(prot: Protonation) -> ChargeState | None:
    """Sb/halogen species charged from connectivity. Mononuclear: -1 for SbX6-, else
    ``3 - degree``. Polynuclear clusters are always Sb(III) + halides, so the total
    is ``3 * n_Sb - n_halogens`` with Sb neutral and the charge on the halogens.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    sb_indices = [i for i, label in enumerate(prot.labels) if label == "Sb"]
    halogen_indices = [i for i, label in enumerate(prot.labels) if label in HALOGENS]

    if not sb_indices or (len(sb_indices) + len(halogen_indices)) != prot.natoms:
        logger.warning(
            "Unexpected structure for Sb-halide specie %s; expected only Sb and halogens",
            prot.formula,
        )
        return None

    adjmat = np.asarray(prot.adjmat)

    rwmol = Chem.RWMol()
    for atomic_num in prot.atnums:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    for i in range(prot.natoms):
        for j in range(i + 1, prot.natoms):
            if adjmat[i, j] != 0:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)

    is_polynuclear = len(sb_indices) > 1
    for idx in sb_indices:
        sb_atom = rwmol.GetAtomWithIdx(idx)
        sb_atom.SetNoImplicit(True)

        if not is_polynuclear:
            degree = int(np.count_nonzero(adjmat[idx]))
            sb_charge = -1 if degree == 6 else 3 - degree
            sb_atom.SetFormalCharge(sb_charge)
        # Polynuclear Sb centers stay formally neutral here -- the total
        # charge is distributed across the halogens below instead.

    if is_polynuclear:
        # Every halogen may end up with >1 explicit bond (bridging) or a
        # -1 charge, either of which violates RDKit's default halogen
        # valence table, so bypass implicit-valence handling uniformly.
        for j in halogen_indices:
            rwmol.GetAtomWithIdx(j).SetNoImplicit(True)

        # Spread -1 charges round-robin, one per Sb, to reach
        # 3 * n_Sb - n_halogens. Prefer terminal halogens over bridging ones.
        halogen_sb_degree = {
            j: sum(1 for idx in sb_indices if adjmat[idx, j] != 0)
            for j in halogen_indices
        }
        n_negative = len(halogen_indices) - 3 * len(sb_indices)
        assigned: set[int] = set()
        remaining = n_negative
        while remaining > 0:
            progressed = False
            for idx in sb_indices:
                if remaining <= 0:
                    break
                candidates = [
                    j
                    for j in halogen_indices
                    if j not in assigned and adjmat[idx, j] != 0
                ]
                if not candidates:
                    continue
                j = min(candidates, key=lambda j: halogen_sb_degree[j])
                rwmol.GetAtomWithIdx(j).SetFormalCharge(-1)
                assigned.add(j)
                remaining -= 1
                progressed = True
            if not progressed:
                logger.warning(
                    "Could not distribute full negative charge across "
                    "halogens for %s; %d unit(s) left unassigned",
                    prot.formula,
                    remaining,
                )
                break

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
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_CLEANUP_ORGANOMETALLICS,
        )
    except Exception as e:
        logger.warning(
            "Failed to sanitize Sb-halide structure for %s: %s", prot.formula, e
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    is_correct = True

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


def _valid_fullerene_vertex_count(n: int) -> bool:
    return n >= 20 and n % 2 == 0 and n != 22


def find_fullerene_cage_indices(atoms, AC) -> list[int] | None:
    """Cage atoms of a fullerene, including one inside a substituted derivative, via
    the 3-core of the carbon-only graph: cage carbons are inherently 3-connected
    and survive, substituents strip away. None if there are no carbons.
    """
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


def has_fullerene(atoms, AC) -> tuple[bool, str]:
    """Detect a fullerene cage from connectivity alone. The cage subgraph must be all
    carbon, 3-regular, connected, planar, girth >= 5, and of valid vertex count
    (even, >= 20, != 22) -- which forces 12 pentagons by Euler. Returns
    (is_fullerene, reason).
    """
    atoms = [int(a) for a in atoms]
    # Ligand adjmats inherited/sliced from a parent molecule's matrix can
    # come through as dtype=object (values are still plain ints, just
    # boxed) -- networkx's from_numpy_array rejects that dtype outright,
    # so force a numeric dtype here rather than passing AC through as-is.
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)

    if ac.shape != (n_total, n_total):
        return False, "bad_ac_shape"

    # 1. Composition - find the embedded all-carbon cage core, if any.
    #    (Endohedral fullerenes, with something trapped *inside* the cage
    #    rather than bonded to it, need a separate check - flagged, not
    #    handled here.)
    cage_indices = find_fullerene_cage_indices(atoms, ac)
    if cage_indices is None:
        return False, "no_carbon_atoms"
    if not cage_indices:
        return False, "no_fullerene_cage_core"

    n = len(cage_indices)

    # 2. Vertex count sanity filter (cheap, do before graph algorithms)
    if not _valid_fullerene_vertex_count(n):
        return False, "invalid_fullerene_vertex_count"

    # 3. Degree check - every cage carbon must be exactly 3-connected to
    #    *other cage carbons* (an exocyclic substituent bond, if any,
    #    isn't part of this cage-only adjacency, so doesn't affect it)
    sub_ac = ac[np.ix_(cage_indices, cage_indices)]
    degrees = np.count_nonzero(sub_ac, axis=1)
    if not np.all(degrees == 3):
        # A dimer fails the 3-regular test at its two degree-4 bridge carbons.
        # Accept if both sides are complete cages once cut; each half is an
        # ordinary single cage, so this never recurses.
        if find_fullerene_dimer_split(atoms, ac) is not None:
            return True, "fullerene_dimer"
        return False, "not_all_degree_3"

    graph = nx.from_numpy_array(sub_ac)

    # 4. Must be one connected cage, not fragments/disorder artifacts
    if not nx.is_connected(graph):
        return False, "disconnected_structure"

    # 5. Planarity - required for any genuine closed polyhedral cage
    is_planar, _ = nx.check_planarity(graph)
    if not is_planar:
        return False, "not_planar"

    # 6. Girth >= 5 - rules out graphs with spurious 3/4-membered rings
    #    (e.g. AC-matrix artifacts from disorder, or a non-fullerene cage)
    girth = _graph_girth(graph)
    if girth < 5:
        return False, f"girth_too_small_{girth}"

    return True, "fullerene_topology_confirmed"


def find_fullerene_dimer_split(atoms, AC) -> dict | None:
    """Two cages joined by one direct C-C bond. A 3-core cannot separate them (both
    sides revert to degree 3 once cut), so each bridge edge is tried and
    has_fullerene checked per side. Returns {"cages", "bridge"} or None.
    """
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


def _graph_girth(graph: nx.Graph) -> int:
    """Length of the shortest cycle in the graph. networkx>=3.0 has
    nx.girth(); fall back to a BFS-based computation for older versions."""
    if hasattr(nx, "girth"):
        g = nx.girth(graph)
        return g if g != float("inf") else 10**9
    # Fallback: BFS from every node, shortest cycle through it
    best = 10**9
    for src in graph.nodes():
        dist = {src: 0}
        parent = {src: None}
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


def _cage_face_sizes(graph: "nx.Graph[int]") -> list[int] | None:
    """
    Sizes of every face in a planar embedding of `graph`, or None if the
    graph is not planar. Used to tell a full deltahedron (all triangular
    faces) apart from a deltahedron missing one vertex (one larger open
    face, the rest triangular).
    """
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


def find_borane_cage_indices(atoms, AC) -> list[int] | None:
    """Cage atoms of a closo/nido deltahedron, via the 3-core of the B/C-only graph.
    Every deltahedron vertex has at least 3 cage neighbours, so cage atoms survive
    and substituents strip away. None if there are no B or C atoms.
    """
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


def _classify_cage(graph) -> str | None:
    """ "closo", "nido" or None for one cage subgraph, read off its planar faces:
    closo is all triangles, nido has exactly one open pentagonal face.
    """
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


def find_borane_cages(atoms, AC) -> list[tuple[list[int], str]]:
    """Every closo/nido cage in the specie, as (cage atom indices, "closo"/"nido").

    A specie can carry more than one cage -- two dicarbollides sandwiching a metal,
    or a pendant carborane bonded to another cage -- so the 3-core is split before
    classifying. Disjoint cages fall out as separate components; linked ones are
    cut at their bridges, which is safe because a deltahedron is 3-connected and
    therefore has none of its own. Returns [] when nothing classifies.
    """
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


def is_borane_cage(atoms, AC) -> tuple[bool, str]:
    """Detect a closo- or nido-type deltahedron from connectivity, exocyclic atoms
    dropped first. Read off the cage's planar faces: closo (n >= 5) is all
    triangles, nido (n >= 6) has exactly one open pentagonal face. Returns
    (is_cage, reason).
    """
    atoms = [int(a) for a in atoms]
    # See the matching comment in has_fullerene: ligand adjmats can be
    # dtype=object even though every value is a plain int, which networkx
    # rejects outright.
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)

    if ac.shape != (n_total, n_total):
        return False, "bad_ac_shape"

    # 1. Cage composition -- require at least one B; a pure-carbon deltahedron
    #    is not a known species, and carbon cages go to has_fullerene.
    cage_indices = find_borane_cage_indices(atoms, ac)
    if cage_indices is None:
        return False, "no_boron_or_carbon_atoms"
    if not cage_indices:
        return False, "no_borane_cage_core"
    if not any(atoms[i] == 5 for i in cage_indices):
        return False, "no_boron_atoms"

    n = len(cage_indices)

    # 2. Vertex count sanity filter (cheap, do before graph algorithms) -
    #    the smallest closo deltahedron is the trigonal bipyramid, B5H5^2-.
    if n < 5:
        return False, "too_few_cage_atoms"

    # 3. Build the cage-only subgraph - exocyclic atoms (terminal H, a
    #    coordinated metal, substituents) are excluded entirely.
    sub_ac = ac[np.ix_(cage_indices, cage_indices)]
    graph = nx.from_numpy_array(sub_ac)

    # 4. Classify every cage the specie carries, not just one. A bis(dicarbollide)
    #    sandwich has two disjoint cages and a pendant carborane is bonded to its
    #    neighbour, so requiring a single connected deltahedron rejected both.
    cages = find_borane_cages(atoms, ac)
    if not cages:
        # Distinguish the two ways this fails, since the reasons read very
        # differently in a log.
        if not nx.is_connected(graph):
            return False, "disconnected_cage"
        return False, "not_deltahedral"

    kinds = {kind for _indices, kind in cages}
    if len(cages) == 1:
        return True, f"{kinds.pop()}_cage_confirmed"
    return True, f"multi_cage_confirmed({'+'.join(sorted(kinds))}, n={len(cages)})"


def find_conjuncto_borane_split(atoms, AC) -> dict | None:
    """Split a fused multi-cage cluster on a shared vertex or edge. Identifies only
    THAT a split exists, never each side's closo/nido class -- a naive graph slice
    distorts faces at the fusion seam. Returns {"cage_indices", "cut_indices",
    "components", "o"} (o = 1 for a shared vertex, 0 for an edge), or None.
    """
    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)
    if ac.shape != (n_total, n_total):
        return None

    # See find_borane_cage_indices: strips B/C substituents (alkyl/aryl
    # tails) so they don't get mistaken for cage vertices.
    cage_indices = find_borane_cage_indices(atoms, ac)
    if not cage_indices or not any(atoms[i] == 5 for i in cage_indices):
        return None
    # Smallest sensible 2-polyhedron fusion: two trigonal bipyramids (the
    # smallest closo deltahedron, n=5) sharing an edge is 5+5-2=8 atoms.
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


def check_fullerene_sphericity(coords, tol: float = 0.35) -> bool:
    """
    Optional geometric cross-check using 3D coordinates: fullerene cages are
    near-perfect spheres, so all atoms should sit at nearly the same radius
    from the centroid. Large variance flags a disordered or malformed AC
    matrix even if the topology check above passed.
    """
    coords = np.asarray(coords, dtype=float)
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    return (radii.std() / radii.mean()) < tol


def has_open_fullerene(atoms, AC, coords) -> tuple[bool, str]:
    """Detect a fullerene shell opened at an orifice, which has_fullerene rejects
    because the rim breaks the 3-core. Uses the carbon 2-core plus geometry: >= 40
    atoms, >= 60% still 3-connected, cyclomatic number >= 20, and real 3D thickness
    by SVD. Coordinate-based. Returns (is_open_fullerene, reason).
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


def _find_porphyrin_rings_and_bridges(
    atoms, AC
) -> list[tuple[list[frozenset[int]], list[int | None]]]:
    """Four pyrrole-type rings joined pairwise into one closed loop, through a meso
    atom (C or N) or, for corrole/corrin, one direct ring-to-ring bond. Returns
    (rings, bridges) per disjoint macrocycle in cyclic order, bridges[i] being the
    meso atom joining rings[i] and rings[i+1], or None for a direct bond.
    """
    atoms = [int(a) for a in atoms]
    # See the matching comment in has_fullerene: ligand adjmats can be
    # dtype=object even though every value is a plain int, which networkx
    # rejects outright.
    ac = np.asarray(AC, dtype=int)
    n = len(atoms)

    if ac.shape != (n, n):
        return []

    n_nitrogen = sum(1 for z in atoms if z == 7)
    n_carbon = sum(1 for z in atoms if z == 6)
    # 16 core carbons (4 rings x 4 C) is the hard minimum: carbon meso bridges
    # push it higher, but a phthalocyanine's meso bridges are nitrogen.
    if n_nitrogen < 4 or n_carbon < 16:
        return []

    graph = nx.from_numpy_array(ac)

    # Candidate pyrrole-type rings: 5-membered cycles with exactly 1 N.
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

    # Each candidate ring must contribute a distinct nitrogen -- rings
    # sharing a nitrogen can't both be genuine, separate pyrrole units.
    rings_by_nitrogen: dict[int, frozenset[int]] = {}
    for ring in pyrrole_rings:
        nitrogen = next(i for i in ring if atoms[i] == 7)
        if nitrogen not in rings_by_nitrogen:
            rings_by_nitrogen[nitrogen] = ring
    remaining_rings = list(rings_by_nitrogen.values())

    # Peel off the smallest closed loop at a time, removing its rings before
    # searching on -- smallest-first so a bis-porphyrin resolves into two real
    # 4-macrocycles rather than one spurious 8-membered loop through both.
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


def find_all_porphyrin_macrocycles(
    atoms, AC
) -> list[tuple[list[int], bool, list[int]]]:
    """Every disjoint N4 macrocycle in a structure -- two for a bis-porphyrin, and so
    on. See find_porphyrin_macrocycle for the fields.
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


def find_porphyrin_macrocycle(atoms, AC) -> tuple[list[int], bool, list[int]] | None:
    """The first N4 macrocycle found, or None. Returns (nitrogens, is_contracted,
    core_atoms): the 4 pyrrole-type N in cyclic order (i and i+2 are the trans
    pair), whether one link is a direct bond (corrole/corrin), and the rings plus
    meso bridges without substituents.
    """
    macrocycles = find_all_porphyrin_macrocycles(atoms, AC)
    return macrocycles[0] if macrocycles else None


def porphyrin_reference_protonation_sites(
    macrocycle_nitrogens: list[int], is_contracted: bool
) -> list[int]:
    """Ring nitrogens carrying the proton in the neutral free base: the trans pair for
    a classic N4, 3 of 4 for a corrole/corrin, or alternating N-H for an expanded
    ring -- there only the baseline m0, which the enumerator brackets at m0 +/- 1.
    """
    if len(macrocycle_nitrogens) == 4:
        if is_contracted:
            return macrocycle_nitrogens[:3]
        return [macrocycle_nitrogens[0], macrocycle_nitrogens[2]]
    # Expanded porphyrin: alternating ring nitrogens (the aromatic tautomer).
    return macrocycle_nitrogens[0::2]


def _macrocycle_ring_order_and_bridges(
    rings, graph: "nx.Graph[int]", atoms: list[int]
) -> tuple[list[int], dict[frozenset[int], int | None]] | None:
    """Whether k rings form one closed loop, each joined to exactly two others by a
    meso atom or a direct bond. For k == 4 at most one direct bond is allowed, else
    a bis-porphyrin would stitch into one spurious loop. Returns (order,
    bridge_by_edge) mapping each ring pair to its meso atom, or None per direct bond.
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

    # Direct ring-to-ring bonds (no bridging atom): the corrole/corrin
    # ring contraction, where two adjacent pyrrole rings are joined by a
    # single bond between one atom of each ring instead of a meso bridge.
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
    # At most one direct-bond link for the classic 4-ring case -- more than
    # one would no longer describe a genuine corrole/corrin-type macrocycle.
    # (Expanded porphyrins may carry several bipyrrole direct bonds, so the
    # limit is only imposed for k == 4.)
    if k == 4 and sum(1 for bridge in bridge_by_edge.values() if bridge is None) > 1:
        return None

    # Walk the k-cycle to get the traversal order.
    order = [0]
    prev, current = None, 0
    for _ in range(k - 1):
        nxt = next(n for n in bridge_graph.neighbors(current) if n != prev)
        order.append(nxt)
        prev, current = current, nxt
    return order, bridge_by_edge


def is_porphyrin_macrocycle(atoms, AC) -> tuple[bool, str]:
    """True if the connectivity contains an N4 macrocycle, regardless of substituents."""
    result = find_porphyrin_macrocycle(atoms, AC)
    if result is None:
        return False, "no_porphyrin_macrocycle"
    _nitrogens, is_contracted, _core_atoms = result
    if is_contracted:
        return True, "corrole_or_corrin_macrocycle_confirmed"
    return True, "porphyrin_macrocycle_confirmed"


def generate_porphyrin_charge_state(prot: Protonation) -> ChargeState | None:
    """Charge an N4 macrocycle analytically rather than searching a conjugated system
    this size. Core atoms are all neutral and Kekulization picks a pattern matching
    the N-H set ``prot`` protonates, so the ligand charge is -(number of N-H).
    Substituents are split off and charged independently, then summed.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    macrocycles = find_all_porphyrin_macrocycles(prot.atnums, prot.adjmat)
    if not macrocycles:
        logger.warning("No porphyrin-family macrocycle found for %s", prot.formula)
        return None
    # A ligand may hold more than one macrocycle (a bis-porphyrin bridging
    # two metals) or one expanded macrocycle (penta-/hexaphyrin). The
    # per-ring cores are disjoint, so we union their nitrogens and core
    # atoms and treat the whole set as "the core".
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

    # The pyrrolic (N-H) nitrogens are whichever ring nitrogens this
    # protonation state actually protonates; the rest are neutral imine-type
    # nitrogens. Both are formally neutral in the free base.
    reference_sites = {n for n in nitrogens if _has_explicit_h(n)}

    atom_charges = [0] * prot.natoms
    for n in nitrogens:
        protonated = _has_explicit_h(n)
        if n in reference_sites:
            atom_charges[n] = 0 if protonated else -1
        else:
            atom_charges[n] = 1 if protonated else 0

    # The core is neutral by construction, so check the protons instead: the charge
    # is -(number added) only if every one landed on a nitrogen. Counting N-H would
    # misread a free base that brought its own, or a pocket N outside the ring.
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

    # Substituents beyond a simple terminal H are split off and charged
    # independently, exactly as for fullerene/borane substituents.
    simple_exo = {
        i
        for i in exo_indices
        if prot.labels[i] == "H" and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [i for i in exo_indices if i not in simple_exo]

    fragment_bond_orders: dict[frozenset, Chem.BondType] = {}
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

    # Saturated ring atoms (a chlorin's sp3 carbons) cannot carry a pi system:
    # four aromatic bonds would mean valence 6 and sanitization would reject the
    # molecule. Bond them single, leaving the rest of the ring aromatic.
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
                # A flat conjugated tetrapyrrole is what RDKit's aromaticity
                # model is built for, so leave these bonds aromatic and let it
                # Kekulize. Inter-macrocycle bonds are single (the else below).
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE
            rwmol.AddBond(i, j, bond_type)

    for i in core_atoms:
        if i not in saturated_core:
            rwmol.GetAtomWithIdx(i).SetIsAromatic(True)

    for n in nitrogens:
        # Force RDKit to satisfy this atom's valence from the ring's own
        # bonds (as Kekulized below) rather than silently padding in an
        # implicit H a pyrrolide-type (-1) or bare imine-type (0) nitrogen
        # doesn't actually have.
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
        # Charges and NoImplicit above already encode lone pair vs. double bond,
        # so only KEKULIZE runs -- no re-deriving aromaticity from Hueckel.
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
        )
    except Exception as e:
        # Not fatal -- the specie falls back to general enumeration. The ring
        # matched the topology but is not an aromatic tetrapyrrole after all.
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


def generate_fullerene_charge_state(prot: Protonation) -> ChargeState | None:
    """Kekule structure for a fullerene: a bridgeless 3-regular graph always has a
    perfect matching (Petersen), so promoting matched edges completes every valence
    with no charge separation and no search. Substituted carbons are excluded from
    the matching. The cage charge is always 0; only substituents contribute.
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

    # Substituents past a terminal H/halogen are split off and charged
    # independently, keeping their internal bond orders as well as their
    # charges -- rebuilt all-single, an aromatic ring comes out saturated.
    allowed_simple_labels = {"H"} | HALOGENS
    simple_exo = {
        i
        for i in exo_indices
        if prot.labels[i] in allowed_simple_labels
        and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [i for i in exo_indices if i not in simple_exo]

    atom_charges = [0] * prot.natoms
    fragment_bond_orders: dict[frozenset, Chem.BondType] = {}
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
        # Skip aromaticity perception: RDKit's Hueckel-based model is not
        # meant for curved, fused 5/6-ring cages and can misfire on it. The
        # explicit single/double bonds above already fully describe the
        # structure.
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
    """Two cages joined by one C-C bond. Each bridge atom reaches degree 4 and leaves
    the matching, so an odd count remains per cage and each keeps one unpaired
    position -- the real radical character of such dimers. Carbanion or carbocation
    is not derivable from topology, so all four combinations are returned.
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

    double_bonds: set[frozenset] = set()
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


def _collect_substituent_fragment(
    root: int, cage_indices: list[int], adjmat: np.ndarray
) -> list[int]:
    """
    BFS over everything reachable from `root` without stepping back into
    a cage atom -- i.e. the full atom set of one substituent group hung
    off a single cage vertex (e.g. the N + 2 H of an -NH2 substituent).
    """
    cage_set = set(cage_indices)
    visited = {root}
    frontier = [root]
    while frontier:
        next_frontier = []
        for i in frontier:
            for j in np.nonzero(adjmat[i])[0]:
                j = int(j)
                if j in cage_set or j in visited:
                    continue
                visited.add(j)
                next_frontier.append(j)
        frontier = next_frontier
    return sorted(visited)


def _charge_capped_fragment(
    fragment: list[int],
    attach_atoms: list[int],
    adjmat: np.ndarray,
    atnums,
    formula: str,
) -> tuple[int | None, dict[int, int], dict[frozenset, "Chem.BondType"]]:
    """Charge a substituent alone: each bond back to the cage is capped with a plain
    H, then the general search runs at increasing |charge|. Internal bond orders
    are returned too, not just charges -- rebuilt as all-single bonds, an aromatic
    ring would be silently padded with implicit hydrogens and come out saturated.
    """
    local_index = {g: local for local, g in enumerate(fragment)}
    k = len(fragment)
    n_caps = len(attach_atoms)
    local_ac = np.zeros((k + n_caps, k + n_caps), dtype=int)
    for a in fragment:
        for b in fragment:
            if a != b and adjmat[a, b] != 0:
                local_ac[local_index[a], local_index[b]] = 1
    for cap_offset, attach_atom in enumerate(attach_atoms):
        cap_idx = k + cap_offset
        local_ac[local_index[attach_atom], cap_idx] = 1
        local_ac[cap_idx, local_index[attach_atom]] = 1

    local_atoms = [int(atnums[g]) for g in fragment] + [1] * n_caps
    for charge in (0, -1, 1, -2, 2, -3, 3):
        mol = generate_rdkit_mol_from_AC2mol(
            atoms=local_atoms,
            AC=local_ac,
            charge=charge,
            allow_charged_fragments=True,
            embed_chiral=True,
        )
        if mol is not None:
            is_correct = check_rdkit_obj_connectivity(mol, len(local_atoms), charge)
            if not is_correct:
                continue
            frag_atom_charges = {
                g: mol.GetAtomWithIdx(local_index[g]).GetFormalCharge()
                for g in fragment
            }

            frag_bond_orders = {}
            for a in fragment:
                for b in fragment:
                    if a < b and adjmat[a, b] != 0:
                        bond = mol.GetBondBetweenAtoms(local_index[a], local_index[b])
                        frag_bond_orders[frozenset((a, b))] = bond.GetBondType()
            return charge, frag_atom_charges, frag_bond_orders

    logger.warning(
        "Could not determine a charge for a capped substituent fragment on specie %s",
        formula,
    )
    return None, {}, {}


def generate_borane_charge_state(prot: Protonation) -> ChargeState | None:
    """Charge a closo/nido cage by Wade's rules, since 3c-2e bonding has no ordinary
    Lewis structure to search for. Each vertex gives ``v - 2 + x`` skeletal
    electrons against 2*(n+1) closo or 2*(n+2) nido; the difference is the cage
    charge, placed on the lowest-index boron since it is really delocalised.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    adjmat = np.asarray(prot.adjmat)

    # Separate true cage atoms from B/C substituents; the rest lands in
    # exo_indices, split below into terminal atoms vs. fragments needing charge.
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

    # Wade's rules apply per cage, so a two-cage specie is counted twice over and
    # each cage carries its own charge, on its own lowest-index boron.
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
        # Exo is relative to THIS cage: a bond to the neighbouring cage is an
        # exo substituent for each of them, exactly like any other.
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

    # Substituent fragments' own internal bond orders (e.g. an aromatic
    # ring's Kekule pattern) are kept, not just their formal charges --
    # otherwise every such ring gets rebuilt as all-single bonds below
    # and comes out wrong (see _charge_capped_fragment's docstring).
    fragment_bond_orders: dict[frozenset, Chem.BondType] = {}
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

    # Cage atoms sit at degree 4-6, far past RDKit's valence tables -- 3c-2e
    # bonding a classical model cannot represent. NoImplicit and skipping
    # SANITIZE_PROPERTIES let that stand. Substituents keep implicit-H on.
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
    """Charge a fused multi-cage cluster by Jemmis' mno rule, ``m + n + o + p - q``.
    m, n and o come from the graph cut and q is assumed 0; p is looked up from
    MANUAL_CONJUNCTO_BORANE_P, being unreliable from a naive slice. A shared vertex
    contributes its full valence, and a bridging H its own single electron.
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

    # Substituent fragments' own internal bond orders (e.g. an aromatic
    # ring's Kekule pattern) are kept, not just their formal charges --
    # otherwise every such ring gets rebuilt as all-single bonds below
    # and comes out wrong (see _charge_capped_fragment's docstring).
    fragment_bond_orders: dict[frozenset, Chem.BondType] = {}
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

    # Cage atoms sit at abnormal valence for the same reason as in
    # generate_borane_charge_state. Bridging H's sit at degree 2 (bonded
    # to 2 borons), past RDKit's default H valence of 1 -- same situation.
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


# S and Se: the selenium analogues (TMTSF, BEDT-TSF) oxidise like their parents.
_TTF_CHALCOGENS = {16, 34}


def find_dithiolylidene_units(mol: Chem.Mol) -> list[int]:
    """The 2-position carbons of every 1,3-dithiol-2-ylidene unit.

    A unit is a five-ring with two chalcogens whose carbon between them is doubly
    bonded outside the ring. Oxidation moves that bond, leaving a dithiolium.
    """
    units = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        chalcogens = {
            idx
            for idx in ring
            if mol.GetAtomWithIdx(idx).GetAtomicNum() in _TTF_CHALCOGENS
        }
        if len(chalcogens) != 2:
            continue

        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                continue
            if {n.GetIdx() for n in atom.GetNeighbors()} & set(ring) != chalcogens:
                continue
            exocyclic = [n for n in atom.GetNeighbors() if n.GetIdx() not in ring]
            if len(exocyclic) != 1 or exocyclic[0].GetAtomicNum() != 6:
                continue
            bond = mol.GetBondBetweenAtoms(idx, exocyclic[0].GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE and idx not in units:
                units.append(idx)
    return units


def _alternating_bond_path(mol: Chem.Mol, start: int, end: int) -> list[int] | None:
    """Atom path from ``start`` to ``end`` reading DOUBLE, SINGLE, ..., DOUBLE.

    The conjugated bridge between two units -- one bond in a plain TTF, `C=C-C=C`
    in a vinylogue. Flipping it oxidises both units at once. None if there is no
    such path.
    """
    double, single = Chem.BondType.DOUBLE, Chem.BondType.SINGLE
    queue = deque([(start, double, [start])])
    seen: set[tuple[int, Chem.BondType]] = set()

    while queue:
        atom_idx, expected, path = queue.popleft()
        if (atom_idx, expected) in seen:
            continue
        seen.add((atom_idx, expected))

        for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            next_idx = neighbour.GetIdx()
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond.GetBondType() != expected or next_idx in path:
                continue
            if next_idx == end:
                # Arrive on a double bond: the far unit's exocyclic one.
                if expected is double:
                    return path + [next_idx]
                continue
            queue.append(
                (next_idx, single if expected is double else double, path + [next_idx])
            )
    return None


def _oxidised_donor_mol(mol: Chem.Mol, n_electrons_removed: int) -> Chem.Mol | None:
    """``mol`` with one or two electrons taken off its dithiolylidene units.

    Flipping the bridge turns both units into dithiolium cations. At one electron
    the second charge becomes an unpaired electron instead, since a radical cation
    has no closed-shell drawing.
    """
    units = find_dithiolylidene_units(mol)
    if len(units) != 2:
        return None

    first, second = units
    path = _alternating_bond_path(mol, first, second)
    if path is None:
        return None

    oxidised = Chem.RWMol(mol)
    for begin, end in zip(path, path[1:]):
        bond = oxidised.GetBondBetweenAtoms(begin, end)
        bond.SetBondType(
            Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.DOUBLE
            else Chem.BondType.DOUBLE
        )

    oxidised.GetAtomWithIdx(first).SetFormalCharge(1)
    far_atom = oxidised.GetAtomWithIdx(second)
    if n_electrons_removed == 2:
        far_atom.SetFormalCharge(1)
    else:
        far_atom.SetNumRadicalElectrons(1)
        far_atom.SetNoImplicit(True)

    result = oxidised.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as exc:
        logger.debug("Oxidised donor failed to sanitize: %s", exc)
        return None
    return result


def generate_oxidised_donor_charge_states(
    neutral: ChargeState,
) -> list[ChargeState]:
    """Radical-cation and dication states for a tetrathiafulvalene-type donor.

    The general search reaches neither, so the 0/+1/+2 ladder is built here and
    the balancer picks the rung. Fractional oxidation is not attempted.
    """
    if neutral.specie_total_charge != 0 or neutral.rdkit_obj is None:
        return []

    states = []
    for removed in (1, 2):
        mol = _oxidised_donor_mol(neutral.rdkit_obj, removed)
        if mol is None:
            continue

        atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        total_charge = neutral.protonated_total_charge + removed
        if sum(atom_charges) != total_charge:
            continue

        smiles = Chem.MolToSmiles(mol)
        logger.debug(
            "Oxidised donor state for %s: +%d | SMILES: %s",
            neutral.protonation.formula,
            removed,
            smiles,
        )
        states.append(
            ChargeState.from_positional(
                True,
                total_charge,
                atom_charges,
                mol,
                smiles,
                total_charge,
                True,
                neutral.protonation,
            )
        )
    return states


def _find_negative_moiety(
    spec: Specie,
) -> list[tuple[int, list[int], str]]:
    """COO- and SO3- moieties from ligand connectivity: a C with two, or an S with
    three, oxygens whose only non-metal neighbour is that centre.

    Returns a list of (center_idx, oxygen_idxs, kind).
    """

    atoms = spec.atoms or []
    molecule = spec.get_parent("molecule")
    moieties: list[tuple[int, list[int], str]] = []

    for center in atoms:
        center_idx = center.get_parent_index("molecule")
        if center.label not in ("C", "S"):
            continue
        oxygen_idxs: list[int] = []

        for nbr_idx in center.adjacency:
            nbr = molecule.atoms[nbr_idx]

            if nbr.label != "O":
                continue

            o_nonmetal_neighbors = [
                adj for adj in nbr.adjacency if adj not in nbr.metal_adjacency
            ]

            if o_nonmetal_neighbors == [center_idx]:
                oxygen_idxs.append(nbr_idx)

        if center.label == "C" and len(oxygen_idxs) == 2:
            moieties.append((center_idx, oxygen_idxs, "carboxylate"))

        elif center.label == "S" and len(oxygen_idxs) == 3:
            moieties.append((center_idx, oxygen_idxs, "sulfonate"))

    return moieties


def _classify_charged_moiety(
    label: str, k: int, n_nonmetal: int, n_oxygen: int
) -> tuple[str | None, int]:
    """Classify a centre by element, terminal-oxygen count and neighbour counts,
    returning (kind, net_charge) or (None, 0). Net-neutral look-alikes are
    excluded: C + 2 terminal O with no third substituent is CO2, S + 2 O + 2 C is
    a sulfone.
    """
    # --- Anionic oxo-anions: centre + k terminal O ---
    if label == "C":
        if k == 2 and n_nonmetal == 3:  # R-COO(-): 2 terminal O + 1 substituent
            return "carboxylate", -1
        # k == 2 and n_nonmetal == 2 -> CO2 (O=C=O), neutral: not a moiety.
        if k == 3:
            return "carbonate", -2
    elif label == "S":
        if k == 3:
            return "sulfonate", -1
        if k == 4:
            return "sulfate", -2
        if k == 2 and n_nonmetal == 3:  # 2 O + 1 C -> R-SO2(-)
            return "sulfinate", -1
        # k == 2 and n_nonmetal == 4 -> sulfone (neutral).

    # --- Cationic onium: full sigma valence to C/H only, no oxygen ---
    # if k == 0 and n_oxygen == 0:
    #     if label == "S" and n_nonmetal == 3:
    #         return "sulfonium", 1

    # Cationic N needs ring size, which this signature lacks: _find_cationic_nitrogen.
    return None, 0


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


def _amidinium_rings(
    atoms, neighbor_source, rings_by_atom
) -> list[tuple[int, list[int], str, int]]:
    """Five-ring amidinium cations -- imidazolium, pyrazolium, benzimidazolium --
    pinned by connectivity: two ring nitrogens, every ring atom three-connected.
    That excludes imidazole, imidazolidine, NHCs and cyclic ureas. Reported once
    per ring, since the charge is delocalised over N-C-N; missing it left AZEMEY's
    histidine zwitterion anchored only by its carboxylates, at -2 instead of 0.
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


def _find_cationic_nitrogen(
    atoms, neighbor_source, local_idx_by_id
) -> list[tuple[int, list[int], str, int]]:
    """Nitrogen whose +1 is pinned by connectivity: quaternary (4 sigma bonds),
    pyridinium (a pyridine ring -- one N, five C -- with two three-connected ring
    neighbours), or the five-ring amidinium of :func:`_amidinium_rings`. Keyed on
    bond count and ring shape, not an N-H, since N-alkylated cations carry none.
    Ring size alone is not enough: a diazine or an O/S-containing 6-ring is a
    different species whose charge is not pinned this way. Guanidinium is out of
    scope.
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
        # Smallest ring must be the pyridine itself: a 6-ring fused to a smaller
        # one is a different environment, as before.
        if min(len(ring) for ring in rings) != 6:
            continue
        if not any(_is_pyridine_ring(ring) for ring in rings):
            continue

        sp2_neighbors = sum(1 for n in neighbors if _nonmetal_connectivity(n) == 3)
        if sp2_neighbors >= 2:
            cations.append((i, [i], "pyridinium-N", 1))

    cations.extend(_amidinium_rings(atoms, neighbor_source, rings_by_atom))
    return cations


def _find_charged_moiety(
    spec: Specie,
) -> list[tuple[int, list[int], str, int]]:
    """Groups carrying a nonzero NET formal charge, from connectivity alone; metal
    coordination is ignored. Anionic ``centre + k terminal O``, plus cationic N, so
    the sum is the net rather than the anionic half. Returns (centre_local_idx,
    atom_local_idxs, kind, net_charge) indexed into ``spec.atoms``.
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
