from __future__ import annotations

from typing import TYPE_CHECKING

import itertools
import numpy as np
import networkx as nx
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

# Manually-verified Jemmis mno-rule "p" term (total vertices missing across
# all fused sub-polyhedra, relative to their own closo parents) for specific
# conjuncto/condensed borane cage compositions -- keyed by cage-only
# composition (e.g. "B22" for 22 boron cage atoms, ignoring substituents),
# since the same fused cage can carry different substituents (-OH, -OEt,
# ...) under different overall formulas. See
# generate_conjuncto_borane_charge_state's docstring for why p can't be
# derived generically from connectivity for these: unlike m, n, and o
# (which come straight out of the graph cut in find_conjuncto_borane_split),
# classifying each fused sub-polyhedron's own openness independently of the
# fusion seam is unreliable from a naive graph slice. Each entry here has
# been checked against a literature-reported charge, not guessed.
MANUAL_CONJUNCTO_BORANE_P = {
    # [B22H22]^2- docosaborate: closo-B12 icosahedron (p=0) fused to a
    # nido-B10 cluster (p=1) sharing a common edge. Verified against
    # Volkov, Rath & Barton, J. Organomet. Chem. 2003, 680, 212 (the
    # -OH derivative, [B22H21OH]^2-, charge -2).
    "B22": 1,
}


def generate_special_charge_states(spec: Specie) -> list[ChargeState] | None:
    """Generate charge states for species handled by a closed-form builder
    rather than the general bond-order search: antimony-halide anions
    (SbX3/X4/X5/X6-type), fullerene cages (C20, C60, C70, ..., including
    substituted derivatives), and closo/nido borane or carborane cages
    (B12H12^2-, o-carborane, a dicarbollide ligand, ..., including
    substituted derivatives).

    Return contract (three outcomes the caller must distinguish):
      * ``None``  -- not a special case; the caller should run the general
        charge-state search.
      * ``[cs]``  -- a special case whose closed-form builder succeeded.
      * ``[]``    -- a special case whose builder FAILED.
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
    # Fullerene cage or dimer (has_fullerene is True for both; a dimer is two
    # cages joined by a single direct C-C bond, e.g. [C60-C60], BAQLUC01). Route
    # to the matching closed-form builder: the dimer builder needs its own
    # per-cage Kekule/leftover handling, so a dimer must NOT go through the
    # single-cage builder (which would fail on the degree-4 bridge carbons).
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

    # Borane/carborane cage (possibly substituted), single closo or nido
    # deltahedron. Fused conjuncto (multi-cage) clusters -- e.g. the
    # docosaborate [B22H22]^2- -- fail is_borane_cage's single-deltahedron
    # check (it has no dimer-style fallback the way has_fullerene does for
    # C60-C60), so has_borane is False for them and they fall through to
    # the general search below rather than generate_conjuncto_borane_charge_state.
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
    """
    Builds Sb/halogen-only species directly from connectivity.

    Mononuclear species (single Sb center): each Sb's formal charge is set
    from its own halogen coordination number, since degree directly tracks
    oxidation state here. Hexacoordinate Sb (SbX6-) gets formal charge -1.
    Every other coordination number (3, 4, 5) keeps one lone
    pair, giving formal charge = 3 - degree: neutral SbX3, SbX4-, SbX5(2-).

    Polynuclear species (multiple Sb centers, e.g. bridged iodoantimonate
    clusters like [Sb7I25]4-). These clusters are always built from
    Sb(III) + halide ligands, so the total charge is fixed at
    3 * n_Sb - n_halogens, Sb centers are drawn neutral and the charge
    is instead distributed across only as many halogens as needed.
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

        # Distribute exactly enough -1 charges across the halogens so the
        # total comes out to 3 * n_Sb - n_halogens: one per Sb, round-robin,
        # so it's spread evenly across centers rather than piled onto a
        # few. Bridging halogens aren't required to carry it -- prefer a
        # terminal neighbor (bonded to just this one Sb) and only fall
        # back to a bridging one if this Sb has no terminal neighbor left.
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
    """
    Identify which atoms form a fullerene cage, including when it's
    embedded in a larger substituted derivative (e.g. a methanofullerene
    like PCBM, a fullerenol C60(OH)n, a halofullerene, or a fullerane
    C60H2n) -- by taking the 3-core of the carbon-only bond graph:
    repeatedly strip away any carbon with fewer than 3 *carbon* neighbors.

    Cage carbons are inherently 3-connected to other cage carbons and
    always survive this, regardless of how many of them also carry an
    exocyclic substituent (which adds a 4th bond but never removes one of
    the 3 cage bonds). Any substituent -- a lone terminal atom (H, a
    halogen), an -OH, or an organic tail/ring bonded through the cage at a
    single point -- eventually strips away completely: tracing back along
    any real substituent from its outermost leaves always reaches a point
    where degree drops below 3, and removing it can only lower the degree
    of what's left, cascading until nothing but the cage remains.

    For a bare, unsubstituted fullerene, every atom is already carbon and
    already 3-connected, so this returns every atom -- identical to the
    cage this function replaces determining via `all(z == 6 for z in
    atoms)` before substituent support existed.

    Returns the sorted list of cage atom indices, or None if there are no
    carbon atoms at all.
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
    """
    Detect any fullerene cage (C20, C60, C70, C76, C84, ...) purely from
    connectivity - no bond orders, no coordinates required - including
    when it's embedded in a larger substituted derivative (see
    find_fullerene_cage_indices).

    A graph is a fullerene skeleton iff its cage-only atoms (see
    find_fullerene_cage_indices) form a subgraph that is:
      1. all carbon
      2. 3-regular among themselves (every cage atom has exactly 3 cage
         neighbors -- any exocyclic substituent bond doesn't count here)
      3. a single connected component
      4. planar (embeds on a sphere - a topological requirement for any
         closed convex-ish cage)
      5. girth >= 5 (no 3- or 4-membered rings/faces; fullerene faces are
         only pentagons and hexagons by construction)
      6. has a valid fullerene vertex count (even, >=20, != 22)

    Given 2-6, Euler's formula (V - E + F = 2) combined with the pentagon/
    hexagon face constraint forces exactly 12 pentagonal faces regardless
    of n - that part never needs to be checked separately, it's automatic.

    Returns
    -------
    (is_fullerene, reason) : tuple[bool, str]
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
        # A fullerene dimer -- two closed cages joined by a single direct C-C
        # bond (e.g. [C60-C60]) -- fails the 3-regular test: its two bridge
        # carbons are degree 4 (3 cage bonds + 1 inter-cage bond). Accept it if
        # find_fullerene_dimer_split confirms both sides are complete cages once
        # the bridge is removed. No recursion risk: each half is an ordinary
        # single cage that passes the degree-3 test and never re-enters here.
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
    """
    For two fullerene cages joined by a single direct C-C bond -- e.g.
    1,1'-bi-C60fulleride, [C60-C60]^2- -- find that bridging bond and
    confirm each side is an independently valid, complete fullerene cage
    in its own right.

    This is a different situation from a substituted single fullerene
    (find_fullerene_cage_indices): there, a substituent gets excluded via
    a 3-core because it doesn't sustain degree-3 connectivity on its own.
    Here, BOTH sides of the cut are themselves complete, ordinary
    fullerene topologies once the bridging bond is removed (every atom,
    including the former bridge atom, reverts to degree exactly 3) -- so
    a 3-core can't separate them; the two full cages have to be told
    apart directly by trying each bridge (an edge whose removal
    disconnects the graph -- necessarily true of a direct inter-cage
    bond, since removing any cage-internal edge can't disconnect a
    3-connected polyhedral cage) and checking has_fullerene on each
    side independently.

    Only a plain two-cage split is handled (not three or more cages, and
    not a spacer-mediated link through non-cage atoms).

    Returns {"cages": [[...], [...]], "bridge": (atom_in_cage0,
    atom_in_cage1)} (atom indices into `atoms`), or None if no such split
    is found.
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
    """
    Identify which atoms form a closo/nido borane or carborane deltahedral
    cage, including when it's embedded in a larger substituted derivative
    (e.g. a phosphine-tethered dicarbollide, an alkylated o-carborane, a
    perhalogenated closo-borate) -- by taking the 3-core of the B/C-only
    bond graph: repeatedly strip away any B or C atom with fewer than 3
    B/C neighbors. Mirrors find_fullerene_cage_indices's approach.

    Every vertex of a closo or nido deltahedron has at least 3 cage
    neighbors (the smallest closo deltahedron, the trigonal bipyramid,
    has its two apex atoms at exactly 3 -- see is_borane_cage), so
    genuine cage atoms always survive this regardless of how many of them
    also carry an exocyclic substituent (which adds a bond but never
    removes one of the cage bonds). Any substituent -- a lone carbon
    bonded only through a non-B/C atom, an alkyl chain, or an aromatic
    ring attached at a single point -- strips away completely: tracing
    back from its outermost leaves always reaches a point where B/C
    degree drops below 3, cascading until nothing but the cage remains.

    Returns the sorted list of cage atom indices, or None if there are no
    boron or carbon atoms at all.
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


def is_borane_cage(atoms, AC) -> tuple[bool, str]:
    """
    Detect a closo- or nido-type borane/carborane deltahedral cage
    (closo-B12H12^2-, closo-C2B10H12/o-carborane, nido-[C2B9H11]^2-
    "dicarbollide", closo-B12Cl12^2-/"dodecachloro-closo-dodecaborate",
    ...) purely from connectivity - no bond orders, no coordinates
    required. Mirrors has_fullerene's coordinate-free, planar-graph-
    based approach, including substituent handling (see
    find_borane_cage_indices).

    Cage vertex atoms are B and/or C (a carborane substitutes some cage
    B for C); anything else attached to them - terminal H, a halogen
    (perhalogenated clusters like B12X12^2- are common), an organic
    substituent (e.g. a phosphine-tethered dicarbollide ligand), or a
    metal coordinated through a dicarbollide's open face - is exocyclic
    and is dropped before the cage-only subgraph is checked, regardless
    of what it actually is.

    Two Wade's-rules cluster families are recognized from that subgraph's
    planar face sizes:
      - closo (n >= 5 cage atoms): a full deltahedron - every face is a
        triangle. These are the free, typically dianionic clusters
        (B12H12^2-, closo-carboranes).
      - nido (n >= 6 cage atoms): a closo (n+1)-vertex deltahedron with
        one vertex removed, leaving exactly one open pentagonal face and
        every other face triangular. This is the dicarbollide ligand
        family, which binds a metal eta5 through that open face like Cp-.

    Returns
    -------
    (is_cage, reason) : tuple[bool, str]
    """
    atoms = [int(a) for a in atoms]
    # See the matching comment in has_fullerene: ligand adjmats can be
    # dtype=object even though every value is a plain int, which networkx
    # rejects outright.
    ac = np.asarray(AC, dtype=int)
    n_total = len(atoms)

    if ac.shape != (n_total, n_total):
        return False, "bad_ac_shape"

    # 1. Cage composition - find the embedded B/C cage core, if any (see
    #    find_borane_cage_indices), then require at least one B in it (a
    #    pure-carbon deltahedral cage isn't a known species; pure-carbon
    #    cages are handled by has_fullerene, whose 3-regular/girth>=5
    #    signature is disjoint from a deltahedron's anyway).
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

    # 4. Must be one connected cage, not fragments/disorder artifacts
    if not nx.is_connected(graph):
        return False, "disconnected_cage"

    # 5. Every deltahedron vertex has degree >= 3 (the two apices of the
    #    trigonal bipyramid); rules out spurious low-connectivity AC
    #    artifacts cheaply, before the planarity/face check.
    if any(d < 3 for _, d in graph.degree()):
        return False, "degree_too_low"

    # 6. Planarity + face-size distribution - required for, and fully
    #    characterizes, a genuine closo/nido deltahedral cage.
    face_sizes = _cage_face_sizes(graph)
    if face_sizes is None:
        return False, "not_planar"

    if all(size == 3 for size in face_sizes):
        return True, "closo_cage_confirmed"

    if (
        n >= 6
        and sum(1 for size in face_sizes if size == 5) == 1
        and all(size == 3 for size in face_sizes if size != 5)
    ):
        return True, "nido_cage_confirmed"

    return False, "not_deltahedral"


def find_conjuncto_borane_split(atoms, AC) -> dict | None:
    """
    For a condensed/conjuncto borane cage that fails the single-deltahedron
    check (is_borane_cage) -- e.g. the docosaborate [B22H22]^2- anion, a
    closo-B12 icosahedron fused to a nido-B10 cluster -- look for a small
    vertex cut (a single shared vertex, or a shared edge of 2 atoms) that
    splits the B/C cage graph into exactly two separate, genuinely
    polyhedral pieces.

    This only identifies *that* a valid 2-polyhedron split exists and what
    the shared/cut atoms are; it does NOT classify either side as
    closo/nido/arachno. Isolating one fused sub-polyhedron's own atoms
    (its unique atoms plus the shared cut atoms) and checking its face
    sizes directly does not reliably reveal that classification -- the
    shared cut atoms' bonds to the *other* sub-polyhedron aren't part of
    this side's own subgraph, which distorts the local face pattern at
    the fusion seam (verified directly on the docosaborate case: the
    nido-B10 side comes out with one square and one hexagonal face rather
    than the single pentagonal opening a clean nido shape would show).
    Callers needing that classification (the mno rule's "p" term) use a
    manual, literature-verified lookup instead -- see
    MANUAL_CONJUNCTO_BORANE_P.

    Returns a dict with:
      "cage_indices": all B/C cage atom indices (both sub-polyhedra)
      "cut_indices": the 1 or 2 atoms shared between the two sub-polyhedra
      "components": [[...], [...]], the atoms unique to each side
      "o": 1 if the two sub-polyhedra share a single vertex, else 0 (an
           edge/2-atom share)
    or None if no such split is found.
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
    """
    Detect an *open* fullerene cage: a fullerene-derived carbon shell whose
    cage has been opened at an orifice (often functionalised with O/N at the
    rim, e.g. AFITUH's H16-C73-N2-O2 open-cage C60 derivative).

    ``has_fullerene`` rejects these -- the orifice leaves rim carbons with only
    two cage neighbours, so the carbon 3-core (which a closed cage survives
    intact) cascades away to nothing. An open cage is instead recognised from
    the looser carbon 2-core plus geometry:

      1. a large carbon 2-core (>= 40 atoms; practical open cages are C60/C70-
         derived, and this floor keeps medium polycyclic aromatics out);
      2. mostly sp2 -- >= 60% of the 2-core carbons keep three carbon
         neighbours (the shell), the rest being the 2-connected orifice rim;
      3. a large fused-ring system (cyclomatic number >= 20; a fullerene shell
         has ~30 faces). This rejects calixarene/cryptophane-type covalent
         organic cages, whose aromatic rings are linker-separated (few fused
         rings);
      4. genuine 3D thickness -- the smallest / largest principal-axis extent
         (SVD) is >= 0.25. A closed or open cage is a 3D shell; a flat
         polycyclic aromatic (coronene, a graphene flake) collapses onto a
         plane and is rejected here.

    Coordinate-based (unlike ``has_fullerene``). Used to skip missing-hydrogen
    detection, where curved sp2 cage carbons are otherwise mis-read as
    under-coordinated. Closed fullerenes are already caught by
    ``has_fullerene``; this only adds the opened ones.

    Returns (is_open_fullerene, reason).
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
    """
    Shared detection for porphyrin/porphine-, phthalocyanine-, and
    corrole/corrin-type N4 macrocycles: four five-membered pyrrole-type
    rings (1 N + 4 C each) joined pairwise into one closed macrocycle.
    Three (corrole/corrin) or four (porphyrin/phthalocyanine) of those
    links go through a single meso bridge atom -- carbon for a porphyrin,
    nitrogen (aza) for a phthalocyanine -- while corrole and corrin are
    "ring-contracted": one link is a direct bond between an atom of each
    ring instead of going through a meso atom, giving a 15-membered core
    instead of the 16-membered porphyrin/phthalocyanine core. Substituents
    (aryl/alkyl groups, fused benzo rings on each pyrrole for
    phthalocyanine, H, a coordinated metal, etc.) are ignored -- only the
    core ring topology is checked, mirroring has_fullerene's
    coordinate-free approach.

    Returns one (rings, bridges) entry per disjoint macrocycle found --
    e.g. two entries for a bis-porphyrin ligand bridging two metals -- or
    an empty list if none exists. In each entry the rings/bridges are in
    cyclic traversal order: rings[i] and rings[(i+1) % 4] are joined by
    bridges[i], the meso atom index, or by a direct bond if bridges[i] is
    None. Macrocycles are peeled off greedily and never share a pyrrole
    ring, so their cores are disjoint.
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
    # 16 core carbons (4 pyrrole rings * 4 C) is the hard minimum for any
    # porphyrin-family macrocycle. Carbon meso bridges push this higher (20 for
    # a full porphyrin, 19 for a ring-contracted corrole/corrin), but a
    # porphyrazine / phthalocyanine has *nitrogen* meso bridges, so its core can
    # bottom out at exactly 16 C when no carbon-adding fused rings are present
    # (e.g. GAFMUW01, a tetrakis(thiadiazole)porphyrazine: C16-N16-S4).
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

    # Peel off one closed macrocycle at a time. A classic porphyrin has
    # exactly 4 candidate rings; a bis-porphyrin has 8 forming two disjoint
    # 4-macrocycles; an expanded porphyrin (penta-/hexaphyrin, ...) closes a
    # single loop of 5, 6, or more pyrroles. Each iteration finds the
    # smallest closed loop among the remaining rings -- smallest-first so a
    # bis-porphyrin still resolves into two genuine 4-macrocycles rather than
    # one spurious 8-membered loop threaded through both -- records it, and
    # removes those rings before searching the rest, so a pyrrole is never
    # reused across macrocycles.
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
    """
    Detect every disjoint porphyrin/porphine-, phthalocyanine-, or
    corrole/corrin-type N4 macrocycle in a structure purely from
    connectivity (see _find_porphyrin_rings_and_bridges). Returns one
    (nitrogens, is_contracted, core_atoms) tuple per macrocycle -- two for
    a bis-porphyrin ligand, and so on -- or an empty list if none is found.
    See find_porphyrin_macrocycle for what each field means.
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
    """
    Detect a single porphyrin/porphine-, phthalocyanine-, or
    corrole/corrin-type N4 macrocycle purely from connectivity (see
    _find_porphyrin_rings_and_bridges). Returns the first macrocycle found,
    or None if there is none; use find_all_porphyrin_macrocycles for
    ligands that may hold more than one (e.g. bis-porphyrins).

    Returns (nitrogens, is_contracted, core_atoms) if found, else None:
    - `nitrogens`: the 4 macrocycle (pyrrole-type) nitrogen atom indices
      in cyclic macrocycle order (not sorted by index) -- position i and
      i+2 are the "opposite" (bridge-separated-by-two-rings) pair,
      positions i and i+1 are "adjacent" (single bridge apart).
    - `is_contracted`: True for a corrole/corrin-type macrocycle (one
      ring-to-ring link is a direct bond rather than a meso bridge),
      False for a classic porphyrin/phthalocyanine macrocycle (all 4
      links are meso bridges).
    - `core_atoms`: every atom in the 4 pyrrole/pyrroline rings plus the
      meso bridge atoms (sorted) -- i.e. the whole macrocycle core,
      excluding any exocyclic substituent (aryl/alkyl groups, fused benzo
      rings, H, a coordinated metal, etc.).
    """
    macrocycles = find_all_porphyrin_macrocycles(atoms, AC)
    return macrocycles[0] if macrocycles else None


def porphyrin_reference_protonation_sites(
    macrocycle_nitrogens: list[int], is_contracted: bool
) -> list[int]:
    """
    The macrocycle nitrogens that carry the proton in the neutral
    free-base tautomer -- the opposite (trans) pair for a classic
    porphyrin/phthalocyanine macrocycle (all 4 links are meso bridges),
    or 3 of the 4 nitrogens for a ring-contracted corrole/corrin
    macrocycle (the 4th, structurally tied to the direct ring-to-ring
    bond, stays an unprotonated imine-type nitrogen).

    For an expanded porphyrin (penta-/hexaphyrin, ...) this returns the
    N-H set of the aromatic free base: N-H on alternating ring nitrogens
    around the macrocycle (e.g. 3 of the 6 for a [26]hexaphyrin). This is
    the baseline count m0; the actual number of protons enumerated for such
    a macrocycle is bracketed at m0 +/- 1 to cover the oxidation-level
    variants ([26]/[28], ...) -- see _generate_porphyrin_protonation_states.

    The classic N4 result is what generate_porphyrin_charge_state relies on
    to stay consistent with the enumerator.
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
    """
    Check whether the k given pyrrole-type rings are connected pairwise,
    each to exactly two others, into one single macrocyclic loop rather
    than, say, two separate pairs or an open chain -- via either a
    single-atom meso bridge (carbon for a porphyrin, nitrogen for a
    phthalocyanine; either is accepted here, so this one check covers both
    macrocycle families) or a direct bond between an atom of each ring (the
    ring-contraction found in corrole/corrin, and the bipyrrole links of
    expanded porphyrins such as rubyrin, replacing a meso bridge). k is 4
    for a classic porphyrin/corrole and 5, 6, ... for expanded porphyrins
    (penta-/hexaphyrin, ...).

    For the k == 4 families a further constraint applies: at most one of
    the four links may be a direct ring-to-ring bond (the single
    corrole/corrin contraction) -- two or more would be a spurious loop
    (e.g. two rings from each of two different macrocycles in a
    bis-porphyrin, stitched together by inter-macrocycle bonds), not a
    genuine tetrapyrrole. Larger loops are only accepted once no smaller
    one exists (see _find_porphyrin_rings_and_bridges), so this guard is
    what keeps a bis-porphyrin resolving into two real 4-macrocycles.

    Returns (order, bridge_by_edge) if so, else None: `order` is the
    cyclic traversal order (indices into `rings`); `bridge_by_edge` maps
    each frozenset({ring_idx_a, ring_idx_b}) to the meso atom index
    bridging that pair, or to None if that pair is joined by a direct
    bond instead.
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
    """
    True if the connectivity contains a porphyrin/porphine-,
    phthalocyanine-, or corrole/corrin-type N4 macrocycle -- four pyrrole
    rings bridged into one closed ring via carbon meso bridges
    (porphyrin), nitrogen aza-meso bridges (phthalocyanine), or 3 meso
    bridges plus one direct ring-to-ring bond (the ring-contracted
    corrole/corrin core) -- regardless of substituents or fused benzo
    rings. See find_porphyrin_macrocycle for the detection logic.
    """
    result = find_porphyrin_macrocycle(atoms, AC)
    if result is None:
        return False, "no_porphyrin_macrocycle"
    _nitrogens, is_contracted, _core_atoms = result
    if is_contracted:
        return True, "corrole_or_corrin_macrocycle_confirmed"
    return True, "porphyrin_macrocycle_confirmed"


def generate_porphyrin_charge_state(prot: Protonation) -> ChargeState | None:
    """
    Builds a charge state for a porphyrin/phthalocyanine/corrole/corrin
    N4 macrocycle ligand by fixing the macrocycle core's formal charges
    analytically instead of running the whole ligand through the general
    combinatorial AC2mol bond-order/charge search, which can settle on a
    chemically implausible charge-separated resonance structure for a
    conjugated ring system this size. Substituents (meso-aryl groups,
    beta-pyrrole substituents, axial groups, ...) are split off and
    charged independently via the same fragment-capping search used for
    fullerene/borane substituents (_collect_substituent_fragment /
    _charge_capped_fragment), then summed with the core's charge.

    Core charge: the neutral free base carries one N-H on each pyrrolic
    nitrogen; the remaining ring nitrogens are imine-type (=N-, part of a
    ring double bond). Both are formally neutral, so we take the pyrrolic
    (reference) set to be exactly the nitrogens `prot` actually protonates
    and leave every core atom neutral, letting RDKit's Kekulization pick a
    closed-shell bond pattern consistent with that N-H arrangement. If none
    exists (a wrong-parity expanded-porphyrin tautomer, say), sanitization
    fails and the state is dropped. The core is therefore always neutral,
    and the ligand's metal-bound charge (uncorrected total minus the added
    protons) comes out as -(number of N-H) -- -2 for a porphyrin, -3 for a
    corrole, and -(m0-1..m0+1) across the enumerated hexaphyrin states --
    with no combinatorial search needed for the core, only for any
    non-trivial substituent.
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

    core_charge = sum(atom_charges[n] for n in nitrogens)
    expected_core_charge = prot.n_protons_added - len(reference_sites)
    if core_charge != expected_core_charge:
        logger.warning(
            "Porphyrin-family core charge mismatch for %s: %d from nitrogen "
            "protonation vs %d expected from proton count; using the "
            "proton-count value",
            prot.formula,
            core_charge,
            expected_core_charge,
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
            # logger.debug(
            #     "Porphyrin-family specie %s has a substituent fragment with "
            #     "bond(s) back to the macrocycle at core atom(s) %s",
            #     prot.formula,
            #     [c for c, _s in attach_bonds],
            # )
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

    # Ring atoms that are saturated (four connections) cannot carry a pi system:
    # a reduced macrocycle -- chlorin, bacteriochlorin, or a bridged/fused
    # variant -- has sp3 carbons sitting inside the ring set. Four aromatic
    # bonds on such a carbon means an explicit valence of 4 x 1.5 = 6, and
    # sanitization rejects the whole molecule. Bond them as single instead and
    # leave the rest of the macrocycle aromatic.
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
                # A macrocycle's own bonds (ring + meso bridges) are left
                # aromatic and Kekulized by RDKit below, rather than
                # hand-assigned, since -- unlike a fullerene cage or a
                # borane cluster -- a flat conjugated tetrapyrrole ring is
                # exactly the kind of system RDKit's standard aromaticity
                # model is built for. This only applies within a single
                # ring: a bond directly linking two different macrocycles in
                # a fused bis-porphyrin is an inter-ring single bond (handled
                # by the `else` below), not part of either aromatic system.
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
        # Keep RDKit's own Hueckel-based SETAROMATICITY perception out of
        # it -- our formal charges/NoImplicit flags above already encode
        # which ring atoms have a lone pair vs. a double bond, so only
        # SANITIZE_KEKULIZE needs to run, picking bond orders consistent
        # with what we've already fixed rather than re-deriving them.
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
        )
    except Exception as e:
        # Not fatal: the specie falls back to the general charge enumeration.
        # Reaching here means the ring matched the macrocycle topology but is
        # not an aromatic tetrapyrrole after all -- e.g. a cyclopropane-fused or
        # otherwise reduced core, where the remaining ring bonds are localized
        # imines rather than one delocalized system.
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
    """
    Builds a closed-shell Kekule structure for a fullerene cage, including
    substituted derivatives (methanofullerenes like PCBM, fullerenols,
    halofullerenes, fulleranes, ...).

    A bare fullerene skeleton is a bridgeless 3-regular graph, which by
    Petersen's theorem always has a perfect matching. Promoting each
    matched edge to a double bond gives every carbon its 4th bond
    directly, with no charge separation and no combinatorial bond-order
    search -- the kind of search that AC2mol/rdDetermineBonds cannot
    afford to run over 60+ atoms.

    A cage carbon bearing an exocyclic substituent already has 4 sigma
    bonds (3 cage + 1 exocyclic) and so is already valence-satisfied
    without a double bond at all -- it's excluded from the matching graph
    entirely, and all 3 of its cage-neighbor bonds are forced single. The
    perfect matching is instead computed over just the *unsubstituted*
    cage atoms (see find_fullerene_cage_indices for how the cage itself
    is identified within a larger substituted structure).

    The cage's own charge is always 0 here: unlike a borane cage, a
    fullerene's bonding is ordinary 2c-2e covalent bonding once the
    Kekule structure is fixed, so every cage atom (substituted or not)
    ends up with a complete, neutral valence by construction. Any actual
    charge comes only from a substituent that isn't itself neutral (e.g.
    a carboxylate tail) -- exactly as with borane cages, anything beyond
    a simple terminal H/halogen is split off and charged independently
    via the general AC2mol search (_collect_substituent_fragment /
    _charge_capped_fragment).
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

    # Substituents beyond a simple terminal H/halogen are split off and
    # charged independently, exactly as for borane cages. Their own
    # internal bond orders (e.g. a pyridyl ring's aromatic Kekule
    # pattern) are kept, not just their formal charges -- otherwise every
    # such ring would get rebuilt as all-single bonds below and come out
    # wrong (see _charge_capped_fragment's docstring).
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
    """
    Builds a two-fullerene-cage dimer joined by a single direct C-C bond
    -- e.g. 1,1'-bi-C60fulleride, [C60-C60]^2- -- from connectivity (see
    find_fullerene_dimer_split for how the two cages and the bridge bond
    are identified).

    Each cage's own bridge atom goes from degree 3 (bare fullerene, needs
    1 double bond to complete its valence) to degree 4 once the
    inter-cage bond forms (3 cage sigma bonds + 1 more to the other cage
    = valence 4 already, no double bond needed or possible) -- exactly
    like a substituted position in generate_fullerene_charge_state, so
    it's excluded from its own cage's perfect-matching computation.

    That leaves an *odd* number of atoms per cage needing pairing (a
    fullerene always has an even vertex count, so vertex count minus the
    1 excluded bridge atom is always odd) -- there is no way to pair all
    of them, so each cage is left with exactly one genuine unpaired
    position. This mirrors real (C60)2-type dimers, which are known to
    have real radical/weak-bond character at the link rather than a
    simple, fully-paired closed shell.

    Unlike a borane cage's charge (fixed by Wade's rules) or a bare/
    substituted fullerene's charge (always neutral by construction),
    there's no way to derive from connectivity alone whether each
    leftover position resolves as a closed-shell carbanion (formal charge
    -1, gaining an electron) or carbocation (+1, losing one) -- that
    depends on the actual electron count of the crystal, not the
    topology. So this returns multiple candidate ChargeStates (both
    anionic, both cationic, and one of each) for the resolver's ordinary
    charge-reconciliation logic to pick from downstream, the same way it
    already does for ordinary ligands via get_candidate_charges --
    instead of committing to a single guessed answer.
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
    """
    Charges a substituent fragment on its own: every bond that used to go
    to the cage is capped with its own plain H instead (usually one, but
    a fragment attached to the cage at multiple points -- e.g. a
    methanofullerene's cyclopropane bridgehead, bonded to 2 cage carbons
    at once -- gets one capping H per broken bond, so `attach_atoms` may
    repeat the same fragment atom). The general AC2mol bond-order search
    is then tried at increasing |charge| until one succeeds -- this is
    the same search the resolver runs for ordinary ligands, just applied
    to the cut-off fragment rather than a whole specie.

    Returns (charge, {global_atom_idx: formal_charge}, {frozenset({i, j}):
    bond_type}) for every atom/internal bond in `fragment` (the capping
    H's and the bonds to them are discarded -- those get replaced by the
    real cage bond in the caller's merged molecule). The bond orders
    matter, not just the charges: a fragment like a pyridyl ring needs
    its own alternating single/double Kekule pattern to be a valid
    aromatic ring at all -- rebuilding it as all-single bonds in the
    merged molecule (using only the formal charges from here) leaves
    every ring atom a bond short, which RDKit then silently "fixes" by
    padding in extra implicit hydrogens instead of raising a valence
    error, producing a saturated ring where an aromatic one belongs.

    Returns (None, {}, {}) if no charge produced a sanitizable structure.
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
    """
    Builds a closo- or nido-type borane/carborane deltahedral cage
    directly from connectivity, using Wade's rules skeletal-electron
    counting to fix the cage's own charge in closed form instead of
    running the general bond-order search over it -- the cage's
    3-center-2-electron bonding isn't expressible as ordinary 2c-2e
    bonds/localized formal charges in the first place, so there is no
    "correct" Lewis structure for AC2mol to search for.

    Each cage vertex atom (B or C) contributes (v - 2 + x) skeletal
    electrons, where v is its main-group valence electron count (B: 3,
    C: 4) and x is its number of exocyclic (non-cage) substituents. This
    count only depends on *how many* terminal substituents a vertex has,
    not on what they are: whether a vertex's terminal position is H, a
    halogen, or the attachment point of a larger substituent group, it's
    still one ordinary 2c-2e exocyclic bond and so still contributes the
    same +1. The required skeletal electron count is 2*(n+1) for a closo
    cage (n vertices, a full deltahedron) or 2*(n+2) for a nido cage (n
    vertices, one vertex short of a closo deltahedron). The cage's own
    charge is whatever is left over: contributed - required.

    A lone terminal H or halogen substituent needs nothing further -- its
    single exocyclic bond is already accounted for above. Anything else
    (e.g. the -NH2 in 1-amino-closo-dodecaborate, [1-NH2-B12H11]^2-) is
    the root of a larger substituent fragment, which is split off, capped
    with a plain H where the cage bond was cut, and charged independently
    via the general AC2mol search (see _charge_capped_fragment) -- since
    the cage's Wade's-rule contribution above already doesn't care what's
    on the far end of that bond, the cage charge and every substituent's
    charge are independent and simply add.

    This reproduces the standard reference points exactly: closo-B12H12
    and closo-B12X12 (X = F/Cl/Br/I) both at -2, closo-C2B10H12
    (o-carborane) at 0, nido-[C2B9H11] ("dicarbollide") at -2, and
    [1-NH2-B12H11] at -2 (neutral -NH2 substituent, so unchanged from
    plain B12H12^2-).

    The cage's own charge has no single real localized site -- it's
    delocalized across the cluster -- so it is placed entirely on one
    arbitrarily chosen boron atom (the lowest-index cage B) purely to
    give RDKit a valid formal-charge assignment to sanitize against.
    Substituent fragment charges, in contrast, are real localized Lewis
    charges and are placed wherever their own AC2mol search puts them.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    adjmat = np.asarray(prot.adjmat)

    # Isolate the true cage atoms from any B/C substituents (e.g. a
    # phosphine-tethered dicarbollide's alkyl/aryl carbons) -- see
    # find_borane_cage_indices. Everything else lands in exo_indices and
    # is split further below into simple terminal substituents vs. larger
    # fragments that need their own independent charge.
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

    n = len(cage_indices)

    is_cage, reason = is_borane_cage(prot.atnums, adjmat)
    if reason == "closo_cage_confirmed":
        required_se = 2 * (n + 1)
    elif reason == "nido_cage_confirmed":
        required_se = 2 * (n + 2)
    else:
        logger.warning(
            "Borane/carborane specie %s failed cage topology re-check (%s)",
            prot.formula,
            reason,
        )
        return None

    contributed_se = 0
    for i in cage_indices:
        valence_electrons = 3 if prot.atnums[i] == 5 else 4
        exo_count = int(np.count_nonzero(adjmat[i][exo_indices])) if exo_indices else 0
        contributed_se += valence_electrons - 2 + exo_count

    core_charge = contributed_se - required_se

    allowed_simple_labels = {"H"} | HALOGENS
    simple_exo = {
        i
        for i in exo_indices
        if prot.labels[i] in allowed_simple_labels
        and int(np.count_nonzero(adjmat[i])) == 1
    }
    complex_roots = [i for i in exo_indices if i not in simple_exo]

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

    # Cage atoms sit at degree 4-6 here, far past RDKit's default valence
    # tables (B: 3, C: 4) -- the 3c-2e cage bonding a classical valence
    # model can't represent. NoImplicit + skipping SANITIZE_PROPERTIES
    # below both exist to let that stand rather than have RDKit "correct"
    # it by adding implicit Hs or rejecting the structure outright.
    # Substituent atoms keep implicit-H handling on: reinstating the real
    # cage bond in place of the fragment's capping H leaves their degree
    # and valence exactly as they were when charged as a fragment.
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
    """
    Builds a conjuncto (fused multi-cage) borane/carborane cluster -- e.g.
    the docosaborate [B22H22]^2- anion, a closo-B12 icosahedron fused to a
    nido-B10 cluster sharing a common edge -- directly from connectivity,
    using Jemmis' mno-rule extension of Wade's rules:

        required skeletal electron pairs = m + n + o + p - q

    where m = number of fused sub-polyhedra, n = total unique cage
    vertices, o = number of single-*vertex*-sharing condensations (0 for
    a shared edge/2-atom share, which is what find_conjuncto_borane_split
    currently detects), p = total vertices missing across all
    sub-polyhedra relative to their own closo parents (e.g. 1 for a
    closo+nido pair), and q = number of capped vertices (assumed 0 here --
    capped/hypercloso conjuncto systems aren't handled).

    m, n, and o are determined generically from connectivity by
    find_conjuncto_borane_split: m is fixed at 2 (the only fusion pattern
    currently detected), n is the total B/C cage atom count, and o follows
    directly from whether the detected cut is a single shared vertex
    (o=1) or a shared edge (o=0).

    p is NOT derived from graph structure. Classifying each fused
    sub-polyhedron's own openness (closo/nido/arachno) independently of
    the fusion seam is unreliable from a naive graph slice -- see
    find_conjuncto_borane_split's docstring, which documents this exact
    failure on the docosaborate case. So p is looked up from
    MANUAL_CONJUNCTO_BORANE_P, keyed by cage composition (e.g. "B22"),
    the same way MANUAL_CHARGE_ASSIGN_SPECIES handles other cases this
    resolver can't derive generically -- each entry there has been
    checked against a literature-reported charge, not guessed.

    Once required_SE is fixed, contributed skeletal electrons are counted
    the same way as generate_borane_charge_state, extended for two
    wrinkles specific to fused multi-cage topology (both verified
    directly against the docosaborate reference structure):
      - A cage vertex belonging to *both* sub-polyhedra typically has no
        exocyclic substituent at all (every one of its bonds goes to
        other cage atoms). Such a vertex contributes its full valence
        electron count rather than the usual (v - 2 + x), since it has
        no exocyclic bond to "spend" 2 electrons on.
      - A bridging (mu-) hydrogen shared between two cage atoms
        contributes its own 1 electron directly to the pool. It is not
        counted as an exocyclic substituent on either neighbor (that
        would double-count its single electron across both vertices).

    As with generate_borane_charge_state, any other non-H/halogen
    exocyclic substituent (e.g. -OH) is split off and charged
    independently via _charge_capped_fragment, and the cage's own
    (delocalized) charge is placed on one arbitrarily chosen boron atom.
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


def _find_negative_moiety(
    spec: Specie,
) -> list[tuple[int, list[int], str]]:
    """
    Finds COO- and SO3- moieties based on ligand connectivity.

    COO-:
        C connected to two O atoms.
        Each O has only that C as its non-metal neighbor.

    SO3-:
        S connected to three O atoms.
        Each O has only that S as its non-metal neighbor.

    Returns
    -------
    list[tuple[int, list[int], str]]
        center_idx, oxygen_idxs, kind
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
    """Classify a candidate centre by element ``label``, its number of terminal
    (=O / -O(-)) oxygens ``k``, its total non-metal neighbour count
    ``n_nonmetal`` and its total non-metal oxygen count ``n_oxygen``. Returns
    ``(kind, net_charge)`` or ``(None, 0)`` if the centre is a net-neutral group.

    Net-neutral look-alikes are deliberately excluded: ``C`` + 2 terminal O
    with no third substituent is CO2 (O=C=O, neutral) rather than a carboxylate,
    and ``S`` + 2 O + 2 C is a sulfone.
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

    return None, 0


def _find_charged_moiety(
    spec: Specie,
) -> list[tuple[int, list[int], str, int]]:
    """Detect functional groups carrying a nonzero *net* formal charge purely
    from connectivity.

    Anionic groups are ``centre + k terminal O`` patterns, where a terminal O
    has the centre as its only non-metal neighbour: carboxylate/carbonate and
    sulfinate/sulfonate/sulfate. Net-neutral look-alikes (CO2, sulfone) are
    excluded -- see ``_classify_charged_moiety``.

    This is a pure substructure match: metal coordination is ignored, so a
    group is detected whether or not its centre or terminal oxygen binds a
    metal.

    Returns ``(centre_local_idx, atom_local_idxs, kind, net_charge)`` with
    indices into ``spec.atoms`` (the ordering used to build the RDKit mol),
    ``atom_local_idxs`` being the terminal oxygens (anionic) or the centre
    (cationic).
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

    return moieties
