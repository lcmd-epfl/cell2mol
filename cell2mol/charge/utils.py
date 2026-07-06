from cell2mol.my_types import RDKitObject
import numpy as np
import networkx as nx

# Monatomic noble gases
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

# Ligand or molecules requiring manual charge assignment
MANUAL_CHARGE_ASSIGN_SPECIES = {
    "O4-Cl",
    "N3",
    "N2",
    "N-O",
    "I3",
    "I4",
    "I5",
    "I6",
    "H",
    "H2",
    "N-O3",
    "Te2",
}

# Pentafluorooxotellurate(VI) ("teflate", -OTeF5) ligand
TEFLATE = {"O-F5-Te"}

HALOGENS = {"F", "Cl", "Br", "I"}


def is_sb_halide_only(labels) -> bool:
    """
    True for species made up of only Sb and halogens (SbX3/X4/X5/X6-type),
    regardless of halogen identity or count.
    """
    return "Sb" in labels and all(
        label in HALOGENS or label == "Sb" for label in labels
    )


# Plausible oxidation states by atomic symbol
# Source: Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
METAL_OXIDATION_STATES = {
    # Alkali metals
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Fr": [1],
    # Alkaline earth metals
    "Be": [2],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "Ra": [2],
    # 1st-row transition metals
    "Sc": [3],
    "Ti": [2, 3, 4],
    "V": [1, 2, 3, 4, 5],
    "Cr": [0, 2, 3],  # +5 intentionally excluded
    "Mn": [1, 2, 3],
    "Fe": [2, 3],
    "Co": [1, 2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2],
    "Zn": [2],
    # 2nd-row transition metals
    "Y": [3],
    "Zr": [2, 3, 4],
    "Nb": [1, 3, 4, 5],
    "Mo": [0, 2, 4, 5, 6],
    "Tc": [1, 2, 3, 4, 5],
    "Ru": [2, 3],
    "Rh": [1, 2, 3],
    "Pd": [0, 2],
    "Ag": [1],
    "Cd": [2],
    # 3rd-row transition metals
    "Hf": [4],
    "Ta": [2, 3, 4, 5],
    "W": [0, 2, 4, 5, 6],
    "Re": [1, 2, 3, 4, 5, 7],
    "Os": [2, 3, 4, 5, 6],
    "Ir": [1, 3],
    "Pt": [0, 2, 4],
    "Au": [1, 3],
    "Hg": [2],
    # Post-transition metals
    "Al": [1, 3],
    "Ga": [3],
    "Ge": [2, 4],
    "In": [3],
    "Sn": [2, 4],
    "Tl": [1, 3],
    "Pb": [2, 4],
    "Bi": [3],
    # Lanthanides
    "La": [3],
    "Ce": [3, 4],
    "Pr": [3],
    "Nd": [2, 3],
    "Pm": [3],
    "Sm": [2, 3],
    "Eu": [3],
    "Gd": [3],
    "Tb": [3],
    "Dy": [3],
    "Ho": [3],
    "Er": [3],
    "Tm": [3],
    "Yb": [2, 3],
    "Lu": [3],
    # Actinides
    "Ac": [3],
    "Th": [3, 4],
    "Pa": [3, 4, 5],
    "U": [3, 4, 5, 6],
    "Np": [3, 4, 5, 6],
    "Pu": [3, 4, 5, 6],
    "Am": [3, 4, 5, 6],
    "Cm": [3, 4],
    "Bk": [3],
    "Cf": [3],
    "Es": [3],
    "Fm": [2, 3],
    "Md": [2, 3],
    "No": [2, 3],
    "Lr": [3],
}


def aromatic_info(mol: RDKitObject, added_indices=None):
    if added_indices is None:
        added_indices = []

    # Count aromatic atoms
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    # Count aromatic rings
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    aromatic_ring_count = 0
    for bond_indices in bond_rings:
        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_indices):
            aromatic_ring_count += 1

    # Check if hydrogen atoms are added in to aromatic atoms
    check = any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in added_indices)

    return {
        "Aromatic atoms": aromatic_atoms,
        "Number of aromatic rings": aromatic_ring_count,
        "Added to aromatic atoms": check,
    }


def _valid_fullerene_vertex_count(n: int) -> bool:
    return n >= 20 and n % 2 == 0 and n != 22


def is_fullerene_cage(atoms, AC) -> tuple[bool, str]:
    """
    Detect any fullerene cage (C20, C60, C70, C76, C84, ...) purely from
    connectivity - no bond orders, no coordinates required.

    A graph is a fullerene skeleton iff it is:
      1. pure carbon
      2. 3-regular (every atom has exactly 3 neighbors)
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
    n = len(atoms)

    if ac.shape != (n, n):
        return False, "bad_ac_shape"

    # 1. Composition - classic crystallographic fullerene entries are bare
    #    carbon cages. (Endohedral/exohedral-functionalized fullerenes need
    #    a separate, more careful check - flagged, not handled here.)
    if not all(z == 6 for z in atoms):
        return False, "not_pure_carbon"

    # 2. Vertex count sanity filter (cheap, do before graph algorithms)
    if not _valid_fullerene_vertex_count(n):
        return False, "invalid_fullerene_vertex_count"

    # 3. Degree check - every carbon must be exactly 3-connected
    degrees = np.count_nonzero(ac, axis=1)
    if not np.all(degrees == 3):
        return False, "not_all_degree_3"

    graph = nx.from_numpy_array(ac)

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


def _find_porphyrin_rings_and_bridges(
    atoms, AC
) -> tuple[list[frozenset[int]], list[int]] | None:
    """
    Shared detection for porphyrin/porphine- and phthalocyanine-type N4
    macrocycles: four five-membered pyrrole-type rings (1 N + 4 C each)
    bridged pairwise by four meso atoms into one closed 16-membered
    macrocycle -- carbon meso bridges for a porphyrin, nitrogen (aza)
    meso bridges for a phthalocyanine. Substituents (aryl/alkyl groups,
    fused benzo rings on each pyrrole for phthalocyanine, H, a coordinated
    metal, etc.) are ignored -- only the core ring topology is checked,
    mirroring is_fullerene_cage's coordinate-free approach.

    Returns (rings, bridges) in cyclic traversal order -- rings[i] and
    rings[(i+1) % 4] are joined by the single meso atom bridges[i] -- or
    None if no such macrocycle exists.
    """
    atoms = [int(a) for a in atoms]
    # See the matching comment in is_fullerene_cage: ligand adjmats can be
    # dtype=object even though every value is a plain int, which networkx
    # rejects outright.
    ac = np.asarray(AC, dtype=int)
    n = len(atoms)

    if ac.shape != (n, n):
        return None

    n_nitrogen = sum(1 for z in atoms if z == 7)
    n_carbon = sum(1 for z in atoms if z == 6)
    if n_nitrogen < 4 or n_carbon < 20:
        return None

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
        return None

    # Each candidate ring must contribute a distinct nitrogen -- rings
    # sharing a nitrogen can't both be genuine, separate pyrrole units.
    rings_by_nitrogen: dict[int, frozenset[int]] = {}
    for ring in pyrrole_rings:
        nitrogen = next(i for i in ring if atoms[i] == 7)
        if nitrogen not in rings_by_nitrogen:
            rings_by_nitrogen[nitrogen] = ring
    distinct_rings = list(rings_by_nitrogen.values())

    if len(distinct_rings) < 4:
        return None

    # Try every combination of 4 candidate rings and check if they close
    # into one meso-bridged macrocycle (real porphyrins won't have more
    # than 4 genuine candidates, so this stays cheap in practice).
    import itertools

    for combo in itertools.combinations(distinct_rings, 4):
        result = _macrocycle_ring_order_and_bridges(combo, graph, atoms)
        if result is not None:
            ring_order, bridge_by_edge = result
            ordered_rings = [combo[r] for r in ring_order]
            ordered_bridges = [
                bridge_by_edge[frozenset((ring_order[i], ring_order[(i + 1) % 4]))]
                for i in range(4)
            ]
            return ordered_rings, ordered_bridges

    return None


def find_porphyrin_macrocycle_nitrogens(atoms, AC) -> list[int] | None:
    """
    Detect a porphyrin/porphine- or phthalocyanine-type N4 macrocycle
    purely from connectivity (see _find_porphyrin_rings_and_bridges).

    Returns the 4 macrocycle (pyrrole-type) nitrogen atom indices if
    found, else None.
    """
    result = _find_porphyrin_rings_and_bridges(atoms, AC)
    if result is None:
        return None
    ordered_rings, _bridges = result
    atoms = [int(a) for a in atoms]
    # Return the 4 nitrogens in cyclic macrocycle order (not sorted by
    # index): position i and i+2 are the "opposite" (meso-bridge-
    # separated-by-two-rings) pair, positions i and i+1 are "adjacent"
    # (single meso bridge apart) -- callers that need to distinguish the
    # two rely on this ordering.
    return [next(i for i in ring if atoms[i] == 7) for ring in ordered_rings]


def _macrocycle_ring_order_and_bridges(
    rings, graph: "nx.Graph[int]", atoms: list[int]
) -> tuple[list[int], dict[frozenset[int], int]] | None:
    """
    Check whether 4 given pyrrole-type rings are connected pairwise, each
    to exactly two others, via single-atom meso bridges -- i.e. they close
    into one macrocyclic loop rather than, say, two separate pairs or an
    open chain. The bridge atom is carbon for a porphyrin (meso-C) or
    nitrogen for a phthalocyanine (aza-meso-N); either is accepted here,
    so this one check covers both macrocycle families.

    Returns (order, bridge_by_edge) if so, else None: `order` is the
    cyclic traversal order (indices into `rings`); `bridge_by_edge` maps
    each frozenset({ring_idx_a, ring_idx_b}) to the meso atom index
    bridging that pair.
    """
    ring_atoms = [set(r) for r in rings]
    all_ring_atoms: set[int] = set().union(*ring_atoms)

    outside_candidates: set[int] = set()
    for atom_set in ring_atoms:
        for i in atom_set:
            for j in graph.neighbors(i):
                if j not in all_ring_atoms:
                    outside_candidates.add(j)

    bridge_graph: "nx.Graph[int]" = nx.Graph()
    bridge_graph.add_nodes_from(range(4))
    bridge_by_edge: dict[frozenset[int], int] = {}

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

    if bridge_graph.number_of_edges() != 4:
        return None
    if any(degree != 2 for _, degree in bridge_graph.degree()):
        return None
    if not nx.is_connected(bridge_graph):
        return None

    # Walk the 4-cycle to get the traversal order.
    order = [0]
    prev, current = None, 0
    for _ in range(3):
        nxt = next(n for n in bridge_graph.neighbors(current) if n != prev)
        order.append(nxt)
        prev, current = current, nxt
    return order, bridge_by_edge


def is_porphyrin_macrocycle(atoms, AC) -> tuple[bool, str]:
    """
    True if the connectivity contains a porphyrin/porphine- or
    phthalocyanine-type N4 macrocycle (four pyrrole rings bridged into one
    16-membered ring via either carbon meso bridges (porphyrin) or
    nitrogen aza-meso bridges (phthalocyanine)), regardless of
    substituents or fused benzo rings. See
    find_porphyrin_macrocycle_nitrogens for the detection logic.
    """
    macrocycle_nitrogens = find_porphyrin_macrocycle_nitrogens(atoms, AC)
    if macrocycle_nitrogens is None:
        return False, "no_porphyrin_macrocycle"
    return True, "porphyrin_macrocycle_confirmed"
