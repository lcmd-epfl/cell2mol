from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict
from collections.abc import Collection, Sequence
from cell2mol.my_types import NDArray
import numpy as np
import warnings
import logging
import networkx as nx
from cell2mol.utils import config
from cell2mol.element_utils import (
    get_radii,
    get_df_block_metal_idxs,
    get_post_transition_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    symbols2formula,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from cell2mol.operations import inv, extract_from_list, point_along_vector
from cell2mol.elementdata import ElementData

if TYPE_CHECKING:
    from cell2mol.classes.atom import Atom
    from cell2mol.classes.metal import Metal
    from cell2mol.classes.ligand import Ligand


class TopologyResult(TypedDict):
    is_connected: bool
    n_rings: int
    ring_sizes: list[int]
    ring_atoms: list[int]
    ring_sets: list[list[int]]
    all_atoms_in_rings: bool
    is_single_simple_ring: bool
    is_fused_5_5: bool
    is_fused_5_6: bool
    is_fused_5_7: bool


logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def scale_covalent_radii(
    radii, df_block_metal_idxs, s_block_metal_idxs, metal_factor, cov_factor
):
    """Scale radii based on whether atom is metal, alkali/alkaline earth metal, or non-metal"""
    scaled_radii = np.zeros_like(radii, dtype=float)
    for k in range(len(radii)):
        if k in df_block_metal_idxs or k in s_block_metal_idxs:
            scaled_radii[k] = radii[k] * metal_factor
        else:
            scaled_radii[k] = radii[k] * cov_factor
    return scaled_radii


def has_adjacency_mismatch(
    labels: list[str],
    positions: np.ndarray,
    atom_site_labels: list[str] | None = None,
    bond_data: list[tuple[str, str, float]] | None = None,
    cutoff: float | None = None,
    cov_factor: float | None = None,
    metal_factor: float | None = None,
    metal_only: bool = False,
) -> bool | None:
    """
    Checks for discrepancies between distance-based and CIF-bond-based adjacency matrices.

    Returns:
        True:  Mismatch detected (Inconsistent connectivity).
        False: Matrices are identical (Consistent connectivity).
        None:  CIF bond data is missing (Comparison not possible).
    """
    # Use global config if specific parameters are not provided
    cutoff = cutoff if cutoff is not None else config.CUTOFF
    cov_factor = cov_factor if cov_factor is not None else config.COV_FACTOR
    metal_factor = metal_factor if metal_factor is not None else config.METAL_FACTOR

    # 1. Build distance-based matrix (Geometric Baseline)
    _, adj_dist, _ = build_distance_adjacency(
        labels,
        positions,
        atom_site_labels=atom_site_labels,
        cutoff=cutoff,
        cov_factor=cov_factor,
        metal_factor=metal_factor,
        metal_only=metal_only,
    )

    # 2. Build CIF-based matrix (Metadata Reference)
    adj_conn = None
    if bond_data and atom_site_labels:
        adj_conn = build_cif_bond_adjacency(
            labels,
            positions,
            atom_site_labels=atom_site_labels,
            bond_data=bond_data,
            metal_only=metal_only,
        )

    # 3. Handle the 'None' case: Comparison cannot be performed
    if adj_conn is None:
        return None

    # 4. Handle Case: Structural Mismatch (Different atom counts or shapes)
    if adj_dist is None or adj_dist.shape != adj_conn.shape:
        return True

    # 5. Handle Numerical Mismatch: True if NOT identical
    return not np.allclose(adj_dist, adj_conn)


def build_adjacency(
    labels: list[str],
    positions: np.ndarray,
    atom_site_labels: list[str] | None = None,
    bond_data: list[tuple[str, str, float]] | None = None,
    use_bond_info: bool | None = None,
    cutoff: float | None = None,
    cov_factor: float | None = None,
    metal_factor: float | None = None,
    metal_only: bool = False,
    warn_on_mismatch: bool = False,
    detail: bool = False,
) -> np.ndarray | None:
    """
    Build an adjacency matrix using distance-based or connectivity-based criteria.

    Distance-based adjacency is always constructed. If CIF bond information is
    provided, it is used for validation or as the canonical adjacency depending
    on `canonical`.
    """
    if cutoff is None:
        cutoff = config.CUTOFF
    if cov_factor is None:
        cov_factor = config.COV_FACTOR
    if metal_factor is None:
        metal_factor = config.METAL_FACTOR
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    canonical = "bond_info" if use_bond_info else "distance"

    isgood = True
    warning = False
    adj_dist = None
    # --- distance-based adjacency (default) ---
    if canonical == "distance" or warn_on_mismatch:
        isgood, adj_dist, warning = build_distance_adjacency(
            labels,
            positions,
            atom_site_labels=atom_site_labels,
            cutoff=cutoff,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            metal_only=metal_only,
        )

    # --- connectivity-based adjacency (optional) ---
    adj_conn = None
    if bond_data is not None:
        if atom_site_labels is None:
            raise ValueError("atom_site_labels must be provided with bond_data")

        adj_conn = build_cif_bond_adjacency(
            labels,
            positions,
            atom_site_labels=atom_site_labels,
            bond_data=bond_data,
            metal_only=metal_only,
        )
    if warn_on_mismatch and adj_dist is not None and adj_conn is not None:
        logger.debug("Sum of Formula: %s", symbols2formula(labels))
        is_consistent = np.allclose(adj_dist, adj_conn)
        if not is_consistent:
            logger.info(
                "Discrepancy detected: The distance-based and CIF-bond-based adjacency matrices are not identical."
            )
        else:
            logger.info(
                "The distance-based and CIF-bond-based adjacency matrices are identical."
            )

    # --- choose canonical ---
    if canonical == "distance":
        adj = adj_dist
        if not isgood:
            logger.error("Clash detected in distance-based adjacency matrix")
            return None
        if warning:
            logger.warning("Valence violation detected in distance-based adjacency")
    elif canonical == "bond_info":
        if adj_conn is None:
            logger.error("canonical='bond_info' requires connectivity-based adjacency")
            return None
        adj = adj_conn
    else:
        logger.error(f"Unknown canonical mode: {canonical}")
        return None

    # --- validate distance vs bond_info ---
    if warn_on_mismatch and adj_conn is not None and adj_dist is not None:
        compare_adjacency_and_warn(
            adj_dist=adj_dist,
            adj_conn=adj_conn,
            labels=labels,
            pos=positions,
            atom_site_labels=atom_site_labels,
            warn=True,
            detail=detail,
        )

    return adj


def extract_bond_pairs(adj: np.ndarray) -> set[tuple[int, int]]:
    """
    Convert adjacency matrix to bond set.
    """
    bonds = set()
    N = adj.shape[0]

    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                bonds.add((i, j))

    return bonds


def compare_adjacency_and_warn(
    adj_dist: np.ndarray,
    adj_conn: np.ndarray | None,
    labels: list[str],
    pos: np.ndarray,
    *,
    atom_site_labels: list[str] | None = None,
    warn: bool = True,
    detail: bool = False,
    use_logger: bool = True,
) -> None:
    """
    Compare distance-based (canonical) and connectivity-based adjacency matrices
    and emit warnings if they differ.
    """
    if adj_conn is None:
        return

    bonds_dist = extract_bond_pairs(adj_dist)
    bonds_conn = extract_bond_pairs(adj_conn)

    extra = bonds_dist - bonds_conn
    missing = bonds_conn - bonds_dist

    if not extra and not missing:
        return

    # --- summary ---
    if warn:
        msg = (
            "Adjacency mismatch detected:"
            "extra_bonds=%d (distance), "
            "missing_bonds=%d (bond_info). "
        )
        if use_logger:
            logger.warning(msg, len(extra), len(missing))
        else:
            warnings.warn(msg % (len(extra), len(missing)), RuntimeWarning)

    # --- detailed info ---
    if detail and use_logger:
        if extra:
            logger.debug("Extra bonds (distance-only):")
            for line in format_bond_info(extra, labels, pos, atom_site_labels):
                logger.debug("extra bonds  %s", line)

        if missing:
            logger.debug("Missing bonds (connectivity-only):")
            for line in format_bond_info(missing, labels, pos, atom_site_labels):
                logger.debug("missing bonds  %s", line)


def format_bond_info(
    bonds: set[tuple[int, int]],
    labels: list[str],
    pos: np.ndarray,
    atom_site_labels: list[str] | None = None,
    *,
    cutoff: float | None = None,
    metal_factor: float | None = None,
    cov_factor: float | None = None,
) -> list[str]:
    """
    Format bond indices with element labels, distances, and distance margins.

    Example
    -------
    (4-7) [Zn-O] [Zn1-O3]: dist=2.043, thres=2.120, margin=-0.077
    """
    if cutoff is None:
        cutoff = config.CUTOFF
    if cov_factor is None:
        cov_factor = config.COV_FACTOR
    if metal_factor is None:
        metal_factor = config.METAL_FACTOR

    df_block_metal_idxs = get_df_block_metal_idxs(labels)
    s_block_metal_idxs = get_alkali_alkaline_earth_metal_idxs(labels)

    radii = scale_covalent_radii(
        np.asarray(get_radii(labels)),
        df_block_metal_idxs,
        s_block_metal_idxs,
        metal_factor,
        cov_factor,
    )

    formatted: list[str] = []
    for i, j in sorted(bonds):
        dist = np.linalg.norm(np.asarray(pos[i]) - np.asarray(pos[j]))
        thres = radii[i] + radii[j] + cutoff
        margin = dist - (radii[i] + radii[j])

        elem_pair = f"{labels[i]}-{labels[j]}"
        site_pair = (
            f"{atom_site_labels[i]}-{atom_site_labels[j]}"
            if atom_site_labels is not None
            else None
        )

        if site_pair:
            formatted.append(
                f"({i}-{j}) [{elem_pair}] [{site_pair}]: "
                f"dist={dist:.3f}, thres={thres:.3f}, margin={margin:.3f}"
            )
        else:
            formatted.append(
                f"({i}-{j}) [{elem_pair}]: "
                f"dist={dist:.3f}, thres={thres:.3f}, margin={margin:.3f}"
            )

    return formatted


def pair_contains_metal(
    i: int,
    j: int,
    labels: list[str],
    df_block_metal_idxs: Collection[int],
    s_block_metal_idxs: Collection[int],
) -> bool:
    """
    Return True if the atom pair includes at least one metal element.
    """
    block_i = elemdatabase.elementblock[labels[i]]
    block_j = elemdatabase.elementblock[labels[j]]

    # d- or f-block metals
    if block_i in ("d", "f") or block_j in ("d", "f"):
        return True

    # alkali / alkaline earth metals
    if i in s_block_metal_idxs or j in s_block_metal_idxs:
        return True

    # fallback: post-transition metals if no other metals present
    if not df_block_metal_idxs and not s_block_metal_idxs:
        if get_post_transition_metal_idxs([labels[i], labels[j]]):
            return True

    return False


SPECIAL_DONORS = ["N", "O", "S", "Cl", "Br", "I", "F"]

PAIR_CUTOFF_075_METALS = ["Cu", "Hg"]
PAIR_CUTOFF_065_METALS = ["Zn", "Ag", "Cd"]


def get_pair_cutoff(
    label_i: str,
    label_j: str,
    default_cutoff: float,
) -> float:
    """Return pair-specific distance cutoff added to the sum of atomic radii.

    Special rules:
    - Cu/Hg bonded to N/O/S/halogens: use 0.75 Å
    - Zn/Ag/Cd bonded to N/O/S/halogens: use 0.65 Å
    - otherwise: use default_cutoff, usually 0.45 Å
    """

    pair = {label_i, label_j}

    has_special_donor = any(donor in pair for donor in SPECIAL_DONORS)

    if has_special_donor:
        if any(metal in pair for metal in PAIR_CUTOFF_075_METALS):
            return 0.75

        if any(metal in pair for metal in PAIR_CUTOFF_065_METALS):
            return 0.65
    return default_cutoff


def build_distance_adjacency(
    labels: list[str],
    pos: np.ndarray,
    atom_site_labels: list[str] | None = None,
    radii: NDArray | list[float] | None = None,
    cutoff: float | None = None,
    cov_factor: float | None = None,
    metal_factor: float | None = None,
    metal_only: bool = False,
    add_atom: bool = False,
    diagnostics: dict | None = None,
):
    """Generates adjacency matrix from atomic positions and covalent radii.
    Args:
        labels (list): List of atomic labels.
        pos (list): List of atomic positions.
        atom_site_labels (list[str] | None, optional): List of atomic site labels. Defaults to None.
        cov_factor (float, optional): Scaling factor for covalent radii. Defaults to
            1.0.
        radii (str or np.ndarray, optional): Radii to use. If "default", uses default
            covalent radii. Defaults to "default".
        metal_factor (float, optional): Scaling factor for metal radii. Defaults to
            1.0.
        metal_only (bool, optional): If True, only considers metal-metal bonds.
            Defaults to False.
        add_atom (bool, optional): If True, force adding atoms regardless of valence.
            Defaults to False.
        diagnostics (dict, optional): Filled in place with the valence-correction
            "removals" and "protections" (see correct_valence_violations).
    Returns:
        isgood (int): 1 if molecule is valid, 0 otherwise.
        adjacency_matrix (list): Adjacency matrix.
        warning (bool): True if valence violations were found, False otherwise.
    """
    isgood = True
    clash_threshold = 0.3
    if cutoff is None:
        cutoff = config.CUTOFF
    if cov_factor is None:
        cov_factor = config.COV_FACTOR
    if metal_factor is None:
        metal_factor = config.METAL_FACTOR

    natoms = len(labels)
    adjacency_matrix = np.zeros((natoms, natoms), dtype=int)
    metal_contact_matrix = np.zeros((natoms, natoms), dtype=int)

    df_block_metal_idxs = get_df_block_metal_idxs(labels)
    s_block_metal_idxs = get_alkali_alkaline_earth_metal_idxs(labels)

    # --- radii normalization ---
    if radii is None:
        radii = get_radii(labels)
    radii = scale_covalent_radii(
        np.asarray(radii),
        df_block_metal_idxs,
        s_block_metal_idxs,
        metal_factor,
        cov_factor,
    )

    # --- adjacency construction ---
    for i in range(natoms - 1):
        a = np.asarray(pos[i])
        for j in range(i + 1, natoms):
            b = np.asarray(pos[j])
            dist = np.linalg.norm(a - b)

            pair_cutoff = get_pair_cutoff(labels[i], labels[j], default_cutoff=cutoff)
            thres = radii[i] + radii[j] + pair_cutoff

            if dist <= clash_threshold:
                isgood = False
                logger.error(
                    "Adjacency clash: dist=%.3f < %.3f for atoms (%s%s:%s)-(%s%s:%s)",
                    dist,
                    clash_threshold,
                    labels[i],
                    f"({atom_site_labels[i]})" if atom_site_labels else "",
                    pos[i],
                    labels[j],
                    f"({atom_site_labels[j]})" if atom_site_labels else "",
                    pos[j],
                )
                continue

            if dist > thres:
                continue

            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

            if metal_only and pair_contains_metal(
                i, j, labels, df_block_metal_idxs, s_block_metal_idxs
            ):
                metal_contact_matrix[i, j] = metal_contact_matrix[j, i] = 1

    if not isgood:
        return False, (metal_contact_matrix if metal_only else adjacency_matrix), False

    # --- valence correction ---
    warning = False
    if not add_atom:
        adjacency_matrix, metal_contact_matrix, warning = correct_valence_violations(
            adjacency_matrix,
            metal_contact_matrix,
            labels,
            pos,
            radii,
            atom_site_labels=atom_site_labels,
            diagnostics=diagnostics,
        )
    return True, (metal_contact_matrix if metal_only else adjacency_matrix), warning


# Elements whose valence check is skipped as a center: they are routinely
# hypervalent or bridging (cage B, halide bridges).
VALENCE_EXEMPT_ELEMENTS = {"F", "Cl", "Br", "I", "B"}
# Coordination numbers above the xyz2mol valence table that are ordinary here,
# such as PF6-, SiF6(2-), AsF6-. Local so the charge code's table is unchanged.
HYPERVALENT_MAX_COORDINATION = {"P": 6, "Si": 6, "Ge": 6, "As": 6, "Sb": 6}


def protect_boron_edges(
    labels: Sequence[str],
    nonmetal_adjacency: np.ndarray,
    *,
    protect_cages: bool = True,
) -> tuple[set[tuple[int, int]], set[int], list[dict]]:
    """Find nonmetal edges that valence counting must not remove.

    These are the edges of closo/nido boranes and carboranes, unclassified B/C
    cores with at least three B atoms ("possible" cages), H bridging two B atoms
    (B-H-B), and terminal H on a cage atom. Cage atoms are hypervalent by
    classical counting, so a valence limit would cut real cage edges.

    ``nonmetal_adjacency`` must have all metal rows and columns zeroed. Return
    (protected sorted edges, cage atom indices, protection records). With
    ``protect_cages=False`` only B-H-B bridges are protected.
    """
    protected: set[tuple[int, int]] = set()
    cage_atoms: set[int] = set()
    protections: list[dict] = []
    if "B" not in labels:
        return protected, cage_atoms, protections

    graph = nx.from_numpy_array(np.asarray(nonmetal_adjacency, dtype=int))
    if protect_cages:
        # Use the index-returning detectors rather than Species.has_borane:
        # cached species flags can describe a different (already corrected)
        # graph. Load classes first to respect the classes/charge import cycle.
        import cell2mol.classes  # noqa: F401
        from cell2mol.charge.special_cases.borane import (
            find_borane_cages,
            find_borane_cage_indices,
        )

        atomic_numbers = [elemdatabase.elementnr[label] for label in labels]
        confirmed_edges: set[tuple[int, int]] = set()
        for indices, kind in find_borane_cages(atomic_numbers, nonmetal_adjacency):
            cage_atoms.update(indices)
            edges = {tuple(sorted(e)) for e in graph.subgraph(indices).edges()}
            protected.update(edges)
            confirmed_edges.update(edges)
            protections.append(
                {
                    "atoms": indices,
                    "reason": "borane_cage",
                    "cage_type": kind,
                    "detector": "find_borane_cages",
                }
            )
        core = find_borane_cage_indices(atomic_numbers, nonmetal_adjacency) or []
        bc = graph.subgraph(core)
        for block in nx.biconnected_components(bc):
            if len(block) >= 4 and sum(labels[i] == "B" for i in block) >= 3:
                edges = {tuple(sorted(e)) for e in bc.subgraph(block).edges()}
                if edges <= confirmed_edges:
                    continue
                cage_atoms.update(block)
                protected.update(edges)
                protections.append(
                    {
                        "atoms": sorted(block),
                        "reason": "possible_boron_cage",
                        "detector": "find_borane_cage_indices",
                        "cage_type": None,
                    }
                )
    for i, label in enumerate(labels):
        if label != "H":
            continue
        neighbors = list(graph.neighbors(i))
        if (len(neighbors) == 2 and all(labels[j] == "B" for j in neighbors)) or (
            len(neighbors) == 1 and neighbors[0] in cage_atoms
        ):
            protected.update(tuple(sorted((i, j))) for j in neighbors)
            protections.append(
                {"atoms": [i, *sorted(neighbors)], "reason": "boron_hydrogen"}
            )
    return protected, cage_atoms, protections


def correct_valence_violations(
    adjacency_matrix,
    metal_contact_matrix,
    labels,
    pos,
    radii,
    atom_site_labels=None,
    diagnostics: dict | None = None,
):
    """Detect and correct valence violations in an adjacency matrix.

    Edges found by protect_boron_edges (borane/carborane cages, B-H-B bridges,
    terminal cage H) are never removed and do not count toward an atom's
    valence, so cage atoms and bridging H are not flagged. If ``diagnostics``
    is given, it receives "removals" (atoms, labels, distance, margin, reason),
    "protections", and "violations" (atom, initial/final degree, limit, status:
    resolved, unresolved or exempt) for every atom that started over its limit.
    Six-coordinate P, Si, Ge, As and Sb are allowed, and metal atoms are not
    checked at all (their coordination is validated separately). Only unresolved violations
    set the returned warning flag and are logged.
    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix.
        labels (list): List of atomic labels.
        pos (list): List of atomic positions.
        radii (list): List of atomic radii.
        atom_site_labels (list, optional): List of atomic site labels. Defaults to None.
        diagnostics (dict, optional): Filled in place; see above.
    Returns:
        adjacency_matrix (np.ndarray): Corrected adjacency matrix.
        metal_contact_matrix (np.ndarray): Corrected metal adjacency matrix.
        warning (bool): True if a valence violation remains after correction
            (halogens and B, which are exempt, do not count), False otherwise.
    """
    from cell2mol.charge.xyz2mol import atomic_valence

    natoms = len(labels)
    warning = False

    df_block_metal_idxs = set(get_df_block_metal_idxs(labels))
    post_transition_metal_idxs = set(get_post_transition_metal_idxs(labels))
    s_block_metal_idxs = set(get_alkali_alkaline_earth_metal_idxs(labels))
    metal_indices = (
        df_block_metal_idxs | post_transition_metal_idxs | s_block_metal_idxs
    )

    nonmetal_ac = np.array(adjacency_matrix, dtype=int)
    for m in metal_indices:
        nonmetal_ac[m, :] = 0
        nonmetal_ac[:, m] = 0
    protected, _, protections = protect_boron_edges(labels, nonmetal_ac)
    if diagnostics is not None:
        diagnostics.setdefault("removals", [])
        diagnostics.setdefault("protections", []).extend(protections)

    def nonmetal_degree(i: int) -> int:
        """Nonmetal neighbors that count toward valence (protected bonds do not)."""
        return sum(
            int(j) not in metal_indices and tuple(sorted((i, int(j)))) not in protected
            for j in np.flatnonzero(adjacency_matrix[i])
        )

    def valence_limit(i: int) -> int:
        if i in metal_indices:
            # Metal coordination is validated separately; the xyz2mol table
            # gives post-transition metals (Al, Sn, ...) organic-style limits.
            return natoms
        limit = max(atomic_valence[elemdatabase.elementnr[labels[i]]], default=0)
        return max(limit, HYPERVALENT_MAX_COORDINATION.get(labels[i], 0))

    limits = [valence_limit(i) for i in range(natoms)]
    initial_degrees = [nonmetal_degree(i) for i in range(natoms)]

    for i in range(natoms):
        excess = initial_degrees[i] - limits[i]
        if excess <= 0 or labels[i] in VALENCE_EXEMPT_ELEMENTS:
            continue
        # The current degree can already be lower: a bond removed from a
        # neighbor also lowers this atom's count.
        excess = nonmetal_degree(i) - limits[i]
        if excess <= 0:
            continue

        # Only bonds to non-metals are candidates: the valence that is violated
        # counts non-metal neighbors only, and metal coordination is validated
        # separately during coordination analysis. Metal bonds are usually the
        # most elongated, so leaving them in would cut them first.
        connections = []
        a = np.asarray(pos[i])
        for j in np.flatnonzero(adjacency_matrix[i]):
            if j in metal_indices or tuple(sorted((i, int(j)))) in protected:
                continue
            b = np.asarray(pos[j])
            dist = np.linalg.norm(a - b)
            margin = dist - (radii[i] + radii[j])
            connections.append((j, margin))

            logger.debug(
                "Bond %d-%d %s%s:%s-%s%s:%s: dist=%.3f, margin=%.3f",
                i,
                j,
                labels[i],
                f"({atom_site_labels[i]})" if atom_site_labels else "",
                pos[i],
                labels[j],
                f"({atom_site_labels[j]})" if atom_site_labels else "",
                pos[j],
                dist,
                margin,
            )

        connections.sort(key=lambda x: x[1], reverse=True)

        for j, margin in connections[:excess]:
            if margin > 0.2:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 0
                distance = float(
                    np.linalg.norm(np.asarray(pos[i]) - np.asarray(pos[j]))
                )
                if diagnostics is not None:
                    diagnostics["removals"].append(
                        {
                            "atoms": [i, int(j)],
                            "labels": [labels[i], labels[j]],
                            "stage": "valence",
                            "reason": "excess_nonmetal_valence",
                            "distance": distance,
                            "margin": float(margin),
                        }
                    )
                logger.info(
                    "Removed bond %d-%d %s%s:%s-%s%s:%s, dist=%.3f, margin=%.3f",
                    i,
                    j,
                    labels[i],
                    f"({atom_site_labels[i]})" if atom_site_labels else "",
                    pos[i],
                    labels[j],
                    f"({atom_site_labels[j]})" if atom_site_labels else "",
                    pos[j],
                    distance,
                    margin,
                )

    # Report every atom that started over its limit, once the pass is finished.
    violations = []
    for i in range(natoms):
        if initial_degrees[i] <= limits[i]:
            continue
        final = nonmetal_degree(i)
        if final <= limits[i]:
            status = "resolved"
        elif labels[i] in VALENCE_EXEMPT_ELEMENTS:
            status = "exempt"
        else:
            status = "unresolved"
            warning = True
            logger.warning(
                "Unresolved valence violation: %s%s %s(index:%d), "
                "nonmetal neighbors=%d > max_valence=%d",
                labels[i],
                f"({atom_site_labels[i]})" if atom_site_labels else "",
                pos[i],
                i,
                final,
                limits[i],
            )
        violations.append(
            {
                "atom": i,
                "label": labels[i],
                "initial_degree": initial_degrees[i],
                "final_degree": final,
                "limit": limits[i],
                "status": status,
            }
        )
    if diagnostics is not None:
        diagnostics["violations"] = violations

    return adjacency_matrix, metal_contact_matrix, warning


def build_cif_bond_adjacency(
    labels: list[str],
    pos: np.ndarray,
    atom_site_labels: list[str],
    bond_data: list[tuple[str, str, float]],
    metal_only: bool = False,
    tol: float = 1e-3,
):
    """
    Build a connectivity-based adjacency matrix from CIF bond information.

    Bond distances from the CIF are validated against Cartesian coordinates
    within a numerical tolerance. If `metal_only` is True, only metal-involving
    bonds are included.

    Returns:
        adjacency_matrix (np.ndarray): Adjacency matrix.
    """
    indices = {atom: i for i, atom in enumerate(atom_site_labels)}
    natoms = len(labels)

    adjacency_matrix = np.zeros((natoms, natoms), dtype=int)

    df_block_metal_idxs = set(get_df_block_metal_idxs(labels))
    s_block_metal_idxs = set(get_alkali_alkaline_earth_metal_idxs(labels))

    for atom1, atom2, bond_dist in bond_data:
        if atom1 not in indices or atom2 not in indices:
            continue

        i, j = indices[atom1], indices[atom2]

        a = np.asarray(pos[i])
        b = np.asarray(pos[j])
        dist = np.linalg.norm(a - b)

        if abs(dist - bond_dist) > tol:
            continue

        if metal_only and not pair_contains_metal(
            i, j, labels, df_block_metal_idxs, s_block_metal_idxs
        ):
            continue

        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

    return adjacency_matrix


def count_bonds_by_element_pair(label: list[str], conmat: np.ndarray) -> np.ndarray:
    elems = elemdatabase.elementnr.keys()
    natoms = len(label)
    bondtypes = np.zeros((len(elems), len(elems)), dtype=int)
    found = np.zeros((natoms, natoms))

    for i in range(0, natoms):
        for j in range(i, natoms):
            if i != j:
                if (conmat[i, j] == 1) and (found[i, j] == 0):
                    for p, elem1 in enumerate(elems):
                        if label[i] == elem1:
                            for q, elem2 in enumerate(elems):
                                if label[j] == elem2:
                                    bondtypes[p, q] += 1
                                    if elem1 != elem2:
                                        bondtypes[q, p] += 1
                                    found[i, j] = 1
                                    found[j, i] = 1
                                    break
                            break
    return bondtypes


def find_block_boundaries(matrix: np.ndarray):
    """Function that detects blocks in a block diagonal matrix."""
    # retrieves the blocks from a diagonal block matrix
    startlist = []  # List including the starting atom for all blocks
    endlist = []  # List including the final atom for all blocks
    start = 1
    pos = start
    posold = 0
    blockcount = 0
    j = 1
    while j < len(matrix):
        if matrix[pos - 1, j] != 0.0:
            pos = j + 1
        if j == len(matrix) - 1:
            blockcount = blockcount + 1
            startlist.append(posold)
            endlist.append(pos - 1)
            posold = pos
            pos = pos + 1
            j = pos - 1
            continue
        j += 1

    # if a 1x1 matrix is provided, it then finds 1 block
    if (blockcount == 0) and (len(matrix) == 1):
        startlist.append(0)
        endlist.append(0)
    return startlist, endlist


def split_species(
    labels: list[str],
    positions: np.ndarray,
    *,
    radii: list[float] | None = None,
    indices: list[int] | None = None,
    atom_site_labels: list[str] | None = None,
    bond_data: list[tuple[str, str, float]] | None = None,
    use_bond_info: bool | None = None,
    cov_factor: float | None = None,
    metal_factor: float | None = None,
    warn_on_mismatch: bool = False,
    detail: bool = False,
    count_species_only: bool = False,
    apply_graph: bool = False,
):
    """Function that identifies connected groups of atoms from their atomic coordinates and labels."""

    if radii is None:
        radii = get_radii(labels)
    if indices is None:
        indices = [*range(0, len(labels), 1)]
    if cov_factor is None:
        cov_factor = config.COV_FACTOR
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    adjacency_matrix = build_adjacency(
        labels=labels,
        positions=positions,
        atom_site_labels=atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
        cov_factor=cov_factor,
        metal_factor=metal_factor,
        warn_on_mismatch=warn_on_mismatch,
        detail=detail,
    )
    if adjacency_matrix is None:
        logger.warning("Adjacency matrix is None. Returning empty blocklist.")
        return []
    neighbor_counts = adjacency_matrix.sum(axis=1)

    degree = np.diag(neighbor_counts)
    lap = adjacency_matrix - degree

    # creates block matrix
    graph = csr_matrix(lap)
    perm = reverse_cuthill_mckee(graph)
    gp1 = graph[perm, :]
    gp2 = gp1[:, perm]
    dense = gp2.toarray()

    # detects blocks in the block diagonal matrix called "dense"
    startlist, endlist = find_block_boundaries(dense)

    nblocks = len(startlist)

    if count_species_only:
        return nblocks

    # keeps track of the atom movement within the matrix. Needed later
    atomlist = np.zeros((len(dense)))
    for b in range(0, nblocks):
        for i in range(0, len(dense)):
            if (i >= startlist[b]) and (i <= endlist[b]):
                atomlist[i] = b + 1
    invperm = inv(perm)
    atomlistperm = [int(atomlist[i]) for i in invperm]

    # assigns atoms to molecules
    blocklist = []
    for b in range(0, nblocks):
        atlist = []  # atom indices in the original ordering
        for i in range(0, len(atomlistperm)):
            if atomlistperm[i] == b + 1:
                atlist.append(indices[i])
        blocklist.append(atlist)
    if use_bond_info:
        return blocklist

    if apply_graph:
        new_blocklist = split_blocks_by_rings(
            blocklist,
            labels,
            positions,
            conn_atom_site_labels=atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
        )
        log_block_changes(blocklist, new_blocklist)
        return new_blocklist
    return blocklist


def split_blocks_by_rings(
    blocklist,
    conn_labels,
    conn_coord,
    *,
    conn_atom_site_labels,
    bond_data,
    use_bond_info: bool | None = None,
    cov_factor: float | None = None,
    metal_factor: float | None = None,
):
    """Split a list of atoms into blocks of connected atoms.

    If a connected block contains one or more cycles, each cycle is saved as
    a separate block. Non-cycle atoms are split into connected components after
    removing all cycle atoms.
    """

    new_blocklist = []
    logger.debug("Applying graph analysis to blocklist: %s", blocklist)
    for b in blocklist:
        logger.debug("block=%s", b)

        gr_labels = extract_from_list(b, conn_labels, dimension=1)
        gr_coord = extract_from_list(b, conn_coord, dimension=1)
        if conn_atom_site_labels is not None:
            gr_atom_site_labels = extract_from_list(
                b, conn_atom_site_labels, dimension=1
            )
        else:
            gr_atom_site_labels = None

        adjacency_matrix = build_adjacency(
            labels=gr_labels,
            positions=np.asarray(gr_coord),
            atom_site_labels=gr_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            warn_on_mismatch=False,
            detail=False,
        )

        G = nx.from_numpy_array(np.array(adjacency_matrix))

        if not nx.is_connected(G):
            new_blocklist.append(b)
            continue

        cycle_basis = nx.cycle_basis(G)

        if len(cycle_basis) == 0:
            new_blocklist.append(b)
            continue

        logger.debug(
            "Found %d cycle(s) in block %s: %s", len(cycle_basis), b, cycle_basis
        )

        cycle_nodes = set()

        for i, cycle in enumerate(cycle_basis):
            cycle_nodes.update(cycle)

            cycle_block = sorted([b[idx] for idx in cycle])
            new_blocklist.append(cycle_block)

            logger.debug("Cycle %d block indices=%s", i, cycle_block)

        remaining_nodes = [n for n in G.nodes if n not in cycle_nodes]

        logger.debug(
            "Remaining non-cycle node indices in local graph=%s",
            remaining_nodes,
        )

        if not remaining_nodes:
            continue

        G_rem = G.subgraph(remaining_nodes)

        for comp in nx.connected_components(G_rem):
            remaining_block = sorted([b[idx] for idx in comp])
            new_blocklist.append(remaining_block)

            logger.debug(
                "Remaining connected block=%s",
                remaining_block,
            )

    logger.debug("Final new_blocklist=%s", new_blocklist)

    return new_blocklist


def log_block_changes(blocklist, new_blocklist):
    old = {tuple(sorted(b)) for b in blocklist}
    new = {tuple(sorted(b)) for b in new_blocklist}

    if old == new:
        return

    logger.debug("Blocklist differences detected.")

    removed = old - new
    added = new - old

    if removed:
        logger.debug("Removed / replaced blocks:")
        for b in removed:
            logger.debug("  %s", list(b))

    if added:
        logger.debug("Added / new blocks:")
        for b in added:
            logger.debug("  %s", list(b))


def _atom_site_labels(atoms: list[Atom]) -> list[str] | None:
    """Return CIF site labels when present; require every atom to have one."""
    if not atoms or not atoms[0].atom_site_label:
        return None
    labels: list[str] = []
    for atom in atoms:
        if atom.atom_site_label is None:
            raise ValueError("Incomplete CIF atom site labels")
        labels.append(atom.atom_site_label)
    return labels


def identify_haptic_mode(atoms: list[Atom], use_bond_info: bool | None = None):
    """Classify a group of atoms coordinating the same metal.

    Apply C2-C8 composition rules, using ring names such as Cp and COT, and
    the C8/C9/C10 fused ring systems. Otherwise, label a single simple ring
    etaN(formula)-ring; other open chains are not haptic.

    ``use_bond_info`` controls CIF bond use and defaults to config.USE_BOND_INFO.
    Return (is_haptic, haptic_type, topology); haptic_type is None if unmatched.
    """
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    results = analyze_topology(atoms, use_bond_info=use_bond_info)
    is_haptic, haptic_type = classify_haptic_group(
        [atom.label for atom in atoms], results
    )
    return is_haptic, haptic_type, results


def classify_haptic_group(
    labels: list[str], results: TopologyResult
) -> tuple[bool, str | None]:
    """Apply the identify_haptic_mode rules to a precomputed group topology.

    Callers that already hold a corrected graph pass graph_topology(subgraph)
    instead of rebuilding adjacency from the group's coordinates.
    """
    group_formula = symbols2formula(labels)
    is_haptic = False
    haptic_type = None

    totnum = len(labels)

    # Group formula -> known haptic type.
    HAPTIC_RULES = {
        **{f"C{n}": f"eta{n}(C{n})" for n in range(2, 9)},
    }
    # Group formula -> (fused-ring flag, haptic type); C9/C10 qualify only here.
    FUSED_RING_RULES = {
        "C8": ("is_fused_5_5", "eta5,eta5(pentalene)"),
        "C9": ("is_fused_5_6", "eta5,eta6(indene)"),
        "C10": ("is_fused_5_7", "eta5,eta7(azulene)"),
    }
    fused_rule = FUSED_RING_RULES.get(group_formula)

    # Use fused ring systems and known haptic types first, then check rings and
    # open chains separately.
    if fused_rule and results["n_rings"] == 2 and results[fused_rule[0]]:
        haptic_type = fused_rule[1]
        is_haptic = True
    elif group_formula in HAPTIC_RULES:
        haptic_type = HAPTIC_RULES[group_formula]
        is_haptic = True
        if results["is_single_simple_ring"]:
            if group_formula == "C5":
                haptic_type = "eta5(Cp)"  # cyclopentadienyl anion
            elif group_formula == "C6":
                haptic_type = "eta6(benzene)"  # benzene
            elif group_formula == "C7":
                haptic_type = "eta7(CHT)"  # cycloheptatrienyl
            elif group_formula == "C8":
                haptic_type = "eta8(COT)"  # cyclooctatetraenyl dianion
    elif results["is_single_simple_ring"]:
        haptic_type = f"eta{totnum}({group_formula})-ring"
        is_haptic = True
    # Open chains with heteroatoms or H (agostic C-H, B-H-M, P-N chelates) are
    # not haptic: they stay subject to contact pruning. Conjugated C/N chains
    # are recognized at charge assignment by _is_conjugated_pi_chain instead.
    logger.debug(
        "Identified haptic mode: is_haptic=%s, haptic_type=%s", is_haptic, haptic_type
    )

    return is_haptic, haptic_type


def is_single_ring(atoms: list[Atom], use_bond_info: bool | None = None) -> bool:
    """Check if a given list of atoms is a ring
    Args:
        atoms: list of Atom objects
        use_bond_info: whether to use bond information from CIF data
    Returns:
        bool: True if the atoms form a single ring, False otherwise
    """
    labels = [atom.label for atom in atoms]
    positions = [atom.coord for atom in atoms]
    atom_site_labels = _atom_site_labels(atoms)
    refcell = atoms[0].get_parent("reference")
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    adjacency_matrix = build_adjacency(
        labels=labels,
        positions=np.asarray(positions),
        atom_site_labels=atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
    )
    if adjacency_matrix is None:
        return False
    G = nx.from_numpy_array(np.array(adjacency_matrix))

    # Check if the graph is connected
    if not nx.is_connected(G):
        return False  # If not connected, can't form a single ring

    # Check for cycles and ensure the graph forms a simple cycle
    cycle_basis = nx.cycle_basis(G)

    # Check if there's exactly one cycle that includes all nodes (simple ring)
    if len(cycle_basis) == 1 and len(cycle_basis[0]) == len(G.nodes):
        return True

    return False  # Otherwise, not a ring compound


def analyze_topology(
    atoms: list[Atom], use_bond_info: bool | None = None
) -> TopologyResult:
    """
    Analyze graph topology for a group of atoms.

    Detects:
    - graph connectivity
    - number of rings
    - ring sizes
    - whether all atoms belong to rings
    - simple single ring
    - fused ring systems (5–5, 5–6, 5–7)
    """

    if not atoms:
        return _empty_topology()

    labels = [atom.label for atom in atoms]
    positions = [atom.coord for atom in atoms]
    atom_site_labels = _atom_site_labels(atoms)

    refcell = atoms[0].get_parent("reference")
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    adjacency_matrix = build_adjacency(
        labels=labels,
        positions=np.asarray(positions),
        atom_site_labels=atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
    )

    if adjacency_matrix is None:
        return _empty_topology()

    return graph_topology(nx.from_numpy_array(np.array(adjacency_matrix)))


def _empty_topology(is_connected: bool = False) -> TopologyResult:
    return {
        "is_connected": is_connected,
        "n_rings": 0,
        "ring_sizes": [],
        "ring_atoms": [],
        "ring_sets": [],
        "all_atoms_in_rings": False,
        "is_single_simple_ring": False,
        "is_fused_5_5": False,
        "is_fused_5_6": False,
        "is_fused_5_7": False,
    }


def graph_topology(G: nx.Graph) -> TopologyResult:
    """Analyze an existing graph; ring atoms are reported as its node labels."""
    if G.number_of_nodes() == 0 or not nx.is_connected(G):
        return _empty_topology()

    # --- Ring detection ---
    cycle_basis = nx.cycle_basis(G)
    ring_sets = [set(cycle) for cycle in cycle_basis]
    n_rings = len(ring_sets)
    ring_atoms = set().union(*ring_sets) if ring_sets else set()
    ring_atoms = list(ring_atoms)
    all_atoms_in_rings = len(ring_atoms) == G.number_of_nodes()
    is_single_simple_ring = (n_rings == 1) and all_atoms_in_rings

    # --- Fused ring detection ---
    fused_flags = {
        "is_fused_5_5": False,
        "is_fused_5_6": False,
        "is_fused_5_7": False,
    }

    if n_rings == 2:
        r1, r2 = ring_sets
        shared = r1 & r2
        sizes = (min(len(r1), len(r2)), max(len(r1), len(r2)))
        total_unique = len(r1 | r2)

        fused_rules = {
            (5, 5): ("is_fused_5_5", 8),
            (5, 6): ("is_fused_5_6", 9),
            (5, 7): ("is_fused_5_7", 10),
        }

        if sizes in fused_rules and len(shared) == 2:
            flag_name, expected_total = fused_rules[sizes]
            if total_unique == expected_total:
                fused_flags[flag_name] = True

    ring_lists = [list(r) for r in ring_sets]
    return {
        "is_connected": True,
        "n_rings": n_rings,
        "ring_sizes": [len(r) for r in ring_lists],
        "ring_atoms": ring_atoms,
        "ring_sets": ring_lists,
        "all_atoms_in_rings": all_atoms_in_rings,
        "is_single_simple_ring": is_single_simple_ring,
        "is_fused_5_5": fused_flags["is_fused_5_5"],
        "is_fused_5_6": fused_flags["is_fused_5_6"],
        "is_fused_5_7": fused_flags["is_fused_5_7"],
    }


def merge_coordination_groups(parent_molecule, groups, parent_ligand):
    from cell2mol.classes.group import Group

    """
    Merge multiple Group objects.
    Returns a LIST of Group objects (usually 1, but can be 2 if splitting occurs).
    """
    all_atoms = []
    seen_atom_ids = set()
    all_metals = []
    haptic_denticities = []

    # --- 1. Aggregate Atoms & Metals ---
    for grp in groups:
        if grp.is_haptic:
            haptic_denticities.append(grp.haptic_type.split("(", 1)[0])

        for atom in grp.atoms:
            idx = atom.get_parent_index("molecule")
            if idx not in seen_atom_ids:
                seen_atom_ids.add(idx)
                all_atoms.append(atom)

        for m in grp.metals:
            if m not in all_metals:
                all_metals.append(m)

    # --- 2. Topology Analysis ---
    results = analyze_topology(all_atoms)
    if not results["is_connected"]:
        return []

    # --- Helper: Filter metals connected to specific atoms ---
    def get_bonded_metals(atom_subset, metal_candidates):
        """
        Return a list of metals from candidates that are bonded
        to at least one atom in atom_subset based on adjacency matrix.
        """
        bonded = []
        atom_indices = [a.get_parent_index("molecule") for a in atom_subset]
        adj = parent_molecule.adjacency_matrix

        for m in metal_candidates:
            m_idx = m.get_parent_index("molecule")
            is_connected = False
            # Check bond between metal and ANY atom in the subset
            for a_idx in atom_indices:
                if adj[m_idx, a_idx]:  # Check adjacency (works for dense or sparse)
                    is_connected = True
                    break

            if is_connected:
                bonded.append(m)
        return bonded

    # --- Helper: Create and link a group ---
    def create_linked_group(atom_subset, metals_subset, forced_haptic_type=None):
        """Creates a Group object, sets parents, and links specific metals."""
        new_grp = Group.from_atom_list(atom_subset)
        new_grp.origin = "_merge_groups"

        # Hapticity
        if forced_haptic_type:
            new_grp.haptic_type = forced_haptic_type
            new_grp.is_haptic = True
            new_grp.topology = dict(results)
        else:
            new_grp.update_hapticity()

        # Map Indices
        grp_mol_indices = [a.get_parent_index("molecule") for a in atom_subset]
        lig_mol_indices = [
            atom.get_parent_index("molecule") for atom in parent_ligand.atoms
        ]
        grp_lig_indices = [lig_mol_indices.index(m_idx) for m_idx in grp_mol_indices]

        # Establish Parents
        new_grp.add_parent(parent_molecule, indices=grp_mol_indices)
        new_grp.inherit_adjacency("molecule")
        new_grp.add_parent(parent_ligand, indices=grp_lig_indices)

        # Link Specific Metals
        if new_grp.metals is None:
            new_grp.metals = []
        new_grp.metals.extend(metals_subset)

        return new_grp

    # --- 3. Decision: Split or Keep Merged? ---
    ring_sets = results.get("ring_sets", [])

    # Find a specific ring that fits the criteria (Size 5 to 8)
    # We look for the first ring in the set that meets the condition.
    primary_ring_indices = next((r for r in ring_sets if 5 <= len(r) <= 8), None)

    # CONDITION: Not all atoms in rings AND we found a valid primary ring
    if (not results.get("all_atoms_in_rings")) and (primary_ring_indices is not None):
        target_indices = set(primary_ring_indices)  # Convert to set for O(1) lookup
        ring_size = len(target_indices)

        logger.debug(
            "Splitting: Isolated Primary Ring (size %d) from merged system", ring_size
        )

        # A. Ring Group (Only atoms in the identified 5-8 membered ring)
        # Note: In your example, this keeps [0, 1, 2, 3, 4]
        ring_atoms = [atom for i, atom in enumerate(all_atoms) if i in target_indices]
        ring_metals = get_bonded_metals(ring_atoms, all_metals)
        group_ring = create_linked_group(
            ring_atoms, ring_metals, forced_haptic_type=None
        )

        # B. Rest Group (Tails AND atoms from other smaller fused rings)
        # Note: In your example, atom [6] (from the 3-ring) moves here.
        non_ring_atoms = [
            atom for i, atom in enumerate(all_atoms) if i not in target_indices
        ]
        non_ring_metals = get_bonded_metals(non_ring_atoms, all_metals)
        group_rest = create_linked_group(
            non_ring_atoms, non_ring_metals, forced_haptic_type=None
        )

        return [group_ring, group_rest]

    # --- 4. Standard Merged Path (One Group) ---
    else:
        labels = [a.label for a in all_atoms]
        formula = symbols2formula(labels)

        # Determine Name
        ring_map = {
            ("single", "C5"): "Cp",
            ("single", "C6"): "benzene",
            ("single", "C7"): "CHT",
            ("single", "C8"): "COT",
            ("fused_5_5", "C8"): "pentalene",
            ("fused_5_6", "C9"): "indene",
            ("fused_5_7", "C10"): "azulene",
        }

        key = None
        if results.get("all_atoms_in_rings"):
            if results.get("is_single_simple_ring"):
                key = ("single", formula)
            elif results.get("n_rings") == 2:
                if results.get("is_fused_5_5"):
                    key = ("fused_5_5", formula)
                elif results.get("is_fused_5_6"):
                    key = ("fused_5_6", formula)
                elif results.get("is_fused_5_7"):
                    key = ("fused_5_7", formula)

        group_name = ring_map.get(key, formula) if key is not None else formula
        denticity_str = ",".join(haptic_denticities)
        new_haptic_type = f"{denticity_str}({group_name})" if denticity_str else None

        # Create single unified group with ALL metals
        group_unified = create_linked_group(
            all_atoms, all_metals, forced_haptic_type=new_haptic_type
        )

        logger.info(
            "Merged group created: %s %s with metals %s",
            group_unified.formula,
            group_unified.haptic_type,
            [m.label for m in (group_unified.metals or [])],
        )
        return [group_unified]


def try_add_ligand_atom(
    labels: list[str],
    coords: Sequence[Sequence[float]] | NDArray,
    site: int,
    ligand: Ligand,
    element: str = "H",
    metal: Metal | None = None,
    removed_idx: list[int] | None = None,
    unconditional: bool = False,
) -> tuple[bool, list[str], list[Sequence[float]]]:
    """
    Add one atom of type `element` to a given ligand atom site.

    The atom is placed along the vector pointing toward a given metal atom.
    """
    isadded = False
    posadded = len(labels)

    newlab = list(labels)
    newcoord = list(coords)

    newlab.append(element)  # one atom will be added

    for idx, atom in enumerate(ligand.atoms or []):
        if idx != site:
            continue

        apos = np.array(atom.coord, copy=True)
        if metal is not None:
            tgt = metal

        else:
            tgt = atom.find_closest_metal()

        if tgt is None:
            raise ValueError("A metal is required to place the added ligand atom")
        if atom.radii is None:
            raise ValueError(f"Missing covalent radius for {atom.label}")
        idealdist = atom.radii + elemdatabase.CovalentRadius3[element]
        added_coords = point_along_vector(apos, tgt.coord, idealdist)

        newcoord.append([added_coords[0], added_coords[1], added_coords[2]])

        isgood, tmpconmat, _ = build_distance_adjacency(
            newlab,
            np.asarray(newcoord),
            cov_factor=ligand.cov_factor,
            add_atom=True,
        )
        tmpconnec = tmpconmat.sum(axis=1)
        logger.debug("tmpconnec at added position=%d", int(tmpconnec[posadded]))

        # Case 1: unconditional addition
        if unconditional:
            isadded = True
            logger.debug(
                "%s added unconditionally at site %d of ligand %s",
                element,
                site,
                ligand.formula,
            )

            # The newly added atom is at the last index: posadded
            # 1. Create a mask or manually reset the row and column for the new atom
            # First, store the specific connection we want to keep
            site_to_new_bond = tmpconmat[site, posadded]
            logger.debug(
                "site_to_new_bond between site %d and new atom at pos %d: %d",
                site,
                posadded,
                site_to_new_bond,
            )
            # 2. Clear all connections for the new atom (row and column)
            tmpconmat[posadded, :] = 0
            tmpconmat[:, posadded] = 0

            # 3. Restore ONLY the bond between the target site and the new atom
            # We use 1 (or site_to_new_bond if you want to preserve the bond order/value)
            tmpconmat[site, posadded] = 1
            tmpconmat[posadded, site] = 1

            logger.debug(
                "%s added unconditionally at site %d. All other new adjacencies cleared.",
                element,
                site,
            )
        elif not isgood:
            logger.info(
                "Reset at site (ligand index: %d) of atom %s %s. Clash detected when adding dummy %s",
                site,
                atom.label,
                atom.atom_site_label,
                element,
            )
            isadded = False
            newlab = list(labels)
            newcoord = list(coords)

        # Case 2: acceptable connectivity
        elif tmpconnec[posadded] <= 1:
            isadded = True

        # Case 3: excessive connectivity but some atoms were removed
        elif (
            tmpconnec[posadded] > 1 and removed_idx is not None and len(removed_idx) > 0
        ):
            connected = {i for i, c in enumerate(tmpconmat[posadded]) if c != 0}
            removed = set(removed_idx)

            logger.debug(
                "Dummy %s attached to atom %s %s %s %s connected to indices %s %s %s; previously removed indices%s %s %s",
                element,
                atom.label,
                atom.atom_site_label,
                apos,
                atom.frac_coord,
                connected,
                [labels[i] for i in connected],
                [ligand.atom_site_labels[i] for i in connected]
                if ligand.atom_site_labels is not None
                else None,
                removed,
                [labels[i] for i in removed],
                [ligand.atom_site_labels[i] for i in removed]
                if ligand.atom_site_labels is not None
                else None,
            )
            logger.debug(f"connected: {connected} | removed: {removed}")
            remaining = list(connected - removed)

            logger.debug("remaining connections after removal=%s", remaining)

            if len(remaining) <= 1:
                isadded = True
                logger.debug(
                    "%s added at site %d after removal of %s",
                    element,
                    site,
                    removed_idx,
                )
            else:
                logger.info(
                    "Reset at site (ligand index: %d) of atom %s %s due to dummy %s connectivity=%d",
                    site,
                    atom.label,
                    atom.atom_site_label,
                    element,
                    tmpconnec[posadded],
                )

                isadded = False
                newlab = list(labels)
                newcoord = list(coords)

        else:
            logger.info(
                "Reset at site (ligand index: %d) of atom %s %s due to dummy %s connectivity=%d",
                site,
                atom.label,
                atom.atom_site_label,
                element,
                tmpconnec[posadded],
            )
            isadded = False
            newlab = list(labels)
            newcoord = list(coords)

    return isadded, newlab, newcoord
