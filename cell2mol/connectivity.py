from typing import Tuple
import numpy as np
import warnings
import logging
import networkx as nx
from cell2mol.utils import config
from cell2mol.element_utils import (
    get_radii,
    get_metal_idxs,
    get_post_transition_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    labels2formula,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from cell2mol.operations import inv, extract_from_list, point_along_vector
from cell2mol.elementdata import ElementData

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def get_scaled_radii(radii, metal_idxs, alkali_idxs, metal_factor, cov_factor):
    """Scale radii based on whether atom is metal, alkali/alkaline earth metal, or non-metal"""
    scaled_radii = np.zeros_like(radii, dtype=float)
    alkali_alkaline_factor = 1.0  # additional factor for alkali/alkaline earth metals
    for k in range(len(radii)):
        if k in metal_idxs:
            scaled_radii[k] = radii[k] * metal_factor
        elif k in alkali_idxs:
            scaled_radii[k] = radii[k] * metal_factor * alkali_alkaline_factor
        else:
            scaled_radii[k] = radii[k] * cov_factor
    return scaled_radii


def is_mismatch_adjacency(
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
    _, adj_dist, _ = get_adjmatrix(
        labels,
        positions,
        cutoff=cutoff,
        cov_factor=cov_factor,
        metal_factor=metal_factor,
        metal_only=metal_only,
    )

    # 2. Build CIF-based matrix (Metadata Reference)
    adj_conn = None
    if bond_data and atom_site_labels:
        adj_conn = get_adjmatrix_from_cif_bonds(
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
) -> np.ndarray:
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

    adj_dist = None
    # --- distance-based adjacency (default) ---
    if canonical == "distance" or warn_on_mismatch:
        isgood, adj_dist, warning = get_adjmatrix(
            labels,
            positions,
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

        adj_conn = get_adjmatrix_from_cif_bonds(
            labels,
            positions,
            atom_site_labels=atom_site_labels,
            bond_data=bond_data,
            metal_only=metal_only,
        )
    if warn_on_mismatch and adj_dist is not None and adj_conn is not None:
        logger.debug("Sum of Formula: %s", labels2formula(labels))
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
    if warn_on_mismatch and adj_conn is not None:
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


def bonds_from_adj(adj: np.ndarray) -> set[tuple[int, int]]:
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

    bonds_dist = bonds_from_adj(adj_dist)
    bonds_conn = bonds_from_adj(adj_conn)

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
                logger.debug("  %s", line)

        if missing:
            logger.debug("Missing bonds (connectivity-only):")
            for line in format_bond_info(missing, labels, pos, atom_site_labels):
                logger.debug("  %s", line)


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

    metal_idxs = get_metal_idxs(labels)
    alkali_idxs = get_alkali_alkaline_earth_metal_idxs(labels)

    radii = get_scaled_radii(
        np.asarray(get_radii(labels)),
        metal_idxs,
        alkali_idxs,
        metal_factor,
        cov_factor,
    )

    formatted: list[str] = []
    for i, j in sorted(bonds):
        dist = np.linalg.norm(np.asarray(pos[i]) - np.asarray(pos[j]))
        thres = radii[i] + radii[j] + cutoff
        margin = dist - thres

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


def includes_metal(
    i: int,
    j: int,
    labels: list[str],
    metal_idxs: list[int],
    alkali_alkaline_earth_metal_idxs: list[int],
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
    if i in alkali_alkaline_earth_metal_idxs or j in alkali_alkaline_earth_metal_idxs:
        return True

    # fallback: post-transition metals if no other metals present
    if not metal_idxs and not alkali_alkaline_earth_metal_idxs:
        if get_post_transition_metal_idxs([labels[i], labels[j]]):
            return True

    return False


SPECIAL_METALS = ["Cu", "Zn", "Ag", "Cd", "Hg"]
SPECIAL_DONORS = ["N", "O", "S", "Cl", "Br", "I", "F", "P"]


def get_pair_cutoff(
    label_i: str,
    label_j: str,
    default_cutoff: float,
) -> float:
    """Return pair-specific distance cutoff added to the sum of atomic radii.

    Special rules:
    - Cu/Zn/Ag/Cd/Hg bonded to N/O/S/P/halogens: use 0.75 Å
    - d/f-block metal with Si: use 0.25 Å
    - otherwise: use default_cutoff 0.45 Å
    """

    # pair = {label_i, label_j}
    # block_i = elemdatabase.elementblock[label_i]
    # block_j = elemdatabase.elementblock[label_j]

    # d/f-block metal-Si contacts should be stricter
    # if "Si" in pair and (block_i in {"d", "f"} or block_j in {"d", "f"}):
    #     return 0.25

    # selected metal-donor coordination bonds should be more permissive
    is_special_metal_donor = (
        label_i in SPECIAL_METALS and label_j in SPECIAL_DONORS
    ) or (label_j in SPECIAL_METALS and label_i in SPECIAL_DONORS)

    if is_special_metal_donor:
        return 0.75

    return default_cutoff


def get_adjmatrix(
    labels: list[str],
    pos: np.ndarray,
    radii: np.ndarray | list | None = None,
    cutoff: float | None = None,
    cov_factor: float | None = None,
    metal_factor: float | None = None,
    metal_only: bool = False,
    add_atom: bool = False,
):
    """Generates adjacency matrix from atomic positions and covalent radii.
    Args:
        labels (list): List of atomic labels.
        pos (list): List of atomic positions.
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
    Returns:
        isgood (int): 1 if molecule is valid, 0 otherwise.
        adjmat (list): Adjacency matrix.
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
    adjmat = np.zeros((natoms, natoms), dtype=int)
    madjmat = np.zeros((natoms, natoms), dtype=int)

    metal_idxs = get_metal_idxs(labels)
    alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(labels)

    # --- radii normalization ---
    if radii is None:
        radii = get_radii(labels)
    radii = get_scaled_radii(
        np.asarray(radii),
        metal_idxs,
        alkali_alkaline_earth_metal_idxs,
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
                    "Adjacency clash: dist=%.3f < %.3f for atoms (%s:%s)-(%s:%s)",
                    dist,
                    clash_threshold,
                    labels[i],
                    pos[i],
                    labels[j],
                    pos[j],
                )
                continue

            if dist > thres:
                continue

            adjmat[i, j] = adjmat[j, i] = 1

            if metal_only and includes_metal(
                i, j, labels, metal_idxs, alkali_alkaline_earth_metal_idxs
            ):
                madjmat[i, j] = madjmat[j, i] = 1

    if not isgood:
        return False, (madjmat if metal_only else adjmat), False

    # --- valence correction ---
    warning = False
    if not add_atom:
        adjmat, madjmat, warning = correct_valence_violation(
            adjmat, madjmat, labels, pos, radii
        )
    return True, (madjmat if metal_only else adjmat), warning


def correct_valence_violation(adjmat, madjmat, labels, pos, radii):
    """Detect and correct valence violations in an adjacency matrix.
    Args:
        adjmat (np.ndarray): Adjacency matrix.
        labels (list): List of atomic labels.
        pos (list): List of atomic positions.
        radii (list): List of atomic radii.
    Returns:
        adjmat (np.ndarray): Corrected adjacency matrix.
        madjmat (np.ndarray): Corrected metal adjacency matrix.
        warning (bool): True if any valence violations were found and corrected,
            False otherwise.
    """
    from cell2mol.charge.xyz2mol import atomic_valence

    natoms = len(labels)
    warning = False

    metal_idxs = set(get_metal_idxs(labels))
    post_metal_idxs = set(get_post_transition_metal_idxs(labels))
    alkali_idxs = set(get_alkali_alkaline_earth_metal_idxs(labels))
    metal_indices = metal_idxs | post_metal_idxs | alkali_idxs

    for i in range(natoms):
        indices = np.where(adjmat[i])[0]
        n_connec_metals = len(set(indices) & metal_indices)
        if n_connec_metals == len(indices):
            continue

        valence = len(indices)
        atomic_num = elemdatabase.elementnr[labels[i]]
        max_valence = max(atomic_valence[atomic_num], default=0)

        if valence - n_connec_metals <= max_valence:
            continue

        logger.warning(
            "Valence violation: %s %s(index:%d), valence=%d > max_valence=%d",
            labels[i],
            pos[i],
            i,
            valence,
            max_valence,
        )

        logger.debug(
            "connections: labels=%s, indices=%s, n_connec_metals=%d",
            [labels[j] for j in indices],
            indices.tolist(),
            n_connec_metals,
        )
        warning = True

        if labels[i] in {"F", "Cl", "Br", "I", "B"}:
            logger.info("Skipping correction for element %s", labels[i])
            continue

        connections = []
        a = np.asarray(pos[i])
        for j in indices:
            b = np.asarray(pos[j])
            dist = np.linalg.norm(a - b)
            margin = dist - (radii[i] + radii[j])
            connections.append((j, margin))

            logger.debug(
                "Bond %d-%d (%s:%s)-(%s:%s): dist=%.3f, margin=%.3f",
                i,
                j,
                labels[i],
                pos[i],
                labels[j],
                pos[j],
                dist,
                margin,
            )

        connections.sort(key=lambda x: x[1], reverse=True)
        to_remove = valence - max_valence - n_connec_metals

        for j, margin in connections[:to_remove]:
            if margin > 0.2:
                adjmat[i, j] = adjmat[j, i] = 0
                madjmat[i, j] = madjmat[j, i] = 0
                logger.info(
                    "Removed bond %d-%d (%s:%s)-(%s:%s), dist=%.3f, margin=%.3f",
                    i,
                    j,
                    labels[i],
                    pos[i],
                    labels[j],
                    pos[j],
                    np.linalg.norm(np.asarray(pos[i]) - np.asarray(pos[j])),
                    margin,
                )

    return adjmat, madjmat, warning


def get_adjmatrix_from_cif_bonds(
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
        adjmat (np.ndarray): Adjacency matrix.
    """
    indices = {atom: i for i, atom in enumerate(atom_site_labels)}
    natoms = len(labels)

    adjmat = np.zeros((natoms, natoms), dtype=int)

    metal_idxs = set(get_metal_idxs(labels))
    alkali_alkaline_earth_metal_idxs = set(get_alkali_alkaline_earth_metal_idxs(labels))

    for atom1, atom2, bond_dist in bond_data:
        if atom1 not in indices or atom2 not in indices:
            continue

        i, j = indices[atom1], indices[atom2]

        a = np.asarray(pos[i])
        b = np.asarray(pos[j])
        dist = np.linalg.norm(a - b)

        if abs(dist - bond_dist) > tol:
            continue

        if metal_only and not includes_metal(
            i, j, labels, metal_idxs, alkali_alkaline_earth_metal_idxs
        ):
            continue

        adjmat[i, j] = adjmat[j, i] = 1

    return adjmat


def get_adjacency_types(label: list, conmat: np.ndarray) -> np.ndarray:
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


def get_blocks(matrix: np.ndarray):
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

    adjmat = build_adjacency(
        labels=labels,
        positions=positions,
        atom_site_labels=atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
        cov_factor=cov_factor,
        metal_factor=metal_factor,
        warn_on_mismatch=warn_on_mismatch,
    )
    if adjmat is None:
        logger.warning("Adjacency matrix is None. Returning empty blocklist.")
        return []
    adjnum = adjmat.sum(axis=1)

    degree = np.diag(adjnum)
    lap = adjmat - degree

    # creates block matrix
    graph = csr_matrix(lap)
    perm = reverse_cuthill_mckee(graph)
    gp1 = graph[perm, :]
    gp2 = gp1[:, perm]
    dense = gp2.toarray()

    # detects blocks in the block diagonal matrix called "dense"
    startlist, endlist = get_blocks(dense)

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
        new_blocklist = apply_graph_to_blocklist(
            blocklist,
            labels,
            positions,
            conn_atom_site_labels=atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
        )
        log_blocklist_diff(blocklist, new_blocklist)
        return new_blocklist
    return blocklist


def apply_graph_to_blocklist(
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
    """Split a list of atoms into blocks of connected atoms."""

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

        adjmat = build_adjacency(
            labels=gr_labels,
            positions=gr_coord,
            atom_site_labels=gr_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            warn_on_mismatch=False,
            detail=False,
        )

        G = nx.from_numpy_array(np.array(adjmat))

        if not nx.is_connected(G):
            new_blocklist.append(b)
            continue

        cycle_basis = nx.cycle_basis(G)

        if len(cycle_basis) != 1:
            new_blocklist.append(b)
            continue

        cycle = cycle_basis[0]
        logger.debug("Found single cycle in block %s: %s", b, cycle)

        # Full cycle covers all atoms
        if len(cycle) == len(G.nodes):
            new_blocklist.append(b)
            continue

        # Partial cycle: split cycle and remaining components
        cycle_block = sorted([b[idx] for idx in cycle])
        new_blocklist.append(cycle_block)

        logger.debug("Cycle block indices=%s", cycle_block)

        remaining = [b[n] for n in G.nodes if n not in cycle]
        logger.debug("Remaining nodes in block=%s", remaining)

        rem_labels = extract_from_list(remaining, conn_labels, dimension=1)
        rem_coord = extract_from_list(remaining, conn_coord, dimension=1)

        if conn_atom_site_labels is not None:
            rem_atom_site_labels = extract_from_list(
                remaining, conn_atom_site_labels, dimension=1
            )
        else:
            rem_atom_site_labels = None

        adjmat_rem = build_adjacency(
            labels=rem_labels,
            positions=rem_coord,
            atom_site_labels=rem_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            warn_on_mismatch=False,
            detail=False,
        )

        G_rem = nx.from_numpy_array(np.array(adjmat_rem))

        for comp in nx.connected_components(G_rem):
            remaining_block = [remaining[idx] for idx in comp]
            new_blocklist.append(remaining_block)
            logger.debug(
                "Remaining connected block=%s",
                remaining_block,
            )

    logger.debug("Final new_blocklist=%s", new_blocklist)

    return new_blocklist


def log_blocklist_diff(blocklist, new_blocklist):
    old = {tuple(sorted(b)) for b in blocklist}
    new = {tuple(sorted(b)) for b in new_blocklist}

    if old == new:
        # logger.debug("Blocklist unchanged.")
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


def identify_haptic_mode(atoms: list, use_bond_info: bool | None = None):
    """
    Determine haptic coordination mode(s) for a given atoms list.
    Args:
        atoms (list): List of Atom objects
        use_bond_info (bool, optional): Whether to use bond information from CIF
    Returns:
        tuple: (is_haptic (bool), haptic_type (str), topology analysis (dict))
    """
    labels = [atom.label for atom in atoms]
    group_formula = labels2formula(labels)
    is_haptic = False  # old self.hapticity
    haptic_type = None  # old self.hapttype

    totnum = len(labels)

    counts = {
        "C": labels.count("C"),
        "As": labels.count("As"),
        "P": labels.count("P"),
        "O": labels.count("O"),
    }

    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    # (numC, numAs, numP, numO, total_atoms) → haptic modes
    HAPTIC_RULES = {
        # eta2
        (2, 0, 0, 0, 2): "eta2(C2)",
        # eta3
        (3, 0, 0, 0, 3): "eta3(C3)",
        # eta4
        (4, 0, 0, 0, 4): "eta4(C4)",
        # eta5
        (5, 0, 0, 0, 5): "eta5(C5)",
        (0, 5, 0, 0, 5): "eta5(As5)",
        (0, 0, 5, 0, 5): "eta5(P5)",
        # eta6+
        (6, 0, 0, 0, 6): "eta6(C6)",
        (7, 0, 0, 0, 7): "eta7(C7)",
        (8, 0, 0, 0, 8): "eta8(C8)",
    }

    key = (
        counts["C"],
        counts["As"],
        counts["P"],
        counts["O"],
        totnum,
    )

    # Fallback: generic single-ring hapticity
    results = analyze_topology(atoms, use_bond_info=use_bond_info)
    if key in HAPTIC_RULES:
        haptic_type = HAPTIC_RULES[key]
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
        elif results["n_rings"] == 2 and results["is_fused_5_5"]:
            if group_formula == "C8":
                haptic_type = "eta5,eta5(pentalene)"  # pentalene
        elif results["n_rings"] == 2 and results["is_fused_5_6"]:
            if group_formula == "C9":
                haptic_type = "eta5,eta6(indene)"  # indene
        elif results["n_rings"] == 2 and results["is_fused_5_7"]:
            if group_formula == "C10":
                haptic_type = "eta5,eta7(azulene)"  # azulene
    else:
        if results["is_single_simple_ring"]:
            haptic_type = f"eta{totnum}({group_formula})"
            is_haptic = True
    logger.debug(
        "Identified haptic mode: is_haptic=%s, haptic_type=%s", is_haptic, haptic_type
    )

    return is_haptic, haptic_type, results


def is_single_ring(atoms: list, use_bond_info: bool | None = None) -> bool:
    """Check if a given list of atoms is a ring
    Args:
        atoms: list of Atom objects
        use_bond_info: whether to use bond information from CIF data
    Returns:
        bool: True if the atoms form a single ring, False otherwise
    """
    labels = [atom.label for atom in atoms]
    positions = [atom.coord for atom in atoms]
    atom_site_labels = (
        [atom.atom_site_label for atom in atoms] if atoms[0].atom_site_label else None
    )
    refcell = atoms[0].get_parent("reference")
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    # logger.debug("Checking if group is a single ring...")
    # logger.debug("Labels: %s", labels)
    # logger.debug("Positions: %s", positions)
    # logger.debug("Atom site labels: %s", atom_site_labels)

    adjmat = build_adjacency(
        labels=labels,
        positions=positions,
        atom_site_labels=atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
    )
    if adjmat is None:
        return False
    G = nx.from_numpy_array(np.array(adjmat))

    # Check if the graph is connected
    if not nx.is_connected(G):
        return False  # If not connected, can't form a single ring

    # Check for cycles and ensure the graph forms a simple cycle
    cycle_basis = nx.cycle_basis(G)

    # Check if there's exactly one cycle that includes all nodes (simple ring)
    if len(cycle_basis) == 1 and len(cycle_basis[0]) == len(G.nodes):
        return True
    #     return True, None  # The graph represents a ring compound
    # elif len(cycle_basis) == 1 and len(cycle_basis[0]) != len(G.nodes):
    #     return False, cycle_basis[0]  # The graph has a cycle but not all nodes are included

    return False  # Otherwise, not a ring compound


def analyze_topology(atoms: list, use_bond_info: bool | None = None):
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

    def _empty_result(is_connected=False):
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

    if not atoms:
        return _empty_result(is_connected=False)

    labels = [atom.label for atom in atoms]
    positions = [atom.coord for atom in atoms]
    atom_site_labels = (
        [atom.atom_site_label for atom in atoms] if atoms[0].atom_site_label else None
    )

    refcell = atoms[0].get_parent("reference")
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    adjmat = build_adjacency(
        labels=labels,
        positions=positions,
        atom_site_labels=atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
    )

    if adjmat is None:
        return _empty_result(is_connected=False)

    G = nx.from_numpy_array(np.array(adjmat))

    is_connected = nx.is_connected(G)
    if not is_connected:
        return _empty_result(is_connected=False)

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
        sizes = tuple(sorted([len(r1), len(r2)]))
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

    ring_sets = [list(r) for r in ring_sets]
    return {
        "is_connected": True,
        "n_rings": n_rings,
        "ring_sizes": [len(r) for r in ring_sets],
        "ring_atoms": ring_atoms,
        "ring_sets": ring_sets,
        "all_atoms_in_rings": all_atoms_in_rings,
        "is_single_simple_ring": is_single_simple_ring,
        **fused_flags,
    }


def merge_multiple_groups(parent_molecule, groups, parent_ligand):
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
        adj = parent_molecule.adjmat

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
            new_grp.topology = results
        else:
            new_grp.get_hapticity()

        # Map Indices
        grp_mol_indices = [a.get_parent_index("molecule") for a in atom_subset]
        lig_mol_indices = [
            atom.get_parent_index("molecule") for atom in parent_ligand.atoms
        ]
        grp_lig_indices = [lig_mol_indices.index(m_idx) for m_idx in grp_mol_indices]

        # Establish Parents
        new_grp.add_parent(parent_molecule, indices=grp_mol_indices)
        new_grp.set_inherit_adjmatrix("molecule")
        new_grp.add_parent(parent_ligand, indices=grp_lig_indices)

        # Link Specific Metals
        if getattr(new_grp, "metals", None) is None:
            object.__setattr__(new_grp, "metals", [])
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
        formula = labels2formula(labels)

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

        group_name = ring_map.get(key, formula)
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
            [m.label for m in group_unified.metals],
        )
        return [group_unified]


def add_atom(
    labels: list,
    coords: list,
    site: int,
    ligand: object,
    element: str = "H",
    metal: object | None = None,
    removed_idx: list | None = None,
    unconditional: bool = False,
) -> Tuple[bool, list, list]:
    """
    Add one atom of type `element` to a given ligand atom site.

    The atom is placed along the vector pointing toward a given metal atom.
    """
    isadded = False
    posadded = len(labels)

    newlab = list(labels)
    newcoord = list(coords)

    newlab.append(element)  # one atom will be added

    # logger.debug("number of ligand atoms=%d", len(ligand.atoms))
    # logger.debug("target site=%d (%s)", site, ligand.atoms[site].label)

    for idx, atom in enumerate(ligand.atoms):
        if idx != site:
            continue

        apos = np.array(atom.coord, copy=True)
        if metal is not None:
            tgt = metal

        else:
            tgt = atom.get_closest_metal()

        # metal_idx = tgt.get_parent_index("molecule")

        # logger.debug("Evaluating atom at %s with closest metal at %s", apos, tgt.coord)

        idealdist = atom.radii + elemdatabase.CovalentRadius3[element]
        added_coords = point_along_vector(apos, tgt.coord, idealdist)

        newcoord.append([added_coords[0], added_coords[1], added_coords[2]])

        isgood, tmpconmat, warning = get_adjmatrix(
            newlab,
            newcoord,
            cov_factor=ligand.cov_factor,
            add_atom=True,
        )
        tmpconnec = tmpconmat.sum(axis=1)
        logger.debug("tmpconnec at added position=%d", int(tmpconnec[posadded]))
        # newlab_with_metal = newlab + [tgt.label]
        # newcoord_with_metal = newcoord + [tgt.coord]

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
            # logger.debug(
            #     "Chosen metal index %d. %s added at site %d",
            #     metal_idx,
            #     element,
            #     site,
            # )

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
                [ligand.atom_site_labels[i] for i in connected],
                removed,
                [labels[i] for i in removed],
                [ligand.atom_site_labels[i] for i in removed],
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
                # import os
                # from cell2mol.write_results import writexyz
                # writexyz(
                #     os.getcwd(),
                #     f"target_atom_{atom.label}_{apos[0]}_newcoord_with_{element}.xyz",
                #     newlab_with_metal,
                #     newcoord_with_metal,
                # )

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
