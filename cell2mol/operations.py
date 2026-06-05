#!/usr/bin/env python

import numpy as np
import itertools
import logging
from cell2mol.elementdata import ElementData
from cell2mol.element_utils import (
    TRANSITION_METALS,
    ALKALI_AND_ALKALINE_EARTH_METALS,
    LANTHANIDES,
    ACTINIDES,
    POST_TRANSITION_METALS,
    METALLOIDS,
    labels2formula,
)
from cell2mol.write_results import save_coordination_report
import re

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def frac2cart_fromcellvec(frac_coord, cellvec):
    """Convert fractional coordinates to cartesian coordinates
    Args:
        frac_coord (list): list of fractional coordinates
        cellvec (list): list of cell vectors
    Returns:
        cartesian (list): list of cartesian coordinates
    """
    cartesian = []
    for idx, frac in enumerate(frac_coord):
        xcar = (
            frac[0] * cellvec[0][0] + frac[1] * cellvec[1][0] + frac[2] * cellvec[2][0]
        )
        ycar = (
            frac[0] * cellvec[0][1] + frac[1] * cellvec[1][1] + frac[2] * cellvec[2][1]
        )
        zcar = (
            frac[0] * cellvec[0][2] + frac[1] * cellvec[1][2] + frac[2] * cellvec[2][2]
        )
        cartesian.append([float(xcar), float(ycar), float(zcar)])
    return cartesian


def frac2cart_fromparam(frac_coord, cellparam):
    """Convert fractional coordinates to cartesian coordinates
    Args:
        frac_coord (list): list of fractional coordinates
        cellparam (list): list of cell parameters
    Returns:
        cartesian (list): list of cartesian coordinates
    """

    a = cellparam[0]
    b = cellparam[1]
    c = cellparam[2]
    alpha = np.radians(cellparam[3])
    beta = np.radians(cellparam[4])
    gamma = np.radians(cellparam[5])

    volume = (
        a
        * b
        * c
        * np.sqrt(
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
    )

    m = np.zeros((3, 3))
    m[0][0] = a
    m[0][1] = b * np.cos(gamma)
    m[0][2] = c * np.cos(beta)
    m[1][0] = 0
    m[1][1] = b * np.sin(gamma)
    m[1][2] = c * ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))
    m[2][0] = 0
    m[2][1] = 0
    m[2][2] = volume / (a * b * np.sin(gamma))

    cartesian = []
    for idx, frac in enumerate(frac_coord):
        xcar = frac[0] * m[0][0] + frac[1] * m[0][1] + frac[2] * m[0][2]
        ycar = frac[0] * m[1][0] + frac[1] * m[1][1] + frac[2] * m[1][2]
        zcar = frac[0] * m[2][0] + frac[1] * m[2][1] + frac[2] * m[2][2]
        cartesian.append([float(xcar), float(ycar), float(zcar)])
    return cartesian


def cart2frac(cartCoords, cellvec):
    """Convert cartesian coordinates to fractional coordinates
    Args:
        cartCoords (list): list of cartesian coordinates
        cellvec (list): list of cell vectors
    Returns:
        fracCoords (list): list of fractional coordinates
    """

    latCnt = [x[:] for x in [[None] * 3] * 3]
    for a in range(3):
        for b in range(3):
            latCnt[a][b] = cellvec[b][a]

    fracCoords = []
    detLatCnt = det3(latCnt)

    for i in cartCoords:
        aPos = (
            det3(
                [
                    [i[0], latCnt[0][1], latCnt[0][2]],
                    [i[1], latCnt[1][1], latCnt[1][2]],
                    [i[2], latCnt[2][1], latCnt[2][2]],
                ]
            )
        ) / detLatCnt
        bPos = (
            det3(
                [
                    [latCnt[0][0], i[0], latCnt[0][2]],
                    [latCnt[1][0], i[1], latCnt[1][2]],
                    [latCnt[2][0], i[2], latCnt[2][2]],
                ]
            )
        ) / detLatCnt
        cPos = (
            det3(
                [
                    [latCnt[0][0], latCnt[0][1], i[0]],
                    [latCnt[1][0], latCnt[1][1], i[1]],
                    [latCnt[2][0], latCnt[2][1], i[2]],
                ]
            )
        ) / detLatCnt
        fracCoords.append([aPos, bPos, cPos])
    return fracCoords


def det3(mat):
    """Calculate the determinant of a 3x3 matrix
    Args:
        mat (list): list of 3x3 matrix
    Returns:
        determinant (float): determinant of the matrix
    """
    return (
        (mat[0][0] * mat[1][1] * mat[2][2])
        + (mat[0][1] * mat[1][2] * mat[2][0])
        + (mat[0][2] * mat[1][0] * mat[2][1])
        - (mat[0][2] * mat[1][1] * mat[2][0])
        - (mat[0][1] * mat[1][0] * mat[2][2])
        - (mat[0][0] * mat[1][2] * mat[2][1])
    )


def translate(vector, coords, cellvec):
    """Translate coordinates by a vector
    Args:
        vector (list): list of vector components
        coords (list): list of coordinates
        cellvec (list): list of cell vectors
    Returns:
        newcoord (list): list of translated coordinates
    """

    newcoord = []
    for idx, coord in enumerate(coords):
        newx = (
            coord[0]
            + vector[0] * cellvec[0][0]
            + vector[1] * cellvec[1][0]
            + vector[2] * cellvec[2][0]
        )
        newy = (
            coord[1]
            + vector[0] * cellvec[0][1]
            + vector[1] * cellvec[1][1]
            + vector[2] * cellvec[2][1]
        )
        newz = (
            coord[2]
            + vector[0] * cellvec[0][2]
            + vector[1] * cellvec[1][2]
            + vector[2] * cellvec[2][2]
        )
        newcoord.append([float(newx), float(newy), float(newz)])

    return newcoord


def extract_from_list(entrylist: list, old_array: list, dimension: int = 2) -> list:
    """Extract a 1D or 2D sub-array using a list of indices.

    Args:
        entrylist (list): indices to extract
        old_array (list): source list
        dimension (int): 2 for 2D extraction, 1 for 1D extraction
    Returns:
        list: extracted sub list
    """
    length = len(entrylist)
    if dimension == 2:
        new_array = np.empty((length, length), dtype=object)
        for idx, row in enumerate(entrylist):
            for jdx, col in enumerate(entrylist):
                new_array[idx, jdx] = old_array[row][col]
    elif dimension == 1:
        new_array = np.empty((length), dtype=object)
        for idx, val in enumerate(entrylist):
            new_array[idx] = old_array[val]
    return list(new_array)


def reorder_element(lst: list, old_idx: int, new_idx: int) -> list:
    """Moves an element from old_idx to new_idx and returns a new list."""
    new_lst = list(lst)
    new_lst.insert(new_idx, new_lst.pop(old_idx))
    return new_lst


def additem(item, vector):
    """Append an item to a list if not already present.

    Args:
        item: item to add
        vector (list): target list

    Returns:
        list: updated list
    """
    if item not in vector:
        vector.append(item)
    return vector


def absolute_value(num):
    """Calculate the absolute value of the sum of the absolute values of elements in a list."""
    sum = 0
    for i in num:
        sum += np.abs(i)
    return abs(sum)


def inv(perm: list) -> list:
    """Compute the inverse of a permutation.

    Args:
        perm (list): permutation mapping

    Returns:
        list: inverse permutation
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def compute_centroid(arr: np.ndarray) -> list:
    """Compute the centroid of a set of 3D coordinates.

    Args:
        arr (np.ndarray): array of shape (N, 3)

    Returns:
        np.ndarray: centroid coordinates
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    centroid = np.around(np.array([sum_x / length, sum_y / length, sum_z / length]), 7)
    return np.array(centroid)


def get_dist(atom1_pos: list, atom2_pos: list) -> float:
    """Compute the Euclidean distance between two points.

    Args:
        atom1_pos (list): first point coordinates
        atom2_pos (list): second point coordinates

    Returns:
        float: distance between points
    """
    dist = np.linalg.norm(np.array(atom1_pos) - np.array(atom2_pos))
    dist = round(float(dist), 3)
    return dist


def get_angle(vec1, vec2) -> float:
    """Compute the angle between two vectors in radians.

    Args:
        vec1: first vector
        vec2: second vector

    Returns:
        float: angle in radians
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    dotprod = np.dot(vec1, vec2)
    factor = dotprod / (norm1 * norm2)
    angle = np.arccos(factor)
    if np.isnan(angle):
        logger.error(
            "GET_ANGLE nan Problem %s %s %s %s %s",
            norm1,
            norm2,
            dotprod,
            factor,
            angle,
        )
        logger.error("GET_ANGLE nan Problem, vecs: %s %s", vec1, vec2)
    return float(angle)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v.

    For zero vectors, returns the original vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def tmatgenerator(centroid, thres=0.40, full=False):
    """Generates a list of translation vectors for fragment reconstruction.

    This function determines the necessary translations for a molecular fragment
    based on the position of its centroid within the unit cell's fractional
    coordinates. The goal is to generate translation vectors that will help in
    reconstructing a whole molecule that is split across unit cell boundaries.

    If a fragment's centroid is near a cell boundary (e.g., close to 0.0 or 1.0
    along an axis), translations along that axis are suggested. Fragments near
    the center of the cell are less likely to need translation.

    Args:
        centroid (np.ndarray): The fractional coordinates (x, y, z) of the
            fragment's centroid.
        thres (float, optional): The threshold defining the boundary region.
            Translations are suggested if a centroid coordinate is less than
            `thres` or greater than `1 - thres`. Defaults to 0.40.
        full (bool, optional): If True, generates all 27 possible translation
            vectors (from -1 to 1 in each dimension), ignoring the centroid
            position. Defaults to False.

    Returns:
        list: A sorted list of translation vectors as tuples (e.g., [(0, 0, 0),
              (1, 0, 0), ...]). The list is sorted by the magnitude of the
              translation.
    """

    tmax = 1 - thres
    tmin = thres

    if not full:
        tmatrix = []
        tmatrix = additem((0, 0, 0), tmatrix)

        # X positive
        if centroid[0] >= tmax:
            tmatrix = additem((-1, 0, 0), tmatrix)
            if centroid[1] >= tmax:
                tmatrix = additem((-1, -1, 0), tmatrix)
                tmatrix = additem((0, -1, 0), tmatrix)
                if centroid[2] >= tmax:
                    tmatrix = additem((-1, -1, -1), tmatrix)
                    tmatrix = additem((0, -1, -1), tmatrix)
                    tmatrix = additem((0, 0, -1), tmatrix)
                if centroid[2] <= tmin:
                    tmatrix = additem((-1, -1, 1), tmatrix)
                    tmatrix = additem((0, -1, 1), tmatrix)
                    tmatrix = additem((0, 0, 1), tmatrix)
            if centroid[1] <= tmin:
                tmatrix = additem((-1, 1, 0), tmatrix)
                tmatrix = additem((0, 1, 0), tmatrix)
                if centroid[2] >= tmax:
                    tmatrix = additem((-1, 1, -1), tmatrix)
                    tmatrix = additem((0, 1, -1), tmatrix)
                    tmatrix = additem((0, 0, -1), tmatrix)
                if centroid[2] <= tmin:
                    tmatrix = additem((-1, 1, 1), tmatrix)
                    tmatrix = additem((0, 1, 1), tmatrix)
                    tmatrix = additem((0, 0, 1), tmatrix)
            if centroid[2] >= tmax:
                tmatrix = additem((-1, 0, -1), tmatrix)
                tmatrix = additem((0, 0, -1), tmatrix)
            if centroid[2] <= tmin:
                tmatrix = additem((-1, 0, 1), tmatrix)
                tmatrix = additem((0, 0, 1), tmatrix)

        if centroid[1] >= tmax:
            tmatrix = additem((0, -1, 0), tmatrix)
            if centroid[2] >= tmax:
                tmatrix = additem((0, -1, -1), tmatrix)
                tmatrix = additem((0, 0, -1), tmatrix)
            if centroid[2] <= tmin:
                tmatrix = additem((0, -1, 1), tmatrix)
                tmatrix = additem((0, 0, 1), tmatrix)

        if centroid[2] >= tmax:
            tmatrix = additem((0, 0, -1), tmatrix)

        if centroid[0] <= tmin:
            tmatrix = additem((1, 0, 0), tmatrix)
            if centroid[1] <= tmin:
                tmatrix = additem((1, 1, 0), tmatrix)
                tmatrix = additem((0, 1, 0), tmatrix)
                if centroid[2] <= tmin:
                    tmatrix = additem((1, 1, 1), tmatrix)
                    tmatrix = additem((0, 1, 1), tmatrix)
                    tmatrix = additem((0, 0, 1), tmatrix)
                if centroid[2] >= tmax:
                    tmatrix = additem((1, 1, -1), tmatrix)
                    tmatrix = additem((0, 1, -1), tmatrix)
                    tmatrix = additem((0, 0, -1), tmatrix)
            if centroid[1] >= tmax:
                tmatrix = additem((1, -1, 0), tmatrix)
                tmatrix = additem((0, -1, 0), tmatrix)
                if centroid[2] >= tmax:
                    tmatrix = additem((1, -1, -1), tmatrix)
                if centroid[2] <= tmin:
                    tmatrix = additem((1, -1, 1), tmatrix)
            if centroid[2] <= tmin:
                tmatrix = additem((1, 0, 1), tmatrix)
                tmatrix = additem((0, 0, 1), tmatrix)
            if centroid[2] >= tmax:
                tmatrix = additem((1, 0, -1), tmatrix)
                tmatrix = additem((0, 0, -1), tmatrix)

        if centroid[1] <= tmin:
            tmatrix = additem((0, 1, 0), tmatrix)
            if centroid[2] <= tmin:
                tmatrix = additem((0, 1, 1), tmatrix)
                tmatrix = additem((0, 0, 1), tmatrix)
            if centroid[2] >= tmax:
                tmatrix = additem((0, 1, -1), tmatrix)
                tmatrix = additem((0, 0, -1), tmatrix)
        if centroid[2] <= tmin:
            tmatrix = additem((0, 0, 1), tmatrix)

        if (centroid[0] > tmin) and (centroid[0] < tmax):
            if centroid[1] <= tmin:
                tmatrix = additem((0, 1, 0), tmatrix)
                if centroid[2] >= tmax:
                    tmatrix = additem((0, 1, -1), tmatrix)
                if centroid[2] <= tmin:
                    tmatrix = additem((0, 1, 1), tmatrix)
            if centroid[1] >= tmax:
                tmatrix = additem((0, -1, 0), tmatrix)
                if centroid[2] >= tmax:
                    tmatrix = additem((0, -1, -1), tmatrix)
                if centroid[2] <= tmin:
                    tmatrix = additem((0, -1, 1), tmatrix)
            if centroid[2] <= tmin:
                tmatrix = additem((0, 0, 1), tmatrix)
                if centroid[1] >= tmax:
                    tmatrix = additem((0, -1, 1), tmatrix)
                if centroid[1] <= tmin:
                    tmatrix = additem((0, 1, 1), tmatrix)
            if centroid[2] >= tmax:
                tmatrix = additem((0, 0, -1), tmatrix)
                if centroid[1] >= tmax:
                    tmatrix = additem((0, -1, -1), tmatrix)
                if centroid[1] <= tmin:
                    tmatrix = additem((0, 1, -1), tmatrix)
    elif full:
        x = [-1, 0, 1]
        tmatrix = [p for p in itertools.product(x, repeat=3)]

    tmatrix.sort(key=absolute_value)

    return tmatrix


def point_along_vector(point1, point2, distance):
    """
    Calculate the coordinates of a point along the vector between two points
    with a specified distance from the first point.

    Args:
    - point1: Coordinates of the first point (numpy array or list)
    - point2: Coordinates of the second point (numpy array or list)
    - distance: Distance from the first point to the new point (float)

    Returns:
    - Coordinates of the new point (numpy array)
    """
    # Convert input to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the vector between the two points
    vector = point2 - point1

    # Normalize the vector
    normalized_vector = vector / np.linalg.norm(vector)

    # Calculate the coordinates of the new point
    new_point = point1 + normalized_vector * distance

    return new_point


def kabsch_rotation(P, Q):
    """
    Find rotation R that best aligns P to Q (both 3xN).
    Returns 3x3 rotation matrix.
    """
    H = P @ Q.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Right-handed fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def perp_unit(u):
    """Deterministic unit vector perpendicular to u."""
    u = unit_vector(u)
    # choose a global axis least aligned with u
    g = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = g - (g @ u) * u
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        g = np.array([0.0, 0.0, 1.0])
        v = g - (g @ u) * u
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            raise ValueError("Cannot construct a perpendicular direction.")
    return v / nv


def get_moiety_indices_from_labels(atom_site_labels, moiety_list):
    flat_list = [atom for moiety in moiety_list for atom in moiety]

    atom_site_labels = np.array(atom_site_labels)  # ensure it's a numpy array
    moiety_indices = [
        np.where(np.isin(atom_site_labels, moiety))[0].tolist()
        for moiety in moiety_list
    ]
    for i, atom in enumerate(atom_site_labels):
        if atom not in flat_list:
            moiety_indices.append([i])

    return moiety_indices


def count_metals(labels: list[str]) -> dict[str, int]:
    metal_counts = {
        "tm": 0,
        "f_block": 0,
        "alkali_alkaline": 0,
        "post_tm": 0,
        "metalloid": 0,
    }

    for label in labels:
        if label in TRANSITION_METALS:
            metal_counts["tm"] += 1
        elif label in LANTHANIDES or label in ACTINIDES:
            metal_counts["f_block"] += 1
        elif label in ALKALI_AND_ALKALINE_EARTH_METALS:
            metal_counts["alkali_alkaline"] += 1
        elif label in POST_TRANSITION_METALS:
            metal_counts["post_tm"] += 1
        elif label in METALLOIDS:
            metal_counts["metalloid"] += 1

    log_map = {
        "Transition metals": metal_counts["tm"],
        "Lanthanides/Actinides": metal_counts["f_block"],
        "Alkali/Alkaline earth metals": metal_counts["alkali_alkaline"],
        "Post-transition metals": metal_counts["post_tm"],
        "Metalloids": metal_counts["metalloid"],
    }
    return metal_counts, log_map


def is_polynuclear_over_limit(labels: list[str], max_metal_centers: int = 6) -> bool:
    """
    Check whether a complex exceeds the metal center limit.
    Logs details if the limit is exceeded or if non-transition metals are present.
    """

    metal_counts, log_map = count_metals(labels)

    total_metal_count = (
        metal_counts["tm"]
        + metal_counts["f_block"]
        + metal_counts["alkali_alkaline"]
        + metal_counts["post_tm"]
    )

    if total_metal_count > max_metal_centers:
        logger.warning(
            "Detected polynuclear complex exceeding metal center limit (%d): %s",
            max_metal_centers,
            labels2formula(labels),
        )
        logger.debug("Total metal counts: %d", total_metal_count)
        logger.debug("Metal type counts: %s", log_map)
        # for name, count in log_map.items():
        #     if count > 0:
        #         logger.debug("  %s: %d", name, count)
        return True

    return False


def has_mixed_metal_types(labels: list[str]) -> bool:
    metal_counts, log_map = count_metals(labels)

    tm_count = metal_counts["tm"]
    total_metal_count = (
        metal_counts["tm"]
        + metal_counts["f_block"]
        + metal_counts["alkali_alkaline"]
        + metal_counts["post_tm"]
    )

    # No metals → not mixed
    if total_metal_count == 0:
        return False

    # Only transition metals → not mixed
    if total_metal_count == tm_count:
        logger.info(
            "TM-only complex detected: %s",
            labels2formula(labels),
        )
        return False
    logger.warning("Mixed metal types detected in complex: %s", labels2formula(labels))
    logger.debug("Total metal counts: %d", total_metal_count)
    logger.debug("Metal type counts: %s", log_map)
    # Mixed metal types detected
    return True


def has_different_metal_coordination(
    refmoleculist, bond_data, refcode, report_csv=None
):
    """
    Compare final metal coordination spheres against bond_data.

    Tracks atoms removed during validate_coordinated_atoms() via metal.removed_from_coordination.

    Notes
    -----
    - removed=True does not always mean final_match=False (removal may fix the result).
    - If a removed atom was expected in bond_data:
        status="missing_in_cell2mol", bond_change="removed_but_expected"
    """
    if not bond_data:
        logger.info("No bond data provided for coordination verification.")
        return None

    report, overall_difference = [], False

    for mol_idx, molecule in enumerate(refmoleculist):
        if molecule.metals is None:
            logger.info(
                "No metals in Molecule %s (%s). Skipping.", mol_idx, molecule.formula
            )
            continue

        for met in molecule.metals:
            met_label = met.atom_site_label
            removed_atoms = getattr(met, "removed_from_coordination", None) or []
            set_removed = {
                a.atom_site_label for a in removed_atoms if a.atom_site_label
            }

            neighbors = [
                b[1] if b[0] == met_label else b[0]
                for b in bond_data
                if met_label in (b[0], b[1])
            ]
            set_data = set(neighbors)
            set_current = {
                a.atom_site_label for a in met.coord_sphere_atoms if a.atom_site_label
            }

            group_dict = {}
            for group in met.groups:
                if group.is_haptic:
                    for atom in group.atoms:
                        group_dict[atom.atom_site_label] = group.haptic_type

            final_match = set_data == set_current
            missing, extra = set_data - set_current, set_current - set_data

            other_metal_labels = [
                other_met.atom_site_label
                for other_met in molecule.metals
                if other_met != met
            ]
            other_metal_labels = set(other_metal_labels)
            set_data_without_metal = set_data - other_metal_labels
            set_current_without_metal = set_current - other_metal_labels

            final_match_wo_mm = set_data_without_metal == set_current_without_metal

            if not final_match:
                overall_difference = True
                logger.warning(
                    "Molecule %s (%s): Mismatch for Metal %s",
                    mol_idx,
                    molecule.formula,
                    met_label,
                )
                logger.warning(
                    "  Expected: %s | Final: %s | Missing: %s | Extra: %s | Removed: %s",
                    sorted(set_data),
                    sorted(set_current),
                    sorted(missing),
                    sorted(extra),
                    sorted(set_removed),
                )
            elif set_removed:
                logger.debug(
                    "Metal %s: removed atoms %s but final coord still matches.",
                    met_label,
                    sorted(set_removed),
                )

            recorded_keys = set()

            def add_row(atom_obj, site, status, bond_change):
                key = (met_label, site, status, bond_change)
                if key in recorded_keys:
                    return
                recorded_keys.add(key)
                report.append(
                    {
                        "refcode": refcode,
                        "molecule_index": mol_idx,
                        "formula": molecule.formula,
                        "metal": met.label,
                        "coord_atom": atom_obj.label,
                        "metal_site_label": met_label,
                        "coord_atom_site_label": site,
                        "bond": f"{met_label}-{site}",
                        "distance": get_dist(met.coord, atom_obj.coord),
                        "status": status,
                        "bond_change": bond_change,
                        "removed": site in set_removed,
                        "final_match": final_match,
                        "final_match_wo_mm": final_match_wo_mm,
                        "haptic_type": group_dict.get(site, None)
                        if site in group_dict.keys()
                        else "not_haptic",
                        "is_haptic": site in group_dict.keys(),
                        "n_metals": len(molecule.metals),
                    }
                )

            def add_completely_missing_row(site):
                key = (met_label, site, "missing_in_cell2mol", "found_other_molecule")
                if key in recorded_keys:
                    return
                recorded_keys.add(key)
                for b in bond_data:
                    if met_label in (b[0], b[1]) and site in (b[0], b[1]):
                        distance = b[2] if len(b) > 2 else None
                        break
                report.append(
                    {
                        "refcode": refcode,
                        "molecule_index": mol_idx,
                        "formula": molecule.formula,
                        "metal": met.label,
                        "coord_atom": re.sub(r"\d+$", "", site),
                        "metal_site_label": met_label,
                        "coord_atom_site_label": site,
                        "bond": f"{met_label}-{site}",
                        "distance": distance,
                        "status": "missing_in_cell2mol",
                        "bond_change": "found_other_molecule",
                        "removed": False,
                        "final_match": final_match,
                        "final_match_wo_mm": final_match_wo_mm,
                        "haptic_type": None,
                        "is_haptic": False,
                        "n_metals": len(molecule.metals),
                    }
                )

            # Normal comparison rows
            for atom in molecule.atoms:
                site = atom.atom_site_label
                if not site:
                    continue
                if site in missing:
                    add_row(
                        atom,
                        site,
                        "missing_in_cell2mol",
                        "removed_but_expected" if site in set_removed else "missing",
                    )
                elif site in extra:
                    add_row(atom, site, "extra_in_cell2mol", "extra")
                elif site in set_current and site in set_data:
                    add_row(atom, site, "match", "kept")

            # Removed atom trace rows
            for atom in removed_atoms:
                site = atom.atom_site_label
                if not site:
                    continue
                if site in set_data and site not in set_current:
                    status, bond_change = "missing_in_cell2mol", "removed_but_expected"
                elif site not in set_data and site not in set_current:
                    status, bond_change = (
                        "removed_from_coordination",
                        "removed_extra_contact",
                    )
                elif site in set_current:
                    status, bond_change = (
                        "removed_but_still_current",
                        "inconsistent_removed_record",
                    )
                else:
                    status, bond_change = "removed_from_coordination", "removed_unknown"
                add_row(atom, site, status, bond_change)

            for site in missing:
                if site not in set_removed and site not in {
                    a.atom_site_label for a in molecule.atoms
                }:
                    logger.warning(
                        "Atom %s expected in bond_data but missing in current molecule %s.",
                        site,
                        molecule.formula,
                    )
                    add_completely_missing_row(site)

    logger.info("Coordination report for %s: %d records.", refcode, len(report))
    if report_csv is not None:
        save_coordination_report(report, report_csv)
    return overall_difference
