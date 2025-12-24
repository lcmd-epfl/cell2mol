#!/usr/bin/env python

import numpy as np


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
        debug (int): debug verbosity level

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


def compute_centroid(arr: np.array) -> list:
    """Compute the centroid of a set of 3D coordinates.

    Args:
        arr (np.array): array of shape (N, 3)

    Returns:
        np.array: centroid coordinates
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
        print("GET_ANGLE nan Problem", norm1, norm2, dotprod, factor, angle)
        print("GET_ANGLE nan Problem, vecs:", vec1, vec2)
    return float(angle)


def get_unit_vector(v):
    """Normalize a vector to unit length.

    Args:
        v : input vector

    Returns:
        np.array: unit vector in the same direction
    """
    return v / np.linalg.norm(v)
