#!/usr/bin/env python

import numpy as np

#######################################################
def frac2cart_fromcellvec(frac_coord, cellvec):
    """ Convert fractional coordinates to cartesian coordinates
    Args:
        frac_coord (list): list of fractional coordinates
        cellvec (list): list of cell vectors
    Returns:
        cartesian (list): list of cartesian coordinates
    """
    cartesian = []
    for idx, frac in enumerate(frac_coord):
        xcar = (frac[0] * cellvec[0][0] + frac[1] * cellvec[1][0] + frac[2] * cellvec[2][0])
        ycar = (frac[0] * cellvec[0][1] + frac[1] * cellvec[1][1] + frac[2] * cellvec[2][1])
        zcar = (frac[0] * cellvec[0][2] + frac[1] * cellvec[1][2] + frac[2] * cellvec[2][2])
        cartesian.append([float(xcar), float(ycar), float(zcar)])
    return cartesian

#######################################################
def frac2cart_fromparam(frac_coord, cellparam):
    """ Convert fractional coordinates to cartesian coordinates
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

    volume = (a*b*c* np.sqrt(1- np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)))

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

############################;###########################
def translate(vector, coords, cellvec):
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

#######################################################
def cart2frac(cartCoords, cellvec):
    """ Translate coordinates by a vector
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
