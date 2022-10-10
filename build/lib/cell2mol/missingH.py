#!/usr/bin/env python

import numpy as np

##############################
def unit_vector(v):
    return v / np.linalg.norm(v)


##############################
def getangle(vec1, vec2):

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    dotprod = np.dot(vec1, vec2)
    factor = dotprod / (norm1 * norm2)
    angle = np.arccos(factor)

    if np.isnan(angle):
        print("GET_ANGLE nan Problem", norm1, norm2, dotprod, factor, angle)
        print("GET_ANGLE nan Problem, vecs:", vec1, vec2)

    return float(angle)


##############################
def get_missingH(Z, valence, center, charge, edges, points):

    missingH = False

    sum_bond_order = np.sum(edges)
    lonepairs = (valence - charge - sum_bond_order) / 2
    num_adj_atoms = len(edges)
    report = ""

    # Creates and sends Vectors
    vecs = []
    for coord in points:
        vec = np.subtract(coord, center)
        vecs.append(unit_vector(vec))
    shape, shapeval, report_shape = find_shape(vecs)

    # Evaluates geometry
    val_e = num_adj_atoms  # - charge
    if val_e < shapeval:
        missingH = True
    if val_e == shapeval:
        missingH = False
    if val_e > shapeval:
        missingH = True

    # Saves report
    # print(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs.")
    report += str(
        f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs. \n"
    )
    report += report_shape

    #missingH = False
 
    return missingH, report


##############################
def get_missingH_from_adjacency(Z, center, points):

    missingH = False

    num_adj_atoms = len(points)
    report = ""

    # Creates and sends Vectors
    vecs = []
    for coord in points:
        vec = np.subtract(coord, center)
        vecs.append(unit_vector(vec))
    shape, shapeval, report_shape = find_shape(vecs)

    # Evaluates geometry
    val_e = num_adj_atoms
    if val_e < shapeval:
        missingH = True
    if val_e == shapeval:
        missingH = False
    if val_e > shapeval:
        missingH = True

    # Saves report
    # print(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs.")
    # report += str(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs. \n")
    report += str(
        f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -arranged in a shape {shape} that suggests coordination {shapeval} \n"
    )
    report += report_shape

    #missingH = False

    return missingH, report


######################
def find_shape(vecs):

    atol = 4e-1
    shape = "Unassigned"
    shapeval = 0
    report_shape = ""

    if len(vecs) == 1:
        shape = "Point"
        shapeval = 1
    else:
        angles = []
        for idx, a in enumerate(vecs):
            for jdx, b in enumerate(vecs):
                if idx != jdx:
                    # print("sending:",idx, jdx, a, b)
                    tmp = getangle(a, b)
                    if tmp not in angles:
                        angles.append(tmp)
                        # print(idx, jdx, tmp, "added")

        avg_angle = np.mean(angles)
        report_shape += str(f"Angles {angles} Avg: {avg_angle}\n")
        report_shape += str(
            f"Angles {np.degrees(angles)} Avg: {np.degrees(avg_angle)}\n"
        )

        diffs = list(
            [
                np.abs(avg_angle - np.pi),
                np.abs(avg_angle - 2.094395),
                np.abs(avg_angle - 1.570796),
                np.abs(avg_angle - 1.911136),
            ]
        )

        minshape = np.argmin(diffs)
        minval = np.min(diffs)

        report_shape += str(f"Diffs: {diffs} Minval: {minval}\n")

        if minval <= atol:
            if minshape == 0:
                shape = "Linear"
                shapeval = 2
            elif minshape == 1:
                shape = "Triangular"
                shapeval = 3
            elif minshape == 2:
                shape = "SquarePlanar"
                shapeval = 4
            elif minshape == 3:
                shape = "Tetrahedron"
                shapeval = 4

    return shape, shapeval, report_shape


######################

def check_missingH(refmoleclist: list, debug: int=0):

    Missing_H_in_C = False
    Missing_H_in_CoordWater = False
    ismissingH = False
    Warning = False

    # List of Metal Atoms for which O atoms might appear connected directly.
    Exceptions_for_CoordWater = ["Re"]

    if debug >= 2: print("")
    if debug >= 2: print("##################")
    if debug >= 2: print("Checking Missing H")
    if debug >= 2: print("##################")
    for idx, ref in enumerate(refmoleclist):
        if ref.type != "Complex":
            if ref.natoms == 1 and "O" in ref.labels: 
                Missing_H_in_CoordWater = True
                if debug >= 2: print(f"WARNING found isolated O atom in the cell. This tends to be a water with missing H, so stopping")
            else:
                for kdx, a in enumerate(ref.atoms):
                    if a.label == "C":
                        bonded_atom_coord = []
                        for adj in a.adjacency:
                            bonded_atom_coord.append(ref.coord[adj])
                        ismissingH, report = get_missingH_from_adjacency(a.atnum, a.coord, bonded_atom_coord)
                        if ismissingH:
                            if debug >= 2: print("")
                            if debug >= 2: print(f"WARNING in Missing H function for: {ref.type}, {idx}, {ref.labels}")
                            if debug >= 2: print(f"C Atom {kdx} has missing H atoms")
                            if debug >= 2: print(report)
                            Missing_H_in_C = True
        elif ref.type == "Complex":
            for jdx, lig in enumerate(ref.ligandlist):
                if lig.natoms == 1 and "O" in lig.labels and lig.totmconnec <= 1:
                    if any(m.label in Exceptions_for_CoordWater for m in lig.metalatoms): pass
                    else:
                        Missing_H_in_CoordWater = True
                        if debug >= 2: print("")
                        if debug >= 2: print("WARNING in Missing H function for ligand",lig.natoms,lig.labels)
                else:
                    for kdx, a in enumerate(lig.atoms):
                        if a.label == "C" and a.mconnec == 0:
                            bonded_atom_coord = []
                            for adj in a.adjacency:
                                bonded_atom_coord.append(lig.coord[adj])
                            ismissingH, report = get_missingH_from_adjacency(a.atnum, a.coord, bonded_atom_coord)
                            if ismissingH:
                                if debug >= 2: print("")
                                if debug >= 2: print(f"WARNING in Missing H function for: {ref.type}, {idx}, {jdx}, {lig.labels}")
                                if debug >= 2: print(f"Atom {kdx} has missing H atoms")
                                if debug >= 2: print(report)
                                Missing_H_in_C = True

    if Missing_H_in_C or Missing_H_in_CoordWater:
        Warning = True

    if not Warning:
        if debug >= 2: print("Not a Single Molecule has Missing H atoms (apparently)")

    return Warning, ismissingH, Missing_H_in_C, Missing_H_in_CoordWater
