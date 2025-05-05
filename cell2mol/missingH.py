#!/usr/bin/env python

import numpy as np
from cell2mol.other import get_angle, get_dist

##############################
def unit_vector(v):
    return v / np.linalg.norm(v)

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
    if val_e < shapeval:  missingH = True
    if val_e == shapeval: missingH = False
    if val_e > shapeval:  missingH = True

    # Saves report
    report += str(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs. \n")
    report += report_shape
 
    return missingH, report


##############################
def get_missingH_from_adjacency(Z, center, points, bonded_atom_labels):
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
    if val_e == 1:
        if bonded_atom_labels[0] == "O" or bonded_atom_labels[0] == "N": # CO or CN
            missingH = False
        else:
            shapeval = "more than 1 (possibly missing H in methyl)"
            missingH = True
    elif val_e == shapeval :  
        missingH = False
    elif val_e > shapeval:  # carbon has more than 4 bonds       
        missingH = False
    else : # val_e < shapeval
        missingH = True

    # Saves report
    # print(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs.")
    # report += str(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -with total bond order {sum_bond_order} \n -arranged in a shape {shape} that suggests coordination {shapeval} \n -with formal charge {charge} \n -valence {valence} and {lonepairs} lone pairs. \n")
    report += str(f"Summary of facts:\n -Atom has {num_adj_atoms} adjacent atoms \n -arranged in a shape {shape} that suggests coordination {shapeval} \n")
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
                    tmp = get_angle(a, b)
                    if tmp not in angles:    angles.append(tmp)
        avg_angle = np.mean(angles)
        report_shape += str(f"Angles {angles} Avg: {avg_angle}\n")
        report_shape += str(f"Angles {np.degrees(angles)} Avg: {np.degrees(avg_angle)}\n")
        diffs = list([np.abs(avg_angle - np.pi),np.abs(avg_angle - 2.094395),np.abs(avg_angle - 1.570796),np.abs(avg_angle - 1.911136)])
        minshape = np.argmin(diffs)
        minval = np.min(diffs)
        report_shape += str(f"Diffs: {diffs} Minval: {minval}\n")
        if minval <= atol:
            if minshape == 0:     shape = "Linear";       shapeval = 2
            elif minshape == 1:   shape = "Triangular";   shapeval = 3
            elif minshape == 2:   shape = "SquarePlanar"; shapeval = 4
            elif minshape == 3:   shape = "Tetrahedron";  shapeval = 4
    return shape, shapeval, report_shape

######################
def check_missingH(refmoleclist: list, debug: int=0):

    Missing_H_in_C = False
    Missing_H_in_Water = False
    Missing_H_in_CoordWater = False
    ismissingH = False
    Warning = False

    # List of Metal Atoms for which O atoms might appear connected directly.
    Exceptions_for_CoordWater = ["Re", "V", "Mo", "W", "Fe", "Tc"]

    # List of fullerenes
    fullerene = ["C60", "C72", "C80"]

    if debug >= 1: print("")
    if debug >= 1: print("##################")
    if debug >= 1: print("Checking Missing H")
    if debug >= 1: print("##################")
    for idx, ref in enumerate(refmoleclist):
        if not ref.iscomplex and not ref.has_IA_IIA:
            if ref.natoms == 1 and "O" in ref.labels: 
                Missing_H_in_Water = True
                # Missing_H_in_CoordWater = True
                if debug >= 1: print(f"WARNING found isolated O atom in the cell. This tends to be a water with missing H, so stopping")
            elif ref.formula == "CO" or ref.formula == "CN":
                pass
            elif ref.formula in fullerene:
                if debug >= 1: print(f"Found fullerene {ref.formula} in the cell. skipping missing H check in carbon atoms")
            else:
                for kdx, a in enumerate(ref.atoms):
                    if not hasattr(a,"adjacency"): continue 
                    if a.label == "C":
                        bonded_atom_coord = []
                        bonded_atom_labels = []
                        
                        for adj in a.adjacency:
                            bonded_atom_coord.append(ref.coord[adj])
                            bonded_atom_labels.append(ref.atoms[adj].label)
                        if debug >= 2: print("Adjacency", a.adjacency, bonded_atom_labels)
                        ismissingH, report = get_missingH_from_adjacency(a.atnum, a.coord, bonded_atom_coord, bonded_atom_labels)
                        if ismissingH:
                            for label, coord in zip(bonded_atom_labels, bonded_atom_coord):
                                print("Dist", f"{a.label}-{label}", get_dist(a.coord, coord))
                        if ismissingH:
                            if debug >= 1: print("")
                            if debug >= 1: print(f"WARNING in Missing H function for: {ref.type}, molecule index {idx}, {ref.formula}")
                            if debug >= 1: print(f"C Atom {kdx} {a.get_parent_index('molecule')} ref_idx={a.get_parent_index('reference')} has missing H atoms")
                            if debug >= 1: print(report)
                            Missing_H_in_C = True
        else:
            for jdx, lig in enumerate(ref.ligands):
                if lig.natoms == 1 and "O" in lig.labels and lig.denticity <= 1:
                    if any(m.label in Exceptions_for_CoordWater for m in lig.metals): pass
                    # else:
                    #     Missing_H_in_CoordWater = True
                    #     if debug >= 1: print("")
                    #     if debug >= 1: print("WARNING in Missing H function for ligand", lig.natoms, lig.labels)
                elif lig.formula == "CO" or lig.formula == "CN":
                    pass
                elif ref.formula in fullerene:
                    pass
                    if debug >= 1: print(f"Found fullerene {ref.formula} in the cell. skipping missing H check in carbon atoms")               
                else:
                    for kdx, a in enumerate(lig.atoms):
                        if a.label == "C" and a.mconnec == 0:
                            bonded_atom_coord = []
                            bonded_atom_labels = []
                            
                            for adj in a.adjacency:
                                bonded_atom_coord.append(lig.get_parent("molecule").coord[adj])
                                bonded_atom_labels.append(lig.get_parent("molecule").atoms[adj].label)
                            if debug >= 2: print("Adjacency", a.adjacency, bonded_atom_labels)
                            ismissingH, report = get_missingH_from_adjacency(a.atnum, a.coord, bonded_atom_coord, bonded_atom_labels)
                            if ismissingH:
                                print(a.label, a.mconnec, a.coord)
                                if debug >= 1: print("")
                                if debug >= 1: print(f"WARNING in Missing H function for: {ref.type}, molecule index {idx}, ligand index {jdx}, {lig.formula}")
                                if debug >= 1: print(f"C Atom {kdx} {a.get_parent_index('molecule')} ref_idx={a.get_parent_index('reference')} has missing H atoms")
                                if debug >= 1: print(report)
                                Missing_H_in_C = True

    if Missing_H_in_C or Missing_H_in_CoordWater or Missing_H_in_Water :  Warning = True
    if not Warning:
        if debug >= 1: print("Not a Single Molecule has Missing H atoms (apparently)")

    return Warning, ismissingH, Missing_H_in_C, Missing_H_in_CoordWater, Missing_H_in_Water
