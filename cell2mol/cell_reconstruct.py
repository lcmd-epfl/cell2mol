#!/usr/bin/env python

import os
import numpy as np
import sys
import networkx as nx
import copy
import scipy
import time
import itertools
import random
import math
from math import fsum
import pickle

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

import cellconversions
from cellconversions import frac2cart_fromparam
from cellconversions import cart2frac
from cellconversions import det3
from cellconversions import translate


import tmcharge_common
from tmcharge_common import getelementcount
from tmcharge_common import getHvcount
from tmcharge_common import get_adjacency_types
from tmcharge_common import getradii
from tmcharge_common import getcentroid
from tmcharge_common import find_groups_within_ligand
from tmcharge_common import checkchemistry
from tmcharge_common import getconec
from tmcharge_common import getblocks
from tmcharge_common import inv
from tmcharge_common import extract_from_matrix

# Imports Classes
import tmcharge_common
from tmcharge_common import atom
from tmcharge_common import molecule
from tmcharge_common import ligand
from tmcharge_common import metal
from tmcharge_common import group

from elementdata import ElementData

elemdatabase = ElementData()


def get_reference_molecules(labels, pos):

    debug = 1
    Warning = False

    # Tries to adjust factor of the covalent radii
    found_covalent_factor = False
    max_covalent_factor = 1.6
    min_covalent_factor = 1.3
    increase_covalent_factor = 0.02

    found_metal_factor = False
    max_metal_factor = 1.2
    min_metal_factor = 0.8
    change_metal_factor = 0.02

    covalent_factor = min_covalent_factor
    metal_factor = 1.0

    # Initiates while that adjusts the two factors (metal and covalent)
    found_both_factors = False
    maxiter = 20
    iteration = 1
    # execute as long as found_both_factor = False, iteration <= maxiter, and Warning = False
    while (not found_both_factors) and (iteration <= maxiter) and not Warning:

        # Tries to find the reference molecules
        print("")
        if debug == 1:
            print(
                "GETREFS: sending listofreferences with", covalent_factor, metal_factor
            )

        Warning, listofreferences = getmolecs(
            labels, pos, covalent_factor, metal_factor
        )

        # Condition to accept the covalent_factor:
        valid_list_of_references = True

        for ref in listofreferences:
            if ref.natoms == 1:
                if (
                    (
                        elemdatabase.elementgroup[ref.atoms[0].label]
                        and ref.atoms[0].label != "H"
                    )
                    or elemdatabase.elementgroup[ref.atoms[0].label] == 2
                    or elemdatabase.elementgroup[ref.atoms[0].label] == 17
                ):
                    pass
                else:
                    if debug == 1:
                        print(
                            "GETREFS: found ref molecule with only one atom", ref.labels
                        )
                    valid_list_of_references = False

        if not valid_list_of_references:
            if covalent_factor < max_covalent_factor:
                found_covalent_factor = False
                covalent_factor += increase_covalent_factor
                if debug == 1:
                    print("GETREFS: Increasing covalent_factor to:", covalent_factor)
            else:
                print("GETREFS: Reached Maximum Covalent_factor:", max_covalent_factor)
                Warning = True
        else:
            found_covalent_factor = True

        # Condition to accept the metal_factor. Runs for all complexes in the list of reference molecules:
        glist = []
        ilist = []
        dlist = []
        for ref in listofreferences:
            if ref.type == "Complex":
                # print("sending", ref.type, ref.natoms, ref.labels)
                ref.ligandlist, ref.metalist = splitcomplex(
                    ref, covalent_factor, metal_factor
                )

                for lig in ref.ligandlist:
                    print(
                        "Lig in Ref molec", lig.natoms, lig.labels, lig.totmconnec
                    )  # , len(lig.metalatoms))

                # Checks Hapticity
                potential_hapticity = get_hapticity(ref)
                print("Potential hapticity=", potential_hapticity)

                # Checks Shared Ligands in Polymetallic complexes
                if any(len(lig.metalatoms) >= 2 for lig in ref.ligandlist):
                    ispolymetallic_and_shared = True
                    print("Molecule is polymetallic and has shared ligands")
                else:
                    ispolymetallic_and_shared = False

                if (
                    not potential_hapticity and not ispolymetallic_and_shared
                ):  # then tries to adjust the metal_factor
                    for a in ref.atoms:
                        if a.block == "d" or a.block == "f":
                            if debug == 1:
                                print(
                                    "GETREFS: sending",
                                    a.label,
                                    a.mconnec,
                                    "to coordcheck",
                                )
                            good, increase, decrease = metalcoordcheck(
                                a.label, a.mconnec
                            )
                            glist.append(good)
                            ilist.append(increase)
                            dlist.append(decrease)
                            if debug == 1:
                                print(
                                    "GETREFS: received",
                                    good,
                                    increase,
                                    decrease,
                                    "from coordcheck",
                                )
                else:
                    glist.append(True)
                    ilist.append(False)
                    dlist.append(False)

        if len(glist) > 0:
            if (
                any((item == True for item in ilist))
                and all((item2 == False for item2 in dlist))
                and (metal_factor < max_metal_factor)
            ):
                metal_factor += change_metal_factor
                if debug == 1:
                    print("GETREFS: Increasing metal_factor to:", metal_factor)
            if (
                all((item == False for item in ilist))
                and any((item2 == True for item2 in dlist))
                and (metal_factor > min_metal_factor)
            ):
                metal_factor -= change_metal_factor
                if debug == 1:
                    print("GETREFS: Decreasing metal_factor to:", metal_factor)
            if all((item == True for item in glist)):
                found_metal_factor = True
                if debug == 1:
                    print("GETREFS: Metal_factor set at:", metal_factor)

        if found_covalent_factor and found_metal_factor:
            found_both_factors = True
            Warning = False
            print("GETREFS: Found both factors. Breaking")
            break
        elif (metal_factor > max_metal_factor) or (metal_factor < min_metal_factor):
            print("GETREFS: metal_factor outside the limits", metal_factor)
            Warning = True
            break
        else:
            iteration += 1
            continue

        if iteration == maxiter:
            Warning = True

    # if (debug == 1): print("GETREFS: final lists G/I/D:", glist, ilist, dlist)
    return listofreferences, covalent_factor, metal_factor, Warning


#######################################################
def metalcoordcheck(label, coordination):

    debug = 0

    from collections import defaultdict

    # Data Obtained from:
    # Venkataraman, D.; Du, Y.; Wilson, S. R.; Hirsch, K. A.; Zhang, P.; Moore, J. S. A
    # Coordination Geometry Table of the D-Block Elements and Their Ions.
    # J. Chem. Educ. 1997, 74, 915.

    good = False
    increase = False
    decrease = False

    # atnum = int_atom(label)
    # print("Metalchargecheck:",label[0])
    atnum = elemdatabase.elementnr[label]
    if debug == 1:
        print("Metalcoordcheck function: got atnum", atnum, "for label", label)

    coordnum = defaultdict(list)
    # adding 1st-row transition metals.
    coordnum[21] = [6]  # Sc
    coordnum[22] = [3, 4, 6]  # Ti
    coordnum[23] = [3, 4, 5, 6]  # V
    coordnum[24] = [4, 5, 6]  # Cr
    coordnum[25] = [
        3,
        4,
        5,
        6,
    ]  # Mn #some strange cases of heptacoordination in Mn exist. Not sure is a good idea to have it
    coordnum[26] = [3, 4, 5, 6]  # Fe
    coordnum[27] = [3, 4, 6]  # Co
    coordnum[28] = [3, 4, 5, 6]  # Ni
    coordnum[29] = [3, 4, 5, 6]  # Cu
    coordnum[30] = [3, 4, 6]  # Zn
    # 2nd-row transition metals.
    coordnum[39] = [6]  # Y
    coordnum[40] = [4, 6]  # Zr
    coordnum[41] = [4, 6]  # Nb
    coordnum[42] = [4, 5, 6]  # Mo
    coordnum[43] = [5, 6]  # Tc
    coordnum[44] = [3, 4, 5, 6]  # Ru
    coordnum[45] = [4, 5, 6]  # Rh
    coordnum[46] = [4, 5]  # Pd
    coordnum[47] = [2, 3, 4]  # Ag
    coordnum[48] = [4, 6]  # Cd
    # 3rd-row transition metals.
    coordnum[57] = []  # La
    coordnum[72] = [6]  # Hf
    coordnum[73] = [5, 6]  # Ta
    coordnum[74] = [4, 5, 6]  # W
    coordnum[75] = [4, 5, 6]  # Re
    coordnum[76] = [4, 5, 6]  # Os
    coordnum[77] = [3, 4, 5, 6]  # Ir
    coordnum[78] = [4, 5, 6]  # Pt
    coordnum[79] = [2, 4]  # Au
    coordnum[80] = [2, 3, 4, 5]  # Hg

    if len(coordnum[atnum]) == 0:
        print(
            "Metalcoordcheck function: Atom with label",
            label,
            "has an empty list of possible coordination",
        )

    if any((coordination == c) for c in coordnum[atnum]):
        good = True
        increase = False
        decrease = False
    elif coordination > np.max(coordnum[atnum]):
        good = False
        increase = False
        decrease = True
    elif coordination < np.min(coordnum[atnum]):
        good = False
        increase = True
        decrease = False
    elif (
        all((coordination != c) for c in coordnum[atnum])
        and (coordination <= np.max(coordnum[atnum]))
        and (coordination >= np.min(coordnum[atnum]))
    ):
        good = False
        increase = True
        decrease = False
    else:
        print(
            "Metalcoordcheck function: Atom with label",
            label,
            "has strange coordination value:",
            coordination,
            coordnum[atnum],
        )

    return good, increase, decrease


#######################################################
def getmolecs(
    labels, pos, factor=1.3, metal_factor=1.0
):  ##Simplified Version of the getmolecs

    Warning = False
    debug = 0
    
    # Gets the covalent radii, and modifies that of the metal if necessary
    radii = getradii(labels)

    if metal_factor != 1.0:
        for idx, r in enumerate(radii):
            if (
                elemdatabase.elementblock[labels[idx]] == "d"
                or elemdatabase.elementblock[labels[idx]] == "f"
            ):
                radii[idx] = (
                    r * metal_factor
                )  # the covalent radii of the metal is modified
                if debug >= 1:
                    print("GETMOLECS: new radii for", labels[idx], radii[idx])

    # Computes the adjacency matrix of what is received
    status, conmat, connec, mconmat, mconnec = getconec(labels, pos, factor, radii)

    if status == 1:
        Warning = False
        degree = np.diag(
            connec
        )  # creates a matrix with connec as diagonal values. Needed for the laplacian
        lap = conmat - degree  # computes laplacian

        # creates block matrix
        graph = csr_matrix(lap)
        perm = reverse_cuthill_mckee(graph)
        gp1 = graph[perm, :]
        gp2 = gp1[:, perm]
        dense = gp2.toarray()

        # detects blocks in the block diagonal matrix called "dense"
        startlist, endlist = getblocks(dense)

        nmolec = len(startlist)

        # keeps track of the atom movement within the matrix. Needed later
        atomlist = np.zeros((len(dense)))
        for b in range(0, nmolec):
            for i in range(0, len(dense)):
                if (i >= startlist[b]) and (i <= endlist[b]):
                    atomlist[i] = b + 1
        invperm = inv(perm)
        atomlistperm = [int(atomlist[i]) for i in invperm]

        # assigns atoms to molecules
        mlist = []
        for b in range(0, nmolec):
            fraglist = []
            labelist = []
            poslist = []
            radiilist = []

            # print("doing", b)
            for i in range(0, len(atomlistperm)):
                if atomlistperm[i] == b + 1:
                    fraglist.append(i)
                    labelist.append(labels[i])
                    poslist.append(pos[i])
                    radiilist.append(radii[i])

            # Generates conmat and mconmat for the molecule
            nidx = 0
            njdx = 0
            nat = len(fraglist)
            #             print(f"length of fragment is {nat}\n")
            conmatlist = np.empty((nat, nat))
            mconmatlist = np.empty((nat, nat))
            for idx in range(0, len(atomlistperm)):
                if atomlistperm[idx] == b + 1:
                    for jdx in range(0, len(atomlistperm)):
                        if atomlistperm[jdx] == b + 1:
                            #                             print(nidx,njdx,idx,jdx)
                            conmatlist[nidx, njdx] = conmat[idx, jdx]
                            mconmatlist[nidx, njdx] = mconmat[idx, jdx]
                            njdx += 1
                    njdx = 0
                    nidx += 1

            # Creates the objects
            molec = molecule(
                b, fraglist, labelist, poslist, radiilist
            )  # Creates Object Molecule
            molec.information(
                factor, metal_factor
            )  # Creates Information about the construction
            molec.adjacencies(
                conmatlist, mconmatlist
            )  # Creates the Connectivity Information
            mlist.append(molec)  # Appends it to the final list of molecules
    #  status = 1 #good molecule, no clashes yet
    if status == 0:
        Warning = True
        mlist = []
        print("GETMOLECS: steric clashes found. Printing Molecule")
        for idx, lab in enumerate(labels):
            # print(lab, pos[idx])
            print(
                "%s   %.6f   %.6f   %.6f" % (lab, pos[idx][0], pos[idx][1], pos[idx][2])
            )
        print("")

    return Warning, mlist


#######################################################
def splitcomplex(molecule, factor=1.3, metal_factor=1.0):  ##Similar to getmolecs

    if hasattr(molecule, "factor"):
        factor = molecule.factor

    if hasattr(molecule, "metal_factor"):
        metal_factor = molecule.metal_factor

    origatoms = molecule.natoms
    # variables for complex without metal (metalfree)
    mfreeradii = []
    mfreelabels = []
    mfreepos = []
    mfreeconnec = []
    mfreeatlist = []

    # variables for metal atoms
    metalist = []
    matoms = []

    # Splits the variables into metal and metal-free
    number_of_metal_atoms = 0
    for idx, a in enumerate(molecule.atoms):
        if a.block == "d" or a.block == "f":
            matoms.append(
                a
            )  # This information is for the ligands. To generate lig.metalatoms
            number_of_metal_atoms += 1
            met = metal(number_of_metal_atoms, idx, a.label, a.coord, a.radii)
            met.information(factor, metal_factor)

            # Extracts the metal adjacency from the molecule.adjacency matrix
            tmp_mconnec = extract_from_matrix(list([idx]), molecule.mconnec, 1)
            tmp_mconnec = tmp_mconnec.astype(int)
            met.adjacencies(tmp_mconnec)
            metalist.append(met)

            connec_atoms_label = (
                []
            )  # collects the labels of all atoms connected to this metal
            # print(molecule.mconmat)
            for jdx, at2 in enumerate(molecule.atoms):
                if molecule.mconmat[idx, jdx] == 1:
                    connec_atoms_label.append(str(at2.label))
            met.coord_sphere = connec_atoms_label
            met.coord_sphere_ID = getelementcount(connec_atoms_label)

        else:
            mfreelabels.append(a.label)
            mfreepos.append(a.coord)
            mfreeconnec.append(a.mconnec)
            mfreeradii.append(a.radii)
            mfreeatlist.append(idx)

    # print("SPLIT: Metal and ligand atoms:", len(mlabels), len(mfreelabels), mfreeradii)
    # Uses the Metal-free coordinates to find the ligands. Notice that, when creating their metal connectivity, it uses that of the original molecule
    status, conmat, connec, dummy, dummy = getconec(
        mfreelabels, mfreepos, factor, mfreeradii
    )

    if status == 1:
        degree = np.diag(
            connec
        )  # creates a matrix with connec as diagonal values. Needed for the laplacian
        lap = conmat - degree  # computes laplacian

        graph = csr_matrix(lap)
        perm = reverse_cuthill_mckee(graph)
        gp1 = graph[perm, :]
        gp2 = gp1[:, perm]
        dense = gp2.toarray()

        startlist, endlist = getblocks(dense)
        nmolec = len(startlist)

        atomlist = np.zeros((len(dense)))
        for b in range(0, nmolec):
            for i in range(0, len(dense)):
                if (i >= startlist[b]) and (i <= endlist[b]):
                    atomlist[i] = b + 1

        invperm = inv(perm)
        atomlistperm = [int(atomlist[i]) for i in invperm]

        ligandlist = []
        for b in range(0, nmolec):
            atlist = []
            labelist = []
            poslist = []
            radiilist = []
            for i in range(0, len(atomlistperm)):
                if atomlistperm[i] == b + 1:
                    atlist.append(mfreeatlist[i])
                    labelist.append(mfreelabels[i])  # replace by molecule.labels
                    poslist.append(mfreepos[i])
                    radiilist.append(mfreeradii[i])

            tmp_conmat = extract_from_matrix(atlist, molecule.conmat, dimension=2)
            tmp_mconnec = extract_from_matrix(atlist, molecule.mconnec, dimension=1)
            tmp_conmat = tmp_conmat.astype(int)
            tmp_mconnec = tmp_mconnec.astype(int)

            # Creates the Ligands
            lig = ligand(
                b, atlist, labelist, poslist, radiilist
            )  # Creates Object Molecule
            lig.information(
                factor, metal_factor
            )  # Creates Information about the construction
            lig.adjacencies(
                tmp_conmat, tmp_mconnec
            )  # Creates the Adjacency Information
            for a in matoms:  # only adds metal atoms that are connected to the ligand
                for idx, ligat in enumerate(lig.atoms):
                    if lig.atlist[idx] in a.adjacency:
                        lig.metalatoms.append(
                            a
                        )  # Saves Metal-Atom Information to the Ligand Object
            ligandlist.append(lig)  # Appends it to the final list of ligand

    finatoms = 0
    for lig in ligandlist:
        finatoms += lig.natoms
    for met in metalist:
        finatoms += met.natom

    if origatoms != finatoms:
        print(
            "WARNING: different initial and final atoms in splitcomplex, for molecule",
            molecule.natoms,
            molecule.labels,
        )

    return ligandlist, metalist


#######################################################
def additem(item, vector):
    if item not in vector:
        vector.append(item)
    return vector


#######################################################
def absolute_value(num):
    sum = 0
    for i in num:
        sum += np.abs(i)
    return abs(sum)


#######################################################
def tmatgenerator(centroid, thres=0.40, full=False):

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
                    # elif (centroid[2] <= tmin):
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
        import itertools

        x = [-1, 0, 1]
        tmatrix = [p for p in itertools.product(x, repeat=3)]

    tmatrix.sort(key=absolute_value)
    #     if (thres == 1.0):
    #         print("TMATGENERATOR: centroid and tmatrix:", centroid, tmatrix)

    return tmatrix


#######################################################
def sequential(
    fragmentlist, refmoleclist, cellvec, debug, factor, metal_factor, typ="All"
):

    if debug == 1:
        print("Entered sequential with", len(fragmentlist), "fragments to reconstruct")

    # Finds How many atoms, at max, can a molecule have. It is used to skip meaningless combinations
    maxatoms = 0
    for ref in refmoleclist:
        if ref.natoms > maxatoms:
            maxatoms = ref.natoms

    molecsfoundlist = []
    remainingfragments = []
    ###################################################
    #### INPUT THAT CONTROLS THE ITERATIVE PROCESS ####
    ###################################################
    threshold_tmat = 0.40
    increase_tmat = 0.20
    fragtoallocate = 0
    Htoallocate = 0
    niter = 1
    maxiter = 300
    mixsize = 1
    lastiter = 0
    lastitermargin = maxiter
    ###################################################

    for frag in fragmentlist:
        frag.tmatrix = tmatgenerator(frag.centroid, threshold_tmat)

    remlist = []
    Hlist = []
    for frag in fragmentlist:
        if (frag.natoms == 1) and (frag.numH == 1):
            frag.type = "H"
            Hlist.append(frag)
        else:
            frag.type = "Heavy"
            remlist.append(frag)

    if debug == 1:
        print(
            "Found",
            len(remlist),
            "and",
            len(Hlist),
            "Heavy and Hydrogen fragments, respectively",
        )

    if typ == "Heavy":
        list1 = remlist.copy()
        list2 = remlist.copy()
    elif typ == "All":
        list1 = remlist.copy()
        list2 = Hlist.copy()

    ## Initial Fragments
    Frag1_toallocate = 0
    Frag2_toallocate = 0

    #################
    ### MAIN LOOP ###
    #################
    while (len(list1) > 0) and (len(list2) > 0):

        ####
        # This part decides which molecules in the two lists are sent to combine
        STOP = False
        Last_Attempt = False

        if niter > 1:
            Frag2_toallocate += 1

        if (
            Frag2_toallocate > len(list2) - 1
        ):  # Reaches the end of the second list. Restarts it and moves forward in the first list
            Frag1_toallocate += 1
            Frag2_toallocate = 0

        if (
            Frag1_toallocate > len(list1) - 1
        ):  # Reaches the end of the first list. Restarts both
            Frag1_toallocate = 0
            Frag2_toallocate = 0

        if typ == "Heavy":
            if Frag1_toallocate == Frag2_toallocate:
                Frag2_toallocate += 1

        if typ == "Heavy":
            if (Frag1_toallocate >= len(list1) - 1) and (
                Frag2_toallocate >= len(list2) - 2
            ):
                STOP = True
        elif typ == "All":
            if (Frag1_toallocate >= len(list1) - 1) and (
                Frag2_toallocate >= len(list2) - 1
            ):
                STOP = True
        ####

        # if (STOP == False):
        if debug == 1:
            print(" ")
        if debug == 1:
            print(
                "Fragments to allocate this iteration:",
                Frag1_toallocate,
                Frag2_toallocate,
                "out of",
                len(list1) - 1,
                len(list2) - 1,
            )

        sublist = []
        keeplist1 = []
        keeplist2 = []
        if typ == "Heavy":
            for i in range(0, len(list1)):
                if i == Frag1_toallocate:
                    sublist.append(list1[i])
                elif i != Frag2_toallocate:
                    keeplist1.append(list1[i])
            for i in range(0, len(list2)):
                if i == Frag2_toallocate:
                    sublist.append(list2[i])
                elif i != Frag1_toallocate:
                    keeplist2.append(list2[i])
        elif typ == "All":
            for i in range(0, len(list1)):
                if i == Frag1_toallocate:
                    sublist.append(list1[i])
                elif i != Frag1_toallocate:
                    keeplist1.append(list1[i])
            for i in range(0, len(list2)):
                if i == Frag2_toallocate:
                    sublist.append(list2[i])
                elif i != Frag2_toallocate:
                    keeplist2.append(list2[i])

        if list1[Frag1_toallocate].natoms + list2[Frag2_toallocate].natoms <= maxatoms:

            if debug == 1:
                print(
                    "SEQUENTIAL",
                    typ,
                    "iteration",
                    niter,
                    "with",
                    len(list1),
                    "and",
                    len(list2),
                    "Remaining in each list",
                )
            if debug == 1:
                print(
                    "SEQUENTIAL",
                    typ,
                    "sending",
                    list1[Frag1_toallocate].labels,
                    "and",
                    list2[Frag2_toallocate].labels,
                    "to combine",
                )
            goodlist, avglist, badlist = combine(
                sublist, refmoleclist, cellvec, threshold_tmat, factor, metal_factor
            )

            if (len(goodlist) > 0) or (len(avglist) > 0):
                # it means that the function combine worked. Thus, it restarts the fragments to allocate
                lastiter = niter
                Frag1_toallocate = 0
                Frag2_toallocate = 0

            # Updates the Type of the molecules that are completely reconstructed
            for g in goodlist:
                g.type = assigntype(g, refmoleclist)
                molecsfoundlist.append(g)
            if len(goodlist) >= 1:
                print("SEQUENTIAL: Molecules found so far:")
                for i, item in enumerate(molecsfoundlist):
                    print(i, item.natoms, item.labels)

            # Reconstructs list1 and list2
            list1 = []
            list2 = []
            for a in avglist:
                list1.append(a)
                if typ == "Heavy":
                    list2.append(a)

            if len(badlist) > 1:
                if typ == "Heavy":
                    list1.append(badlist[0])
                    list1.append(badlist[1])
                    list2.append(badlist[0])
                    list2.append(badlist[1])
                elif typ != "Heavy":
                    list1.append(badlist[0])
                    list2.append(badlist[1])

            for k1 in keeplist1:
                list1.append(k1)

            for k2 in keeplist2:
                list2.append(k2)

            if len(list1) + len(list2) == 0:
                print("FINISHED succesfully")
                break

            if typ == "Heavy":
                if len(list1) == 1:
                    for l in list1:
                        remainingfragments.append(l)
                        print("FINISHED with Remaining Fragment")
                    break

            if (len(list1) == 0) and (len(list2) == 0):
                print("FINISHED succesfully")
                break

        else:
            if debug == 1:
                print(
                    "SEQUENTIAL",
                    typ,
                    "SKIPPED",
                    list1[Frag1_toallocate].natoms,
                    "and",
                    list2[Frag2_toallocate].natoms,
                )

        niter += 1
        if niter > maxiter:
            # print("STOP: Maximum Number of Iterations Reached")
            STOP = True
        if niter > (lastiter + lastitermargin):
            # print("STOP: Too many failed attempts since lastiter")
            STOP = True

        if STOP:
            if (threshold_tmat <= (1.0 - increase_tmat)) or Last_Attempt:
                STOP = False
                # threshold_tmat = fsum([threshold_tmat, incr])
                threshold_tmat += increase_tmat
                if threshold_tmat >= 1:
                    Last_Attempt = True
                    print("Launching Last Attempt")
                if not Last_Attempt:
                    maxsize = 0
                    for l in list1:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat)
                        if len(l.tmatrix) > maxsize:
                            maxsize = len(l.tmatrix)
                    for l in list2:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat)
                        if len(l.tmatrix) > maxsize:
                            maxsize = len(l.tmatrix)
                    print(" Increased Threshold_tmat. Now:", threshold_tmat)
                    print(" Maxsize of the translation matrix is=", maxsize)
                elif Last_Attempt:
                    for l in list1:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat, full=True)
                        if len(l.tmatrix) != 27:
                            print("error when generating the full tmatrix")
                    for l in list2:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat, full=True)
                        if len(l.tmatrix) != 27:
                            print("error when generating the full tmatrix")
                    print("Trying Full Tmatrix for all Items in list")

                niter = 1
                Frag1_toallocate = 0
                Frag2_toallocate = 0

            else:
                for l in list1:
                    print("Sequential: list1 end:", l.labels)
                    # print("Sequential: list1 end:", l.centroid, l.tmatrix)
                    remainingfragments.append(l)
                for l in list2:
                    if typ == "All":
                        print("Sequential: list2 end:", l.labels)
                    # print("Sequential: list2 end:", l.centroid, l.tmatrix)
                    if typ == "All":
                        remainingfragments.append(l)
                break
        else:
            continue

    return molecsfoundlist, remainingfragments


#######################################################
def combine(
    tobeallocated, references, cellvec, threshold_tmat, factor, metal_factor, debug=0
):

    goodlist = []
    avglist = []
    badlist = []
    available = np.ones((len(tobeallocated)))

    mergedatoms = 0  # SERGI
    for mol in tobeallocated:
        mergedatoms += mol.natoms  # SERGI
        # print("COMBINE received molec with:", mol.natoms)

    combinations = [(1, 1)]
    # Main loop
    for idx, c in enumerate(combinations):
        if np.sum(c) <= np.sum(available):

            mergelist = []
            goodcombination = True
            for jdx, times in enumerate(c):
                if (times == 1) and (available[jdx] == 1):
                    mergelist.append(jdx)
                if (times == 1) and (available[jdx] == 0):
                    goodcombination = False

            if goodcombination:
                found, newmoleclist = reconstruct(
                    tobeallocated, mergelist, references, cellvec, factor, metal_factor
                )

                if found == 1:
                    for m in mergelist:
                        available[m] = 0

                    if newmoleclist[0].natoms != mergedatoms:
                        if debug == 1:
                            print(
                                "COMBINE WARNING: I sent",
                                mergedatoms,
                                "atoms but received a molecule with",
                                newmoleclist[0].natoms,
                            )

                    number = 0
                    newmolec = newmoleclist[0]
                    newmolec.frac = cart2frac(newmolec.coord, cellvec)
                    newmolec.centroid = getcentroid(newmolec.frac)
                    newmolec.tmatrix = tmatgenerator(newmolec.centroid, threshold_tmat)

                    # ASSIGNS NEWMOLEC TO EITHER:
                    #   - REC.FRAGMENT to continue reconstring later with H
                    #   - Molec or Complex
                    shit = 0
                    for ref in references:
                        if (
                            ref.elemcountvec == newmolec.elemcountvec
                        ).all() and shit == 0:
                            if (ref.adjtypes == newmolec.adjtypes).all():
                                shit = 1
                                newmolec.type = ref.type
                                goodlist.append(newmolec)
                                if debug == 1:
                                    print(
                                        "COMBINE: Fragment",
                                        newmolec.labels,
                                        "added to goodlist",
                                    )
                    if shit == 0:
                        newmolec.type = "Rec. Fragment"
                        avglist.append(newmolec)
                        if debug == 1:
                            print(
                                "COMBINE: Fragment", newmolec.labels, "added to avglist"
                            )

        else:
            for kdx, a in enumerate(available):
                if a == 1:
                    badlist.append(tobeallocated[kdx])
            break

        if idx == len(combinations) - 1:
            for kdx, a in enumerate(available):
                if a == 1:
                    badlist.append(tobeallocated[kdx])
            break

    return goodlist, avglist, badlist


#######################################################
def reconstruct(fraglist, listofids, reflist, cellvec, factor, metal_factor):
    # function also used fraglist

    tmatlist = []
    status = 0

    # finds biggest fragment and keeps it in the original cell
    sizes = []
    for l in listofids:
        size = fraglist[l].natoms
        sizes.append(size)
    keep = np.argmax(sizes)

    listwithoutkeep = []
    for idx, frag in enumerate(listofids):
        if idx != keep:
            listwithoutkeep.append(frag)

    for l in listwithoutkeep:
        tmatlist.append(fraglist[l].tmatrix)

    applytranspose = list(itertools.product(*tmatlist))

    if (status == 0) and (len(applytranspose) > 0):
        for jdx, tmol in enumerate(applytranspose):

            reccoord = []
            reclabels = []
            reccoord.extend(fraglist[listofids[keep]].coord)
            reclabels.extend(fraglist[listofids[keep]].labels)

            for kdx, mol in enumerate(tmol):
                reclabels.extend(fraglist[listwithoutkeep[kdx]].labels)
                if mol != (0, 0, 0):
                    # indicates that the molecule needs translation in direction defined by "tma"
                    newcoord = translate(
                        mol, fraglist[listwithoutkeep[kdx]].coord, cellvec
                    )
                    if len(newcoord) != len(fraglist[listwithoutkeep[kdx]].coord):
                        print("error 1 in Reconstruct")
                    reccoord.extend(newcoord)
                if mol == (0, 0, 0):
                    reccoord.extend(fraglist[listwithoutkeep[kdx]].coord)

            Warning, reclist = getmolecs(reclabels, reccoord, factor, metal_factor)
            if len(reclist) == 1:
                status = checkchemistry(reclist[0], reflist, "Max")
                break
    else:
        reclist = []

    return status, reclist


#######################################################
def indentify_frag_molec_H(blocklist, moleclist, refmoleclist, cellvec):

    init_natoms = 0

    fraglist = []
    Hlist = []

    # Convert blocks' coordinates and get centroid
    for b in blocklist:
        b.frac = cart2frac(b.coord, cellvec)
        b.centroid = getcentroid(b.frac)
        init_natoms += b.natoms

    for idx, block in enumerate(blocklist):
        if any((block.elemcountvec == ref.elemcountvec).all() for ref in refmoleclist):
            if any((atom.block == "d" or atom.block == "f") for atom in block.atoms):
                block.type = "Complex"
            else:
                block.type = "Molecule"
            moleclist.append(block)
        else:
            if (block.natoms == 1) and (block.numH == 1):
                block.type = "H"
                Hlist.append(block)
            else:
                block.type = "Fragment"
                fraglist.append(block)
    print("")
    print("#################")
    print("Blocks found are:")
    print("#################")
    for b in blocklist:
        print("Block in Info", b.natoms, b.labels, b.type)
    print(
        len(blocklist),
        "Blocks sorted for reconstructrion as (Molec, Frag, H):",
        len(moleclist),
        len(fraglist),
        len(Hlist),
    )
    print("With a total of", init_natoms, "atoms")

    return moleclist, fraglist, Hlist, init_natoms


#######################################################
def fragments_reconstruct(
    moleclist, fraglist, Hlist, refmoleclist, cellvec, debug, factor, metal_factor
):

    Warning = False

    # Reconstruct Heavy Fragments
    if len(fraglist) > 1:
        print("")
        print("##############################################")
        print(len(fraglist), "molecules submitted to SEQUENTIAL with Heavy")
        print("##############################################")
        newmols, remfrag = sequential(
            fraglist, refmoleclist, cellvec, debug, factor, metal_factor, "Heavy"
        )
        print(len(newmols), len(remfrag), "molecules out of SEQUENTIAL with Heavy")
        moleclist.extend(newmols)
        fraglist = []
        fraglist.extend(remfrag)
        fraglist.extend(Hlist)

        print(" ")
        # Prints molecules after Heavy Fragment Reconstruction
        if len(newmols) > 0:
            for mol in newmols:
                print(
                    "Molec reconstructed after Heavy", mol.natoms, mol.labels, mol.type
                )
        else:
            print("NO Molecules reconstructed after Heavy")
        if len(remfrag) > 0:
            for rem in remfrag:
                print("Remaining after Heavy", rem.natoms, rem.labels, rem.type)
        else:
            print("NO remaining Molecules after Heavy")

        print(" ")

    else:
        print("Only 0 or 1 heavy fragments. Skipping Heavy")

    # Reconstruct Hydrogens with remaining Fragments
    if len(remfrag) > 0 and len(Hlist) > 0:
        print("")
        print("##############################################")
        print(len(fraglist), "molecules submitted to sequential with All")
        print("##############################################")
        finalmols, remfrag = sequential(
            fraglist, refmoleclist, cellvec, debug, factor, metal_factor, "All"
        )
        if len(remfrag) > 0:
            Warning = True
            for rem in remfrag:
                print(
                    "Remaining after Hydrogen reconstruction",
                    rem.natoms,
                    rem.labels,
                    rem.type,
                )
        else:
            print("NO remaining Molecules after Hydrogen reconstruction")
            Warning = False
        print(" ")
    else:
        finalmols = fraglist.copy()  # IF not Hidrogen fragments, then is done
        remfrag = []

    return moleclist, finalmols, Warning


#######################################################
def assigntype(molecule, references):
    Found = False
    for ref in references:
        if (
            (ref.elemcountvec == molecule.elemcountvec).all()
            and (ref.adjtypes == molecule.adjtypes).all()
            and not Found
        ):
            molectype = ref.type
            Found = True
    if not Found:
        molectype = "Other"
        for a in molecule.atoms:
            if (a.block == "d") or (a.block == "f"):
                molectype = "Complex"
    return molectype


#######################################################
def split_complexes_reassign_type(cell, moleclist):

    if not all(cell.warning_list):
        # Split Complexes
        for mol in moleclist:
            if mol.type == "Complex":
                mol.ligandlist, mol.metalist = splitcomplex(
                    mol, mol.factor, mol.metal_factor
                )
                dummy = get_hapticity(mol)

        # Reassign Type of molecules and store information
        for mol in moleclist:
            mol.type = assigntype(mol, cell.refmoleclist)
            mol.refcode = cell.refcode
            mol.name = str(
                cell.refcode + "_" + mol.type + "_" + str(moleclist.index(mol))
            )
            if mol.type == "Complex":
                for lig in mol.ligandlist:
                    lig.refcode = cell.refcode
                    lig.name = str(
                        cell.refcode
                        + "_"
                        + mol.type
                        + "_"
                        + str(moleclist.index(mol))
                        + "_"
                        + lig.type
                        + "_"
                        + str(mol.ligandlist.index(lig))
                    )
                for met in mol.metalist:
                    met.refcode = cell.refcode
                    met.name = str(
                        cell.refcode
                        + "_"
                        + mol.type
                        + "_"
                        + str(moleclist.index(mol))
                        + "_"
                        + met.type
                        + "_"
                        + str(mol.metalist.index(met))
                    )

    cell.moleclist = moleclist

    return cell


#######################################################
def get_hapticity(molecule):

    debug = 0

    if molecule.type == "Complex":
        for lig in molecule.ligandlist:
            groups = find_groups_within_ligand(lig)
            #             print(len(groups), "groups found for ligand:", lig.labels)

            for g in groups:
                has_hapticity = False
                group_hapttype = []

                list_of_coord_atoms = []
                for idx, a in enumerate(lig.atoms):
                    if idx in g and a.mconnec > 0:
                        list_of_coord_atoms.append(a.label)

                numC = list_of_coord_atoms.count(
                    "C"
                )  # Carbon is the most common connected atom in ligands with hapticity
                numAs = list_of_coord_atoms.count(
                    "As"
                )  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
                numO = list_of_coord_atoms.count("O")  # For h4-Enone
                ## Carbon-based Haptic Ligands
                if numC == 2:
                    group_hapttype = ["h2-Benzene", "h2-Butadiene"]
                    has_hapticity = True
                elif numC == 3 and numO == 0:
                    group_hapttype = ["h3-Allyl", "h3-Cp"]
                    has_hapticity = True
                elif numC == 3 and numO == 1:
                    group_hapttype = ["h4-Enone"]
                    has_hapticity = True
                elif numC == 4:
                    group_hapttype = ["h4-Butadiene", "h4-Benzene"]
                    has_hapticity = True
                elif numC == 5:
                    group_hapttype = ["h5-Cp"]
                    has_hapticity = True
                elif numC == 6:
                    group_hapttype = ["h6-Benzene"]
                    has_hapticity = True
                elif numC == 7:
                    group_hapttype = ["h7-Cicloheptatrienyl"]
                    has_hapticity = True
                elif numC == 8:
                    group_hapttype = ["h8-Ciclooctatetraenyl"]
                    has_hapticity = True

                # Other less common types of haptic ligands
                elif numC == 0 and numAs == 5:
                    group_hapttype = ["h5-AsCp"]
                    has_hapticity = True

                # Creates Group
                newgroup = group(g, has_hapticity, group_hapttype)
                lig.grouplist.append(newgroup)

            # Sets Ligand hapticity
            if any(g.hapticity == True for g in lig.grouplist):
                lig.hapticity = True
                for g in lig.grouplist:
                    # lig.haptgroups.append(g.atlist)
                    for typ in g.hapttype:
                        if typ not in lig.hapttype:
                            lig.hapttype.append(typ)
            else:
                lig.hapticity = False

        # Sets molecule hapticity
        if any(lig.hapticity == True for lig in molecule.ligandlist):
            molecule.hapticity = True
            for lig in molecule.ligandlist:
                for typ in lig.hapttype:
                    if typ not in molecule.hapttype:
                        molecule.hapttype.append(typ)
        else:
            molecule.hapticity = False

    elif molecule.type != "Complex":
        molecule.hapticity = False

    return molecule.hapticity
