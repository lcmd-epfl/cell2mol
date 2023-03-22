#!/usr/bin/env python

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from cell2mol.elementdata import ElementData
from typing import Tuple

elemdatabase = ElementData()

################################
def labels2formula(labels):
    elems = elemdatabase.elementnr.keys()
    formula=[]
    for z in elems:
        nz = labels.count(z)
        if nz > 1:
            formula.append(f"{z}{nz}-")
        if nz == 1:
            formula.append(f"{z}-")
    formula = ''.join(formula)[:-1] 
    return formula 

################################
def getelementcount(labels: list) -> np.ndarray:
    elems = elemdatabase.elementnr.keys()
    times = np.zeros((len(elems)),dtype=int)
    for l in labels:
        for jdx, elem in enumerate(elems):
            if l == elem:
                times[jdx] += 1
    return times


################################
def getHvcount(labels: list) -> np.ndarray:
    elems = elemdatabase.elementnr.keys()
    times = np.zeros((len(elems)),dtype=int)
    for l in labels:
        if l != "H":
            for jdx, elem in enumerate(elems):
                if l == elem:
                    times[jdx] += 1
    return times


################################
def get_adjacency_types(label: list, conmat: np.ndarray) -> np.ndarray:
    elems = elemdatabase.elementnr.keys()
    bondtypes = np.zeros((len(elems), len(elems)),dtype=int)
    #bondtypes = np.zeros((len(elems), len(elems))).astype(int)
    natoms = len(label)
    found = np.zeros((natoms, natoms))

    for i in range(0, natoms):
        for j in range(i, natoms):
            if i != j:
                if (conmat[i, j] == 1) and (found[i, j] == 0):
                    for k, elem1 in enumerate(elems):
                        if label[i] == elem1:
                            for l, elem2 in enumerate(elems):
                                if label[j] == elem2:
                                    bondtypes[k, l] += 1
                                    if elem1 != elem2:
                                        bondtypes[l, k] += 1
                                    found[i, j] = 1
                                    found[j, i] = 1
                                    break
                            break
    return bondtypes


################################
def checkchemistry(molecule: object, references: list, typ: str="Max") -> int:

    elems = elemdatabase.elementnr.keys()
    maxval = np.zeros((len(references[0].adjtypes), len(references[0].adjtypes)))

    for i in range(0, len(references[0].adjtypes)):
        for j in range(0, len(references[0].adjtypes)):
            lst = []
            for ref in references:
                lst.append(ref.adjtypes[i, j])
            maxval[i, j] = np.max(lst)

    status = 1  # Good
    if (
        typ == "Max"
    ):  # the bonds[j,k] of the molecule cannot be larger than the maximum within references
        for i in range(0, len(molecule.adjtypes)):
            for j in range(0, len(molecule.adjtypes)):
                if molecule.adjtypes[i, j] > 0:
                    if molecule.adjtypes[i, j] > maxval[i, j]:
                        status = 0  # bad
    return status


################################
def getradii(labels: list) -> np.ndarray:
    radii = []
    for l in labels:
        radii.append(elemdatabase.CovalentRadius2[l])
    return np.array(radii)


################################
def getcentroid(frac: list) -> list:
    natoms = len(frac)
    x = 0
    y = 0
    z = 0
    for idx, l in enumerate(frac):
        x += frac[idx][0]
        y += frac[idx][1]
        z += frac[idx][2]
    centroid = [float(x / natoms), float(y / natoms), float(z / natoms)]
    return centroid


################################
def extract_from_matrix(entrylist: list, old_array: np.ndarray, dimension: int=2) -> np.ndarray:

    length = len(entrylist)

    if dimension == 2:
        new_array = np.empty((length, length))
        for idx, row in enumerate(entrylist):
            for jdx, col in enumerate(entrylist):
                new_array[idx, jdx] = old_array[row][col]

    elif dimension == 1:
        new_array = np.empty((length))
        for idx, val in enumerate(entrylist):
            new_array[idx] = old_array[val]

    return new_array


####################################
def getconec(labels: list, pos: list, factor: float, radii="default") -> Tuple[int, list, list, list, list]:
    status = 1  # good molecule, no clashes yet
    clash = 0.3
    natoms = len(labels)
    conmat = np.zeros((natoms, natoms))
    connec = np.zeros((natoms))
    mconmat = np.zeros((natoms, natoms))
    mconnec = np.zeros((natoms))


    covalent_factor_for_metal = {
        "H": 1.30,
        "D": 1.30,
        "He": 1.30,
        "Li": 1.30,  
        "Be": 1.30,  
        "B": 1.15,
        "C": 1.15,
        "N": 1.25,
        "O": 1.25,
        "F": 1.15,
        "Ne": 1.30,
        "Na": 1.30,  
        "Mg": 1.30,  
        "Al": 1.30,
        "Si": 1.10,
        "P": 1.25,   
        "S": 1.25,
        "Cl": 1.25,
        "Ar": 1.30,
        "K": 1.30,  
        "Ca": 1.30,
        "Sc": 1.30,
        "Ti": 1.30,
        "V": 1.30,
        "Cr": 1.30,
        "Mn": 1.30,
        "Fe": 1.30,
        "Co": 1.30,
        "Ni": 1.30,
        "Cu": 1.30,
        "Zn": 1.30,
        "Ga": 1.30,
        "Ge": 1.30,
        "As": 1.15,
        "Se": 1.15,
        "Br": 1.25,
        "Kr": 1.30,
        "Rb": 1.30,
        "Sr": 1.30,
        "Y": 1.30,
        "Zr": 1.30,
        "Nb": 1.30,
        "Mo": 1.30,
        "Tc": 1.30,
        "Ru": 1.30,
        "Rh": 1.30,
        "Pd": 1.30,
        "Ag": 1.30,
        "Cd": 1.30,
        "In": 1.30,
        "Sn": 1.30,
        "Sb": 1.15,  
        "Te": 1.15,  
        "I": 1.25,
        "Xe": 1.30,
        "Cs": 1.30,
        "Ba": 1.30,
        "La": 1.30,
        "Ce": 1.30,
        "Pr": 1.30,
        "Nd": 1.30,
        "Pm": 1.30,
        "Sm": 1.30,
        "Eu": 1.30,
        "Gd": 1.30,
        "Tb": 1.30,
        "Dy": 1.30,
        "Ho": 1.30,
        "Er": 1.30,
        "Tm": 1.30,
        "Yb": 1.30,
        "Lu": 1.30,
        "Hf": 1.30,
        "Ta": 1.30,
        "W": 1.30,
        "Re": 1.30,
        "Os": 1.30,
        "Ir": 1.30,
        "Pt": 1.30,
        "Au": 1.30,
        "Hg": 1.30,
        "Tl": 1.30,
        "Pb": 1.30,
        "Bi": 1.30,
        "Po": 1.30,
        "At": 1.30,
        "Rn": 1.30,
        "Fr": 1.30,
        "Ra": 1.30,
        "Ac": 1.30,
        "Th": 1.30,
        "Pa": 1.30,
        "U": 1.30,
        "Np": 1.30,
        "Pu": 1.30,
        "Am": 1.30,
        "Cm": 1.30,
        "Bk": 1.30,
        "Cf": 1.30,
        "Es": 1.30,
        "Fm": 1.30,
        "Md": 1.30,
        "No": 1.30,
        "Lr": 1.30,
        "Rf": 1.30,
        "Db": 1.30,
        "Sg": 1.30,
        "Bh": 1.30,
        "Hs": 1.30,
        "Mt": 1.30,
    }


    # Sometimes argument radii np.ndarry, or list
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        if radii == "default":
            radii = getradii(labels)

    for i in range(0, natoms - 1):
        for j in range(i, natoms):
            if i != j:
                # print(i,j)
                a = np.array(pos[i])
                b = np.array(pos[j])
                dist = np.linalg.norm(a - b)
                
                if (
                        elemdatabase.elementblock[labels[i]] != "d"
                        and elemdatabase.elementblock[labels[i]] != "f"
                        and elemdatabase.elementblock[labels[j]] != "d"
                        and elemdatabase.elementblock[labels[j]] != "f"
                    ):

                        thres = (radii[i] + radii[j]) * factor

                elif (
                        elemdatabase.elementblock[labels[i]] == "d"
                        or elemdatabase.elementblock[labels[i]] == "f"
                        or elemdatabase.elementblock[labels[j]] == "d"
                        or elemdatabase.elementblock[labels[j]] == "f"
                    ):
                        factor_i = covalent_factor_for_metal[labels[i]]
                        factor_j = covalent_factor_for_metal[labels[j]]

                        if factor_i < factor_j  :
                            new_factor = factor_i
                        
                        elif factor_i == factor_j :
                            new_factor = factor_i

                        else : 
                            new_factor = factor_j
                        #print(factor_i, factor_j, new_factor, labels[i], labels[j])
                        thres = (radii[i] + radii[j]) * new_factor

                if dist <= clash:
                    status = 0  # invalid molecule
                    print("GETCONEC: Distance", dist, "smaller than clash for atoms", i, j)
                elif dist <= thres:
                    conmat[i, j] = 1
                    conmat[j, i] = 1
                    if (
                        elemdatabase.elementblock[labels[i]] == "d"
                        or elemdatabase.elementblock[labels[i]] == "f"
                        or elemdatabase.elementblock[labels[j]] == "d"
                        or elemdatabase.elementblock[labels[j]] == "f"
                    ):
                        mconmat[i, j] = 1
                        mconmat[j, i] = 1

    for i in range(0, natoms):
        connec[i] = np.sum(conmat[i, :])
        mconnec[i] = np.sum(mconmat[i, :])

    conmat = conmat.astype(int)
    mconmat = mconmat.astype(int)
    connec = connec.astype(int)
    mconnec = mconnec.astype(int)
    # return status, np.array(conmat), np.array(connec), np.array(mconmat), np.array(mconnec)
    return status, conmat, connec, mconmat, mconnec


####################################
def getconec_original(labels: list, pos: list, factor: float, radii="default") -> Tuple[int, list, list, list, list]:
    status = 1  # good molecule, no clashes yet
    clash = 0.3
    natoms = len(labels)
    conmat = np.zeros((natoms, natoms))
    connec = np.zeros((natoms))
    mconmat = np.zeros((natoms, natoms))
    mconnec = np.zeros((natoms))
    # Sometimes argument radii np.ndarry, or list
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        if radii == "default":
            radii = getradii(labels)

    for i in range(0, natoms - 1):
        for j in range(i, natoms):
            if i != j:
                # print(i,j)
                a = np.array(pos[i])
                b = np.array(pos[j])
                dist = np.linalg.norm(a - b)
                thres = (radii[i] + radii[j]) * factor
                if dist <= clash:
                    status = 0  # invalid molecule
                    print("GETCONEC: Distance", dist, "smaller than clash for atoms", i, j)
                elif dist <= thres:
                    conmat[i, j] = 1
                    conmat[j, i] = 1
                    if (
                        elemdatabase.elementblock[labels[i]] == "d"
                        or elemdatabase.elementblock[labels[i]] == "f"
                        or elemdatabase.elementblock[labels[j]] == "d"
                        or elemdatabase.elementblock[labels[j]] == "f"
                    ):
                        mconmat[i, j] = 1
                        mconmat[j, i] = 1

    for i in range(0, natoms):
        connec[i] = np.sum(conmat[i, :])
        mconnec[i] = np.sum(mconmat[i, :])

    conmat = conmat.astype(int)
    mconmat = mconmat.astype(int)
    connec = connec.astype(int)
    mconnec = mconnec.astype(int)
    # return status, np.array(conmat), np.array(connec), np.array(mconmat), np.array(mconnec)
    return status, conmat, connec, mconmat, mconnec

def inv(perm: list) -> list:
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def getblocks(matrix: np.ndarray) -> Tuple[list, list]:
    # retrieves the blocks from a diagonal block matrix
    startlist = []
    endlist = []

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

    if (blockcount == 0) and (
        len(matrix) == 1
    ):  # if a 1x1 matrix is provided, it then finds 1 block
        startlist.append(0)
        endlist.append(0)

    return startlist, endlist


def find_groups_within_ligand(ligand: object) -> list:

    debug = 0
    if debug >= 1:
        print(f"DOING LIGAND: {ligand.labels}")
    if debug >= 1:
        print(f"FIND GROUPS received conmat shape: {ligand.conmat.shape}")
    if debug >= 2:
        print(f"FIND GROUPS received conmat: {ligand.conmat}")

    connected_atoms = []
    unconnected_atoms = []
    cutmolec = []
    for idx, a in enumerate(ligand.atoms):
        if a.mconnec >= 1:
            connected_atoms.append(idx)
            cutmolec.append([a.label, idx])
        elif a.mconnec == 0:
            unconnected_atoms.append(idx)

    rowless = np.delete(ligand.conmat, unconnected_atoms, 0)
    columnless = np.delete(rowless, unconnected_atoms, axis=1)

    if debug >= 1:
        print(f"FIND GROUPS: connected are: {connected_atoms}")
    if debug >= 1:
        print(f"FIND GROUPS: unconnected are: {unconnected_atoms}")

    # Regenerates the truncated lig.connec
    connec = []
    for idx, c in enumerate(ligand.connec):
        if idx in connected_atoms:
            connec.append(ligand.connec[idx])

    # Does the
    degree = np.diag(connec)
    lap = columnless - degree

    # Copied from split_complex
    graph = csr_matrix(lap)
    perm = reverse_cuthill_mckee(graph)
    gp1 = graph[perm, :]
    gp2 = gp1[:, perm]
    dense = gp2.toarray()

    startlist, endlist = getblocks(dense)
    ngroups = len(startlist)

    atomlist = np.zeros((len(dense)))
    for b in range(0, ngroups):
        for i in range(0, len(dense)):
            if (i >= startlist[b]) and (i <= endlist[b]):
                atomlist[i] = b + 1
    invperm = inv(perm)
    atomlistperm = [int(atomlist[i]) for i in invperm]

    if debug >= 1:
        print(f"FIND GROUPS: the {ngroups} groups start at: {startlist}")

    groups = []
    for b in range(0, ngroups):
        atlist = []
        for i in range(0, len(atomlistperm)):
            if atomlistperm[i] == b + 1:
                atlist.append(cutmolec[i][1])
        groups.append(atlist)

    if debug >= 1:
        print(f"FIND GROUPS finds {ngroups} as {groups}")

    return groups

#######################################################
def find_closest_metal(atom: object, metalist: list, debug: int=0) -> Tuple[np.ndarray, np.ndarray, list]:

    apos = np.array(atom.coord)
    dist = []
    for tm in metalist:
        bpos = np.array(tm.coord)
        dist.append(np.linalg.norm(apos - bpos))

    # finds the closest Metal Atom (tgt)
    tgt = np.argmin(dist)
    return tgt, apos, dist[tgt]

############################ CLASSES #########################

###############
### ATOM ######
###############
class atom(object):
    def __init__(self, index: int, label: str, coord: list, radii: float) -> None:
        self.version = "V1.0"
        self.index = index
        self.label = label
        self.coord = coord
        self.frac = []
        self.atnum = elemdatabase.elementnr[label]
        self.val = elemdatabase.valenceelectrons[label]  # currently not used
        self.weight = elemdatabase.elementweight[label]  # currently not used
        self.block = elemdatabase.elementblock[label]
        self.radii = radii

    # Adjacency part is simultaneously to creating the ligand or molecule object
    ### Changed in V14
    def adjacencies(self, conmat: np.ndarray, mconmat: np.ndarray, type: str="Molecule") -> None:
        self.adjacency = []  
        self.metal_adjacency = []

        self.connec = np.sum(conmat)
        if conmat.shape:
            for idx, c in enumerate(conmat):
                if c >= 1:
                    self.adjacency.append(idx)
        else:
            self.adjacency.append(conmat)

        if type == "Molecule":
            self.mconnec = np.sum(mconmat)
            if mconmat.shape:
                for idx, c in enumerate(mconmat):
                    if c >= 1:
                        self.metal_adjacency.append(idx)
            else:
                self.adjacency.append(conmat)

        elif type == "Ligand" or type == "Metal":
            self.mconnec = mconmat  # this has to be improved, now it only receives a number, should receive a vector as above for "molecule"

    # Bonds part is created after the formal charge for each molecule/ligand/metal is decided
    def bonds(self, start: list, end: list, order: list) -> None:
        self.bond = []
        self.bond_start_idx = []
        self.bond_end_idx = []
        self.bond_order = []

        for a in start:
            self.bond_start_idx.append(a)
        for b in end:
            self.bond_end_idx.append(b)
        for c in order:
            self.bond_order.append(c)

        for group in zip(start, end, order):
            self.bond.append(group)

        self.nbonds = len(self.bond)
        self.totbondorder = np.sum(self.bond_order)

    def atom_charge(self, charge: int) -> None:
        self.charge = charge
        
###############
### MOLECULE ##
###############
class molecule(object):
    def __init__(self, name: str, atlist: list, labels: list, coord: list, radii:list) -> None:
        self.version = "V1.0"
        self.refcode = ""  
        self.name = name
        self.atlist = atlist
        self.labels = labels
        self.coord = coord
        self.radii = radii
        self.formula = labels2formula(labels)
        self.occurrence = 0   # How many times the molecule appears in a unit cell

        self.natoms = len(atlist)
        self.elemcountvec = getelementcount(labels)
        self.Hvcountvec = getHvcount(labels)

        self.frac = []
        self.centroid = []
        self.tmatrix = []

        # Creates Atoms
        self.atnums = []
        self.atoms = []
        for idx, l in enumerate(self.labels):
            newatom = atom(idx, l, self.coord[idx], self.radii[idx])
            self.atnums.append(newatom.atnum)
            self.atoms.append(newatom)

        self.type = "Other"
        self.eleccount = 0
        self.numH = 0
        for a in self.atoms:
            if a.atnum == 1:
                self.numH += 1
            self.eleccount += a.atnum
            if (a.block == "d") or (a.block == "f"):
                self.type = "Complex"

        if self.type == "Complex":
            self.ligandlist = []
            self.metalist = []
            self.hapticity = False  # V13
            self.hapttype = []  # V13

        # Lists of potentially good variables for this molecule
        self.poscharge = []
        self.posatcharge = []
        self.posobjlist = []
        self.posspin = []
        self.possmiles = []

    # Stores the covalentradii factor and metal factor that were used to generate the molecule
    def information(self, factor: float, metal_factor: float) -> None:
        self.factor = factor
        self.metal_factor = metal_factor

    # Actual variables for the molecule in the crystal where it comes from:
    def charge(self, atcharge: np.ndarray, totcharge: int, rdkit_mol: list, smiles: list) -> None:
        self.atcharge = atcharge
        self.totcharge = totcharge
        self.smiles_with_H = " "                # If "Complex", it will be a list of ligand smiles
        self.smiles = smiles                    # If "Complex", it will be a list of ligand smiles 
        self.rdkit_mol = rdkit_mol              # If "Complex", it is an empty list

        if self.type != "Complex":
            for idx, a in enumerate(self.atoms):
                a.atom_charge(self.atcharge[idx])

    # Spin State Variables
    def magnetism(self, spin):
        self.spin = spin

    # Connectivity = Adjacency Matrix. Potentially expandable to Include Bond Types
    def adjacencies(self, conmat: np.ndarray, mconmat: np.ndarray) -> None:
        self.conmat = conmat
        self.mconmat = mconmat
        self.connec = np.zeros((self.natoms))
        self.mconnec = np.zeros((self.natoms))
        for i in range(0, self.natoms):
            self.connec[i] = np.sum(self.conmat[i, :])
            self.mconnec[i] = np.sum(self.mconmat[i, :])

        self.totconnec = int(np.sum(self.connec) / 2)
        self.totmconnec = int(np.sum(self.mconnec) / 2)
        self.adjtypes = get_adjacency_types(self.labels, self.conmat)  # V14
        # self.nbonds = int(np.sum(self.bonds)/2)

        for idx, a in enumerate(self.atoms):
            # print("adjacencies sent with", np.array(self.conmat[idx].astype(int)), np.array(self.mconmat[idx].astype(int)))
            a.adjacencies(np.array(self.conmat[idx].astype(int)),np.array(self.mconmat[idx].astype(int)),type="Molecule")

    # def repr_CM(self, ):
    # self.CM = coulomb_matrix(self.atnums, self.coord)
    # NOTE: don't forget to copy to ligand object when ready


###############
### LIGAND ####
###############
class ligand(object):
    def __init__(self, name: str, atlist: list, labels: list, coord: list, radii: list) -> None:
        self.version = "V1.0"
        self.refcode = ""  
        self.name = name      # creates as ligand index, later modified
        self.atlist = atlist  # atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.labels = labels  # elements
        self.coord = coord    # coordinates
        self.radii = radii
        self.formula = labels2formula(labels)
        self.occurrence = 0   # How many times the ligand appears in a molecule

        self.natoms = len(atlist)  # number of atoms
        self.type = "Ligand"
        self.elemcountvec = getelementcount(labels)
        self.Hvcountvec = getHvcount(labels)

        # Lists of potentially good variables for this molecule
        self.poscharge = []
        self.posatcharge = []
        self.posobjlist = []
        self.posspin = []
        self.possmiles = []

        # Stores information about the metal to which it is attached, about groups of connected atoms, and hapticity
        self.metalatoms = []
        self.grouplist = []  # V14, this info is generated in get_hapticity
        self.hapticity = False  # V13
        self.hapttype = []  # V13
        # self.haptgroups = []      #V14, replaced by grouplist

        self.atnums = []
        self.atoms = []
        for idx, l in enumerate(self.labels):
            newatom = atom(idx, l, self.coord[idx], self.radii[idx])
            self.atnums.append(newatom.atnum)
            self.atoms.append(newatom)

        # Creates atoms and defines charge
        self.eleccount = 0
        self.numH = 0
        for a in self.atoms:
            if a.atnum == 1:
                self.numH += 1
            self.eleccount += a.atnum

    # Stores the covalentradii factor and metal factor that were used to generate the molecule
    def information(self, factor: float, metal_factor: float) -> None:
        self.factor = factor
        self.metal_factor = metal_factor

    def charge(self, atcharge: list, totcharge: int, rdkit_mol: object, smiles: str) -> None:
        self.atcharge = atcharge
        self.totcharge = totcharge
        self.smiles_with_H = " "                # Smiles of the ligand that includes any added H atom, created at "build_bonds" function
        self.smiles = smiles                    # Now Empty, created later as a smiles without any added H atom 
        self.rdkit_mol = rdkit_mol              # Rdkit mol object 

        for idx, a in enumerate(self.atoms):
            a.atom_charge(int(self.atcharge[idx]))

    # Spin State Variables
    def magnetism(self, spin):
        self.spin = spin

    def adjacencies(self, conmat: np.ndarray, mconnec: np.ndarray) -> None:
        self.conmat = conmat
        self.connec = np.zeros((self.natoms))
        for i in range(0, self.natoms):
            self.connec[i] = np.sum(self.conmat[i, :])

        # For ligand, mconmat is all zero np.ndarray --> not used, Can I remove? 
        # so metal_adjacency in Class atom is also empty because mconmat doesn't contain metal-ligand connectivity
        self.mconmat = np.zeros((self.natoms, self.natoms)).astype(int) 

        self.mconnec = mconnec
        self.totconnec = int(np.sum(self.connec))
        self.totmconnec = int(np.sum(self.mconnec))
        self.adjtypes = get_adjacency_types(self.labels, self.conmat)  # V14

        for idx, a in enumerate(self.atoms):
            a.adjacencies(np.array(self.conmat[idx].astype(int)), int(mconnec[idx]), type="Ligand")

###############
class group(object):
    def __init__(self, atlist: list, hapticity: bool, hapttype: list) -> None:
        self.version = "V1.0"
        self.atlist = atlist  # atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.hapticity = hapticity
        self.hapttype = hapttype


###############
#### METAL ####
###############
class metal(object):
    def __init__(self, name: int, atlist: int, label: str, coord: list, radii: float) -> None:
        self.version = "V1.0"
        self.refcode = ""  
        self.name = name  # creates as metal index, later modified
        self.atlist = atlist  # atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.label = label
        self.coord = coord
        self.radii = radii
        self.natom = int(1)  # number of atoms
        self.type = "Metal"
        self.poscharge = []
        self.coord_sphere = []
        self.coord_sphere_ID = []
        self.coordinating_atoms = []
        self.coordinating_atoms_sites = []
        self.occurrence = 0   # How many times the metal appears in a unit cell

        self.atom = atom(name, label, self.coord, self.radii)

    # Stores the covalentradii factor and metal factor that were used to generate the molecule
    def information(self, factor: float, metal_factor: float) -> None:
        self.factor = factor
        self.metal_factor = metal_factor

    def charge(self, metal_charge: int) -> None:
        self.totcharge = metal_charge

    # def magnetism(self, spin):
    #     self.spin = spin

    def adjacencies(self, mconnec: np.ndarray) -> None:
        self.mconnec = mconnec  # adjacencies matrix with only metal bonds
        self.totmconnec = int(np.sum(mconnec))
        self.atom.adjacencies(np.array(int(0)), int(mconnec), type="Metal")

    def coordination (self, hapticity: bool, posgeom_dev: dict) -> None:
        self.hapticity = hapticity
        if self.hapticity == False :
            self.coordination_number=len(self.coordinating_atoms)
            self.posgeom_dev=posgeom_dev
            self.geometry=min(posgeom_dev, key=posgeom_dev.get)
            self.deviation=min(posgeom_dev.values())
        else:
            self.coordination_number=""
            self.posgeom_dev={}
            self.geometry=""
            self.deviation=""

##############
#### CELL ####
##############
class cell(object):
    def __init__(self, refcode: str, labels: list, pos: list, cellvec: list, cellparam: list, warning_list: list) -> None:

        self.version = "V1.0"
        self.refcode = refcode

        self.cellvec = cellvec
        self.cellparam = cellparam

        self.labels = labels 
        self.atom_coord = pos  # Atom cartesian coordinates from info file
        
        self.natoms = len(labels)
        self.coord = [] 

        self.speclist = []
        self.refmoleclist = []
        self.moleclist = []
        self.warning_list = warning_list
        self.warning_after_reconstruction = []

        self.charge_distribution_list = []
   
    def arrange_cell_coord(self): 
        ## Updates the cell coordinates preserving the original atom ordering
        ## Do do so, it uses the variable atlist stored in each molecule
        self.coord = np.zeros((self.natoms,3))
        for mol in self.moleclist:
            for z in zip(mol.atlist, mol.coord):
                for i in range(0,3):
                    self.coord[z[0]][i] = z[1][i]
        self.coord = np.ndarray.tolist(self.coord)


    def data_for_postproc(self, molecules, indices, options):
        self.pp_molecules = molecules
        self.pp_indices = indices
        self.pp_options = options

    def print_charge_assignment(self):
        print("=====================================Charges for all species in the unit cell=====================================")
        print("[Ref.code] : {}".format(self.refcode))
        for unit in self.moleclist:
            if unit.type == "Complex":
                print("\n{}, totcharge {}, spin multiplicity {}".format(unit.type, unit.totcharge, unit.spin))
                for metal in unit.metalist:
                    print("\t Metal: {} charge {}".format(metal.label, metal.totcharge))
                for ligand in unit.ligandlist:
                    print("\t Ligand charge {}, {}".format(ligand.totcharge, ligand.smiles))
            elif unit.type == "Other":
                print("\n{} totcharge {}, {}".format(unit.type, unit.totcharge, unit.smiles))
        print("\n")

    def print_Warning(self):
        reason_of_Warning = [
            "Warning 1! Errors received from getting reference molecules (disorder or strange coordination)",
            "Warning 2! Missing H atoms from C in reference molecules",
            "Warning 3! Missing H atoms from coordinated water in reference molecules",
            "Warning 4! Steric clashes while blocking molecules",
            "Warning 5! Errors in cell reconstruction",
            "Warning 6! Empty list of possible charges received for molecule or ligand",
            "Warning 7! More than one valid possible charge distribution found",
            "Warning 8! No valid possible charge distribution found",
            "Warning 9! Error while preparing molecules.",
        ]

        # printing original list
        print("The Warning type list is : " + str(self.warning_list))

        res = [i for i, val in enumerate(self.warning_list) if val]
        # printing result
        #         print ("The list indices having True values are : " + str(res))

        if len(res) == 0:
            print("No Warnings!")
        else:
            for i in res:
                print(reason_of_Warning[i])
########## END OF CLASSES ###########
