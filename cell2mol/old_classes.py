#!/usr/bin/env python

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Tuple

from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

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

############################ CLASSES #########################

###############
### ATOM ######
###############
class old_atom(object):
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
class old_molecule(object):
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
    def magnetism(self, spin: int) -> None:
        self.spin = spin

    def ml_prediction(self, spin_rf: int) -> None:
        self.spin_rf = spin_rf

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
class old_ligand(object):
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
    def magnetism(self, spin: int) -> None:
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
class old_group(object):
    def __init__(self, atlist: list, hapticity: bool, hapttype: list) -> None:
        self.version = "V1.0"
        self.atlist = atlist  # atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.hapticity = hapticity
        self.hapttype = hapttype


###############
#### METAL ####
###############
class old_metal(object):
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
        self.coordinating_atlist = []
        self.group_list = []
        self.group_atoms_list = []

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

    def coordination (self, coordination_number: int, hapticity: bool, hapttype: list, posgeom_dev: dict) -> None:
        self.hapticity = hapticity
        self.hapttype = hapttype
        self.coordination_number=coordination_number
        self.posgeom_dev=posgeom_dev
        if len(posgeom_dev) == 0:
            self.geometry = "Undefined"
            self.deviation = "Undefined"
        else:
            self.geometry=min(posgeom_dev, key=posgeom_dev.get)
            self.deviation=min(posgeom_dev.values())

    def relative_radius(self, rel: float, rel_g: float, rel_c: float) -> None:
        self.rel = rel
        self.rel_g = rel_g 
        self.rel_c = rel_c
        

##############
#### CELL ####
##############
class old_cell(object):
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
