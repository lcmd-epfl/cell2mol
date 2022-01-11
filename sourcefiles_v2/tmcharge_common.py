#!/usr/bin/env python
import sys
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from elementdata import ElementData

elemdatabase = ElementData()

################################
def getelementcount(labels):
    elems = elemdatabase.elementnr.keys()
    times = np.zeros((len(elems)))
    for l in labels:
        for jdx, elem in enumerate(elems):
            if (l == elem): times[jdx] += 1   
    return times

################################
def getHvcount(labels):
    elems = elemdatabase.elementnr.keys()
    times = np.zeros((len(elems)))
    for l in labels:
        if (l != 'H'):
            for jdx, elem in enumerate(elems):
                if (l == elem): times[jdx] += 1    
    return times

################################
def get_adjacency_types(label, conmat):
    elems = elemdatabase.elementnr.keys()
    bondtypes = np.zeros((len(elems), len(elems))).astype(int)
    natoms = len(label)
    found = np.zeros((natoms, natoms))
    
    for i in range(0, natoms):
        for j in range(i, natoms):
            if(i != j):
                if (conmat[i,j] == 1) and (found[i,j] == 0):
                    for k, elem1 in enumerate(elems):
                        if (label[i] == elem1):
                            for l, elem2 in enumerate(elems):
                                if (label[j] == elem2):
                                    bondtypes[k,l] += 1
                                    if (elem1 != elem2): bondtypes[l,k] += 1
                                    found[i,j] = 1
                                    found[j,i] = 1
                                    break
                            break
    return bondtypes    

################################
def checkchemistry(molecule, references, typ="Max"):
    
    elems = elemdatabase.elementnr.keys()
    maxval = np.zeros((len(references[0].adjtypes), len(references[0].adjtypes)))
    
    for i in range(0,len(references[0].adjtypes)):
        for j in range(0,len(references[0].adjtypes)):
            lst = []
            for ref in references:
                lst.append(ref.adjtypes[i,j])
            maxval[i,j] = np.max(lst)
            
    status = 1  #Good
    if (typ == "Max"): #the bonds[j,k] of the molecule cannot be larger than the maximum within references
        for i in range(0,len(molecule.adjtypes)):
            for j in range(0,len(molecule.adjtypes)):
                if (molecule.adjtypes[i,j] > 0):
                    if (molecule.adjtypes[i,j] > maxval[i,j]):
                        status = 0   #bad
    return status

################################
def getradii(labels):
    radii = []
    for l in labels:
        radii.append(elemdatabase.CovalentRadius2[l])
    return np.array(radii)

################################
def getcentroid(frac):
    natoms = len(frac)
    x = 0
    y = 0
    z = 0
    for idx, l in enumerate(frac):
        x += frac[idx][0]
        y += frac[idx][1]
        z += frac[idx][2]           
    centroid = [float(x/natoms), float(y/natoms), float(z/natoms)]
    return centroid

################################
def extract_from_matrix(entrylist, old_array, dimension=2):

    length = len(entrylist)
 
    if (dimension == 2):
        new_array = np.empty((length, length))
        for idx, row in enumerate(entrylist):
            for jdx, col in enumerate(entrylist):
                new_array[idx,jdx] = old_array[row][col]

    elif (dimension == 1):
        new_array = np.empty((length))
        for idx, val in enumerate(entrylist):
            new_array[idx] = old_array[val]

    return new_array

####################################
def getconec(labels, pos, factor, radii="default"):
    status = 1 #good molecule, no clashes yet
    clash = 0.3
    natoms = len(labels)
    conmat = np.zeros((natoms,natoms))
    connec = np.zeros((natoms))
    mconmat = np.zeros((natoms,natoms))
    mconnec = np.zeros((natoms))
    
    if (radii == "default"): 
        radii = getradii(labels)

    for i in range(0,natoms-1):
        for j in range(i,natoms):
            if (i != j):
                #print(i,j)
                a = np.array(pos[i])
                b = np.array(pos[j])
                dist = np.linalg.norm(a-b)
                thres = (radii[i] + radii[j])*factor
                if (dist <= clash):
                    status = 0   #invalid molecule 
                    print("GETCONEC: Distance", dist, "smaller than clash for atoms", i, j)
                elif (dist <= thres):
                    conmat[i,j] = 1
                    conmat[j,i] = 1
                    if (elemdatabase.elementblock[labels[i]] == 'd' or elemdatabase.elementblock[labels[i]] == 'f' or elemdatabase.elementblock[labels[j]] == 'd' or elemdatabase.elementblock[labels[j]] == 'f'):
                        mconmat[i,j] = 1
                        mconmat[j,i] = 1

    for i in range(0,natoms):
        connec[i]=np.sum(conmat[i,:])
        mconnec[i]=np.sum(mconmat[i,:])

    conmat = conmat.astype(int)
    mconmat = mconmat.astype(int)
    connec = connec.astype(int)
    mconnec = mconnec.astype(int)
    #return status, np.array(conmat), np.array(connec), np.array(mconmat), np.array(mconnec)
    return status, conmat, connec, mconmat, mconnec


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

#######################################################
def get_blocks_V2(matrix):
    # retrieves the blocks from a diagonal block matrix
    startlist = []
    endlist = []

    row=0
    rowold=0
    maxcol=0
    blockcount=0
    col=1
    while col < len(matrix):   #moving within a row
        #print("Evaluating", row, col, maxcol)
        if (matrix[row,col] != 0 ):
            #print("Searching maxcol between", row, col+1, "and", row, len(matrix)-1)
            for idx in range(col+1,len(matrix)):
                if matrix[row, idx] == 1: 
                    if idx > maxcol: maxcol = idx
                    #print("maxcol found at", maxcol)
            row=row+1
            if col > maxcol: 
                maxcol = col
                #print("maxcol updated at", maxcol)
        if (col == len(matrix)-1): # and (row >= maxcol):
            blockcount=blockcount+1
            startlist.append(rowold)
            endlist.append(np.max([row-1, maxcol]))
            #print("adds block with", startlist, endlist, maxcol)
            rowold=row
            row=np.max([row-1, maxcol])+1
            col=np.max([row-1, maxcol])+1
            #print("restarting at", row, col)
            continue
#         elif (col == len(matrix)-1) and (row < maxcol):
#             row += 1
        col += 1

    if (blockcount == 0) and (len(matrix) == 1):     # if a 1x1 matrix is provided, it then finds 1 block
        startlist.append(0)
        endlist.append(0)
        
    return startlist, endlist

def getblocks(matrix):
    # retrieves the blocks from a diagonal block matrix
    startlist = []
    endlist = []

    start=1
    pos=start
    posold=0
    blockcount=0
    j=1
    while j < len(matrix):
        if (matrix[pos-1,j] != 0. ):
            pos=j+1
        if (j == len(matrix)-1):
            blockcount=blockcount+1
            startlist.append(posold)
            endlist.append(pos-1)
            posold=pos
            pos=pos+1
            j=pos-1
            continue
        j += 1

    if (blockcount == 0) and (len(matrix) == 1):     # if a 1x1 matrix is provided, it then finds 1 block
        startlist.append(0)
        endlist.append(0)
        
    return startlist, endlist

def find_groups_within_ligand(ligand):
    
    debug = 0
    if debug >= 1: print(f"DOING LIGAND: {ligand.labels}")
    if debug >= 1: print(f"FIND GROUPS received conmat shape: {ligand.conmat.shape}")
    if debug >= 2: print(f"FIND GROUPS received conmat: {ligand.conmat}")
        
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
    
    if debug >= 1: print(f"FIND GROUPS: connected are: {connected_atoms}")
    if debug >= 1: print(f"FIND GROUPS: unconnected are: {unconnected_atoms}")
    
    # Regenerates the truncated lig.connec
    connec = []
    for idx, c in enumerate(ligand.connec):
        if idx in connected_atoms: connec.append(ligand.connec[idx])
    
    # Does the 
    degree = np.diag(connec)
    lap = columnless - degree
    
    #Copied from split_complex
    graph = csr_matrix(lap)
    perm = reverse_cuthill_mckee(graph)
    gp1 = graph[perm,:]
    gp2 = gp1[:,perm]
    dense=gp2.toarray()

    startlist, endlist = getblocks(dense)
    ngroups = len(startlist)
    
#     np.save("C:/Users/sergi/Documents/PostDoc/Marvel_TM_Database/Get_TMCharge-Tests/BADWOT/Output/matrix", dense)
    
    atomlist= np.zeros((len(dense)))
    for b in range(0,ngroups):
        for i in range(0,len(dense)):
            if (i >= startlist[b]) and (i <= endlist[b]):
                atomlist[i]=b+1
    invperm = inv(perm)
    atomlistperm = [int(atomlist[i]) for i in invperm] 
    
    if debug >= 1: print(f"FIND GROUPS: the {ngroups} groups start at: {startlist}")    

#     CUTMOLEC [['C', 5], ['C', 6], ['C', 8], ['C', 22], ['C', 24], ['C', 30], ['C', 31], ['C', 33], ['C', 34], ['C', 35]]
#     ATOMLISTPERM: [2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        
    groups = []
    for b in range(0,ngroups):
        atlist = []
        for i in range(0,len(atomlistperm)):
            if (atomlistperm[i] == b+1):
                atlist.append(cutmolec[i][1])
        groups.append(atlist)
    
    if debug >= 1: print(f"FIND GROUPS finds {ngroups} as {groups}")
    
    return groups


############################ CLASSES #########################

###############
### ATOM ######
###############
class atom(object):
    def __init__(self, index, label, coord, radii):
        self.version = 'V16'
        self.index = index 
        self.label = label
        self.coord = coord
        self.frac = []
        self.atnum = elemdatabase.elementnr[label]
        self.val = elemdatabase.valenceelectrons[label]  #currently not used
        self.weight = elemdatabase.elementweight[label]  #currently not used
        self.block = elemdatabase.elementblock[label]
        self.radii = radii

    #Adjacency part is simultaneously to creating the ligand or molecule object
    ### Changed in V14
    def adjacencies(self, conmat, mconmat, type="Molecule"):
        self.adjacency = []
        self.metal_adjacency = []

        self.connec=np.sum(conmat)
        if conmat.shape:
            for idx, c in enumerate(conmat):
                if c >= 1: 
                    self.adjacency.append(idx)
        else: 
            self.adjacency.append(conmat)

        if type == "Molecule":
            self.mconnec=np.sum(mconmat)
            if mconmat.shape:
                for idx, c in enumerate(mconmat):
                    if c >= 1: 
                        self.metal_adjacency.append(idx)
            else:
                self.adjacency.append(conmat)

        elif type == "Ligand" or type == "Metal":
            self.mconnec=mconmat    #this has to be improved, now it only receives a number, should receive a vector as above for "molecule"

    #Bonds part is created after the formal charge for each molecule/ligand/metal is decided
    def bonds(self, start, end, order):
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

        for group in zip(start,end,order):
            self.bond.append(group)

        self.nbonds = len(self.bond)
        self.totbondorder = np.sum(self.bond_order)

    def charge(self, charge):
        self.charge = charge

###############
### MOLECULE ##
###############
class molecule(object):
    def __init__(self, name, atlist, label, coord, radii):
        self.version = 'V16'
        self.refcode = ""         #V14
        self.name = name
        self.atlist = atlist
        self.labels = label
        self.coord = coord
        self.radii = radii
        
        self.natoms = len(atlist)
        self.elemcountvec = getelementcount(label)
        self.Hvcountvec = getHvcount(label)
        
        self.frac = []
        self.centroid = []        
        self.tmatrix = []
        
        #Creates Atoms
        self.atnums = []
        self.atoms = []
        for idx, l in enumerate(self.labels):
            newatom = atom(idx, l, self.coord[idx], self.radii[idx])
            self.atnums.append(newatom.atnum)
            self.atoms.append(newatom)

        self.type = 'Other'
        self.eleccount = 0
        self.numH = 0
        for a in self.atoms:
            if (a.atnum == 1): self.numH += 1
            self.eleccount += a.atnum
            if (a.block == 'd') or (a.block == 'f'): self.type = 'Complex'

        if (self.type == 'Complex'):
            self.ligandlist = []
            self.metalist = []
            self.hapticity = False    #V13
            self.hapttype = []        #V13

        # Lists of potentially good variables for this molecule
        self.poscharge = []
        self.posatcharge = []
        self.posobjlist = []
        self.posspin = []
        self.possmiles = []

    #Stores the covalentradii factor and metal factor that were used to generate the molecule
    def information(self, factor, metal_factor):
        self.factor = factor
        self.metal_factor = metal_factor

    # Actual variables for the molecule in the crystal where it comes from:
    def charge(self, atcharge, totcharge, object, smiles):
        self.atcharge = atcharge
        self.totcharge = totcharge
        self.smiles = smiles
        self.object = object

        if (self.type != 'Complex'):
            for idx, a in enumerate(self.atoms):
                a.charge(self.atcharge[idx])

    # Spin State Variables
    def magnetism(self, spin):
        self.spin = spin 

    # Connectivity = Adjacency Matrix. Potentially expandable to Include Bond Types
    def adjacencies(self, conmat, mconmat):
        self.conmat = conmat
        self.mconmat = mconmat
        self.connec = np.zeros((self.natoms))
        self.mconnec = np.zeros((self.natoms))
        for i in range(0,self.natoms):
            self.connec[i]=np.sum(self.conmat[i,:])
            self.mconnec[i]=np.sum(self.mconmat[i,:])
            
        self.totconnec = int(np.sum(self.connec)/2)
        self.totmconnec = int(np.sum(self.mconnec)/2)
        self.adjtypes = get_adjacency_types(self.labels, self.conmat)   #V14
        #self.nbonds = int(np.sum(self.bonds)/2)
    
        for idx, a in enumerate(self.atoms):
            #print("adjacencies sent with", np.array(self.conmat[idx].astype(int)), np.array(self.mconmat[idx].astype(int)))
            a.adjacencies(np.array(self.conmat[idx].astype(int)), np.array(self.mconmat[idx].astype(int)), type = "Molecule")

    #def repr_CM(self, ):
        #self.CM = coulomb_matrix(self.atnums, self.coord)
        # NOTE: don't forget to copy to ligand object when ready

###############
### LIGAND ####
###############
class ligand(object):
    def __init__(self, name, atlist, labels, coord, radii):
        self.version = 'V16'
        self.refcode = ""         #V14
        self.name = name         #creates as ligand index, later modified
        self.atlist = atlist     #atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.labels = labels      #elements
        self.coord = coord       #coordinates
        self.radii = radii

        self.natoms = len(atlist) #number of atoms
        self.type = 'Ligand'
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
        self.grouplist = []          #V14, this info is generated in get_hapticity
        self.hapticity = False    #V13
        self.hapttype = []        #V13
        #self.haptgroups = []      #V14, replaced by grouplist

        self.atnums = []
        self.atoms = []
        for idx, l in enumerate(self.labels):
            newatom = atom(idx, l, self.coord[idx], self.radii[idx])
            self.atnums.append(newatom.atnum)
            self.atoms.append(newatom)

        #Creates atoms and defines charge
        self.eleccount = 0
        self.numH = 0
        for a in self.atoms:
            if (a.atnum == 1): self.numH += 1
            self.eleccount += a.atnum

    #Stores the covalentradii factor and metal factor that were used to generate the molecule
    def information(self, factor, metal_factor):
        self.factor = factor
        self.metal_factor = metal_factor

    def charge(self, atcharge, totcharge, object, smiles):
        self.atcharge = atcharge
        self.totcharge = totcharge
        self.smiles = smiles
        self.object = object

        for idx, a in enumerate(self.atoms):
            a.charge(self.atcharge[idx])

    # Spin State Variables
    def magnetism(self, spin):
        self.spin = spin 

    def adjacencies(self, conmat, mconnec):
        self.conmat = conmat
        self.connec = np.zeros((self.natoms))
        for i in range(0,self.natoms):
            self.connec[i]=np.sum(self.conmat[i,:])

        self.mconnec = mconnec
        self.mconmat = np.zeros((self.natoms, self.natoms)).astype(int)
        self.totconnec = int(np.sum(self.connec))
        self.totmconnec = int(np.sum(self.mconnec))
        self.adjtypes = get_adjacency_types(self.labels, self.conmat)   #V14

        for idx, a in enumerate(self.atoms):
            a.adjacencies(np.array(self.conmat[idx].astype(int)), int(mconnec[idx]), type = "Ligand")

###############
class group(object):
    def __init__(self, atlist, hapticity, hapttype):
        self.version = 'V16'
        self.atlist = atlist     #atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.hapticity = hapticity
        self.hapttype = hapttype

###############
#### METAL ####
###############
class metal(object):
    def __init__(self, name, atlist, label, coord, radii):
        self.version = 'V16'
        self.name = name         #creates as metal index, later modified
        self.atlist = atlist     #atom index list. Numbers refer to the original molecule from where the subroutine is launched
        self.label = label
        self.coord = coord
        self.radii = radii
        self.natom = int(1) #number of atoms
        self.type = 'Metal'
        self.poscharge = []
        self.coord_sphere = [] 
        self.coord_sphere_ID = [] 

        self.atom = atom(name, label, self.coord, self.radii)

    #Stores the covalentradii factor and metal factor that were used to generate the molecule
    def information(self, factor, metal_factor):
        self.factor = factor
        self.metal_factor = metal_factor

    def charge(self, charge):
        self.totcharge = charge
   
    def magnetism(self, spin):
        self.spin = spin 

    def adjacencies(self, mconnec): 
        self.mconnec = mconnec                      #adjacencies matrix with only metal bonds
        self.totmconnec = int(np.sum(mconnec))
        self.atom.adjacencies(np.array(int(0)), int(mconnec), type = "Metal")

##############
#### CELL ####
##############
class Cell (object):
    def __init__(self, refcode, labels, pos, cellvec, cellparam, warning_list):
        
        self.version = 'V16'
        self.refcode = refcode

        self.cellvec = cellvec
        self.cellparam = cellparam 

        self.labels = labels # original_cell_labels
        self.pos = pos       # original_cell_pos
        
        self.refmoleclist = []
        self.moleclist = []
        self.warning_list = warning_list
        
    def print_charge_assignment(self):
        print("=====================================Charges for all species in the unit cell=====================================")
        for unit in self.moleclist:
            if unit.type == "Complex":
                print("\n{}, totcharge {}".format(unit.type, unit.totcharge))
                for metal in unit.metalist:
                    print("\t Metal: {} charge {}".format(metal.label, metal.totcharge))
                for ligand in unit.ligandlist:
                    print("\t Ligand charge {}, {}".format(ligand.totcharge, ligand.smiles))
                    
            elif unit.type == "Other":
                print("{} totcharge {}, {}".format(unit.type, unit.totcharge, unit.smiles))
                
                
    def print_Warning (self):
        
        reason_of_Warning = [
        "Warning! Errors received from getting reference molecules",
        "Warning! Missing H atoms from C in reference molecules", 
        "Warning! Missing H atoms from coordinated water in reference molecules",
        "Warning! Steric clashes while blocking molecules",
        "Warning! Remaining fragments after reconstruction",
        "Warning! No coincidence in the number of atoms between final and initial state",
        "Warning! Empty list of possible charges received for molecule or ligand",
        "Warning! More than one OR no valid possible charge distribution found",
        "Warning! Total net charge does not exist among possible charges"
        ]
        
        # printing original list 
        print ("The Warning type list is : " + str(self.warning_list))

        res = [i for i, val in enumerate(self.warning_list) if val]

        # printing result
#         print ("The list indices having True values are : " + str(res))

        if len(res) == 0:
            print ("No Warnings!")
        else:
            for i in res:
                print(reason_of_Warning[i])
            
########## END OF CLASSES ###########
