import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Tuple
from cell2mol.other import inv
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

#######################################################
def add_atom(labels: list, coords: list, site: int, ligand: object, metalist: list, element: str="H", debug: int=0) -> Tuple[bool, list, list]:
    from cell2mol.other import get_dist
    # This function adds one atom of a given "element" to a given "site=atom index" of a "ligand".
    # It does so at the position of the closest "metal" atom to the "site"
    #:return newlab: labels of the original ligand, plus the label of the new element
    #:return newcoord: same as above but for coordinates

    # Original labels and coordinates are copied
    isadded = False
    posadded = len(labels)
    newlab = labels.copy()
    newcoord = coords.copy()
    newlab.append(str(element))  # One H atom will be added

    if debug >= 2: print("ADD_ATOM: Metalist length", len(metalist))
    # It is adding the element (H, O, or whatever) at the vector formed by the closest TM atom and the "site"
    for idx, a in enumerate(ligand.atoms):
        if idx == site:
            apos = a.coord.copy()
            tgt  = a.get_closest_metal(metalist)
            dist = get_dist(apos, tgt.coord)
            idealdist = a.radii + elemdatabase.CovalentRadius2[element]
            addedHcoords = apos + (metalist[tgt].coord - apos) * (idealdist / dist)  # the factor idealdist/dist[tgt] controls the distance
            newcoord.append([addedHcoords[0], addedHcoords[1], addedHcoords[2]])     # adds H at the position of the closest Metal Atom

            # Evaluates the new adjacency matrix.
            dummy, tmpconmat, tmpconnec = get_adjmatrix(newlab, newcoord, ligand.factor)
            # If no undesired adjacencies have been created, the coordinates are kept
            if tmpconnec[posadded] <= 1:
                isadded = True
                if debug >= 2: print(f"ADD_ATOM: Chosen {tgt} Metal atom. {element} is added at site {site}")
            # Otherwise, coordinates are reset
            else:
                if debug >= 1: print(f"ADD_ATOM: Chosen {tgt} Metal atom. {element} was added at site {site} but RESET due to connec={tmpconnec[posadded]}")
                isadded = False
                newlab = labels.copy()
                newcoord = coords.copy()
    return isadded, newlab, newcoord

#######################################################
def find_closest_metal(atom: object, metalist: list, debug: int=0):
    apos = np.array(atom.coord)
    dist = []
    for met in metalist:
        bpos = np.array(met.coord)
        dist.append(np.linalg.norm(apos - bpos))
    # returns the closest metal atom
    return np.argmin(dist)

################################
def labels2formula(labels: list):
    elems = elemdatabase.elementnr.keys()
    formula=[]
    for z in elems:
        nz = labels.count(z)
        if nz > 1:   formula.append(f"{z}{nz}-")
        if nz == 1:  formula.append(f"{z}-")
    formula = ''.join(formula)[:-1] 
    return formula 

################################
def labels2ratio(labels):
    elems = elemdatabase.elementnr.keys()
    ratio=[]
    for z in elems:
        nz = labels.count(z)
        if nz > 0: ratio.append(nz)
    return ratio

################################
def labels2electrons(labels):
    if type(labels) == list:
        eleccount = 0
        for l in labels:
            eleccount += elemdatabase.elementnr[l]
    elif type(labels) == str:
        eleccount = elemdatabase.elementnr[labels]
    return eleccount 

################################
def get_metal_idxs(labels: list, debug: int=0):
    from cell2mol.elementdata import ElementData
    elemdatabase = ElementData()
    metal_indices = []
    for idx, l in enumerate(labels):
        if (elemdatabase.elementblock[l] == 'd' or elemdatabase.elementblock[l] == 'f'): metal_indices.append(idx)
    return metal_indices

################################
def get_metal_species(labels: list):
    from cell2mol.elementdata import ElementData
    elemdatabase = ElementData()
    metal_species = []
    elems = list(set(labels))
    for idx, l in enumerate(elems):
        if l[-1].isdigit(): label = l[:-1]
        else: label = l
        if (elemdatabase.elementblock[label] == 'd' or elemdatabase.elementblock[label] == 'f') and l not in metal_species: metal_species.append(l)
    return metal_species

################################
def get_element_count(labels: list, heavy_only: bool=False) -> np.ndarray:
    elems = list(elemdatabase.elementnr.keys())
    count = np.zeros((len(elems)),dtype=int)
    for l in labels:
        for jdx, elem in enumerate(elems):
            if l == elem:                             count[jdx] += 1
            if (l == 'H' or l == 'D') and heavy_only: count = 0
    return count

################################
def get_adjacency_types(label: list, conmat: np.ndarray) -> np.ndarray:
    elems = elemdatabase.elementnr.keys()
    natoms = len(label)
    bondtypes = np.zeros((len(elems), len(elems)),dtype=int)
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
def get_radii(labels: list) -> np.ndarray:
    radii = []
    for l in labels:
        if l[-1].isdigit(): label = l[:-1]
        else: label = l
        radii.append(elemdatabase.CovalentRadius3[label])
    return np.array(radii)

####################################
def get_adjmatrix(labels: list, pos: list, cov_factor: float=1.3, radii="default", metal_only: bool=False) -> Tuple[int, list, list]:
    isgood = True 
    clash_threshold = 0.3
    natoms = len(labels)
    adjmat = np.zeros((natoms, natoms))
    adjnum = np.zeros((natoms))

    # Sometimes argument radii np.ndarry, or list
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        if type(radii) == str:
            if radii == "default":
                radii = get_radii(labels)

    # Creates Adjacency Matrix
    for i in range(0, natoms - 1):
        for j in range(i, natoms):
            if i != j:
                a = np.array(pos[i])
                b = np.array(pos[j])
                dist = np.linalg.norm(a - b)
                thres = (radii[i] + radii[j]) * cov_factor
                if dist <= clash_threshold:
                    isgood = False # invalid molecule
                    print("Adjacency Matrix: Distance", dist, "smaller than clash for atoms", i, j)
                elif dist <= thres:
                    if not metal_only: 
                        adjmat[i, j] = 1
                        adjmat[j, i] = 1
                    if metal_only: 
                        if (elemdatabase.elementblock[labels[i]] == "d"
                        or elemdatabase.elementblock[labels[i]] == "f"
                        or elemdatabase.elementblock[labels[j]] == "d"
                        or elemdatabase.elementblock[labels[j]] == "f"):
                            adjmat[i, j] = 1
                            adjmat[j, i] = 1

    # Sums the adjacencies of each atom to obtain "adjnum" 
    for i in range(0, natoms):
        adjnum[i] = np.sum(adjmat[i, :])

    adjmat = adjmat.astype(int)
    adjnum = adjnum.astype(int)
    return isgood, adjmat, adjnum

####################################
def get_blocks(matrix: np.ndarray) -> Tuple[list, list]:
    # retrieves the blocks from a diagonal block matrix
    startlist = []  # List including the starting atom for all blocks
    endlist = []    # List including the final atom for all blocks
    start = 1
    pos = start
    posold = 0
    blockcount = 0
    j = 1
    while j < len(matrix):
        if matrix[pos - 1, j] != 0.0: pos = j + 1
        if j == len(matrix) - 1:
            blockcount = blockcount + 1
            startlist.append(posold)
            endlist.append(pos - 1)
            posold = pos
            pos = pos + 1
            j = pos - 1
            continue
        j += 1

    if (blockcount == 0) and (len(matrix) == 1):  # if a 1x1 matrix is provided, it then finds 1 block
        startlist.append(0)
        endlist.append(0)
    return startlist, endlist

#########################
def count_species(labels: list, pos: list, radii: list=None, indices: list=None, cov_factor: float=1.3, debug: int=0) -> Tuple[bool, list]:
    # Gets the covalent radii
    if radii is None:    radii = get_radii(labels)
    if indices is None:  indices = [*range(0,len(labels),1)]

    # Computes the adjacency matrix of what is received
    # isgood indicates whether the adjacency matrix could be built normally, or errors were detected. Typically, those errors are steric clashes
    isgood, adjmat, adjnum = get_adjmatrix(labels, pos, cov_factor, radii)
    if not isgood: return int(0)

    degree = np.diag(adjnum)  # creates a matrix with adjnum as diagonal values. Needed for the laplacian
    lap = adjmat - degree     # computes laplacian

    # creates block matrix
    graph = csr_matrix(lap)
    perm = reverse_cuthill_mckee(graph)
    gp1 = graph[perm, :]
    gp2 = gp1[:, perm]
    dense = gp2.toarray()

    # detects blocks in the block diagonal matrix called "dense"
    startlist, endlist = get_blocks(dense)

    nblocks = len(startlist)
    return nblocks

####################################
def split_species(labels: list, pos: list, radii: list=None, indices: list=None, cov_factor: float=1.3, debug: int=0) -> Tuple[bool, list]:
    ## Function that identifies connected groups of atoms from their atomic coordinates and labels.

    # Gets the covalent radii
    if radii is None:    radii = get_radii(labels)
    if indices is None:  indices = [*range(0,len(labels),1)]

    # Computes the adjacency matrix of what is received
    # isgood indicates whether the adjacency matrix could be built normally, or errors were detected. Typically, those errors are steric clashes
    isgood, adjmat, adjnum = get_adjmatrix(labels, pos, cov_factor, radii)
    if not isgood: return None

    degree = np.diag(adjnum)  # creates a matrix with adjnum as diagonal values. Needed for the laplacian
    lap = adjmat - degree     # computes laplacian

    # creates block matrix
    graph = csr_matrix(lap)
    perm = reverse_cuthill_mckee(graph)
    gp1 = graph[perm, :]
    gp2 = gp1[:, perm]
    dense = gp2.toarray()

    # detects blocks in the block diagonal matrix called "dense"
    startlist, endlist = get_blocks(dense)

    nblocks = len(startlist)
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
        atlist = []    # atom indices in the original ordering
        for i in range(0, len(atomlistperm)):
            if atomlistperm[i] == b + 1:
                atlist.append(indices[i])
        blocklist.append(atlist)
    return blocklist

#####################
def merge_atoms(atoms):
    labels = [] 
    coord  = [] 
    for a in atoms:
        labels.append(a.label) 
        coord.append(a.coord) 
    return labels, coord

#################################
def compare_atoms(at1, at2, check_coordinates: bool=False, debug: int=0):
    if debug > 0: 
        print("Comparing Atoms")
        print(at1)
        print(at2)
    # Compares Species, Coordinates, Charge and Spin
    if (at1.label != at2.label): return False
    if check_coordinates:
        if (at1.coord[0] != at2.coord[0]): return False
        if (at1.coord[1] != at2.coord[1]): return False
        if (at1.coord[2] != at2.coord[2]): return False
    if hasattr(at1,"charge") and hasattr(at2,"charge"):
        if (at1.charge != at2.charge): return False
    if hasattr(at1,"spin") and hasattr(at2,"spin"):
        if (at1.spin != at2.spin): return False
    return True

#################################
def compare_metals (at1, at2, check_coordinates: bool=False, debug: int=0):
    if at1.subtype != "metal" or at2.subtype != "metal": return False
    if debug > 0: 
        print("Comparing Metals")
        print(at1.label)
        print(at2.label)

    if (at1.label != at2.label): return False

    if not hasattr(at1,"coord_sphere"): at1.get_coord_sphere()
    if not hasattr(at2,"coord_sphere"): at2.get_coord_sphere()
    at1_coord_sphere_formula = labels2formula ([atom.label for atom in at1.coord_sphere])
    at2_coord_sphere_formula = labels2formula ([atom.label for atom in at2.coord_sphere])    
    if (at1_coord_sphere_formula != at2_coord_sphere_formula) : return False
    
    if check_coordinates:
        if (at1.coord[0] != at2.coord[0]): return False
        if (at1.coord[1] != at2.coord[1]): return False
        if (at1.coord[2] != at2.coord[2]): return False
        
    return True

#################################
def compare_species(mol1, mol2, check_coordinates: bool=False, debug: int=0):
    if debug > 0: 
        print("Comparing Species")
        print(mol1)
        print(mol2)
    # a pair of species is compared on the basis of:
    # 1) the total number of atoms
    if (mol1.natoms != mol2.natoms): return False

    # 2) the total number of electrons (as sum of atomic number)
    if (mol1.eleccount != mol2.eleccount): return False

    # 3) the number of atoms of each type
    if not hasattr(mol1,"element_count"): mol1.set_element_count()
    if not hasattr(mol2,"element_count"): mol2.set_element_count()
    for kdx, elem in enumerate(mol1.element_count):
        if elem != mol2.element_count[kdx]: return False       

    # 4) the number of adjacencies between each pair of element types
    if not hasattr(mol1,"adj_types"):     mol1.set_adj_types()
    if not hasattr(mol2,"adj_types"):     mol2.set_adj_types()
    for kdx, elem in enumerate(mol1.adj_types):
        for ldx, elem2 in enumerate(elem):
            if elem2 != mol2.adj_types[kdx, ldx]: return False

    if check_coordinates:
        # 5) Finally, the coordinates if the user wants it
        for idx in range(0,mol1.natoms,1):
            if (mol1.coord[idx][0] !=  mol2.coord[idx][0]): return False
            if (mol1.coord[idx][1] !=  mol2.coord[idx][1]): return False
            if (mol1.coord[idx][2] !=  mol2.coord[idx][2]): return False
    return True

#################################

