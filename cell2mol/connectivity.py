import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Tuple
from cell2mol.other import inv, extract_from_list
from cell2mol.elementdata import ElementData
from cell2mol.read_write import writexyz
import os
import networkx as nx
import re
from cell2mol.missingH import get_missingH_from_adjacency
elemdatabase = ElementData()

#######################################################
def add_atom(labels: list, coords: list, site: int, ligand: object, metalist: list, element: str="H", removed_idx: list=None, unconditional: bool=False, debug: int=0) -> Tuple[bool, list, list]:
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

    if debug > 2: print("ADD_ATOM: Metalist length", len(metalist))
    if debug > 2: print("ADD_ATOM: Ligand Atoms", len(ligand.atoms))
    if debug >= 2: print("ADD_ATOM: site=", site)
    if debug >= 2: print("ADD_ATOM: target ligand atom ", ligand.atoms[site].label)
    # It is adding the element (H, O, or whatever) at the vector formed by the closest TM atom and the "site"
    for idx, a in enumerate(ligand.atoms):
        if idx == site:
            apos = np.array(a.coord.copy())
            tgt  = a.get_closest_metal(metalist)
            if debug >= 2: print(f"ADD_ATOM: evaluating atom position={apos} and metal position={tgt.coord}")
            # ligand_idx = tgt.get_parent_index("ligand")
            metal_idx = tgt.get_parent_index("molecule")
            dist = get_dist(apos, tgt.coord)
            idealdist = a.radii + elemdatabase.CovalentRadius3[element]
            # addedHcoords = apos + (tgt.coord - apos) * (idealdist / dist)  # the factor idealdist/dist[tgt] controls the distance
            # newcoord.append([addedHcoords[0], addedHcoords[1], addedHcoords[2]])     # adds H at the position of the closest Metal Atom
            addedHcoords = point_along_vector(apos, tgt.coord, idealdist)
            newcoord.append([addedHcoords[0], addedHcoords[1], addedHcoords[2]])

            # Evaluates the new adjacency matrix.
            dummy, tmpconmat, tmpconnec = get_adjmatrix(newlab, newcoord, ligand.cov_factor)
            # if debug >= 2: print(f"ADD_ATOM: received {newlab=}")
            # if debug >= 2: print(f"ADD_ATOM: received {tmpconmat=}")
            # if debug >= 2: print(f"ADD_ATOM: received {tmpconnec=}")
            if debug >= 2: print(f"ADD_ATOM: received tmpconnec[posadded]={int(tmpconnec[posadded])}")
            newlab_with_metal = newlab.copy()
            newcoord_with_metal = newcoord.copy()
            newlab_with_metal.append(tgt.label)
            newcoord_with_metal.append(tgt.coord)
            #writexyz(os.getcwd(), f"target_atom_{a.label}_{apos[0]}_newcoord_with_H_new{addedHcoords[0]}.xyz", newlab_with_metal, newcoord_with_metal)
            # If no undesired adjacencies have been created, the coordinates are kept
            if  unconditional:
                isadded = True
                if debug >= 1: print(f"ADD_ATOM: {element} is added at site {site} of ligand {ligand.formula} to generate a protonation state")
            elif tmpconnec[posadded] <= 1:
                isadded = True
                if debug >= 2: print(f"ADD_ATOM: Chosen Metal index {metal_idx}. {element} is added at site {site}")
            # Otherwise, coordinates are reset
            elif tmpconnec[posadded] > 1 and removed_idx is not None and len(removed_idx) > 0 :
                set1 = set([i for i, c in enumerate(tmpconmat[posadded]) if c != 0 ])
                set2 = set(removed_idx)
                if debug >= 1: print(f"ADD_ATOM: {element} is connected with ligand atoms with indices {set1}. previously removed indices {set2}")
                result = list(set1 - set2)
                if debug >= 2: print(f"ADD_ATOM: {element} is connected with ligand atoms with indices {result=}. ")
                if len(result) <= 1:
                    isadded = True
                    if debug >= 2: print(f"ADD_ATOM: Chosen Metal index {metal_idx}. {element} is added at site {site} after previously removing atom {removed_idx}")
                else:
                    if debug >= 1: print(f"ADD_ATOM: Chosen Metal index {metal_idx}. {element} was added at site {site} but RESET due to connec={tmpconnec[posadded]}")
                    if debug > 2: writexyz(os.getcwd(), f"target_atom_{a.label}_{apos[0]}_newcoord_with_H_new{addedHcoords[0]}.xyz", newlab_with_metal, newcoord_with_metal)
                    isadded = False
                    newlab = labels.copy()
                    newcoord = coords.copy()

    return isadded, newlab, newcoord
#######################################################
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
def is_haptic_ring(labels, coord):
    """ Check if the group is a ring """
    isgood, adjmat, adjnum = get_adjmatrix(labels, coord)

    # Convert adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(np.array(adjmat))

    # Check if the graph is connected
    if not nx.is_connected(G):
        return False  # If not connected, can't form a single ring
    
    # Check for cycles and ensure the graph forms a simple cycle
    cycle_basis = nx.cycle_basis(G)
    
    # Check if there's exactly one cycle that includes all nodes (simple ring)
    if len(cycle_basis) == 1 and len(cycle_basis[0]) == len(G.nodes):
        print("Ring group", len(labels), labels)
        return True  # The graph represents a ring compound

    return False  # Otherwise, not a ring compound

################################
def add_hydrogen_to_carbon (labels: list, coords: list, site: int, ligand: object, element: str="H", debug: int=0) -> Tuple[bool, list, list]:
    # Original labels and coordinates are copied
    isadded = True
    newlab = labels.copy()
    newcoord = coords.copy()   
    for idx, a in enumerate(ligand.atoms):
        if idx == site:
            apos = np.array(a.coord.copy())
            bonded_atom_coord = []
            bonded_atom_labels = []
            
            for adj in a.adjacency:
                n_label = ligand.get_parent("molecule").labels[adj]
                n_coord = ligand.get_parent("molecule").coord[adj]
                if elemdatabase.elementblock[n_label] == 'd' or elemdatabase.elementblock[n_label] == 'f':
                    pass
                else:
                    bonded_atom_coord.append(n_coord)
                    bonded_atom_labels.append(n_label)
            if debug >= 2: print("Adjacency", a.adjacency, bonded_atom_labels)
            ismissingH, report, num_missingH = get_missingH_from_adjacency(a.atnum, a.coord, bonded_atom_coord, bonded_atom_labels)
            print("ADD_H_to_CARBON: ismissingH", ismissingH, report, num_missingH)
            if len(bonded_atom_labels) == 2:
                Hs = place_hydrogens(apos, bonded_atom_coord[0], bonded_atom_coord[1])
                if Hs.shape[0] == 2:
                    newcoord.append(Hs[0])
                    newcoord.append(Hs[1])
                    newlab.extend([str(element), str(element)])
                    if debug >= 2: print(f"ADD_H_to_CARBON: Added two {element} to atom {site} with: a.mconnec={a.mconnec} a.connec={a.connec}  and label={a.label}")
                elif Hs.shape[0] == 1:
                    newcoord.append(Hs[0])
                    newlab.extend([str(element)])
                    if debug >= 2: print(f"ADD_H_to_CARBON: Added one {element} to atom {site} with: a.mconnec={a.mconnec} a.connec={a.connec}  and label={a.label}")

            # print(a.adjacency)
            # neighbors_in_ligand = []
            # for adj in a.adjacency:
            #     n_label = ligand.get_parent("molecule").labels[adj]
            #     if elemdatabase.elementblock[n_label] == 'd' or elemdatabase.elementblock[n_label] == 'f':
            #         pass
            #     else :
            #         if debug >= 2: print(f"ADD_TWO_ATOMS: {n_label} is not a transition metal, adding to neighbors_in_ligand")
            #         neighbors_in_ligand.append(adj)
            # if len(neighbors_in_ligand) == 2:
            #     N1 = ligand.get_parent("molecule").coord[neighbors_in_ligand[0]]
            #     N2 = ligand.get_parent("molecule").coord[neighbors_in_ligand[1]]
            #     H1, H2 = add_two_hydrogens_sp3(apos, N1, N2)
            #     newcoord.append(H1)
            #     newcoord.append(H2)
            #     if debug >= 2:
            #         print(f"ADD_TWO_ATOMS: Added two {element} to atom {site} with: a.mconnec={a.mconnec} a.connec={a.connec}  and label={a.label}")
    return isadded, newlab, newcoord

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

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

def add_two_hydrogens_sp3(C, N1, N2, r_CH=1.09):
    """
    Place two hydrogens on carbon C given two neighbor atoms N1, N2.
    Assumes sp3 tetrahedral around C. Returns positions H1, H2.
    """
    # Unit vectors from C toward existing neighbors
    a = normalize(N1 - C)
    b = normalize(N2 - C)

    # Tetrahedral template (four directions)
    u1 = normalize(np.array([ 1,  1,  1], dtype=float))
    u2 = normalize(np.array([ 1, -1, -1], dtype=float))
    u3 = normalize(np.array([-1,  1, -1], dtype=float))
    u4 = normalize(np.array([-1, -1,  1], dtype=float))

    # Align template u1,u2 to the actual directions a,b via Kabsch
    P = np.stack([u1, u2], axis=1)  # 3x2
    Q = np.stack([a,  b ], axis=1)  # 3x2
    R = kabsch_rotation(P, Q)

    # Rotate the remaining template directions to get H directions
    dH1 = normalize(R @ u3)
    dH2 = normalize(R @ u4)

    # Place hydrogens at tetrahedral distance
    H1 = C + r_CH * dH1
    H2 = C + r_CH * dH2
    return H1, H2


def place_hydrogens(C, N1, N2, r_CH=1.09, hybridization="auto",
                    sp2_angle_window=(95, 145), sp3_angle_window=(95, 125)):
    """
    Place hydrogens on a carbon with two existing neighbors.

    Parameters
    ----------
    C, N1, N2 : (3,) arrays
        3D coordinates of carbon and its two neighbors.
    r_CH : float
        C–H bond length (Å). ~1.09 Å is fine for sp2/sp3.
    hybridization : {'auto','sp2','sp3'}
        - 'auto': detect from angle between N1–C and N2–C
        - 'sp2' : force one H in trigonal planar geometry
        - 'sp3' : force two H in tetrahedral geometry
    sp2_angle_window : (lo, hi) degrees
        Angle window to consider geometry as sp2 in 'auto' mode (default ~120° ±).
    sp3_angle_window : (lo, hi) degrees
        Angle window to consider geometry as sp3 in 'auto' mode (default ~109.5° ±).

    Returns
    -------
    Hs : (k,3) array
        Coordinates of placed hydrogens (k=1 for sp2, k=2 for sp3).

    Raises
    ------
    ValueError
        If geometry is degenerate or 'auto' cannot classify reliably.
    """
    C = np.asarray(C, float)
    N1 = np.asarray(N1, float)
    N2 = np.asarray(N2, float)

    a = normalize(N1 - C)
    b = normalize(N2 - C)

    # Angle between neighbors
    cosang = np.clip(a @ b, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))

    def add_sp2():
        # In-plane bisector opposite to existing bonds
        dH = -(a + b)
        if np.linalg.norm(dH) < 1e-8:
            raise ValueError("Neighbors nearly opposite (sp-like). Cannot place sp2 hydrogen reliably.")
        dH = normalize(dH)
        return np.array([C + r_CH * dH])

    def add_sp3():
        # Tetrahedral template (four directions)
        u1 = normalize(np.array([ 1,  1,  1], float))
        u2 = normalize(np.array([ 1, -1, -1], float))
        u3 = normalize(np.array([-1,  1, -1], float))
        u4 = normalize(np.array([-1, -1,  1], float))

        # Align template u1,u2 to actual directions a,b (order doesn't matter much)
        P = np.stack([u1, u2], axis=1)  # 3x2
        Q = np.stack([a,  b ], axis=1)  # 3x2
        R = kabsch_rotation(P, Q)

        dH1 = normalize(R @ u3)
        dH2 = normalize(R @ u4)
        H1 = C + r_CH * dH1
        H2 = C + r_CH * dH2
        return np.vstack([H1, H2])

    # Decide hybridization
    mode = hybridization.lower()
    if mode == "auto":
        # Prefer sp3 if close to tetrahedral, else sp2 if closer to trigonal
        in_sp3 = (sp3_angle_window[0] <= angle <= sp3_angle_window[1])
        in_sp2 = (sp2_angle_window[0] <= angle <= sp2_angle_window[1])

        if in_sp3 and not in_sp2:
            return add_sp3()
        if in_sp2 and not in_sp3:
            return add_sp2()
        # If ambiguous, pick the closer target angle
        target_sp3 = 109.47
        target_sp2 = 120.0
        if abs(angle - target_sp3) < abs(angle - target_sp2):
            return add_sp3()
        else:
            return add_sp2()

    elif mode == "sp3":
        return add_sp3()
    elif mode == "sp2":
        return add_sp2()
    else:
        raise ValueError("hybridization must be 'auto', 'sp2', or 'sp3'.")

# -----------------------
# Example usage:
# C  = np.array([0.0, 0.0, 0.0])
# N1 = np.array([1.54, 0.0, 0.0])        # e.g., a C–C bond
# N2 = np.array([-0.5, 1.4, 0.0])        # another neighbor
# H1, H2 = add_two_hydrogens_sp3(C, N1, N2)
# print(H1, H2)
################################
def labels2formula(labels: list):
    elems = elemdatabase.elementnr.keys()
    formula=[]
    for z in elems:
        nz = list(labels).count(z)
        if nz > 1:   formula.append(f"{z}{nz}-")
        if nz == 1:  formula.append(f"{z}-")
    formula = ''.join(formula)[:-1] 
    return formula 

################################
def labels2ratio(labels):
    elems = elemdatabase.elementnr.keys()
    ratio=[]
    for z in elems:
        nz = list(labels).count(z)
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
def get_alkali_alkaline_earth_metal_idxs(labels: list, debug: int=0):
    """ alkali metals (Group 1) and alkaline earth metals (Group 2)  """
    non_transition_metal_indices = []
    for idx, l in enumerate(labels):
        if elemdatabase.elementgroup[l]==1 and l != "H" and l != "D": # Alkali Metals
            non_transition_metal_indices.append(idx)
        elif elemdatabase.elementgroup[l]==2 :     # Alkaline Earth Metals
            non_transition_metal_indices.append(idx)
    return non_transition_metal_indices
#################################
def get_non_transition_metal_idxs(labels: list, debug: int=0):
    """ alkali metals (Group 1) and alkaline earth metals (Group 2)  """
    non_transition_metal_indices = []
    for idx, l in enumerate(labels):
        # if elemdatabase.elementgroup[l]==1 and l != "H" and l != "D": # Alkali Metals
        #     non_transition_metal_indices.append(idx)
        # elif elemdatabase.elementgroup[l]==2 :     # Alkaline Earth Metals
        #     non_transition_metal_indices.append(idx)
        if l in ["Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi", "Po", "At"]: # Post-Transition Metals
            non_transition_metal_indices.append(idx)
        elif l in ["B", "Si", "Ge", "As", "Sb", "Te"] : # Metalloids
            non_transition_metal_indices.append(idx)
    return non_transition_metal_indices

#################################
def get_post_transition_metal_idxs(labels: list, debug: int=0):
    """ Post-Transition Metals """
    post_transition_metal_indices = []
    for idx, l in enumerate(labels):
        if l in ["Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi"]: # Post-Transition Metals
            post_transition_metal_indices.append(idx)
    return post_transition_metal_indices

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
        # if elemdatabase.elementgroup[label] == 1 and label != "H":
        #     radii.append(elemdatabase.CovalentRadius2[label])
        # elif elemdatabase.elementgroup[label] == 2:
        #     radii.append(elemdatabase.CovalentRadius2[label])
        # else:
        #     radii.append(elemdatabase.CovalentRadius3[label])
    return np.array(radii)

####################################
def get_adjmatrix(labels: list, pos: list, cov_factor: float=1.3, radii="default", metal_only: bool=False) -> Tuple[int, list, list]:
    
    isgood = True 
    clash_threshold = 0.3
    natoms = len(labels)
    adjmat = np.zeros((natoms, natoms))
    adjnum = np.zeros((natoms))
    madjmat = np.zeros((natoms, natoms))
    madjnum = np.zeros((natoms))

    metal_idxs = get_metal_idxs(labels)
    alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(labels)

    add_factor = 0.45
    #add_factor = 0.3
    # Sometimes argument radii np.ndarry, or list
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        if type(radii) == str:
            if radii == "default":
                radii = get_radii(labels)

    # Creates Adjacency Matrix
    for i in range(natoms - 1):
        for j in range(i+1, natoms):      
            a = np.array(pos[i])
            b = np.array(pos[j])
            dist = np.linalg.norm(a - b)
            thres = (radii[i] + radii[j]) + add_factor
            #thres = min((radii[i] + radii[j]) * cov_factor, (radii[i] + radii[j]) + add_factor)
            # if thres - (radii[i] + radii[j]) > 0.8:
            #     thres = (radii[i] + radii[j]) + add_factor
            if dist <= clash_threshold:
                isgood = False # invalid molecule
                print("Adjacency Matrix: Distance", round(dist, 3), "smaller than clash for atoms", i, j, labels[i], labels[j], a, b, cov_factor)
            elif dist <= thres:
                # if not metal_only: 
                adjmat[i, j] = 1
                adjmat[j, i] = 1
                # if len(get_alkali_alkaline_earth_metal_idxs([labels[i], labels[j]])) > 0:
                #     adjmat[i, j] = 0
                #     adjmat[j, i] = 0
                #     print("Adjacency Matrix: Set Zeros for Alkali or Alkaline Earth Metal", labels[i], labels[j], f"{i=}", f"{j=}", f"{adjmat[i, j]=}")   
                if metal_only: 
                    if (elemdatabase.elementblock[labels[i]] == "d"
                    or elemdatabase.elementblock[labels[i]] == "f"
                    or elemdatabase.elementblock[labels[j]] == "d"
                    or elemdatabase.elementblock[labels[j]] == "f"):
                        madjmat[i, j] = 1
                        madjmat[j, i] = 1
                    # elif len(get_non_transition_metal_idxs([labels[i], labels[j]])) > 0:
                    #     madjmat[i, j] = 1
                    #     madjmat[j, i] = 1
                    elif len(get_alkali_alkaline_earth_metal_idxs([labels[i], labels[j]])) > 0:
                        madjmat[i, j] = 1
                        madjmat[j, i] = 1    
                    if len(metal_idxs)== 0 and len(alkali_alkaline_earth_metal_idxs) == 0:
                        if len(get_post_transition_metal_idxs([labels[i], labels[j]])) > 0:
                            madjmat[i, j] = 1
                            madjmat[j, i] = 1

    # Corrects valence violations
    isgood_valence, adjmat, madjmat = correct_valence_violation(adjmat, madjmat, labels, pos, radii)
    isgood = isgood and isgood_valence

    for i in range(0, natoms):
        adjnum[i] = np.sum(adjmat[i, :])
        madjnum[i] = np.sum(madjmat[i, :])
    
    adjmat = adjmat.astype(int)
    adjnum = adjnum.astype(int)
    madjmat = madjmat.astype(int)
    madjnum = madjnum.astype(int)

    if not metal_only: 
        return isgood, adjmat, adjnum
    else:
        return isgood, madjmat, madjnum

####################################
def correct_valence_violation(adjmat, madjmat, labels: list, pos: list, radii: list):
    from cell2mol.xyz2mol import atomic_valence
    natoms = len(labels)
    isgood = True
    #Checks if the valence of the atoms is correct
    metal_idxs = get_metal_idxs(labels)
    non_transition_metal_idxs = get_non_transition_metal_idxs(labels)
    post_transition_metal_idxs = get_post_transition_metal_idxs(labels)
    alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(labels)
    allowed = set(metal_idxs) | set(post_transition_metal_idxs) | set(alkali_alkaline_earth_metal_idxs)  

    for i in range(0, natoms):
        indices = np.where(adjmat[i, :] != 0)[0]
        num_in_allowed = len(set(indices) & allowed)
        if num_in_allowed == len(indices):
            continue # all connected atoms are metals
        a = np.array(pos[i])
        valence = np.sum(adjmat[i, :])
        atomicNum = elemdatabase.elementnr[labels[i]]
        if atomic_valence[atomicNum] == []:
            max_valence = 0
        else:
            max_valence = max(atomic_valence[atomicNum]) 
        if valence - num_in_allowed > max_valence:
            print("Adjacency Matrix: Atom", i, labels[i], "has", valence, "valence bigger than allowed max valence", max_valence , "with metal bonding",  num_in_allowed, "in allowed total valence", valence)    
            # if (i in alkali_alkaline_earth_metal_idxs or 
            #     i in post_transition_metal_idxs or 
            #     i in non_transition_metal_idxs):
            if i in alkali_alkaline_earth_metal_idxs or i in post_transition_metal_idxs:
                for j in indices:
                    b = np.array(pos[j])
                    dist = np.linalg.norm(a - b)
                    margin = dist - (radii[i] + radii[j])
                    print("Adjacency Matrix: Atom", i, labels[i], "is connected to", j, labels[j], "distance", round(dist, 3), "bond margin", round(margin, 3))                       
            else:
                connections = []                        
                for j in indices:
                    b = np.array(pos[j])
                    dist = np.linalg.norm(a - b)
                    margin = dist - (radii[i] + radii[j])
                    connections.append((j, margin))
                    print("Adjacency Matrix: Atom", i, labels[i], "is connected to", j, labels[j], "distance", round(dist, 3), "bond margin", round(margin, 3))
                sorted_connections = sorted(connections, key=lambda x: x[1], reverse=True)
                num_to_remove = len(connections) - max_valence - num_in_allowed
                for idx in range(num_to_remove):
                    j, rem = sorted_connections[idx]
                    if rem > 0.2: # Only remove bonds with a significant margin
                        adjmat[i, j] = 0
                        adjmat[j, i] = 0
                        madjmat[i, j] = 0
                        madjmat[j, i] = 0
                        print(f"Adjacency Matrix: Removed bond {i} ({labels[i]}) - {j} ({labels[j]}) (margin = {round(rem, 3)})")
                    else:
                        print(f"Adjacency Matrix: Not removing bond {i} ({labels[i]}) - {j} ({labels[j]}) (margin = {round(rem, 3)})")
    return isgood, adjmat, madjmat

####################################
def correct_valence_violation_v1(adjmat, madjmat, labels: list, pos: list, radii: list):

    from cell2mol.xyz2mol import atomic_valence
    natoms = len(labels)
    isgood = True
    #Checks if the valence of the atoms is correct
    metal_idxs = get_metal_idxs(labels)
    non_transition_metal_idxs = get_non_transition_metal_idxs(labels)
    post_transition_metal_idxs = get_post_transition_metal_idxs(labels)
    alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(labels)
    allowed = set(metal_idxs) | set(post_transition_metal_idxs) | set(alkali_alkaline_earth_metal_idxs)
    
    for i in range(0, natoms):
        indices = np.where(adjmat[i, :] != 0)[0]
        num_in_allowed = len(set(indices) & allowed)

        a = np.array(pos[i])
        if set(indices).issubset(allowed):
            pass
        else:
            valence = np.sum(adjmat[i, :])
            atomicNum = elemdatabase.elementnr[labels[i]]
            if atomic_valence[atomicNum] == []:
                max_valence = 0
            else:
                max_valence = max(atomic_valence[atomicNum])    
            num_in_allowed = sum(1 for j in indices if j in allowed)
            
            if valence - num_in_allowed > max_valence:
                print(num_in_allowed)
            #if valence > max_valence:
                print("Adjacency Matrix: Atom", i, labels[i], "has", valence, "valence bigger than allowed max valence", max_valence)                
                if i in non_transition_metal_idxs:
                    for j in indices:
                        if j in metal_idxs or j in alkali_alkaline_earth_metal_idxs:
                            b = np.array(pos[j])
                            dist = np.linalg.norm(a - b)
                            margin = dist - (radii[i] + radii[j])
                            #connections.append((j, margin))                                
                            adjmat[i, j] = 0
                            adjmat[j, i] = 0
                            madjmat[i, j] = 0
                            madjmat[j, i] = 0
                            print(f"Adjacency Matrix: Removed bond {i} ({labels[i]}) - {j} ({labels[j]}) (margin = {round(margin, 3)})")
                    newindices = np.where(adjmat[i, :] != 0)[0]
                    new_valence = np.sum(adjmat[i, :])
                    if new_valence > max_valence:
                        print("Adjacency Matrix: Atom", i, labels[i], "still has", new_valence, "valence bigger than allowed max valence", max_valence)
                        print("Adjacency Matrix: Atom", i, labels[i], "is a non-transition metal with valence bigger than allowed max valence", max_valence, "and is connected to", newindices, [labels[j] for j in newindices])
                        #isgood = False
                        connections = []                        
                        for j in indices:
                            b = np.array(pos[j])
                            dist = np.linalg.norm(a - b)
                            margin = dist - (radii[i] + radii[j])
                            connections.append((j, margin))
                            print("Adjacency Matrix: Atom", i, labels[i], "is connected to", j, labels[j], "distance", round(dist, 3), "bond margin", round(margin, 3))
                        sorted_connections = sorted(connections, key=lambda x: x[1], reverse=True)
                        num_to_remove = len(connections) - max_valence
                        for idx in range(num_to_remove):
                            j, rem = sorted_connections[idx]
                            adjmat[i, j] = 0
                            adjmat[j, i] = 0
                            madjmat[i, j] = 0
                            madjmat[j, i] = 0
                            print(f"Adjacency Matrix: Removed bond {i} ({labels[i]}) - {j} ({labels[j]}) (margin = {round(rem, 3)})")

                    else:
                        print("Adjacency Matrix: Atom", i, labels[i], "is a non-transition metal with valence bigger than allowed max valence", max_valence, \
                              "and is now connected to", newindices, new_valence, [labels[j] for j in newindices], "after removing bonds to metals")
                elif i in alkali_alkaline_earth_metal_idxs:
                    for j in indices:
                        b = np.array(pos[j])
                        dist = np.linalg.norm(a - b)
                        margin = dist - (radii[i] + radii[j])
                        print("Adjacency Matrix: Atom", i, labels[i], "is connected to", j, labels[j], "distance", round(dist, 3), "bond margin", round(margin, 3))                       
                else:
                    connections = []                        
                    for j in indices:
                        b = np.array(pos[j])
                        dist = np.linalg.norm(a - b)
                        margin = dist - (radii[i] + radii[j])
                        connections.append((j, margin))
                        print("Adjacency Matrix: Atom", i, labels[i], "is connected to", j, labels[j], "distance", round(dist, 3), "bond margin", round(margin, 3))
                    sorted_connections = sorted(connections, key=lambda x: x[1], reverse=True)
                    num_to_remove = len(connections) - max_valence
                    for idx in range(num_to_remove):
                        j, rem = sorted_connections[idx]
                        adjmat[i, j] = 0
                        adjmat[j, i] = 0
                        madjmat[i, j] = 0
                        madjmat[j, i] = 0
                        print(f"Adjacency Matrix: Removed bond {i} ({labels[i]}) - {j} ({labels[j]}) (margin = {round(rem, 3)})")
    
    return isgood, adjmat, madjmat

########################################
def get_adjmatrix_from_cif_bonds (labels: list, pos: list,  mol_atom_site_labels: list, bond_data: list, metal_only: bool = False) -> Tuple[int, list, list]:
    isgood = True
    indices = {atom: idx for idx, atom in enumerate(mol_atom_site_labels)}

    natoms = len(labels)
    adjmat = np.zeros((natoms, natoms))
    adjnum = np.zeros((natoms))

    for atom1, atom2, bond_distance in bond_data:
        if atom1 in mol_atom_site_labels and atom2 in mol_atom_site_labels:
            i = indices[atom1]
            j = indices[atom2]
            a = np.array(pos[i])
            b = np.array(pos[j])
            dist = np.linalg.norm(a - b)
            if not metal_only:
                # Allow a small tolerance for floating point comparison
                if round(abs(dist - bond_distance),3) <= 1e-3: 
                    adjmat[i, j] = 1
                    adjmat[j, i] = 1
                    # if len(get_alkali_alkaline_earth_metal_idxs([labels[i], labels[j]])) > 0:
                    #     adjmat[i, j] = 0
                    #     adjmat[j, i] = 0
                    #     print("Adjacency Matrix: Set Zeros for Alkali or Alkaline Earth Metal", labels[i], labels[j], f"{i=}", f"{j=}") 
                # else:
                    # print(f"Adjacency Matrix: Distance {round(dist, 3)} {dist=} is different with the bond distance {round(bond_distance, 3)} {bond_distance=} for atoms {i=} {j=} {labels[i]} {labels[j]} {atom1=} {atom2=}")
            if metal_only:
                if round(abs(dist - bond_distance),3) <= 1e-3:
                    if (elemdatabase.elementblock[labels[i]] == "d"
                    or elemdatabase.elementblock[labels[i]] == "f"
                    or elemdatabase.elementblock[labels[j]] == "d"
                    or elemdatabase.elementblock[labels[j]] == "f"):
                        adjmat[i, j] = 1
                        adjmat[j, i] = 1
                    elif len(get_alkali_alkaline_earth_metal_idxs([labels[i], labels[j]])) > 0:
                        adjmat[i, j] = 1
                        adjmat[j, i] = 1    
                    # if len(get_alkali_alkaline_earth_metal_idxs([labels[i], labels[j]])) > 0:
                    #     adjmat[i, j] = 0
                    #     adjmat[j, i] = 0
                    #     print("Adjacency Matrix: Set Zeros for Alkali or Alkaline Earth Metal", labels[i], labels[j], f"{i=}", f"{j=}") 
                # else:
                    # print(f"Adjacency Matrix: Distance {round(dist, 3)} {dist=} is different with the bond distance {round(bond_distance, 3)} {bond_distance=} for atoms {i=} {j=} {labels[i]} {labels[j]} {atom1=} {atom2=}")

    for i in range(0, natoms):
        adjnum[i] = np.sum(adjmat[i, :])

    adjmat = adjmat.astype(int)
    adjnum = adjnum.astype(int)
    
    return isgood, adjmat, adjnum


#####################################
# def get_adjmatrix(
#     labels: list,
#     pos: list,
#     mol_atom_site_labels: list = None,
#     bond_data: list = None,
#     cov_factor: float = 1.3,
#     radii="default",
#     metal_only: bool = False
# ) -> Tuple[int, list, list]:
#     """
#     Build an adjacency matrix using interatomic distances first,
#     then correct/update it based on provided bond_data if available.
#     """
#     isgood = True
#     clash_threshold = 0.3
#     natoms = len(labels)
#     adjmat = np.zeros((natoms, natoms))
#     adjnum = np.zeros((natoms))

#     # Load radii if needed
#     if isinstance(radii, str) and radii == "default":
#         with warnings.catch_warnings():
#             warnings.simplefilter(action="ignore", category=FutureWarning)
#             radii = get_radii(labels)

#     # Step 1: Build initial adjmat based on interatomic distances
#     add_factor = 0.3
#     for i in range(natoms - 1):
#         for j in range(i + 1, natoms):
#             a = np.array(pos[i])
#             b = np.array(pos[j])
#             dist = np.linalg.norm(a - b)

#             thres = (radii[i] + radii[j]) * cov_factor
#             if thres - (radii[i] + radii[j]) > 0.8:
#                 thres = (radii[i] + radii[j]) + add_factor

#             if dist <= clash_threshold:
#                 isgood = False
#                 print(f"Adjacency Matrix: Distance {round(dist, 3)} smaller than clash for atoms {i} {j} {labels[i]} {labels[j]}")
#             elif dist <= thres:
#                 if not metal_only:
#                     adjmat[i, j] = 1
#                     adjmat[j, i] = 1
#                 else:
#                     if (elemdatabase.elementblock[labels[i]] in ["d", "f"]
#                     or elemdatabase.elementblock[labels[j]] in ["d", "f"]
#                     or len(get_non_transition_metal_idxs([labels[i], labels[j]])) > 0):
#                         adjmat[i, j] = 1
#                         adjmat[j, i] = 1

#     # Step 2: Fix/update adjmat based on bond_data
#     if bond_data is not None and mol_atom_site_labels is not None:
#         indices = {atom: idx for idx, atom in enumerate(mol_atom_site_labels)}
#         for atom1, atom2, _ in bond_data:
#             if atom1 in indices and atom2 in indices:
#                 i = indices[atom1]
#                 j = indices[atom2]
#                 if not metal_only:
#                     adjmat[i, j] = 1
#                     adjmat[j, i] = 1
#                 else:
#                     if (elemdatabase.elementblock[labels[i]] in ["d", "f"]
#                     or elemdatabase.elementblock[labels[j]] in ["d", "f"]
#                     or len(get_non_transition_metal_idxs([labels[i], labels[j]])) > 0):
#                         adjmat[i, j] = 1
#                         adjmat[j, i] = 1

#     # Final step: calculate adjnum
#     for i in range(natoms):
#         adjnum[i] = np.sum(adjmat[i, :])

#     adjmat = adjmat.astype(int)
#     adjnum = adjnum.astype(int)

#     return isgood, adjmat, adjnum
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
def count_species(labels: list, pos: list, radii: list=None, indices: list=None, atom_site_labels : list=None, geom_bond_cif: list=None, cov_factor: float=1.3, debug: int=0) -> Tuple[bool, list]:
    # Gets the covalent radii
    if radii is None:    radii = get_radii(labels)
    if indices is None:  indices = [*range(0,len(labels),1)]

    # Computes the adjacency matrix of what is received
    # isgood indicates whether the adjacency matrix could be built normally, or errors were detected. 
    if atom_site_labels is not None and geom_bond_cif is not None:
        isgood, adjmat, adjnum = get_adjmatrix_from_cif_bonds (labels, pos, atom_site_labels, geom_bond_cif)
    else:
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
def split_species(labels: list, pos: list, radii: list=None, indices: list=None, atom_site_labels : list=None, geom_bond_cif: list=None, cov_factor: float=1.3, debug: int=0) -> Tuple[bool, list]:
    ## Function that identifies connected groups of atoms from their atomic coordinates and labels.
    
    # if debug >= 2:
    #     print(f"SPLIT_SPECIES: {labels=}", len(labels))
    #     print(f"SPLIT_SPECIES: {indices=}")

    # Gets the covalent radii
    if radii is None:    radii = get_radii(labels)
    if indices is None:  indices = [*range(0,len(labels),1)]

    # Computes the adjacency matrix of what is received
    # isgood indicates whether the adjacency matrix could be built normally, or errors were detected. Typically, those errors are steric clashes
    if atom_site_labels is not None and geom_bond_cif is not None:
        isgood, adjmat, adjnum = get_adjmatrix_from_cif_bonds (labels, pos, atom_site_labels, geom_bond_cif)
    else:
        isgood, adjmat, adjnum = get_adjmatrix(labels, pos, cov_factor, radii)
    if not isgood: return None

    degree = np.diag(adjnum)  # creates a matrix with adjnum as diagonal values. Needed for the laplacian
    lap = adjmat - degree     # computes laplacian

    # creates block matrix
    graph = csr_matrix(lap)
    # print(f"SPILT_SPECIES: Laplacian {lap=}")
    # print(f"SPILT_SPECIES: {graph=}")
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
    if debug > 0: 
        print("COMPARE_METALS. Comparing:")
        print(at1.label)
        print(at2.label)

    if at1.subtype != "metal" or at2.subtype != "metal": 
        if debug > 0: print("COMPARE_METALS. Different subtype")
        if debug > 0: print(at1.subtype)
        if debug > 0: print(at1.subtype)
        return False

    if (at1.label != at2.label): 
        if debug > 0: print("COMPARE_METALS. Different label")
        return False

    if at1.coord_sphere_formula is None: at1.get_coord_sphere_formula()
    if at2.coord_sphere_formula is None: at2.get_coord_sphere_formula()
    if (at1.coord_sphere_formula != at2.coord_sphere_formula):
        if debug > 0: print("COMPARE_METALS. Different coordination sphere")
        if debug > 0: print(at1.coord_sphere_formula)
        if debug > 0: print(at2.coord_sphere_formula)
        return False
    
    if check_coordinates:
        if (at1.coord[0] != at2.coord[0]): return False
        if (at1.coord[1] != at2.coord[1]): return False
        if (at1.coord[2] != at2.coord[2]): return False
        
    return True

#################################
def compare_species(mol1, mol2, check_coordinates: bool=False, debug: int=0):
    
    elems = elemdatabase.elementnr.keys()

    if debug > 0: 
        print("COMPARE_SPECIES. Comparing:")
        print(mol1.formula)
        print(mol2.formula)

    
    # a pair of species is compared on the basis of:
    # 1) the total number of atoms
    if (mol1.natoms != mol2.natoms): 
        if debug > 0: print("COMPARE_SPECIES. FALSE, different natoms:")
        return False

    # 2) the total number of electrons (as sum of atomic number)
    if (mol1.eleccount != mol2.eleccount): 
        if debug > 0: print("COMPARE_SPECIES. FALSE, different eleccount:")
        return False

    # 3) the number of atoms of each type
    if mol1.element_count is None: mol1.set_element_count()
    if mol2.element_count is None: mol2.set_element_count()
    for kdx, elem in enumerate(mol1.element_count):
        if elem != mol2.element_count[kdx]: 
            if debug > 0: print(f"COMPARE_SPECIES. FALSE, different {elem} count:")
            return False       
    # writexyz(os.getcwd(), f"reordered.xyz", mol1.labels, mol1.coord)
    # 4) the number of adjacencies between each pair of element types
    if mol1.adj_types is None:     mol1.set_adj_types()
    if mol2.adj_types is None:     mol2.set_adj_types()
    if debug == 2: print(f"{mol1.adj_types=}")
    if debug == 2: print(f"{mol2.adj_types=}")

    count = 0
    if debug > 0: print("COMPARE_SPECIES. kdx ldx elem1 - elem2 : reordered - reference")
    for kdx, (elem, row1) in enumerate(zip(elems, mol1.adj_types)):
        for ldx, (elem2, val1) in enumerate(zip(elems, row1)):
            val2 = mol2.adj_types[kdx, ldx]
            if val1 != val2: 
                count += 1
                if debug > 0: print(f"COMPARE_SPECIES. FALSE, different adjacency count")
                if debug > 0: print(f"COMPARE_SPECIES. {kdx} {ldx} {elem} - {elem2} : {val1} - {val2}")
                
    if count > 0 : return False
    else: return True

    if check_coordinates:
        # 5) Finally, the coordinates if the user wants it
        for idx in range(0,mol1.natoms,1):
            if (mol1.coord[idx][0] !=  mol2.coord[idx][0]): return False
            if (mol1.coord[idx][1] !=  mol2.coord[idx][1]): return False
            if (mol1.coord[idx][2] !=  mol2.coord[idx][2]): return False
    return True
#################################
def compare_reference_indices (ref, mol, debug: int=0):
    if (ref.natoms == mol.natoms) & (ref.formula == mol.formula):
        if (sorted(ref.get_parent_indices("reference")) == sorted(mol.get_parent_indices("reference"))):
            if debug >= 2: 
                print("Matched", mol.formula, ref.formula, ref.get_parent_indices("reference"), mol.get_parent_indices("reference"))
            issame = True
        else:
            if debug >= 2:
                print("Different indices", mol.formula, ref.formula, ref.get_parent_indices("reference"), mol.get_parent_indices("reference"))
            issame = False
    else : 
        if debug >= 2:
            print("Different numbers", mol.formula, ref.formula, ref.get_parent_indices("reference"), mol.get_parent_indices("reference"))
        issame = False
    return issame
#################################
def arrange_data_for_reorder(reference: object, target: object, debug: int=0):
    # To do the reorder, we create new tags that include as much information as possible.
    # Ideally, we aim to include the label + the connectivity + the metal connectivity
    t_totconnec = 0
    t_totmconnec = 0
    for a in target.atoms:
        t_totconnec  += a.connec
        t_totmconnec += a.mconnec
    r_totconnec = 0
    r_totmconnec = 0
    for a in reference.atoms:
        r_totconnec  += a.connec
        r_totmconnec += a.mconnec
    if t_totconnec == r_totconnec:   useconec = True
    else:                            useconec = False
    if t_totmconnec == r_totmconnec: usemconec = True
    else:                            usemconec = False
    # For target
    target_data = []
    for a in target.atoms:
        data = a.label
        if useconec:  data += str(a.connec)
        if usemconec: data += str(a.mconnec)
        target_data.append(data)
    # For reference
    ref_data = []
    for a in reference.atoms:
        data = a.label
        if useconec:  data += str(a.connec)
        if usemconec: data += str(a.mconnec)
        ref_data.append(data)
    return ref_data, target_data

#################################
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

#################################
def split_group(original_group, conn_idx, final_ligand_indices, debug: int=0):
    from cell2mol.classes import group
    # Split the "group" to obtain the groups connected to a specific metal
    splitted_groups = []
    
    if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: {conn_idx=}")
    if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: {original_group.labels=}")
    if debug > 2: print(f"\t\tGROUP.SPLIT_GROUP: {original_group.coord=}")
    conn_labels  = extract_from_list(conn_idx, original_group.labels, dimension=1)
    conn_coord   = extract_from_list(conn_idx, original_group.coord, dimension=1)
    frac_coord = getattr(original_group, "frac_coord", None)
    conn_frac_coord = extract_from_list(conn_idx, frac_coord, dimension=1) if frac_coord is not None else None
    conn_radii   = extract_from_list(conn_idx, original_group.radii, dimension=1)
    conn_atoms   = extract_from_list(conn_idx, original_group.atoms, dimension=1)
    atom_site_labels = getattr(original_group, "atom_site_labels", None)
    if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: original_group.atom_site_labels={atom_site_labels}")
    conn_atom_site_labels = extract_from_list(conn_idx, atom_site_labels, dimension=1) if atom_site_labels is not None else None

    if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: {conn_labels=}")

    cov_factor=original_group.get_parent("ligand").cov_factor
    refcell = original_group.get_parent("reference")
    geom_bond_cif = getattr(refcell, "geom_bond_cif", None)
    if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False) and geom_bond_cif is not None:
        blocklist = split_species(conn_labels, conn_coord, atom_site_labels=conn_atom_site_labels, geom_bond_cif=geom_bond_cif, debug=debug)
    else :
        blocklist = split_species(conn_labels, conn_coord, radii=conn_radii, cov_factor=cov_factor, debug=debug)      
    if debug > 0: print(f"\t\tGROUP.SPLIT_GROUP: {blocklist=}")

    ## Arranges Groups 
    for b in blocklist:
        if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: block={b}")
        gr_indices      = extract_from_list(b, conn_idx, dimension=1)
        ligand_idx      = extract_from_list(b, final_ligand_indices, dimension=1)
        gr_labels       = extract_from_list(b, conn_labels, dimension=1)
        gr_coord        = extract_from_list(b, conn_coord, dimension=1)
        gr_frac_coord   = extract_from_list(b, conn_frac_coord, dimension=1) if frac_coord is not None else None
        gr_radii        = extract_from_list(b, conn_radii, dimension=1)
        gr_atoms        = extract_from_list(b, conn_atoms, dimension=1)
        gr_atom_site_labels = extract_from_list(b, conn_atom_site_labels, dimension=1) if atom_site_labels is not None else None

        if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: {gr_labels=}")
        if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: {gr_atom_site_labels=}")
        # Create Group Object
        newgroup = group.from_positional(gr_labels, gr_coord, gr_frac_coord, radii=gr_radii)
        if debug > 1: print(f"\t\tGROUP.SPLIT_GROUP: {newgroup.labels=}")
        # For debugging
        newgroup.origin = "split_group"
        # Define the GROUP as parent of the group. Bottom-Up hierarchy
        newgroup.add_parent(original_group.get_parent("ligand"), indices=ligand_idx)
        # Pass the GROUP atoms to the groud
        newgroup.set_atoms(atomlist=gr_atoms, atom_site_labels=gr_atom_site_labels)
        # Inherit the adjacencies from molecule
        newgroup.inherit_adjmatrix("ligand")
        # Associate the Groups with the Metals
        newgroup.get_connected_metals(debug=debug)
        newgroup.get_closest_metal(debug=debug)
        newgroup.get_hapticity(debug=debug)
        newgroup.checked_coordination = True
        newgroup.get_denticity(debug=debug)
        # Top-down hierarchy
        splitted_groups.append(newgroup)
    return splitted_groups