import numpy as np
import itertools

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Tuple
from collections import defaultdict

from cell2mol.cellconversions import frac2cart_fromparam, cart2frac, translate
from cell2mol.tmcharge_common import (getelementcount,getradii,getcentroid,find_groups_within_ligand,find_closest_metal,checkchemistry,getconec,getblocks,inv,extract_from_matrix)
from cell2mol.missingH import getangle
# Imports Classes
from cell2mol.tmcharge_common import atom, molecule, ligand, metal, group
from cell2mol.elementdata import ElementData
from cosymlib import Geometry
from cosymlib.shape.tools import shape_structure_references

elemdatabase = ElementData()

#######################################################
def verify_connectivity(ligand: object, molecule: object, debug: int = 0) -> None:

    metalist = molecule.metalist.copy()

    # Original labels and coordinates are copied
    newlab = ligand.labels.copy()
    newlab.append(str("H"))  # One H atom will be added
    newcoord = ligand.coord.copy()

    # position (index) of the added atom
    posadded = len(ligand.labels)

    if debug >= 2: print("")
    if debug >= 2: print(f"VERIFY: checking connectivity of ligand {ligand.formula}")
    if debug >= 2: print(f"VERIFY: initial connectivity is {ligand.totmconnec}")
    for g in ligand.grouplist:
        if g.hapticity is True:
            if debug >= 2: print("VERIFY: group has hapticity, skipping check")
        else:
            for idx, a in enumerate(ligand.atoms):
                if a.mconnec >= 1 and a.index in g.atlist:
                    if debug >= 2: print(f"VERIFY: connectivity={a.mconnec} in atom idx={idx}, label={a.label}")
                    tgt, apos, dist = find_closest_metal(a, metalist)
                    idealdist = a.radii + elemdatabase.CovalentRadius2["H"]
                    addedHcoords = apos + (metalist[tgt].coord - apos) * (idealdist / dist)  # the factor idealdist/dist[tgt] controls the distance
                    newcoord.append([addedHcoords[0], addedHcoords[1], addedHcoords[2]])  # adds H at the position of the closest Metal Atom

                    # Evaluates the new adjacency matrix.
                    tmpradii = getradii(newlab)
                    dummy, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(newlab, newcoord, ligand.factor, tmpradii)
                    # If no undesired adjacencies have been created, the coordinates are kept. Otherwise, data is corrected
                    if tmpconnec[posadded] == 1:
                        if debug >= 2: print(f"VERIFY: connectivity verified for atom {idx} with label {a.label}")
                    else:
                        # Corrects data of atom object
                        a.mconnec = 0
                        if debug >= 2: print(f"VERIFY: corrected mconnec of atom {idx} with label {a.label}")
                        # Now it should correct data of metal, ligand and molecule objects. Not yet implemented

    # Corrects data of ligand object
    ligand.totmconnec = 0
    for a in ligand.atoms:
        ligand.totmconnec += a.mconnec
    if debug >= 2: print(f"VERIFY: final connectivity is {ligand.totmconnec}")

#######################################################
def get_reference_molecules(labels: list, pos: list, debug: int=0) -> Tuple[list, float, float, bool]:
    ## Retrieves the reference molecules from the information in the .cif file
    # Molecules are extracted from the adjacency matrix, and the results are evaluated. How? Well, the adjacency matrix is constructed using a covalent factor.
    # This factor is taken as 1.3, but it can occasionally be too small or too large. If it is too small, it results on some atoms being detached from their molecule.
    # If an isolated atom is found, and is not a H, or a Halogen or alkaline atom, then the covalent factor can be increased.

    # Something similar happens for the Metal atoms. The coordination of an atom to the metal is also evaluated from the adjacency matrix. Again, the covalent factor might fail.
    # This function evaluates that the coordination number of the metal makes 'sense'. That is, is a commmon number for the metal atom. Common coordination numbers are in the function metalcoordcheck
    # If the coordination number is too big, or too small, it the covalent factor of the metal-ligand bonds gets modified. This factor is called metal_factor

    # The covalent and metal factors are stored in all molecule objects, and used throughout the analysis of the whole crystal structure.
    #:return listofreferences: list of reference molecules saved as objects
    #:return covalent_factor: value between 1.3 and 1.6 that will be used to construct the adjacency matrix of the crystal structure
    #:return metal_factor: value between 1.2 and 0.8 that multiplies the covalent radii of the metal atoms in the crystal structure
    #:return Warning: boolean warning. Typically, a warning in this function is raised when getmolecs finds steric clashes in any of the reference molecules

    Warning = False

    # Parameters to adjust the covalent factor
    found_covalent_factor = False
    max_covalent_factor = 1.6
    min_covalent_factor = 1.3
    increase_covalent_factor = 0.02

    # Parameters to adjust the metal factor
    found_metal_factor = False
    max_metal_factor = 1.2
    min_metal_factor = 0.8
    change_metal_factor = 0.02

    # Initial Values
    covalent_factor = min_covalent_factor
    metal_factor = 1.0

    # Initiates while that adjusts the two factors (metal and covalent)
    found_both_factors = False
    maxiter = 11
    iteration = 1
    # execute as long as found_both_factor = False, iteration <= maxiter, and Warning = False
    while (not found_both_factors) and (iteration <= maxiter):#and not Warning:

        # Tries to find the reference molecules
        if debug >= 2: print("")
        if debug >= 2: print(f"GETREFS: sending listofreferences with {covalent_factor} {metal_factor}")
        Warning, listofreferences = getmolecs(labels, pos, covalent_factor, metal_factor)
 
        ##################################################
        # PART 1: Condition to accept the covalent_factor:
        ##################################################
        valid_list_of_references = True

        # checks for isolated atoms, and retrieves warning if there is any. Except if it is H, halogen (group 17) or alkalyne (group 2)
        for ref in listofreferences:
            if ref.natoms == 1:
                if ((elemdatabase.elementgroup[ref.atoms[0].label] and ref.atoms[0].label != "H" )
                    or elemdatabase.elementgroup[ref.atoms[0].label] == 2
                    or elemdatabase.elementgroup[ref.atoms[0].label] == 17):
                    pass
                else:
                    if debug >= 2: print(f"GETREFS: found ref molecule with only one atom {ref.labels}")
                    valid_list_of_references = False

        if not valid_list_of_references:
            if covalent_factor < max_covalent_factor:
                found_covalent_factor = False
                covalent_factor += increase_covalent_factor
                covalent_factor = np.round(covalent_factor,2)
                if debug >= 2: print("GETREFS: Increasing covalent_factor to:", covalent_factor)
            else:
                print("GETREFS: Reached Maximum Covalent_factor:", max_covalent_factor)
                Warning = True
        else:
            found_covalent_factor = True

        #########################################################################################################
        # PART 2: Condition to accept the metal_factor. Runs for all complexes in the list of reference molecules
        #########################################################################################################
        glist = []
        ilist = []
        dlist = []
        for ref in listofreferences:
            if ref.type == "Complex":
                ref.ligandlist, ref.metalist = splitcomplex(ref, covalent_factor, metal_factor)

                # Checks Hapticity
                potential_hapticity = get_hapticity(ref)
                
                # Check coordination geometry around metal
                ref.metalist = get_coordination_geometry (ref.metalist, potential_hapticity, debug=1)

                if debug >= 2:  print(f"Potential hapticity={potential_hapticity} for molecule {ref.formula}")

                for lig in ref.ligandlist:
                    verify_connectivity(lig, ref, debug)
                    if debug >= 2: print(f"Verifying Connectivity for Lig in Ref molec {lig.natoms}, {lig.formula}, {lig.totmconnec}")  # , len(lig.metalatoms))

                # Checks Shared Ligands in Polymetallic complexes
                if any(len(lig.metalatoms) >= 2 for lig in ref.ligandlist):
                    ispolymetallic_and_shared = True
                    print("Molecule is polymetallic and has shared ligands")
                else:
                    ispolymetallic_and_shared = False

                if (not potential_hapticity and not ispolymetallic_and_shared):  # then tries to adjust the metal_factor
                    for a in ref.atoms:
                        if a.block == "d" or a.block == "f":
                            if debug >= 2: print(f"GETREFS: sending {a.label} {a.mconnec} to coordcheck")
                            good, increase, decrease = metalcoordcheck(a.label, a.mconnec, debug)
                            glist.append(good)
                            ilist.append(increase)
                            dlist.append(decrease)
                            if debug >= 2: print(f"GETREFS: received {good}, {increase}, {decrease} from coordcheck")
                else:
                    glist.append(True)
                    ilist.append(False)
                    dlist.append(False)

        if len(glist) > 0:
            if (any((item == True for item in ilist)) and all((item2 == False for item2 in dlist)) and (metal_factor < max_metal_factor)):
                metal_factor += change_metal_factor
                metal_factor = np.round(metal_factor,2)
                if debug >= 2: print("GETREFS: Increasing metal_factor to:", metal_factor)
            if (all((item == False for item in ilist)) and any((item2 == True for item2 in dlist)) and (metal_factor > min_metal_factor)):
                metal_factor -= change_metal_factor
                metal_factor = np.round(metal_factor,2)
                if debug >= 2: print("GETREFS: Decreasing metal_factor to:", metal_factor)
            if all((item == True for item in glist)):
                found_metal_factor = True
                if debug >= 2: print("GETREFS: Metal_factor set at:", metal_factor)

        ##########################
        # Part 3: Takes a decision 
        ##########################
        if found_covalent_factor and found_metal_factor:
            found_both_factors = True
            Warning = False
            if debug >= 2: print("GETREFS: Found both factors. Breaking with", len(listofreferences), "references found")
            break
        elif (metal_factor > max_metal_factor) or (metal_factor < min_metal_factor):
            if debug >= 2: print("GETREFS: metal_factor outside the limits", metal_factor)
            Warning = True
            break
        else:
            if debug >= 2: print(f"GETREFS: finished iteration number {iteration}/{maxiter}")
            iteration += 1
            Warning = True
            continue

        if iteration == maxiter:
            if debug >= 2: print("GETREFS: maximum number of iterations reached")
            Warning = True

    ### RIGHT NOW, we ignore the warning because if there is really a problem, the program will fail later. i
    ### If the problem is not important, a meaningful result can still be reached

    #return listofreferences, covalent_factor, metal_factor, Warning
    return listofreferences, covalent_factor, metal_factor, False


#######################################################
def metalcoordcheck(label: str, coordination: int, debug: int=0) -> Tuple[bool, bool, bool]:
    ## Function that receives the label of a metal atom, and a proposed coordination number, and tells whether this coordination is too high, low, or correct.
    # It uses a database of common coordination numbers from paper below
    #:return good, increase, decrease: booleans for what to do with this metal

    # Data Obtained from:
    # Venkataraman, D.; Du, Y.; Wilson, S. R.; Hirsch, K. A.; Zhang, P.; Moore, J. S. A
    # Coordination Geometry Table of the D-Block Elements and Their Ions.
    # J. Chem. Educ. 1997, 74, 915.

    good = False
    increase = False
    decrease = False

    atnum = elemdatabase.elementnr[label]
    if debug >= 2:
        print("Metalcoordcheck function: got atnum", atnum, "for label", label)

    coordnum = defaultdict(list)
    # adding 1st-row transition metals.
    coordnum[21] = [6]  # Sc
    coordnum[22] = [3, 4, 6]  # Ti
    coordnum[23] = [3, 4, 5, 6]  # V
    coordnum[24] = [4, 5, 6]  # Cr
    coordnum[25] = [3, 4, 5, 6]  # Mn #some strange cases of heptacoordination in Mn exist. Not sure is a good idea to have it
    coordnum[26] = [3, 4, 5, 6]  # Fe
    coordnum[27] = [3, 4, 5, 6]  # Co
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
        print("Metalcoordcheck function: Atom with label",label,"has an empty list of possible coordination")
    else:
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
        else: print("Metalcoordcheck function: Atom with label",label,"has strange coordination value:",coordination,coordnum[atnum])

    return good, increase, decrease




#######################################################
def get_reference_molecules_simple(labels: list, pos: list, debug: int=2) -> Tuple[list, float, float, bool]:

    Warning = False

    # Initial Values
    covalent_factor = 1.3
    metal_factor = 1.0

    # Tries to find the reference molecules
    if debug >= 2: print("")
    if debug >= 2: print(f"GETREFS: sending listofreferences with {covalent_factor} {metal_factor}")
    Warning, listofreferences = getmolecs(labels, pos, covalent_factor, metal_factor)

    valid_list_of_references = True

    # checks for isolated atoms, and retrieves warning if there is any. Except if it is H, halogen (group 17) or alkalyne (group 2)
    for ref in listofreferences:
        if ref.natoms == 1:
            if ((elemdatabase.elementgroup[ref.atoms[0].label] and ref.atoms[0].label != "H" )
                or elemdatabase.elementgroup[ref.atoms[0].label] == 2
                or elemdatabase.elementgroup[ref.atoms[0].label] == 17):
                pass
            else:
                if debug >= 2: print(f"GETREFS: found ref molecule with only one atom {ref.labels}")
                valid_list_of_references = False

    if not valid_list_of_references: # If list of reference is not valid
        Warning = True
    else:
        Warning = False

    for ref in listofreferences:
        if ref.type == "Complex":
            
            ref.ligandlist, ref.metalist = splitcomplex(ref, covalent_factor, metal_factor, debug=1)

            if debug >= 2  :
                print("###### met in ref.metalist ######")
                for met in ref.metalist:
                    print(f"{met.label=}\t{met.mconnec=}\t{met.totmconnec=}")
                    print(f"{met.coord_sphere=}")
                    print(f"{met.coordinating_atoms=}")
 
            
            # Checks Hapticity
            potential_hapticity = get_hapticity(ref)
            
            # Check coordination geometry around metal
            ref.metalist = get_coordination_geometry (ref.metalist, potential_hapticity, debug=1)

            if debug >= 2:  print(f"Potential hapticity={potential_hapticity} for molecule {ref.formula}")

            for lig in ref.ligandlist:
                verify_connectivity(lig, ref, debug=0)
                if debug >= 2: print(f"Verifying Connectivity for Lig in Ref molec {lig.natoms}, {lig.formula}, {lig.totmconnec} {len(lig.metalatoms)}")
    
    if debug >= 2: print(f"{valid_list_of_references=} {Warning=}")

    ### RIGHT NOW, we ignore the warning because if there is really a problem, the program will fail later.
    return listofreferences, covalent_factor, metal_factor, False


##############################################
def correct_metal_coordinating_atoms (lig: object, metalist: list, debug: int=2) -> object :
    
    if debug >= 1 : 
        print(f"{lig.formula=}")
        # print(f"{lig.labels}")
        # print(f"{lig.mconnec}")

    if lig.hapticity == False :
        # Generate index list of metal-coordinating atoms    
        idx_list = [index for index, value in enumerate(lig.mconnec) if value >= 1] 

        for i in idx_list:
            atom = lig.atoms[i] 
            tgt, apos, dist = find_closest_metal(atom, metalist)
            metal = metalist[tgt]

            if debug >= 1 : 
                print(">>> metal-coordinating atoms", atom.label, "\tMetal :", metal.label, metal.coordinating_atoms)#, metal.totmconnec, metal.mconnec, atom.adjacency, atom.mconnec, atom.coord)
            
            if atom.mconnec >= 1 :
                neighbors_coordination = []
                
                for j in atom.adjacency:
                    neighbor = lig.atoms[j]
                    if debug >= 2 : print(f"{atom.label} connected to {neighbor.label}") #{neighbor.adjacency} {neighbor.mconnec} {neighbor.coord}")      
                    neighbors_coordination.append(neighbor.mconnec)

                if sum(neighbors_coordination) > 1 :
                    if debug >= 1 : 
                        print(f"[Check] This coordinating atom {atom.label} connected to more than one coordinating atoms")

                        nb_idx_list = [index for index,value in enumerate(neighbors_coordination) if value == 1]
                        for nb_idx in nb_idx_list :
                            nb = lig.atoms[atom.adjacency[nb_idx]]
                            print(f"        This coordinating atom {atom.label} connected to other coordinating atom {nb.label}") #{nb.adjacency} {nb.mconnec} {nb.coord}")
 
                    if debug >= 1 : 
                        print("Wrong metal-coordination assignment for", lig.labels[i]) #lig.mconnec[i], lig.atoms[i].mconnec)  
                    lig.mconnec[i] = 0
                    lig.adjacencies(lig.conmat, lig.mconnec)
                    metal.adjacencies(np.array([x - 1 for x in metal.mconnec]))

                    wrong = [index for index, value in enumerate(metal.coordinating_atoms_sites) if value == lig.coord[i]][0]
                    if debug >= 1 : 
                        print("Wrong : ", metal.coordinating_atoms[wrong], metal.coordinating_atoms_sites[wrong])

                    del metal.coordinating_atoms[wrong]
                    del metal.coordinating_atoms_sites[wrong]
                    
                    if debug >= 1 :
                        print("After correction : ", metal.coordinating_atoms, metal.totmconnec, metal.mconnec) 
                    
                elif sum(neighbors_coordination) == 1 : # e.g. "S" atom in refcode YOBCUO, PORNOC
                        
                    nb_idx = [index for index,value in enumerate(neighbors_coordination) if value == 1][0]
                    nb = lig.atoms[atom.adjacency[nb_idx]]
                    
                    if debug >= 1 : 
                        print(f"[Check] This coordinating atom {atom.label} connected to another coordinating atom {nb.label}") #{nb.adjacency} {nb.mconnec} {nb.coord}")
                    
                    if (atom.label == "H" and nb.label in ["B", "O", "N"]) :
                        if debug >= 1 : print("Wrong metal-coordination assignment for", lig.labels[i]) #lig.mconnec[i], lig.atoms[i].mconnec)  
                        
                        lig.mconnec[i] = 0
                        lig.adjacencies(lig.conmat, lig.mconnec)
                        metal.adjacencies(np.array([x - 1 for x in metal.mconnec]))

                        wrong = [index for index, value in enumerate(metal.coordinating_atoms_sites) if value == lig.coord[i]][0]
                        if debug >= 1 : 
                            print("Wrong : ", metal.coordinating_atoms[wrong], metal.coordinating_atoms_sites[wrong])

                        del metal.coordinating_atoms[wrong]
                        del metal.coordinating_atoms_sites[wrong]
                        
                        if debug >= 1 :
                            print("After correction : ", metal.coordinating_atoms, metal.totmconnec, metal.mconnec) 

                    elif atom.label in ["B", "O", "N"] and nb.label == "H" :
                        
                        if debug >= 1 : 
                            print("Wrong metal-coordination assignment for neighboring", nb.label) #lig.mconnec[i], lig.atoms[i].mconnec)  

                        lig.mconnec[atom.adjacency[nb_idx]] = 0
                        lig.adjacencies(lig.conmat, lig.mconnec)                        
                        metal.adjacencies(np.array([x - 1 for x in metal.mconnec]))

                        wrong = [index for index, value in enumerate(metal.coordinating_atoms_sites) if value == lig.coord[atom.adjacency[nb_idx]]][0]
                        if debug >= 1 : 
                            print("Wrong : ", metal.coordinating_atoms[wrong], metal.coordinating_atoms_sites[wrong])

                        del metal.coordinating_atoms[wrong]
                        del metal.coordinating_atoms_sites[wrong]
                        
                        if debug >= 1 :
                            print("After correction : ", metal.coordinating_atoms, metal.totmconnec, metal.mconnec) 

                    else :
                        tgt, apos, dist = find_closest_metal(atom, metalist)
                        metal = metalist[tgt]
                        
                        vector1 = np.subtract(np.array(atom.coord), np.array(nb.coord))
                        vector2 = np.subtract(np.array(atom.coord), np.array(metal.coord))
                        
                        angle = np.degrees(getangle(vector1, vector2))
                        if debug >= 1 : 
                            print(f"{metal.label}-{atom.label}-{nb.label} angle {round(angle,2)}")
                        
                        if angle < 50 :
                            if debug >= 1 : 
                                print("Wrong metal-coordination assignment for", lig.labels[i]) #lig.mconnec[i], lig.atoms[i].mconnec)  
                            lig.mconnec[i] = 0
                            lig.adjacencies(lig.conmat, lig.mconnec)
                            metal.adjacencies(np.array([x - 1 for x in metal.mconnec]))

                            wrong = [index for index, value in enumerate(metal.coordinating_atoms_sites) if value == lig.coord[i]][0]
                            if debug >= 1 : 
                                print("Wrong : ", metal.coordinating_atoms[wrong], metal.coordinating_atoms_sites[wrong])

                            del metal.coordinating_atoms[wrong]
                            del metal.coordinating_atoms_sites[wrong]
                            
                            if debug >= 1 :
                                print("After correction : ", metal.coordinating_atoms, metal.totmconnec, metal.mconnec) 
                                        
                else :
                    pass
            else :
                if debug >= 1 : 
                    print(f"Excluded {atom.label} from {metal.coordinating_atoms}")
    else :
        if debug >= 1 : 
            print(f"Ligand hapticity is True {lig.hapttype} => Do not find metal-coordinating atoms")

        # Generate index list of metal-coordinating atoms    
        idx_list = [index for index, value in enumerate(lig.mconnec) if value >= 1] 

        for i in idx_list:
            atom = lig.atoms[i] 
            tgt, apos, dist = find_closest_metal(atom, metalist)
            metal = metalist[tgt]

            if debug >= 2 : 
                print(">>> coordination sphere", atom.label, "\tMetal :", metal.label, metal.coordinating_atoms) #, metal.totmconnec, metal.mconnec) #, atom.adjacency, atom.mconnec, atom.coord)

    if debug >= 2  :
        print("###### met in metalist ######")
        for met in metalist:
            print(f"{met.mconnec=}")
            print(f"{met.totmconnec=}")
            print(f"{met.coord_sphere=}")
            print(f"{met.coord_sphere_ID=}")
            print(f"{met.coordinating_atoms=}")
            print(f"{met.coordinating_atoms_sites=}")    
    if debug >= 1 :
        print("")
    return lig, metalist

      

#######################################################
def getmolecs(labels: list, pos: list, factor: float=1.3, metal_factor: float=1.0, atlist: list=[], debug: int=0) -> Tuple[bool, list]:
    ##Simplified Version of the getmolecs
    ## Function that identifies connected groups of atoms from their positions and labels.
    #:return mlist. List of molecules saved as objects

    Warning = False

    # Gets the covalent radii, and modifies that of the metal if necessary
    radii = getradii(labels)

    # Modifies the radii for metal atoms, if necessary
    if metal_factor != 1.0:
        for idx, r in enumerate(radii):
            if (
                elemdatabase.elementblock[labels[idx]] == "d"
                or elemdatabase.elementblock[labels[idx]] == "f"
            ):
                radii[idx] = (
                    r * metal_factor
                )  # the covalent radii of the metal is modified

    # Computes the adjacency matrix of what is received
    status, conmat, connec, mconmat, mconnec = getconec(labels, pos, factor, radii)

    # status indicates whether the adjacency matrix could be built normally, or errors were detected. Typically, those errors are steric clashes
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

            for i in range(0, len(atomlistperm)):
                if atomlistperm[i] == b + 1:
                    if len(atlist) == len(labels): 
                        fraglist.append(atlist[i])
                    else: 
                        fraglist.append(i)
                    labelist.append(labels[i])
                    poslist.append(pos[i])
                    radiilist.append(radii[i])

            # Generates conmat and mconmat for the molecule
            nidx = 0
            njdx = 0
            nat = len(fraglist)
            conmatlist = np.empty((nat, nat))
            mconmatlist = np.empty((nat, nat))
            for idx in range(0, len(atomlistperm)):
                if atomlistperm[idx] == b + 1:
                    for jdx in range(0, len(atomlistperm)):
                        if atomlistperm[jdx] == b + 1:
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

    # If 0, prints the coordinates for debugging
    elif status == 0:
        Warning = True
        mlist = []
        if debug >=1: print("GETMOLECS: steric clashes found.")
        if debug >=2: 
            print("GETMOLECS: steric clashes found. Printing Coordinates for Debugging")
            for idx, lab in enumerate(labels):
                print("%s   %.6f   %.6f   %.6f" % (lab, pos[idx][0], pos[idx][1], pos[idx][2]))
            print("")

    return Warning, mlist


#######################################################
def splitcomplex(molecule: object, factor: float=1.3, metal_factor: float=1.0, debug: int=0) -> Tuple[list, list]:
    ## Similar function to getmolecs, but with a prelude in which the molecule (it must be a molecule of type "Complex"), is split into ligands and metals.
    #:return ligandlist, metalist. List of ligands and metals, respectively, saved as objects

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
            matoms.append(a)  # This information is for the ligands. To generate lig.metalatoms
            number_of_metal_atoms += 1
            met = metal(number_of_metal_atoms, idx, a.label, a.coord, a.radii)
            met.information(factor, metal_factor)

            # Extracts the metal adjacency from the molecule.adjacency matrix
            tmp_mconnec = extract_from_matrix(list([idx]), molecule.mconnec, 1)
            tmp_mconnec = tmp_mconnec.astype(int)
            met.adjacencies(tmp_mconnec)
            metalist.append(met)

            connec_atoms_label = ([])  # collects the labels of all atoms connected to this metal
            connec_atoms_sites = ([])
            for jdx, at2 in enumerate(molecule.atoms):
                if molecule.mconmat[idx, jdx] == 1:
                    connec_atoms_label.append(str(at2.label))
                    connec_atoms_sites.append(at2.coord)

            met.coord_sphere = connec_atoms_label
            met.coord_sphere_ID = getelementcount(connec_atoms_label)
            met.coordinating_atoms = connec_atoms_label.copy()
            met.coordinating_atoms_sites = connec_atoms_sites
            
        else:
            mfreelabels.append(a.label)
            mfreepos.append(a.coord)
            mfreeconnec.append(a.mconnec)
            mfreeradii.append(a.radii)
            mfreeatlist.append(idx)
    
    # Uses the Metal-free coordinates to find the ligands. Notice that, when creating their metal connectivity, it uses that of the original molecule
    status, conmat, connec, dummy, dummy = getconec(mfreelabels, mfreepos, factor, mfreeradii)

    if status == 1:
        degree = np.diag(connec)  # creates a matrix with connec as diagonal values. Needed for the laplacian
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
            lig = ligand(b, atlist, labelist, poslist, radiilist)  # Creates Object Molecule
            lig.information(factor, metal_factor)  # Creates Information about the construction
            lig.adjacencies(tmp_conmat, tmp_mconnec)  # Creates the Adjacency Information
            lig = get_hapticity_ligand(lig)
            lig, metalist = correct_metal_coordinating_atoms (lig, metalist, debug) # Correct metal-coordinating atoms if there is an error

            for a in matoms:  # only adds metal atoms that are connected to the ligand
                found = False 
                for idx, ligat in enumerate(lig.atoms):
                    if lig.atlist[idx] in a.adjacency and not found:
                        lig.metalatoms.append(a)  # Saves Metal-Atom Information to the Ligand Object
                        found = True
            ligandlist.append(lig)  # Appends it to the final list of ligand   

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
def tmatgenerator(centroid, thres=0.40, full=False, debug: int=0):
    # This function generates a list of the translations that a fragment should undergo depending on the centroid of its fractional coordinates
    # For instance, if the centroid of a fragment is at 0.9 in any given axis, it is unlikely that a one-cell-length translation along such axis (resulting in 1.9) would help.
    # Also, a fragment right at the center of the unit cell (centroid=(0.5, 0.5, 0.5) is unlikely to require reconstruction
    # The threshold defines the window. If thres=0.4, the function will suggest positive translation for any fragment between 0 and 0.4, and negative translation between 0.6 and 1.0.
    # If full is asked, then all translations are applied

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

    return tmatrix


#######################################################
def sequential(fragmentlist: list, refmoleclist: list, cellvec: list, factor: float, metal_factor: float, typ: str="All", debug: int=0) -> Tuple[list, list]:
    # Function that controls the reconstruction process. It is called sequential because pairs of fragments are sent one by one. Ideally, a parallel version would be desirable.
    # Given a list of fragments(fragmentlist), a list of reference molecules(refmoleclist), and some other minor parameters, the function sends pairs of fragments and evaluates if they...
    # ...form a bigger fragment. If so, the bigger fragment is evaluated. If it coincides with one of the molecules in refmoleclist, then it means that it is a full molecule that requires no further work.
    # ...if it does not, then it means that requires further reconstruction, and is again introduced in the loop.
    # typ is a variable that defines how to combine the fragments. To speed up the process, this function is called two times:
    # -First, to combine heavy fragments among themselves (typ="Heavy")
    # -Second, to combie heavy fragments with H atoms (typ="All")
    #:return molecsfoundlist, remainingfragments: lists of molecules and fragments, respectively, saved as objects

    if debug >= 2:
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
    maxiter = 3000
    mixsize = 1
    lastiter = 0
    lastitermargin = maxiter
    ###################################################

    ###################################################
    # Lists (list1 and list2) are created here depending on variable "typ"
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

    if debug >= 2:
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

    ## Initial Fragment indices for each list
    Frag1_toallocate = 0
    Frag2_toallocate = 0

    #################
    ### MAIN LOOP ###
    #################
    while (len(list1) > 0) and (len(list2) > 0):

        #################
        #  This part decides which molecules in the two lists are sent to combine
        #################
        STOP = False
        Last_Attempt = False

        if niter > 1: Frag2_toallocate += 1
        if (Frag2_toallocate > len(list2) - 1):  # Reaches the end of the second list. Restarts it and moves forward in the first list
            Frag1_toallocate += 1
            Frag2_toallocate = 0

        if (Frag1_toallocate > len(list1) - 1):  # Reaches the end of the first list. Restarts both
            Frag1_toallocate = 0
            Frag2_toallocate = 0

        if typ == "Heavy":
            if Frag1_toallocate == Frag2_toallocate:
                Frag2_toallocate += 1

        if typ == "Heavy":
            if (Frag1_toallocate >= len(list1) - 1) and (Frag2_toallocate >= len(list2) - 2):  STOP = True
        elif typ == "All":
            if (Frag1_toallocate >= len(list1) - 1) and (Frag2_toallocate >= len(list2) - 1):  STOP = True
        #################

        #################
        #  This part handles sublist, keeplist1 and keeplist2. They are necessary to handle the results of the function "Combine", which is called later.
        #################
        if debug >= 2: print(" ")
        if debug >= 2: print("Fragments to allocate this iteration:",Frag1_toallocate,Frag2_toallocate,"out of",len(list1) - 1,len(list2) - 1)

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

        #################
        #  This part evaluates that the fragments that are going to be combined, can form one of the reference molecules. The resulting number of atoms is used.
        #################
        if list1[Frag1_toallocate].natoms + list2[Frag2_toallocate].natoms > maxatoms:
            if debug >= 2: print("SEQUENTIAL",typ,"SKIPPED",list1[Frag1_toallocate].natoms,"and",list2[Frag2_toallocate].natoms)
        else:
            if debug >= 2: print("SEQUENTIAL",typ,"iteration",niter,"with",len(list1),"and",len(list2),"Remaining in each list")
            if debug >= 2: print("SEQUENTIAL",typ,"sending",list1[Frag1_toallocate].labels,"and",list2[Frag2_toallocate].labels,"to combine")

            #################
            #  Here, the function "combine" is called. It will try cell translations of one fragment, and check whether it eventually combines with the second fragment into either a bigger fragment or a molecule
            #################
            goodlist, avglist, badlist = combine(sublist, refmoleclist, cellvec, threshold_tmat, factor, metal_factor)

            #################
            #  This part handles the results of combine
            #################
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
                if debug >= 2:
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
                if debug >= 1: print("FINISHED succesfully")
                break

            if typ == "Heavy":
                if len(list1) == 1:
                    for l in list1:
                        remainingfragments.append(l)
                        if debug >= 1: print("FINISHED with Remaining Fragment")
                    break

            if (len(list1) == 0) and (len(list2) == 0):
                if debug >= 1: print("FINISHED succesfully")
                break

        #################
        #  This part decides whether the WHILE loop must finish.
        #################
        niter += 1
        if niter > maxiter:
            STOP = True
        if niter > (lastiter + lastitermargin):
            STOP = True

        if not STOP:
            continue
        else:
            if (threshold_tmat <= (1.0 - increase_tmat)) or Last_Attempt:
                STOP = False
                threshold_tmat += increase_tmat
                if threshold_tmat >= 1:
                    Last_Attempt = True
                    if debug >= 2: print("Launching Last Attempt")
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
                    if debug >= 2:
                        print(" Increased Threshold_tmat. Now:", threshold_tmat)
                        print(" Maxsize of the translation matrix is=", maxsize)
                elif Last_Attempt:
                    for l in list1:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat, full=True)
                    for l in list2:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat, full=True)
                    if debug >= 2:
                        print("Trying Full Tmatrix for all Items in list")

                niter = 1
                Frag1_toallocate = 0
                Frag2_toallocate = 0
            else:
                for l in list1:
                    if debug >= 2: print("Sequential: list1 end:", l.labels)
                    remainingfragments.append(l)
                for l in list2:
                    if typ == "All" and debug >= 2: print("Sequential: list2 end:", l.labels)
                    if typ == "All": remainingfragments.append(l)
                break

    return molecsfoundlist, remainingfragments


#######################################################
def combine(tobeallocated: list, references: list, cellvec: list, threshold_tmat: float, factor: float, metal_factor: float, debug: int=0) -> Tuple[list, list, list]:

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
                found, newmoleclist = merge_fragments(tobeallocated, mergelist, references, cellvec, factor, metal_factor)

                if found == 1:
                    for m in mergelist:
                        available[m] = 0

                    if newmoleclist[0].natoms != mergedatoms:
                        if debug >= 2: print("COMBINE WARNING: I sent",mergedatoms,"atoms but received a molecule with",newmoleclist[0].natoms)

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
                        if (ref.elemcountvec == newmolec.elemcountvec).all() and shit == 0:
                            if (ref.adjtypes == newmolec.adjtypes).all():
                                shit = 1
                                newmolec.type = ref.type
                                goodlist.append(newmolec)
                                if debug >= 2: print("COMBINE: Fragment",newmolec.labels,"added to goodlist")
                    if shit == 0:
                        newmolec.type = "Rec. Fragment"
                        avglist.append(newmolec)
                        if debug >= 2: print("COMBINE: Fragment", newmolec.labels, "added to avglist")

        else:
            for kdx, a in enumerate(available):
                if a == 1: badlist.append(tobeallocated[kdx])
            break

        if idx == len(combinations) - 1:
            for kdx, a in enumerate(available):
                if a == 1: badlist.append(tobeallocated[kdx])
            break

    return goodlist, avglist, badlist


#######################################################
def merge_fragments(fraglist: list, listofids: list, reflist: list, cellvec: list, factor: float, metal_factor: float, debug: int=0) -> Tuple[int, list]:
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
            recatlist = [] ## atlist

            reccoord.extend(fraglist[listofids[keep]].coord)
            reclabels.extend(fraglist[listofids[keep]].labels)
            recatlist.extend(fraglist[listofids[keep]].atlist)  ## atlist

            for kdx, mol in enumerate(tmol):
                reclabels.extend(fraglist[listwithoutkeep[kdx]].labels)
                recatlist.extend(fraglist[listwithoutkeep[kdx]].atlist)
                if mol != (0, 0, 0):
                    # indicates that the molecule needs translation in direction defined by "tma"
                    newcoord = translate(mol, fraglist[listwithoutkeep[kdx]].coord, cellvec)
                    if len(newcoord) != len(fraglist[listwithoutkeep[kdx]].coord):
                        print("error 1 in Reconstruct")
                    reccoord.extend(newcoord)
                if mol == (0, 0, 0):
                    reccoord.extend(fraglist[listwithoutkeep[kdx]].coord)

            Warning, reclist = getmolecs(reclabels, reccoord, factor, metal_factor, atlist=recatlist)
            if len(reclist) == 1:
                status = checkchemistry(reclist[0], reflist, "Max")
                break
    else:
        reclist = []


    return status, reclist


#######################################################
def identify_frag_molec_H(blocklist: list, moleclist: list, refmoleclist: list, cellvec: list, debug: int=0) -> Tuple[list, list, list, int]:

    init_natoms = 0

    fraglist = []
    Hlist = []

    if debug >= 1: 
        for ref in refmoleclist: 
            print(f"{ref.formula} found as reference")

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

    if debug >= 1: print(len(blocklist),"Blocks sorted for reconstruction as (Molec, Frag, H):",len(moleclist),len(fraglist),len(Hlist))
    if debug >= 1: print("With a total of", init_natoms, "atoms")

    return moleclist, fraglist, Hlist, init_natoms


#######################################################
def fragments_reconstruct(moleclist: list, fraglist: list, Hlist: list, refmoleclist: list, cellvec: list, factor: float, metal_factor: float, debug: int=0) -> Tuple[list, list, bool]:

    Warning = False

    # Reconstruct Heavy Fragments
    if len(fraglist) > 1:
        if debug >= 2: print("")
        if debug >= 2: print("##############################################")
        if debug >= 2: print(len(fraglist), "molecules submitted to SEQUENTIAL with Heavy")
        if debug >= 2: print("##############################################")
        newmols, remfrag = sequential(fraglist, refmoleclist, cellvec, factor, metal_factor, "Heavy", debug)
        if debug >= 1: print(f"{len(newmols)} molecules and {len(remfrag)} fragments out of SEQUENTIAL with Heavy")
        moleclist.extend(newmols)
        fraglist = []
        fraglist.extend(remfrag)
        fraglist.extend(Hlist)

        # For debugging
        if debug >= 2:
            print(" ")
            # Prints molecules after Heavy Fragment Reconstruction
            if len(newmols) > 0:
                for mol in newmols:
                    print("Molec reconstructed after Heavy", mol.natoms, mol.formula, mol.type)
            else:
                print("NO Molecules reconstructed after Heavy")
            if len(remfrag) > 0:
                for rem in remfrag:
                    print("Remaining after Heavy", rem.natoms, rem.formula, rem.type)
            else:
                print("NO remaining Molecules after Heavy")
            print(" ")
    else:
        print("Only 0 or 1 heavy fragments. Skipping Heavy")
        remfrag = fraglist.copy()

    # Reconstruct Hydrogens with remaining Fragments
    if len(remfrag) > 0 and len(Hlist) > 0:
        if debug >= 2: print("")
        if debug >= 2: print("##############################################")
        if debug >= 2: print(len(fraglist), "molecules submitted to sequential with All")
        if debug >= 2: print("##############################################")
        finalmols, remfrag = sequential(fraglist, refmoleclist, cellvec, factor, metal_factor, "All", debug)
        if len(remfrag) > 0:
            Warning = True
            for rem in remfrag:
                if debug >= 1: print("Remaining after Hydrogen reconstruction",rem.natoms,rem.formula,rem.type)
        else:
            if debug >= 1: print("NO remaining Molecules after Hydrogen reconstruction")
            Warning = False
        print(" ")
    else:
        if len(remfrag) > 0 and len(Hlist) == 0: 
            Warning = True
            if debug >= 1: print("There are remaining Fragments and no H in list")
            finalmols = []
            remfrag = []
        elif len(remfrag) == 0 and len(Hlist) > 0: 
            Warning = True
            if debug >= 1: print("There are isolated H atoms in cell")
            finalmols = []
            remfrag = []
        elif len(remfrag) == 0 and len(Hlist) == 0: 
            if debug >= 1: print("Not necessary to reconstruct Hydrogens")
            finalmols = fraglist.copy()  # IF not Hidrogen fragments, then is done
            remfrag = []

    return moleclist, finalmols, Warning

#######################################################
#def compare_moleclist_refmoleclist(moleclist: list, refmoleclist: list, debug: int=0) -> bool:
#    
#    found = []
#    not_found = []
#
#    for mol in moleclist:
#        for ref in refmoleclist:
#            if (ref.elemcountvec == mol.elemcountvec).all() and (ref.adjtypes == mol.adjtypes).all():
#                found.append(mol)
#            else:
#                pass
#
#    for mol in moleclist:
#        if mol not in found:
#            not_found.append(mol)
#
#    print(f"Compare molecules in moleclist with molecules in refmoleclist: Right/Wrong {len(found)}/ {len(not_found)}")
#
#    if len(not_found) == 0 :
#        Warning = False
#        print("All molecules in moleclist are found in refmoleclist\n")
#    else : 
#        Warning = True
#        print("Wrong molecules in moleclist")
#        for wrong in not_found: 
#            print("Wrong: the number of atoms", len(wrong.labels), wrong.labels)
#        if debug >= 2 : 
#            print("\nRefmoleclist")
#            for j, ref in enumerate(refmoleclist):
#                print("Ref {}".format(j), "the number of atoms:", len(ref.labels), ref.labels)
#
#    print("")
#
#    return Warning
#
#
#######################################################
def assigntype(molecule: object, references: list, debug: int=0) -> str:
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
def split_complexes_reassign_type(cell: object, moleclist: list, debug: int=0) -> object:

    if not all(cell.warning_list):
        # Split Complexes
        for mol in moleclist:
            if mol.type == "Complex":
                mol.ligandlist, mol.metalist = splitcomplex(
                    mol, mol.factor, mol.metal_factor
                )
                dummy = get_hapticity(mol)
                
                # Check coordination geometry around metal
                mol.metalist = get_coordination_geometry (mol.metalist, dummy, debug=0)

        # Reassign Type of molecules and store information
        for mol in moleclist:
            mol.type = assigntype(mol, cell.refmoleclist)
            mol.refcode = cell.refcode
            mol.name = str(cell.refcode + "_" + mol.type + "_" + str(moleclist.index(mol)))
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

    coord = [None] * cell.natoms # Atom coordinate after cell reconstruction
    for mol in cell.moleclist:
        for z in zip(mol.atlist, mol.coord):
            coord[z[0]] = z[1]
    cell.coord = coord

    return cell


#######################################################
def get_coordination_geometry (metalist: object, hapticity: bool, debug: int=0) -> None:
    # Get coordination geomery in case that there is no hapticity in TM complexes
    # Find the cloest geometry using Shape measurment in cosymlib https://cosymlib.readthedocs.io/en/latest/

    if hapticity == False :
        
        for met in metalist:
            positions=[]
            symbols=[]
            connectivity=[]    
            
            cn =len(met.coordinating_atoms)
            connectivity= [[1, i] for i in range(2, cn+2)]
            
            symbols.append(met.label)
            positions.append(met.coord)

            for a, a_site in zip(met.coordinating_atoms, met.coordinating_atoms_sites):
                symbols.append(a)
                positions.append(a_site)   
            
            geometry = Geometry(positions=positions, 
                        symbols=symbols,
                        name=met.refcode, 
                        connectivity=connectivity)
                       
            ref_geom = np.array(shape_structure_references['{} Vertices'.format(cn)])
            posgeom_dev={}
            
            if debug >= 2 :
                for p, s in zip(symbols, positions):
                    print (p, s)
                print("")
            for idx, rg in enumerate(ref_geom[:,0]):
                shp_measure = geometry.get_shape_measure(rg, central_atom=1)
                geom = ref_geom[:,3][idx]
                posgeom_dev[geom]=round(shp_measure, 3)      

            met.coordination (hapticity, posgeom_dev) 

            if debug >= 1 :
                print(met.label)
                print (f"Coordination number : {met.coordination_number} {met.posgeom_dev}")
                print(f"The most likely geometry : '{met.geometry}' with deviation value {met.deviation} (hapticity : {met.hapticity})")
                print("")      

    else :
        posgeom_dev = {}
        for met in metalist:
            met.coordination (hapticity, posgeom_dev) 
    
    return metalist

#######################################################
def get_hapticity(molecule: object, debug: int=0) -> bool:
    # This function evaluates whether a molecule has any ligand with hapticity and, if so, detects which type of hapticity
    # The information is stored in both the molecule and ligand objects.
    # This function also defines the number of groups in a ligand. A "group" is a group of adjacent atoms that is connected to the metal atom.
    # For instance, a Cp ligand forms a group of 5 C atoms connected to the metal
    # In turn, a Cp ligand that is substituted with a long functional group that is connected to the metal by, say, an O atom at the end of such functional group
    # ... would generate 2 groups. The Cp, and the O atom.
    # This information is useful in the subroutine that decides whether any element must be added to the group to generate a meaningful connectivity and charge. This is done somewhere else

    if molecule.type == "Complex":
        for lig in molecule.ligandlist:
            groups = find_groups_within_ligand(lig)

            for g in groups:
                has_hapticity = False
                group_hapttype = []

                list_of_coord_atoms = []
                for idx, a in enumerate(lig.atoms):
                    if idx in g and a.mconnec > 0:
                        list_of_coord_atoms.append(a.label)

                numC = list_of_coord_atoms.count("C")  # Carbon is the most common connected atom in ligands with hapticity
                numAs = list_of_coord_atoms.count("As")  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
                numP = list_of_coord_atoms.count("P")  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
                numO = list_of_coord_atoms.count("O")  # For h4-Enone
                ## Carbon-based Haptic Ligands
                if numC == 2:
                    group_hapttype = ["h2-Benzene", "h2-Butadiene", "h2-ethylene"]
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
                elif numC == 0 and numP == 5:
                    group_hapttype = ["h5-Pentaphosphole"]
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


#######################################################
def get_hapticity_ligand (lig: object, debug: int=0) -> bool:
    # This function evaluates whether a ligand has hapticity and, if so, detects which type of hapticity

    groups = find_groups_within_ligand(lig)

    for g in groups:
        has_hapticity = False
        group_hapttype = []

        list_of_coord_atoms = []
        for idx, a in enumerate(lig.atoms):
            if idx in g and a.mconnec > 0:
                list_of_coord_atoms.append(a.label)

        numC = list_of_coord_atoms.count("C")  # Carbon is the most common connected atom in ligands with hapticity
        numAs = list_of_coord_atoms.count("As")  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
        numP = list_of_coord_atoms.count("P")  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
        numO = list_of_coord_atoms.count("O")  # For h4-Enone
        ## Carbon-based Haptic Ligands
        if numC == 2:
            group_hapttype = ["h2-Benzene", "h2-Butadiene", "h2-ethylene"]
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
        elif numC == 0 and numP == 5:
            group_hapttype = ["h5-Pentaphosphole"]
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

    return lig

################ END
