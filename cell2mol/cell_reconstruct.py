import numpy as np
import itertools

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Tuple
from collections import defaultdict

from cell2mol.cellconversions import frac2cart_fromparam, cart2frac, translate
from cell2mol.other import compute_centroid, additem, inv
from cell2mol.connectivity import get_radii, get_adjacency_types, get_adjmatrix, get_blocks, get_element_count, compare_atoms, compare_species
from cell2mol.tmcharge_common import find_closest_metal
from cell2mol.missingH import getangle
from cell2mol.coordination_sphere import get_coordination_geometry
# Imports Classes
from cell2mol.classes import specie, molecule, ligand, group, atom, metal, cell
from cosymlib import Geometry
from cosymlib.shape.tools import shape_structure_references


from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

#######################################################
def verify_connectivity(ligand: object, molecule: object, debug: int = 0) -> None:
    """Verifies the connectivity of a ligand. It is used to correct the connectivity of the ligand, if needed.
    
    Args:
        ligand (object): ligand object
        molecule (object): molecule object
        debug (int, optional): debug level. Defaults to 0.
    
    Returns:
        None
    """

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
                    tmpradii = get_radii(newlab)
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
def classify_fragments(blocklist: list, refmoleclist: list, debug: int=0):
    init_natoms = 0
    moleclist = []
    fraglist  = []
    Hlist     = []

    # Classifies blocks and puts them in 3 bags. (1) Full molecules, (2) partial molecules=fragments, (3) Hydrogens
    for idx, block in enumerate(blocklist):
        if not hasattr(block,"numH"): block.numH = block.set_element_count()[4]
        if (block.natoms == 1) and (block.numH == 1):
            block.subtype = "H"
            Hlist.append(block)
        else:
            found = False 
            for ref in refmoleclist:
                issame = compare_species(ref, block)
                if issame: 
                    block.subtype = "molecule"
                    moleclist.append(block)
                    found = True
            if not found:
                block.subtype = "fragment"
                fraglist.append(block)

    if debug > 0: print(len(blocklist),"Blocks sorted as (Molec, Frag, H):",len(moleclist),len(fraglist),len(Hlist))
    return moleclist, fraglist, Hlist

#######################################################
def fragments_reconstruct(moleclist: list, fraglist: list, Hlist: list, refmoleclist: list, cellvec: list, factor: float=1.3, metal_factor: float=1.0, debug: int=0):

    Warning = False
    # Reconstruct Heavy Fragments
    if len(fraglist) > 1:
        print("")
        print("##############################################")
        print(len(fraglist), "molecules submitted to SEQUENTIAL with Heavy")
        print("##############################################")
        newmols, remfrag = sequential(fraglist, refmoleclist, cellvec, factor, metal_factor, "Heavy", debug)
        print(f"{len(newmols)} molecules and {len(remfrag)} fragments out of SEQUENTIAL with Heavy")
        moleclist.extend(newmols)
        fraglist = []
        fraglist.extend(remfrag)
        fraglist.extend(Hlist)

        # For debugging
        if debug >= 1:
            print(" ")
            # Prints molecules after Heavy Fragment Reconstruction
            if len(newmols) > 0:
                for mol in newmols:
                    print("Molec reconstructed after Heavy", mol.natoms, mol.formula, mol.type)
            else:
                print("NO Molecules reconstructed after Heavy")
            if len(remfrag) > 0:
                for rem in remfrag:
                    print("Remaining after Heavy", rem.natoms, rem.formula, rem.subtype)
            else:
                print("NO remaining Molecules after Heavy")
            print(" ")
    else:
        print("Only 0 or 1 heavy fragments. Skipping Heavy")
        remfrag = fraglist.copy()

    # Reconstruct Hydrogens with remaining Fragments
    if len(remfrag) > 0 and len(Hlist) > 0:
        print("")
        print("##############################################")
        print(len(fraglist), "molecules submitted to sequential with All")
        print("##############################################")
        finalmols, remfrag = sequential(fraglist, refmoleclist, cellvec, factor, metal_factor, "All", debug)
        if len(remfrag) > 0:
            Warning = True
            for rem in remfrag:
                print("Remaining after Hydrogen reconstruction",rem.natoms,rem.formula,rem.subtype)
        else:
            print("NO remaining Molecules after Hydrogen reconstruction")
            Warning = False
        print(" ")
    else:
        if len(remfrag) > 0 and len(Hlist) == 0:
            Warning = True
            print("There are remaining Fragments and no H in list")
            finalmols = []
            remfrag = []
        elif len(remfrag) == 0 and len(Hlist) > 0:
            Warning = True
            print("There are isolated H atoms in cell")
            finalmols = []
            remfrag = []
        elif len(remfrag) == 0 and len(Hlist) == 0:
            print("Not necessary to reconstruct Hydrogens")
            finalmols = fraglist.copy()  # IF not Hidrogen fragments, then is done
            remfrag = []

    return moleclist, finalmols, Warning


    latCnt = [x[:] for x in [[None] * 3] * 3]
    for a in range(3):
        for b in range(3):
            latCnt[a][b] = cellvec[b][a]
    fracCoords = []
    detLatCnt = det3(latCnt)
    for i in cartCoords:
        aPos = (det3([
                    [i[0], latCnt[0][1], latCnt[0][2]],
                    [i[1], latCnt[1][1], latCnt[1][2]],
                    [i[2], latCnt[2][1], latCnt[2][2]],
                ]
            )
        ) / detLatCnt
        bPos = (
            det3(
                [
                    [latCnt[0][0], i[0], latCnt[0][2]],
                    [latCnt[1][0], i[1], latCnt[1][2]],
                    [latCnt[2][0], i[2], latCnt[2][2]],
                ]
            )
        ) / detLatCnt
        cPos = (
            det3(
                [
                    [latCnt[0][0], latCnt[0][1], i[0]],
                    [latCnt[1][0], latCnt[1][1], i[1]],
                    [latCnt[2][0], latCnt[2][1], i[2]],
                ]
            )
        ) / detLatCnt
        fracCoords.append([aPos, bPos, cPos])
    return fracCoords

#######################################################
def assign_subtype(molecule: object, references: list) -> str:
    for ref in references:
        issame = compare_species(molecule, ref)
        if issame: 
            if ref.iscomplex: return "Complex"
            else:             return "Other"
    # If not in references
    if molecule.iscomplex: return "Complex"
    else:                  return "Other"

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
                mol = get_coordination_geometry(mol, debug=0)

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
def check_hapticity_hapttype (atoms_list: list, g: list) -> Tuple[bool, list]:
    # Check if the list of atoms in a given group has hapticity and return its hapticity and the haptic type

    has_hapticity = False
    group_hapttype = []

    list_of_coord_atoms = []
    for idx, a in enumerate(atoms_list):
        if idx in g and a.mconnec > 0:
            list_of_coord_atoms.append(a.label)

    numC = list_of_coord_atoms.count("C")  # Carbon is the most common connected atom in ligands with hapticity
    numAs = list_of_coord_atoms.count("As")  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
    numP = list_of_coord_atoms.count("P")  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
    numO = list_of_coord_atoms.count("O")  # For h4-Enone
    numN = list_of_coord_atoms.count("N")

    # print(f"{g=} {list_of_coord_atoms=} {numC=} {numAs=} {numP=} {numO=} {numN=}")

    ## Carbon-based Haptic Ligands
    if numC == 2:
        group_hapttype = ["h2-Benzene", "h2-Butadiene", "h2-ethylene"]
        has_hapticity = True
    # elif numC == 2 and numN == 2 : # 2,2′-bipyridine

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
        group_hapttype = ["h7-Cycloheptatrienyl"]
        has_hapticity = True
    elif numC == 8:
        group_hapttype = ["h8-Cyclooctatetraenyl"]
        has_hapticity = True

        has_hapticity = True

    # Other less common types of haptic ligands
    elif numC == 0 and numAs == 5:
        group_hapttype = ["h5-AsCp"]
        has_hapticity = True
    elif numC == 0 and numP == 5:
        group_hapttype = ["h5-Pentaphosphole"]
        has_hapticity = True

    return has_hapticity, group_hapttype

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
                # Check if the list of atoms in a given group has hapticity and return its hapticity and the haptic type
                has_hapticity, group_hapttype = check_hapticity_hapttype(lig.atoms, g)

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
        # Check if the list of atoms in a given group has hapticity and return its hapticity and the haptic type
        has_hapticity, group_hapttype = check_hapticity_hapttype(lig.atoms, g)

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




#######################################################
def sequential(fragmentlist: list, refmoleclist: list, cellvec: list, factor: float=1.3, metal_factor: float=1.0, typ: str="All", debug: int=1):
    # Crappy function that controls the reconstruction process. It is called sequential because pairs of fragments are sent one by one. Ideally, a parallel version would be desirable.
    # Given a list of fragments(fragmentlist), a list of reference molecules(refmoleclist), and some other minor parameters, the function sends pairs of fragments and evaluates if they...
    # ...form a bigger fragment. If so, the bigger fragment is evaluated. If it coincides with one of the molecules in refmoleclist, than it means that it is a full molecule that requires no further work.
    # ...if it does not, then it means that requires further reconstruction, and is again introduced in the loop.
    # typ is a variable that defines how to combine the fragments. To speed up the process, this function is called twice in main.
    # -First, to combine heavy fragments among themselves (typ="Heavy")
    # -Second, to combie heavy fragments with H atoms (typ="All")
    #:return molecsfoundlist, remainingfragments: lists of molecules and fragments, respectively, saved as objects

    if debug >= 1: print("Entered sequential with", len(fragmentlist), "fragments to reconstruct")

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
        if not hasattr(frag,"frac_centroid"): frag.get_centroid()
        frag.tmatrix = tmatgenerator(frag.frac_centroid, threshold_tmat)

    remlist = []
    Hlist = []
    for frag in fragmentlist:
        if (frag.natoms == 1) and (frag.numH == 1):
            frag.subtype = "H"
            Hlist.append(frag)
        else:
            frag.subtype = "Heavy"
            remlist.append(frag)
    if debug >= 1: print("Found",len(remlist),"and",len(Hlist),"Heavy and Hydrogen fragments, respectively")
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
        if (Frag2_toallocate > len(list2) - 1):  # Reaches the end of the second list. Restarts 2nd and advances 1st
            Frag1_toallocate += 1
            Frag2_toallocate = 0
        if (Frag1_toallocate > len(list1) - 1):  # Reaches the end of the first list. Restarts both
            Frag1_toallocate = 0
            Frag2_toallocate = 0
        if typ == "Heavy":
            if Frag1_toallocate == Frag2_toallocate: Frag2_toallocate += 1
        if typ == "Heavy":
            if (Frag1_toallocate >= len(list1) - 1) and (Frag2_toallocate >= len(list2) - 2): STOP = True
        elif typ == "All":
            if (Frag1_toallocate >= len(list1) - 1) and (Frag2_toallocate >= len(list2) - 1): STOP = True
        #################

        #################
        #  This part handles sublist, keeplist1 and keeplist2. They are necessary to handle the results of the function "Combine", which is called later.
        #################
        if debug >= 1: print(" ")
        if debug >= 1: print("Fragments to allocate:",Frag1_toallocate,Frag2_toallocate,"out of",len(list1)-1,len(list2)-1)
        sublist = []
        keeplist1 = []
        keeplist2 = []
        if typ == "Heavy":
            for i in range(0, len(list1)):
                if i == Frag1_toallocate:   sublist.append(list1[i])
                elif i != Frag2_toallocate: keeplist1.append(list1[i])
            for i in range(0, len(list2)):
                if i == Frag2_toallocate:   sublist.append(list2[i])
                elif i != Frag1_toallocate: keeplist2.append(list2[i])
        elif typ == "All":
            for i in range(0, len(list1)):
                if i == Frag1_toallocate:   sublist.append(list1[i])
                elif i != Frag1_toallocate: keeplist1.append(list1[i])
            for i in range(0, len(list2)):
                if i == Frag2_toallocate:   sublist.append(list2[i])
                elif i != Frag2_toallocate: keeplist2.append(list2[i])

        #################
        #  This part evaluates that the fragments that are going to be combined, can form one of the reference molecules. The resulting number of atoms is used.
        #################
        if list1[Frag1_toallocate].natoms + list2[Frag2_toallocate].natoms > maxatoms:
            if debug >= 1: print("SEQUENTIAL",typ,"SKIPPED",list1[Frag1_toallocate].natoms,"and",list2[Frag2_toallocate].natoms)
        else:
            if debug >= 1: print("SEQUENTIAL",typ,"iteration",niter,"with",len(list1),"and",len(list2),"Remaining in each list")
            if debug >= 1: print("SEQUENTIAL",typ,"sending",list1[Frag1_toallocate].labels,"and",list2[Frag2_toallocate].labels,"to combine")

            #################
            #  Here, the function "combine" is called. It will try cell translations of one fragment, and check whether it eventually combines with the second fragment into either a bigger fragment or a molecule
            #################
            goodlist, avglist, badlist = combine(sublist, refmoleclist, cellvec, threshold_tmat, factor, metal_factor, debug=debug)

            #################
            #  This part handles the results of combine
            #################
            if (len(goodlist) > 0) or (len(avglist) > 0):
                # it means that the function combine worked. Thus, it restarts the fragments to allocate
                lastiter = niter
                Frag1_toallocate = 0
                Frag2_toallocate = 0

            # Adds the found molecule to the appropriate list
            for g in goodlist:
                molecsfoundlist.append(g)
                if debug >= 1: print(f"SEQUENTIAL: Found molecule {g.formula}")

            # Reconstructs list1 and list2
            list1 = []
            list2 = []
            for a in avglist:
                list1.append(a)
                if typ == "Heavy": list2.append(a)

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

        #################
        #  This part decides whether the WHILE loop must finish.
        #################
        niter += 1
        if niter > maxiter:                     STOP = True
        if niter > (lastiter + lastitermargin): STOP = True
        if not STOP: continue
        else:
            if (threshold_tmat <= (1.0 - increase_tmat)) or Last_Attempt:
                STOP = False
                threshold_tmat += increase_tmat
                if threshold_tmat >= 1: Last_Attempt = True
                if not Last_Attempt: 
                    maxsize = 0
                    for l in list1:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat)
                        if len(l.tmatrix) > maxsize: maxsize = len(l.tmatrix)
                    for l in list2:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat)
                        if len(l.tmatrix) > maxsize: maxsize = len(l.tmatrix)
                    if debug >= 1: print(" Increased Threshold_tmat. Now:", threshold_tmat)
                    if debug >= 1: print(" Maxsize of the translation matrix is=", maxsize)
                elif Last_Attempt:
                    for l in list1:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat, full=True)
                    for l in list2:
                        l.tmatrix = tmatgenerator(l.centroid, threshold_tmat, full=True)
                    if debug >= 1: print("Trying Full Tmatrix for all Items in list")

                niter = 1
                Frag1_toallocate = 0
                Frag2_toallocate = 0
            else:
                for l in list1:
                    if debug >= 1: print("Sequential: list1 end:", l.labels)
                    remainingfragments.append(l)
                for l in list2:
                    if typ == "All" and debug >= 1: print("Sequential: list2 end:", l.labels)
                    if typ == "All": remainingfragments.append(l)
                break

    return molecsfoundlist, remainingfragments

#######################################################
def combine(tobemerged: list, references: list, cellvec: list, threshold_tmat: float, cov_factor: float, metal_factor: float, debug: int=0):
    goodlist = []   ## List of molecules coming from the two fragments received
    avglist = []    ## List of bigger fragments coming from the two fragments received
    badlist = []    ## List of fragments as they entered the function

    ## Merges the coordinates of both fragments, and finds species
    newmolec = merge_fragments(tobemerged, references, cellvec, cov_factor, metal_factor, debug=debug)
    if newmolec is not None and debug >= 1: print("COMBINE. received molecule:", newmolec, "from merge fragments")

    ## Steric Clashes, or more than one fragment retrieved
    if newmolec is None: 
        badlist.append(tobemerged[0])
        badlist.append(tobemerged[1])

    ## Single specie retrieved
    if newmolec is not None:
        newmolec.get_fractional_coord(cellvec)
        newmolec.get_centroid()
        newmolec.tmatrix = tmatgenerator(newmolec.frac_centroid, threshold_tmat)

        found = False 
        for ref in references:
            if not found: 
                issame = compare_species(newmolec, ref)
                if issame:    ## Then is a molecule that appears in the reference list 
                    found = True 
                    newmolec.subtype = ref.subtype
                    goodlist.append(newmolec)
                    if debug >= 1: print("COMBINE: Fragment",newmolec.formula,"added to goodlist")
        if not found:        ## Then it is a fragment. A bigger one, but still a fragment
            newmolec.subtype = "Rec. Fragment"
            avglist.append(newmolec)
            if debug >= 1: print("COMBINE: Fragment", newmolec.formula, "added to avglist")

    return goodlist, avglist, badlist

#######################################################
def merge_fragments(frags: list, refs: list, cellvec: list, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
    # finds biggest fragment and keeps it in the original cell
    sizes = []
    for f in frags:
        size = f.natoms
        sizes.append(size)
    keep_idx = np.argmax(sizes)
    if   keep_idx == 0: move_idx = 1
    elif keep_idx == 1: move_idx = 0
    keep_frag = frags[keep_idx]
    move_frag = frags[move_idx]
    if debug > 0: print("MERGE_FRAGMENTS: keep_idx", keep_idx)
    if debug > 0: print("MERGE_FRAGMENTS: move_idx", move_idx)
    if debug > 0: print("MERGE_FRAGMENTS: move_frag.tmatrix", move_frag.tmatrix)

    #applytranspose = list(itertools.product(*move_frag.tmatrix))
    #print("applytranspose", applytranspose)
    if len(move_frag.tmatrix) == 0: return None
    for t in move_frag.tmatrix:
        if debug > 0: print("MERGE_FRAGMENTS: translation", t)
        ## Applies Translations and each time, it checks if a bigger molecule is formed
        ## meaning that the translation was successful
        reclabels = []
        reclabels.extend(keep_frag.labels)
        reclabels.extend(move_frag.labels)
        reccoord = []
        reccoord.extend(keep_frag.coord)
        if t == (0, 0, 0): reccoord.extend(move_frag.coord)
        else:              reccoord.extend(translate(t, move_frag.coord, cellvec))

        ## Evaluate if we get only one fragment. If so, we're ok:
        numspecs  = count_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        if debug > 0: print("MERGE_FRAGMENTS: count_species found", numspecs)
        if numspecs != 1: continue
        blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        if blocklist is None: continue
        else:
            if len(blocklist) != 1: continue
            if len(blocklist) == 1: 
                newmolec = molecule(reclabels, reccoord)
                newmolec.set_adjacency_parameters(cov_factor, metal_factor)
                newmolec.set_adj_types()
                newmolec.set_element_count()
                newmolec.get_adjmatrix()
                newmolec.get_centroid()
                return newmolec
    return None


    return (
        (mat[0][0] * mat[1][1] * mat[2][2])
        + (mat[0][1] * mat[1][2] * mat[2][0])
        + (mat[0][2] * mat[1][0] * mat[2][1])
        - (mat[0][2] * mat[1][1] * mat[2][0])
        - (mat[0][1] * mat[1][0] * mat[2][2])
        - (mat[0][0] * mat[1][2] * mat[2][1]))


