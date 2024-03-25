import numpy as np
import itertools
from cell2mol.cell_operations import translate
from cell2mol.other import additem, absolute_value
from cell2mol.connectivity import compare_species, count_species, split_species
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

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

    ## Prepares Blocks
    for b in blocklist:
        if debug > 0: print(f"CLASSIFY FRAGMENTS, preparing block {b}")
        if not hasattr(b,"centroid"):         b.get_centroid()
        if not hasattr(b,"element_count"):    b.set_element_count()
        if not hasattr(b,"numH"):             b.numH = b.set_element_count()[4] + b.set_element_count()[3] #"Hidrogen + Deuterium atoms"
    ## Prepares Reference Molecules
    for ref in refmoleclist:
        if debug > 0: print(f"CLASSIFY FRAGMENTS, preparing reference {ref}")
        if not hasattr(ref,"element_count"):  ref.set_element_count()
        if not hasattr(ref,"numH"):           ref.numH = ref.set_element_count()[4] + ref.set_element_count()[3] #"Hidrogen + Deuterium atoms"

    # Classifies blocks and puts them in 3 bags. (1) Full molecules, (2) partial molecules=fragments, (3) Hydrogens
    for idx, block in enumerate(blocklist):
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
    ## Moleclist is the list of species which have been identified as 'complete' molecules
    ## Fraglist is the list of species which are not 'complete' molecules
    ## Hlist is the list of species that are only H atoms
    ## Refmoleclist is the list of species that are identified as reference molecules (that is, molecules that will appear in the unit cell once reconstructed)
    Warning = False
    # Reconstruct Heavy Fragments
    if len(fraglist) > 1:
        print("")
        print("##############################################")
        print("FRAG_RECONSTRUCT.", len(fraglist), "molecules submitted to SEQUENTIAL with Heavy")
        print("##############################################")
        newmols, remfrag = sequential(fraglist, refmoleclist, cellvec, factor, metal_factor, "Heavy", debug)
        print(f"FRAG_RECONSTRUCT. {len(newmols)} molecules and {len(remfrag)} fragments out of SEQUENTIAL with Heavy")
        moleclist.extend(newmols)

        # After the first step, fraglist is made of the remaining molecules in the first step, and the list of H atoms
        fraglist = []
        fraglist.extend(remfrag)
        fraglist.extend(Hlist)
    else:  print(f"Only {len(fraglist)} heavy fragments. Skipping Heavy"); remfrag = fraglist.copy()

    # Reconstruct Hydrogens with remaining Fragments
    if len(remfrag) > 0 and len(Hlist) > 0:
        print("FRAG_RECONSTRUCT.", len(fraglist), "molecules submitted to sequential with All")
        finalmols, remfrag = sequential(fraglist, refmoleclist, cellvec, factor, metal_factor, "All", debug)
        moleclist.extend(finalmols)
        if len(remfrag) > 0: Warning = True;  print("FRAG_RECONSTRUCT. Remaining after Hydrogen reconstruction",remfrag)
        else:                Warning = False; print("FRAG_RECONSTRUCT. No remaining Molecules after Hydrogen reconstruction")
    elif len(remfrag) > 0 and len(Hlist) == 0:
        Warning = True
        print("FRAG_RECONSTRUCT. WARNING: There are remaining Fragments and no H in list")
    elif len(remfrag) == 0 and len(Hlist) > 0:
        Warning = True
        print("FRAG_RECONSTRUCT. WARNING: There are isolated H atoms in cell")

    return moleclist, Warning

#######################################################
def assign_subtype(mol: object, references: list) -> str:
    for ref in references:
        issame = compare_species(mol, ref)
        if issame: 
            if ref.iscomplex: return "Complex"
            else:             return "Other"
    # If not in references
    if mol.iscomplex: return "Complex"
    else:                  return "Other"

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
    niter = 1
    maxiter = 3000
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
    from cell2mol.classes import molecule

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
                newmolec.cell_indices = blocklist[0]
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


