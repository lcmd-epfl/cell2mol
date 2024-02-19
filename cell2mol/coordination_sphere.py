import numpy as np
from cell2mol.Elementdata import ElementData  
from cell2mol.other import *

elemdatabase = ElementData()

#######################################################
def get_coordination_geometry (mol: object, debug: int=0) -> object:

    for met in mol.metalist:
        coord_list = []
        for a, a_site in zip(met.coordinating_atoms, met.coordinating_atoms_sites):
            for atom in mol.atoms:
                if (atom.label == a ) and (atom.coord == a_site):
                    coord_list.append(atom)
        met.coordinating_atlist = coord_list

        positions=[]
        symbols=[]
        symbols.append(met.label)
        positions.append(met.coord)
        if debug >= 2 : print(f"{met.label=} {met.coord=} {met.coordinating_atlist=}")

        # atoms_list = [ mol.atoms[i] for i in met.coordinating_atlist ]
        hapticity, hapttype, group_list, group_atoms_list = get_hapticity_within_atoms_list(met.coordinating_atlist)

        if hapticity == False :
            for a, a_site in zip(met.coordinating_atoms, met.coordinating_atoms_sites):
                symbols.append(a)
                positions.append(a_site)
                if debug >= 2 : print(a, a_site)

            posgeom_dev = shape_measure(positions, symbols, debug=debug)
            met.coordination (len(met.coordinating_atlist), hapticity, hapttype, posgeom_dev)
            met.group_list = group_list
            met.group_atoms_list = group_atoms_list

            if debug >= 1 :
                print(f"Molecule : {mol.formula}")
                print(f"Metal : {met.label}\tCoordinating atoms : {met.coordinating_atoms}")
                print (f"Coordination number : {met.coordination_number} {met.posgeom_dev}")
                print(f"The most likely geometry : '{met.geometry}' with deviation value {met.deviation} (hapticity : {met.hapticity})")
                print("")

        else :
            for g, g_atoms in zip(group_list, group_atoms_list):
                if g.hapticity :
                    arr=[]
                    for a in g_atoms:
                        if debug >= 2 : print("\t", a.index, a.label, a.coord, a.adjacency, a.metal_adjacency, a.connec, a.mconnec)
                        arr.append(a.coord)
                    if len(arr) > 1 :
                        arr = np.array(arr)
                        symbols.append(str(g.hapttype))
                        positions.append(compute_centroid(arr))
                        if debug >= 2 : print(f"mid point of {g.hapttype=}", compute_centroid(arr))
                else :
                    for a in g_atoms:
                        symbols.append(a.label)
                        positions.append(a.coord)
                        if debug >= 2 : print(a.label, a.coord)

            posgeom_dev = shape_measure(positions, symbols, debug=debug)

            cn_haptic = len(symbols)-1
            met.coordination (cn_haptic, hapticity, hapttype, posgeom_dev)
            met.group_list = group_list
            met.group_atoms_list = group_atoms_list

            if debug >= 1 :
                print(f"Molecule : {mol.formula}")
                print(f"Metal : {met.label}\tCoordinating atoms : {met.coordinating_atoms}")
                print (f"The number of coordinating points (including the mid point of haptic ligands) : {cn_haptic} {met.posgeom_dev}")
                print(f"The most likely geometry : '{met.geometry}' with deviation value {met.deviation} (hapticity : {met.hapticity})")
                print(f"The type of hapticity : {met.hapttype}")
                print("")

    return mol

#######################################################
def get_hapticity_within_atoms_list (atoms_list: list) -> bool:
    # This function evaluates whether a ligand has hapticity and, if so, detects which type of hapticity

    group_list = []
    group_atoms_list = []
    hapttype = []
    groups = find_groups_within_atoms_list(atoms_list)


    for g in groups:
        # Check if the list of atom objects in a given group has hapticity and return its hapticity and the haptic type
        has_hapticity, group_hapttype = check_hapticity_hapttype (atoms_list, g)

        # Creates Group
        newgroup = group(g, has_hapticity, group_hapttype)
        group_list.append(newgroup)
        group_atoms_list.append([atoms_list[idx]  for idx in g])


    # Sets hapticity
    if any(g.hapticity == True for g in group_list):
        hapticity = True
        for g in group_list:
            for typ in g.hapttype:
                if typ not in hapttype:
                    hapttype.append(typ)
    else:
        hapticity = False

    return hapticity, hapttype, group_list, group_atoms_list

#######################################################
def find_groups_within_atoms_list (atoms_list: list) -> list:

    debug = 0
    labels = [ atom.label for atom in atoms_list ]
    pos = [ atom.coord for atom in atoms_list ]
    factor = 1.3
    radii = get_radii(labels)

    status, conmat, connec, mconmat, mconnec = getconec(labels, pos, factor, radii)

    if debug >= 2 :
        print(f"{status=}")
        print(f"{conmat=}")
        print(f"{connec=}")
        print(f"{mconmat=}")
        print(f"{mconnec=}")

    connected_atoms = []
    unconnected_atoms = []
    cutmolec = []

    for idx, a in enumerate(atoms_list):
        if a.mconnec >= 1:
            connected_atoms.append(idx)
            cutmolec.append([a.label, idx])
        elif a.mconnec == 0:
            unconnected_atoms.append(idx)

    rowless = np.delete(conmat, unconnected_atoms, 0)
    columnless = np.delete(rowless, unconnected_atoms, axis=1)

    if debug >= 1:
        print(f"FIND GROUPS: connected are: {connected_atoms}")
    if debug >= 1:
        print(f"FIND GROUPS: unconnected are: {unconnected_atoms}")

    # Regenerates the truncated lig.connec
    connec_list = []
    for idx, c in enumerate(connec):
        if idx in connected_atoms:
            connec_list.append(connec[idx])

    # # Does the
    degree = np.diag(connec_list)
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
        final_list = []
        for i in range(0, len(atomlistperm)):
            if atomlistperm[i] == b + 1:
                final_list.append(cutmolec[i][1])
        groups.append(final_list)

    if debug >= 1:
        print(f"FIND GROUPS finds {ngroups} as {groups}")

    return groups

#######################################################
def shape_measure (positions: list, symbols: list, debug: int=0) -> dict:
    # Get shape measure of a set of coordinates
    
    shape_structure_references_simplified = {'2 Vertices': [['L-2', 1, 'Dinfh', 'Linear'],
                                            ['vT-2', 2, 'C2v', 'Bent (V-shape, 109.47°)'],
                                            ['vOC-2', 3, 'C2v', 'Bent (L-shape, 90°)']],

                            '3 Vertices': [['TP-3', 1, 'D3h', 'Trigonal planar'],
                                            ['mvOC-3', 4, 'C2v', 'T-shaped']],

                            '4 Vertices': [['T-4', 2, 'Td', 'Tetrahedral'],
                                            ['SP-4', 1, 'D4h', 'Square planar'],
                                            ['SS-4', 3, 'C2v', 'Seesaw']],

                            '5 Vertices': [['PP-5', 1, 'D5h', 'Pentagon'],
                                            ['TBPY-5', 3, 'D3h', 'Trigonal bipyramidal'],
                                            ['SPY-5', 4, 'C4v', 'Square pyramidal']],

                            '6 Vertices': [['HP-6', 1, 'D6h', 'Hexagon'],
                                            ['PPY-6', 2, 'C5v', 'Pentagonal pyramidal'],
                                            ['OC-6', 3, 'Oh', 'Octahedral'],
                                            ['TPR-6', 4, 'D3h', 'Trigonal prismatic']],

                            '7 Vertices': [['HP-7', 1, 'D7h', 'Heptagon'],
                                            ['HPY-7', 2, 'C6v', 'Hexagonal pyramidal'],
                                            ['PBPY-7', 3, 'D5h', 'Pentagonal bipyramidal'],
                                            ['CTPR-7', 5, 'C2v', 'Capped trigonal prismatic']],

                            '8 Vertices': [['OP-8', 1, 'D8h', 'Octagon'],
                                            ['HPY-8', 2, 'C7v', 'Heptagonal pyramidal'],
                                            ['HBPY-8', 3, 'D6h', 'Hexagonal bipyramidal'],
                                            ['CU-8', 4, 'Oh', 'Cube'],
                                            ['SAPR-8', 5, 'D4d', 'Square antiprismatic'],
                                            ['TDD-8', 6, 'D2d', 'Dodecahedral']],

                            '9 Vertices': [['EP-9', 1, 'D9h', 'Enneagon'],
                                            ['OPY-9', 2, 'C8v', 'Octagonal pyramid'],
                                            ['HBPY-9', 3, 'D7h', 'Heptagonal bipyramid'],
                                            ['JTC-9', 4, 'C3v', 'Johnson triangular cupola J3'],
                                            ['JCCU-9', 5, 'C4v', 'Capped cube J8'],
                                            ['CCU-9', 6, 'C4v', 'Spherical-relaxed capped cube'],
                                            ['JCSAPR-9', 7, 'C4v', 'Capped square antiprism J10'],
                                            ['CSAPR-9', 8, 'C4v', 'Spherical capped square antiprism'],
                                            ['JTCTPR-9', 9, 'D3h', 'Tricapped trigonal prism J51'],
                                            ['TCTPR-9', 10, 'D3h', 'Spherical tricapped trigonal prism'],
                                            ['JTDIC-9', 11, 'C3v', 'Tridiminished icosahedron J63'],
                                            ['HH-9', 12, 'C2v', 'Hula-hoop'],
                                            ['MFF-9', 13, 'Cs', 'Muffin']],

                            '10 Vertices': [['DP-10', 1, 'D10h', 'Decagon'],
                                            ['EPY-10', 2, 'C9v', 'Enneagonal pyramid'],
                                            ['OBPY-10', 3, 'D8h', 'Octagonal bipyramid'],
                                            ['PPR-10', 4, 'D5h', 'Pentagonal prism'],
                                            ['PAPR-10', 5, 'D5d', 'Pentagonal antiprism'],
                                            ['JBCCU-10', 6, 'D4h', 'Bicapped cube J15'],
                                            ['JBCSAPR-10', 7, 'D4d', 'Bicapped square antiprism J17'],
                                            ['JMBIC-10', 8, 'C2v', 'Metabidiminished icosahedron J62'],
                                            ['JATDI-10', 9, 'C3v', 'Augmented tridiminished icosahedron J64'],
                                            ['JSPC-10', 10, 'C2v', 'Sphenocorona J87'],
                                            ['SDD-10', 11, 'D2', 'Staggered Dodecahedron (2:6:2)'],
                                            ['TD-10', 12, 'C2v', 'Tetradecahedron (2:6:2)'],
                                            ['HD-10', 13, 'D4h', 'Hexadecahedron (2:6:2) or (1:4:4:1)']],
                            '11 Vertices': [['HP-11', 1, 'D11h', 'Hendecagon'],
                                            ['DPY-11', 2, 'C10v', 'Decagonal pyramid'],
                                            ['EBPY-11', 3, 'D9h', 'Enneagonal bipyramid'],
                                            ['JCPPR-11', 4, 'C5v', 'Capped pentagonal prism J9'],
                                            ['JCPAPR-11', 5, 'C5v', 'Capped pentagonal antiprism J11'],
                                            ['JAPPR-11', 6, 'C2v', 'Augmented pentagonal prism J52'],
                                            ['JASPC-11', 7, 'Cs', 'Augmented sphenocorona J87']],
                            '12 Vertices': [['DP-12', 1, 'D12h', 'Dodecagon'],
                                            ['HPY-12', 2, 'C11v', 'Hendecagonal pyramid'],
                                            ['DBPY-12', 3, 'D10h', 'Decagonal bipyramid'],
                                            ['HPR-12', 4, 'D6h', 'Hexagonal prism'],
                                            ['HAPR-12', 5, 'D6d', 'Hexagonal antiprism'],
                                            ['TT-12', 6, 'Td', 'Truncated tetrahedron'],
                                            ['COC-12', 7, 'Oh', 'Cuboctahedron'],
                                            ['ACOC-12', 8, 'D3h', 'Anticuboctahedron J27'],
                                            ['IC-12', 9, 'Ih', 'Icosahedron'],
                                            ['JSC-12', 10, 'C4v', 'Johnson square cupola J4'],
                                            ['JEPBPY-12', 11, 'D6h', 'Johnson elongated pentagonal bipyramid J16'],
                                            ['JBAPPR-12', 12, 'C2v', 'Biaugmented pentagonal prism J53'],
                                            ['JSPMC-12', 13, 'Cs', 'Sphenomegacorona J88']],
                            '20 Vertices': [['DD-20', 1, 'Ih', 'Dodecahedron']],
                            '24 Vertices': [['TCU-24', 1, 'Oh', 'Truncated cube'],
                                            ['TOC-24', 2, 'Oh', 'Truncated octahedron']],
                            '48 Vertices': [['TCOC-48', 1, 'Oh', 'Truncated cuboctahedron']],
                            '60 Vertices': [['TRIC-60', 1, 'Ih', 'Truncated icosahedron (fullerene)']]}

    cn = len(symbols) - 1 # coordination number of metal center
    connectivity= [[1, i] for i in range(2, cn+2)]
    
    geometry = Geometry(positions=positions, 
                        symbols=symbols, 
                        connectivity=connectivity)            
    
    # ref_geom = np.array(shape_structure_references['{} Vertices'.format(cn)])
    ref_geom = np.array(shape_structure_references_simplified['{} Vertices'.format(cn)])
    
    posgeom_dev={}
    if debug >= 2 :
        for p, s in zip(symbols, positions):
            print (p, s)
        print("")
    
    for idx, rg in enumerate(ref_geom[:,0]):
        shp_measure = geometry.get_shape_measure(rg, central_atom=1)
        geom = ref_geom[:,3][idx]
        posgeom_dev[geom]=round(shp_measure, 3)      
    
    return posgeom_dev


################################
def get_thres_from_two_atoms(label_i, label_j, factor=1.3, debug=0): 
   
    radii_i = elemdatabase.CovalentRadius2[label_i]
    radii_j = elemdatabase.CovalentRadius2[label_j]

    if (
            elemdatabase.elementblock[label_i] != "d"
            and elemdatabase.elementblock[label_i] != "f"
            and elemdatabase.elementblock[label_j] != "d"
            and elemdatabase.elementblock[label_j] != "f"
        ):

            thres = (radii_i + radii_j) * factor
            thres = round(thres, 3)

            if debug >=2 :
                print(f"{label_i} : {radii_i}, {label_j} : {radii_j}, {factor}, {thres=}")
    elif (
            elemdatabase.elementblock[label_i] == "d"
            and elemdatabase.elementblock[label_i] == "f"
            and elemdatabase.elementblock[label_j] == "d"
            and elemdatabase.elementblock[label_j] == "f"
        ):
            factor_i = covalent_factor_for_metal[label_i]
            factor_j = covalent_factor_for_metal[label_j]

            if factor_i < factor_j  :
                new_factor = factor_i
            
            elif factor_i == factor_j :
                new_factor = factor_i

            else : 
                new_factor = factor_j

            thres = (radii_i + radii_j) * new_factor
            thres = round(thres, 3)

            if debug >=2 :
                print(f"{label_i} : {radii_i} ({factor_i}), {label_j} : {radii_j} ({factor_j}), {new_factor=}, {thres=}")
    else :

        thres = (radii_i + radii_j) * factor
        thres = round(thres, 3)

        if debug >=2 :
            print(f"{label_i} : {radii_i}, {label_j} : {radii_j}, {factor}, {thres=}")   
    
    return thres



def coordination_correction_coordination_complexes (lig, g, metalist, remove, debug=1) -> list:
    
    for g_idx in g.atlist:
        atom = lig.atoms[g_idx] 
        tgt, apos, dist = find_closest_metal(atom, metalist)
        metal = metalist[tgt]
        thres = get_thres_from_two_atoms(metal.label, atom.label, debug=debug)
    
        if debug >= 1 : 
            print(f"\tAtom {g_idx} :", atom.label, f"\tMetal {tgt} :", metal.label, "\tdistance :", round(dist, 3), "\tthres :", thres)

        neighbors_totmconnec = []  
        for j in atom.adjacency:
            neighbor = lig.atoms[j]
            neighbors_totmconnec.append(neighbor.mconnec)
            if debug >= 2 : print(f"\t\t{atom.label} connected to {neighbor.label}") #{neighbor.adjacency} {neighbor.mconnec} {neighbor.coord}")      

        if sum(neighbors_totmconnec) >= 2 :
            if debug >= 2 : print(f"\t[Check] This metal-coordinating atom {atom.label} connected to more than one metal-coordinating atoms")               
            nb_idx_list = [index for index, value in enumerate(neighbors_totmconnec) if value >= 1]
            nb_label = []
            for nb_idx in nb_idx_list :
                nb = lig.atoms[atom.adjacency[nb_idx]]
                nb_label.append(nb.label)
                if debug >= 1 : print(f"\tThis metal-coordinating atom {atom.label} connected to other metal-coordinating atom {nb.label}") 
            if set(nb_label) == set(["H"]) :
                pass
            else :
                remove.append(g_idx)
                if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", g_idx, atom.label , get_dist(atom.coord, metal.coord), "due to neighboring atoms")
        
        elif sum(neighbors_totmconnec) == 1 : # e.g. "S" atom in refcode YOBCUO, PORNOC                                         
            nb_idx = [index for index, value in enumerate(neighbors_totmconnec) if value == 1][0]
            nb = lig.atoms[atom.adjacency[nb_idx]]      
            if debug >= 2 : print(f"\t[Check] This metal-coordinating atom {atom.label} connected to another metal-coordinating atom {nb.label}") 

            if (atom.label == "H" and nb.label in ["B", "O", "N", "C"]) :
                if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", g_idx, atom.label , get_dist(atom.coord, metal.coord), "due to H")
                remove.append(g_idx)
            elif (atom.label in ["B", "O", "N", "C"] and nb.label == "H") :
                if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", atom.adjacency[nb_idx], nb.label, get_dist(nb.coord, metal.coord), "due to H")
                remove.append(atom.adjacency[nb_idx])
            else : # Check angle between metal-coordinating atoms               
                tgt, apos, dist = find_closest_metal(atom, metalist)
                metal = metalist[tgt]
                vector1 = np.subtract(np.array(atom.coord), np.array(nb.coord))
                vector2 = np.subtract(np.array(atom.coord), np.array(metal.coord))                        
                angle = np.degrees(getangle(vector1, vector2))
                
                if angle < 55 :
                    if debug >= 1 : print("\t!!! Wrong metal-coordination assignment for Atom", g_idx, atom.label, get_dist(atom.coord, metal.coord), "due to the angle", round(angle,2))
                    remove.append(g_idx)
                else :
                    if debug >= 2 : print(f"\t{metal.label}-{atom.label}-{nb.label} angle {round(angle,2)}")
                    pass                     

        elif round(dist/thres, 3) > 0.95 :
            if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", g_idx, atom.label , get_dist(atom.coord, metal.coord), "due to the long distance")
            remove.append(g_idx)
        else :
            pass
    
    remove = list(set(remove))

    return remove

############################################
def coordination_correction_for_haptic (lig, g, metalist, remove, debug=1) -> list:


    
    ratio_list = []
    for g_idx in g.atlist:
        atom = lig.atoms[g_idx] 
        tgt, apos, dist = find_closest_metal(atom, metalist)
        metal = metalist[tgt]
        thres = get_thres_from_two_atoms(metal.label, atom.label, debug=debug)
        ratio_list.append(round(dist/thres,3))
        if debug >= 1 : 
            print(f"\tAtom {g_idx} :", atom.label, f"\tMetal {tgt} :", metal.label, "\tdistance :", round(dist, 3), "\tthres :", thres)
    std_dev = round(np.std(ratio_list), 3)

    for g_idx, ratio in zip(g.atlist, ratio_list) :
        atom = lig.atoms[g_idx] 
        tgt, apos, dist = find_closest_metal(atom, metalist)
        metal = metalist[tgt]
        if atom.label == "H" : 
            if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", g_idx, atom.label , get_dist(atom.coord, metal.coord), "due to H")
            remove.append(g_idx)            
        elif std_dev > 0.05 and ratio > 0.9 :
            if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", g_idx, atom.label , get_dist(atom.coord, metal.coord), "due to the long distance")
            remove.append(g_idx)
        else:
            pass

    remove = list(set(remove))

    return remove

#######################################################
def reset_adjacencies_lig_metalist (lig: object, metalist: list, i: int, debug: int=0) -> object :    

    lig.mconnec[i] = 0
    lig.adjacencies(lig.conmat, lig.mconnec)
    
    atom = lig.atoms[i] 
    tgt, apos, dist = find_closest_metal(atom, metalist)
    metal = metalist[tgt]
    metal.adjacencies(np.array([x - 1 for x in metal.mconnec]))

    wrong = [index for index, value in enumerate(metal.coordinating_atoms_sites) if value == lig.coord[i]][0]

    if debug >= 2 : 
        print("!!! Wrong : ", i, metal.coordinating_atoms[wrong], "distance", get_dist(metal.coordinating_atoms_sites[wrong], metal.coord)) #metal.coordinating_atoms_sites[wrong])
           
    del metal.coordinating_atoms[wrong]
    del metal.coordinating_atoms_sites[wrong]

    return lig, metalist


def reset_adjacencies_lig_metalist_v2 (lig: object, remove: list, debug: int=0) -> object :    
    
    for i in remove :
        lig.mconnec[i] = 0
        lig.adjacencies(lig.conmat, lig.mconnec)
        
        atom = lig.atoms[i] 
        tgt, apos, dist = find_closest_metal(atom, lig.parent.metals)
        metal = lig.parent.metals[tgt]
        metal.adjacencies(np.array([x - 1 for x in metal.mconnec]))

        wrong = [index for index, value in enumerate(metal.coordinating_atoms_sites) 
                 if value == lig.coord[i]][0]

        if debug >= 2 : 
            print("!!! Wrong : ", i, metal.coordinating_atoms[wrong], 
                  "distance", get_dist(metal.coordinating_atoms_sites[wrong], metal.coord)) #metal.coordinating_atoms_sites[wrong])
            
        del metal.coordinating_atoms[wrong]
        del metal.coordinating_atoms_sites[wrong]

    return lig