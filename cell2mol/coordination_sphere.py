import numpy as np
from cosymlib import Geometry
from cell2mol.other import *
from cell2mol.connectivity import add_atom
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

#######################################################
###     Define coordination geometry from groups    ### 
#######################################################
def get_coordination_geometry (coord_group: list, debug: int=0) -> object:
    
    symbols = []
    positions = []
    coord_haptic_type = []
    for group in coord_group:
        if group.hapticity == False:
            for atom in group.atoms:
                symbols.append(atom.label)
                positions.append(atom.coord)
                if debug >= 2 : print(atom.label, atom.coord)
        else :
            haptic_center_coord = compute_centroid([atom.coord for atom in group.atoms])
            symbols.append(str(group.hapttype))
            positions.append(haptic_center_coord)      
            if debug >= 2 : print(f"mid point of {group.haptic_type=}", haptic_center_coord)      
            coord_haptic_type.append(group.haptic_type)             
    
    posgeom_dev = shape_measure(symbols, positions, debug=debug)

    if len(posgeom_dev) > 0:
        coordination_geometry=min(posgeom_dev, key=posgeom_dev.get)
        geom_deviation=min(posgeom_dev.values())
    else :
        coordination_geometry = "Undefined"
        geom_deviation = "Undefined"

    if debug >= 1 :
        print(f"The number of coordinating points (including the mid point of haptic ligands) : {len(coord_group)}")
        print (f"{posgeom_dev}")
        print(f"The most likely geometry : '{coordination_geometry}' with deviation value {geom_deviation}")
        print(f"The type of hapticity : {coord_haptic_type}")
        print("")

    return coordination_geometry
    # return coordination_geometry, geom_deviation, coord_haptic_type

#######################################################
def shape_measure (symbols: list, positions: list, debug: int=0) -> dict:
    # Get shape measure of a set of coordinates

    cn = len(symbols) # coordination number of metal center
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

#######################################################
shape_structure_references_simplified = {'2 Vertices': [['L-2', 1, 'Dinfh', 'Linear'],
                                        ['vT-2', 2, 'C2v', 'Bent (V-shape, 109.47°)'],
                                        ['vOC-2', 3, 'C2v', 'Bent (L-shape, 90°)']],

                        '3 Vertices': [['TP-3', 1, 'D3h', 'Trigonal planar'],
                                       ['fvOC-3', 3, 'C3v', 'fac-Trivacant octahedron'],
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


#######################################################
###    Make corrcetion for coordination sphere      ###
#######################################################
# TODO: will be done in the future
#######################################################
covalent_factor_for_metal_v2 = {
    'H': 1.19,
    'D': 1.19,
    'He': 1.68,
    'Li': 0.99,
    'Be': 0.95,
    'B': 1.1,
    'C': 1.22,
    'N': 1.19,
    'O': 1.21,
    'F': 1.14,
    'Ne': 1.56,
    'Na': 0.88,
    'Mg': 1.04,
    'Al': 1.32,
    'Si': 1.06,
    'P': 1.14,
    'S': 1.19,
    'Cl': 1.19,
    'Ar': 1.5,
    'K': 0.79,
    'Ca': 0.97,
    'Sc': 1.3,
    'Ti': 1.3,
    'V': 1.3,
    'Cr': 1.3,
    'Mn': 1.3,
    'Fe': 1.3,
    'Co': 1.3,
    'Ni': 1.3,
    'Cu': 1.3,
    'Zn': 1.3,
    'Ga': 1.26,
    'Ge': 1.24,
    'As': 1.12,
    'Se': 1.12,
    'Br': 1.21,
    'Kr': 1.6,
    'Rb': 1.02,
    'Sr': 0.97,
    'Y': 1.3,
    'Zr': 1.3,
    'Nb': 1.3,
    'Mo': 1.3,
    'Tc': 1.3,
    'Ru': 1.3,
    'Rh': 1.3,
    'Pd': 1.3,
    'Ag': 1.3,
    'Cd': 1.3,
    'In': 1.35,
    'Sn': 1.29,
    'Sb': 1.04,
    'Te': 1.06,
    'I': 1.21,
    'Xe': 1.51,
    'Cs': 1.03,
    'Ba': 0.99,
    'La': 1.3,
    'Ce': 1.3,
    'Pr': 1.3,
    'Nd': 1.3,
    'Pm': 1.3,
    'Sm': 1.3,
    'Eu': 1.3,
    'Gd': 1.3,
    'Tb': 1.3,
    'Dy': 1.3,
    'Ho': 1.3,
    'Er': 1.3,
    'Tm': 1.3,
    'Yb': 1.3,
    'Lu': 1.3,
    'Hf': 1.3,
    'Ta': 1.3,
    'W': 1.3,
    'Re': 1.3,
    'Os': 1.3,
    'Ir': 1.3,
    'Pt': 1.3,
    'Au': 1.3,
    'Hg': 1.3,
    'Tl': 1.3,
    'Pb': 1.29,
    'Bi': 1.28,
    'Po': 1.38,
    'At': 1.34,
    'Rn': 1.63,
    'Fr': 1.09,
    'Ra': 1.16,
    'Ac': 1.3,
    'Th': 1.3,
    'Pa': 1.3,
    'U': 1.3,
    'Np': 1.3,
    'Pu': 1.3,
    'Am': 1.3,
    'Cm': 1.3,
    'Bk': 1.3,
    'Cf': 1.3,
    'Es': 1.3,
    'Fm': 1.3,
    'Md': 1.3,
    'No': 1.3,
    'Lr': 1.3,
    'Rf': 1.3,
    'Db': 1.3,
    'Sg': 1.3,
    'Bh': 1.3,
    'Hs': 1.3,
    'Mt': 1.3  
}
#######################################################
def get_thres_from_two_atoms(label_i, label_j, factor=1.3, debug=0): 
   
    radii_i = elemdatabase.CovalentRadius3[label_i]
    radii_j = elemdatabase.CovalentRadius3[label_j]

    if (
            elemdatabase.elementblock[label_i] == "d"
            or elemdatabase.elementblock[label_i] == "f"
            or elemdatabase.elementblock[label_j] == "d"
            or elemdatabase.elementblock[label_j] == "f"
        ):
            factor_i = covalent_factor_for_metal_v2 [label_i]
            factor_j = covalent_factor_for_metal_v2 [label_j]

            if factor_i < factor_j  :   new_factor = factor_i
            elif factor_i == factor_j : new_factor = factor_i
            else :                      new_factor = factor_j

            thres = round( (radii_i + radii_j) * new_factor, 3)
            if debug >=2 :  print(f"{label_i} : {radii_i} ({factor_i}), {label_j} : {radii_j} ({factor_j}), {new_factor=}, {thres=}")
    else :
        thres = round( (radii_i + radii_j) * factor , 3)
        if debug >=2 :  print(f"{label_i} : {radii_i}, {label_j} : {radii_j}, {factor}, {thres=}")   
    
    return thres

#######################################################    
def coordination_correction_for_nonhaptic (group, debug=1) -> list:

    ## First Correction (former verify_connectivity)
    for idx, atom in enumerate(group.atoms):
        if debug > 0: print(f"GROUP.check_denticity: connectivity={a.mconnec} in atom idx={idx}, label={atom.label}")
        isadded, newlab, newcoord = add_atom(group.parent.labels, group.parent.coord, group.parent_indices[idx], group.parent, group.parent.parent.metals, "H", debug=debug)
        if isadded:
            if debug > 0: print(f"GROUP.check_denticity: connectivity verified for atom {idx} with label {atom.label}")
        else:
            if debug > 0: print(f"GROUP.check_denticity: corrected mconnec of atom {idx} with label {atom.label}")
            atom.reset_mconnec(idx)

    ## Second Correction
    for idx, atom in enumerate(group.atoms):
        metal = atom.get_closest_metal()
        thres = get_thres_from_two_atoms(metal.label, atom.label, debug=debug)
        if debug >= 1 : print(f"\tAtom {atom.label} connected to {metal.label} distance {get_dist(atom.coord, metal.coord)} with threshold {thres}")
        
        neighbors = [ atom.parent.atoms[j] for j in atom.adjacency ]
        nb_dist_from_metal = [ get_dist(nb.coord, metal.coord) for nb in neighbors]
        neighbors_mconnec =[]

        for nb, dist in zip(neighbors, nb_dist_from_metal) :
            thres = get_thres_from_two_atoms(metal.label, nb.label, debug=debug)
            if dist > thres :   pass
            else :              neighbors_mconnec.append(nb)   
        
        if debug >= 2 : 
            print(f"\t\t{atom.label} connected to {[nb.label for nb in neighbors]}")
            print(f"\t\tAmong these neighbors, {[nb_m for nb_m in neighbors_mconnec]} are connected to the metal {metal.label}")

        if len(neighbors_mconnec) >= 2 :
            if debug >=1 : print(f"\t[Check] This coordinating atom {atom.label} connected to more than one coordinating atoms to the metal {metal.label}")
            if set([nb_m.label for nb_m in neighbors_mconnec]) == set(["H"]) :  pass  # TODO : Figure out why I put this condition
            else :  
                atom.reset_mconnec(idx)
                if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label, get_dist(atom.coord, metal.coord), "due to neighboring atoms")

        elif len(neighbors_mconnec) == 1 :
            nb_m = neighbors_mconnec[0]
            if debug >=1 : print(f"\t[Check] This coordinating atom {atom.label} connected to another coordinating atom {nb_m.label} to the metal {metal.label}")

            if (atom.label == "H" and nb_m.label in ["B", "O", "N", "C"]) :
                atom.reset_mconnec(idx)
                if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label, get_dist(atom.coord, metal.coord), "due to H")
            elif (atom.label in ["B", "O", "N", "C"] and nb_m.label == "H") :
                if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", nb_m.label, get_dist(nb_m.coord, metal.coord), "due to H")
                nb_m.reset_mconnec() # put an index of nb_m in group
            else : # Check angle between metal-coordinating atoms               
                vector1 = np.subtract(np.array(atom.coord), np.array(nb_m.coord))
                vector2 = np.subtract(np.array(atom.coord), np.array(metal.coord))                        
                angle = np.degrees(get_angle(vector1, vector2))
                if angle < 55 :
                    if debug >= 1 : print("\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label, get_dist(atom.coord, metal.coord), "due to the angle", round(angle,2))
                    atom.reset_mconnec(idx)

        elif round(dist/thres, 3) > 0.95 :
            if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label , get_dist(atom.coord, metal.coord), "due to the long distance")
            atom.reset_mconnec(idx)
        else :
            if debug >=1 :print(f"\tThere is no neighbor atom connected to the metal {metal.label}")
            pass
    
    return group 

#######################################################    
def coordination_correction_for_haptic (group, debug=1) -> list:

    ratio_list = []
    for idx, atom in enumerate(group.atoms):
        metal = atom.get_closest_metal()
        dist = get_dist(atom.coord, metal.coord)
        thres = get_thres_from_two_atoms(metal.label, atom.label, debug=debug)
        ratio_list.append(round(dist/thres,3))
        if debug >= 1 : print(f"\tAtom {idx} :", atom.label, f"\tMetal :", metal.label, "\tdistance :", round(dist, 3), "\tthres :", thres)

    std_dev = round(np.std(ratio_list), 3)
    if debug >= 1 : print(f"{ratio_list=} {std_dev=}")

    count = 0
    for idx, (atom, ratio) in enumerate(zip(group.atoms, ratio_list)) :
        if atom.label == "H" : 
            if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label , get_dist(atom.coord, metal.coord), "due to H")
            atom.reset_mconnec(idx)
            count += 1          
        elif std_dev > 0.05 and ratio > 0.9 :
            if debug >=1 : print("\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label , get_dist(atom.coord, metal.coord), "due to the long distance")
            atom.reset_mconnec(idx) 
            count += 1      
        else :
            pass

    # get group hapticity if there are any changes
    if count == 0 : return group
    else :
        group.get_hapticity()    
        return group