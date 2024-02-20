#!/usr/bin/env python

import numpy as np
from cell2mol.elementdata import ElementData
from cosymlib.shape.tools import shape_structure_references
from cell2mol.tmcharge_common import getelementcount
import ast #ast.literal_eval



elemdatabase = ElementData()

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
################################
def count_d_elec (met_label, m_ox):
    """ Count d electrons for a given transition metal and metal oxidation state
    Args:
        met_label (str): metal symbol
        m_ox (int): metal oxidation state
    Returns:
        d_elec (int): number of d electrons
    """

    v_elec = elemdatabase.valenceelectrons[met_label]
    group = elemdatabase.elementgroup[met_label]

    if v_elec - m_ox >= 0 :
        d_elec = v_elec - m_ox
    elif v_elec - m_ox < 0 :
        d_elec = group - m_ox

    return d_elec

################################
def decide_spin_multiplicity (spin, N):
    """ Decide spin multiplicity for a given spin state and number of electrons
    Args:
        spin (str): spin state
        N (int): number of total electrons
    Returns:
        smul (int): spin multiplicity
    """    
    if N%2 == 0:
        if spin == "LS":
            smul = 1
        elif spin == "IS":
            smul = 3
        elif spin == "HS" :
            smul = 5
        else :
            smul = 1
    else:
        if spin == "LS":
            smul = 2
        elif spin == "IS":
            smul = 4
        elif spin == "HS" :
            smul = 6
        else :
            smul = 2

    return smul

################################
def counts_element (temp) : 
    """ Count number of each element in a given list
    Args:
        temp (list): list of elements
    Returns:
        arr (np.ndarray): array of element counts
    """    
    arr = np.zeros(112)
    coord_sphere = list(ast.literal_eval(temp))
    
    elem_list, count_list = np.unique(coord_sphere, return_counts=True)
    
    for elem, count in zip(elem_list, count_list):
        elem_nr = elemdatabase.elementnr[elem]
#         print(elem, elem_nr, count)
        arr[elem_nr] = count
    return arr

################################
def count_nitrosyl (arr):
    """ Count number of nitrosyl ligands in list of ligands
    Args:
        arr (list): list of ligands
    Returns:
        count_NO (int): number of nitrosyl ligands
    """    
    count_NO = 0 

    for lig in arr :
        if isinstance(lig, str) :
            if sorted(list(ast.literal_eval(lig))) == ['N', 'O'] : # Nitrosyl ligand
                count_NO += 1
        elif isinstance(lig, list) or isinstance(lig, np.ndarray):
            if sorted(lig) == ['N', 'O'] : # Nitrosyl ligand
                count_NO += 1
        else :
            raise ValueError("lig is neither string, list nor np.ndarray type")

    return count_NO

################################
def count_N (mol):
    """ Count number of electrons in a given molecule
    Args:
        mol (obj): molecule object
    Returns:
        N (int): number of total electrons
    """
    N = 0
    for atom in mol.labels:
        N += elemdatabase.elementnr[atom]
    N -= mol.totcharge
    
    return N

################################
def make_geom_list ():

    geom_list = {}
    count = 0
    for i in shape_structure_references_simplified.values():
    #     print(np.array(i)[:,3])
        for geom in np.array(i)[:,3]:
            geom_list[geom] = count
            count +=1
    return geom_list

################################
def get_dist (atom1_pos: list, atom2_pos: list) -> float :
    dist = np.linalg.norm(np.array(atom1_pos) - np.array(atom2_pos))
    return round(dist, 3)

################################
def get_centroid(arr: np.array) -> list:
    # Get centroid of a set of coordinates

    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    centroid = np.around(np.array([sum_x/length, sum_y/length, sum_z/length]),7)
    centroid = list(centroid)

    return centroid

################################
def calcualte_relative_metal_radius_haptic_complexes (metal, debug=0):
    
    diff_list_g = []
    diff_list_c = []
    
    for g, g_atoms in zip(metal.group_list, metal.group_atoms_list):
        if g.hapticity == False :
            for a in g_atoms:
                diff = round(get_dist(metal.coord, a.coord) - elemdatabase.CovalentRadius3[a.label], 3)
                #print(a.label, a.coord, round(get_dist(metal.coord, a.coord),3), elemdatabase.CovalentRadius3[a.label], round(diff,3))
                diff_list_g.append(diff)         
                diff_list_c.append(diff)                                         
        else : # for haptic group
            print(g.hapttype, [a.label for a in g_atoms])
            arr=[a.coord for a in g_atoms]
            if len(arr) > 1 : arr = np.array(arr)   
            haptic_center_label="C"
            haptic_center_coord = get_centroid(arr)
            diff_c = round(get_dist(metal.coord, haptic_center_coord) - elemdatabase.CovalentRadius3[haptic_center_label], 3)
            print(haptic_center_label, haptic_center_coord, round(get_dist(metal.coord, haptic_center_coord),3), elemdatabase.CovalentRadius3[haptic_center_label], round(diff_c,3))
            diff_list_c.append(diff_c)

            sub = [round(get_dist(metal.coord, a.coord) - elemdatabase.CovalentRadius3[a.label], 3) for a in g_atoms]
            sub_average = round(np.average(sub), 3)
            diff_list_g.append(sub_average)      

    average_g = round(np.average(diff_list_g), 3)    
    rel_g = round(average_g/elemdatabase.CovalentRadius3[metal.label], 3)

    average_c = round(np.average(diff_list_c), 3)
    rel_c = round(average_c/elemdatabase.CovalentRadius3[metal.label], 3)

    if debug >=2 : print(f"{len(metal.group_list)=}, {diff_list_g}, {average_g=}, {rel_g=}, {metal.label}, {elemdatabase.CovalentRadius3[metal.label]}")
    if debug >=2 : print(f"{len(metal.group_list)=}, {diff_list_c}, {average_c=}, {rel_c=}, {metal.label}, {elemdatabase.CovalentRadius3[metal.label]}")

    return rel_g, rel_c

################################
def calcualte_relative_metal_radius (metal, debug=0):
    """ Calculate relative metal radius for a given transition metal coordination complex
    Args:
        metal (obj): metal atom object
    Returns:
        rel (float): relative metal radius
    """

    diff_list = []
    for a_label, a_coord in zip(metal.coordinating_atoms, metal.coordinating_atoms_sites): 
        diff = round(get_dist(metal.coord, a_coord) - elemdatabase.CovalentRadius3[a_label], 3)
        #print(a_label, a_coord, round(get_dist(metal.coord, a_coord),3), elemdatabase.CovalentRadius3[a_label], round(diff,3))
        diff_list.append(diff)
    
    average = round(np.average(diff_list), 3)    
    rel = round(average/elemdatabase.CovalentRadius3[metal.label], 3)
    
    if debug >=2 : print(f"{metal.coordinating_atoms}, {diff_list}, {average=}, {rel=}, {metal.label}, {elemdatabase.CovalentRadius3[metal.label]}")
    
    return rel

################################
def generate_feature_vector (metal):
    """ Generate feature vector for a given transition metal coordination complex
    Args:
        metal (obj): metal atom object
    Returns:
        feature (np.ndarray): feature vector
    """
    elem_nr = elemdatabase.elementnr[metal.label]
    m_ox = metal.totcharge
    d_elec = count_d_elec (metal.label, m_ox)

    coord_sphere = metal.coordination_sphere
    CN = len(coord_sphere)
    CN = metal.coordination_number
    geom_nr = make_geom_list()[metal.geometry]

    if metal.hapticity == False :
        rel = calcualte_relative_metal_radius (metal)
        hapticity = 0
    else :
        dummy, rel = calcualte_relative_metal_radius_haptic_complexes (metal)
        hapticity = 1

    print(f"{elem_nr=} {m_ox=} {d_elec=} {rel=} {hapticity=}\n")

    feature = np.array([[elem_nr, m_ox, d_elec, CN, geom_nr, rel, hapticity]])
    
    return feature

################################
def get_posspin_v2 (d_elec: int, m_ox: int, geometry: str, metal: str) -> list:
    """ Get possible spin states for a given transition metal coordination complex
    Args:
        d_elec (int): number of d electrons
        m_ox (int): metal oxidation state
        geometry (str): coordination geometry of the complex
        metal (str): metal symbol
    Returns:
        posspin (list): list of possible spin states
    """
    if d_elec in [0, 1, 9, 10]:
        posspin = ["LS"]
    elif d_elec in [2, 3]:
        posspin = ["IS"]
    else:
        if m_ox == 0 :
            posspin = ["LS"]
        else :
            if d_elec in [4, 5, 6]:
                if d_elec == 4 :
                    if geometry == "Octahedral":
                        posspin = ["IS", "HS"]
                    else :
                        posspin = ["HS"]
                else : 
                    if geometry == "Square planar" or geometry == "Trigonal bipyramidal":
                        posspin = ["IS", "HS"]                   
                    elif metal =="Co" and m_ox == 3 and geometry == "Octahedral":
                        posspin = ["LS"]
                    else :
                        posspin = ["LS", "IS", "HS"]
            elif d_elec in [7, 8] :
                if geometry == "Square planar": 
                    posspin = ["LS"]
                elif metal == "Ni" and m_ox == 3:
                    posspin = ["LS"]
                else :
                    posspin = ["LS", "IS"]
    return posspin

################################
def length_threshold_based_on_CN_geometry_average (CN, geometry, metal) :
    """ Calculate length threshold based on coordination number and geometry and metal type
    Args:
        CN (int): coordination number
        geometry (str): coordination geometry of the complex
        metal (str): metal symbol
    Returns:
        length_thres (float): length threshold
    """

    if CN < 4 :
        length_thres = 0.85
    elif CN == 4 :
        if geometry == "Square planar" :
            length_thres = 0.95
        elif geometry == "Tetrahedral" :
            length_thres = 0.85
        else :
            length_thres = 0.90
    elif CN == 5 :
        length_thres = 0.95
    elif CN >= 6 :
        length_thres = 1.0
        
    if metal == "Cr" or metal == "Mn":
        length_thres = round(length_thres * 0.719/0.757, 2)
    elif metal == "Co":
        length_thres = round(length_thres * 0.793/0.757, 2)
    elif metal == "Ni" :
        length_thres = round(length_thres * 0.806/0.757, 2)        

    return length_thres

################################
def classify_spin_based_on_relative_length_simple (rel, length_thres, window, posspin):
    """ Classify spin state based on relative metal radius and length threshold
    Args:
        rel (float): relative metal radius
        length_thres (float): length threshold
        window (float): window size
        posspin (list): list of possible spin states
    Returns:
        spin (str): ground state spin
        ambiguous (bool): True if the assigned spin state is in an ambiguous region
    """

    if abs(rel-length_thres) < window :
        ambiguous = True
    else :
        ambiguous = False

    if rel < length_thres :
        spin = posspin[0]
    else :
        spin = posspin[-1]  
          
    return spin, ambiguous

################################
def assign_ground_state_spin_empirical (d_elec: int, m_ox: int, geometry: str, metal: str, CN: int, rel: float, N: int) :
    """ Assign ground state spin for a given transition metal coordination complex based on empirical rules
    
    Args:
        d_elec (int): number of d electrons
        m_ox (int): metal oxidation state
        geometry (str): coordination geometry of the complex
        metal (str): metal symbol
        CN (int): coordination number
        rel (float): relative metal radius
        N (int): number of total electrons
    Returns:
        spin (str): ground state spin
        rule (bool): True if the spin state is assigned based on empirical rules
    """

    posspin = get_posspin_v2 (d_elec, m_ox, geometry, metal)
    print(posspin)
    if len(posspin) == 1 :            
        spin = posspin[0]
        rule = True
        threshold = False

    elif len(posspin) > 1 :
        if CN == 2 or CN > 6 :
            spin = posspin[-1]  
            rule = True
            threshold = False            
        else : 
            rule = False
            threshold = True        
            # window = 0.05
            length_thres = length_threshold_based_on_CN_geometry_average (CN, geometry, metal)
            # spin, ambiguous = classify_spin_based_on_relative_length_simple (rel, length_thres, window, posspin)     
            if rel < length_thres :
                spin = posspin[0]
            else :
                spin = posspin[-1]          
    else :
        print("***Error*** No possible spin state", posspin)
        spin = "unknown"
        rule = False
        threshold = False  
    
    smul = decide_spin_multiplicity (spin, N)
    
    return smul, rule, threshold 

################################
def get_posspin_v1 (l, CN, posspin, preferred_spin):

    if (l.count("C") + l.count("P")) / CN > 0.5 :
        posspin = posspin[0] # lowest spin state
    elif (l.count("F") + l.count("Cl") + l.count("Br") + l.count("I") )/ CN > 0.5 :
        posspin =  posspin[-1] # highest spin state
    else :
        posspin = preferred_spin
           
    return posspin

################################
def predict_ground_state_spin_v1 (metal, elec, coord_sphere, geometry, CN, N, nitrosyl) :
    
    block = elemdatabase.elementblock[metal]
    period = elemdatabase.elementperiod[metal]
    
    if block != "d" :
        #print("Element is not transtion metal")
        spin = "unknown"

    elif block == "d" :
        if nitrosyl > 0 : # TMC contains at least one nitrosyl(NO) ligand
            spin = "LS"        
        else :   
            if period == 5 or period == 6 : # 4d or 5d TM ions
                spin = "LS"           
            elif period == 4 : # 3d TM ions
                if elec in [0, 1, 9, 10]:
                    spin = "LS"
                elif elec in [2, 3]:
                    spin = "IS"                
                else :
                    if CN < 4 or CN > 6 :
                        if elec == 4 :
                            spin = "HS"
                        elif elec == 5 :
                            spin = "HS"
                        elif elec == 6 :
                            spin = "HS"
                        elif elec == 7 :
                            spin = "IS"
                        elif elec == 8 :
                            spin = "IS"
                    else :
                        if elec == 4 :             
                            posspin = ["IS", "HS"]
                            if metal == "Cr" : # Cr(+2)
                                if geometry == "Octahedron":
                                    preferred_spin = "HS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)
                                else :
                                    spin = "HS" 

                            elif metal == "Mn" : # Mn(+3):
                                if geometry == "Octahedron":
                                    preferred_spin = "HS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)                        
                                else :
                                    spin = "HS" 

                        elif elec == 5 :
                            posspin = ["LS", "IS", "HS"]

                            if metal == "Mn": # Mn(+2)
                                if geometry == "Octahedron":
                                    preferred_spin = "HS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)                   
                                elif geometry == "Square":
                                    spin = "IS"                  
                                else :
                                    spin = "HS" 

                            elif metal == "Fe": # Fe(+3) 
                                preferred_spin = "HS" 
                                spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)                                  

                        elif elec == 6 :
                            posspin = ["LS", "IS", "HS"]
                            if metal == "Cr" : # Cr(0) 
                                spin = "LS"   

                            elif metal == "Mn" : # Mn(+1)
                                spin = "LS"

                            elif metal == "Fe" : # Fe(+2)
                                if geometry == "Tetrahedron" :
                                    spin = "HS"          
                                elif geometry == "Square" or geometry == "Trigonal bipyramid":
                                    preferred_spin = "HS" 
                                    posspin = ["IS", "HS"]
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)
                                elif geometry == "Octahedron":
                                    preferred_spin = "LS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)         
                                else :
                                    preferred_spin = "HS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)

                            elif metal == "Co" : # Co(+3)
                                if geometry == "Square":    
                                    spin = "IS" 
                                elif geometry == "Octahedron":
                                    spin="LS"
                                else : 
                                    preferred_spin = "LS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)

                        elif elec == 7 :
                            posspin = ["LS", "IS"]
                            if metal == "Co" : # Co(+2)
                                if geometry == "Square":    
                                    spin = "LS"                
                                else :
                                    preferred_spin = "IS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)

                            elif metal == "Ni" : # Ni(+3)
                                spin = "LS"   


                        elif elec == 8 : 
                            posspin = ["LS", "IS"]
                            if metal == "Fe": # Fe(0)
                                spin = "LS"

                            elif metal == "Co" : # Co(+1)
                                if geometry == "Tetrahedron" :
                                    spin = "IS"         
                                elif geometry == "Square":    
                                    spin = "LS"                           
                                else :
                                    preferred_spin = "LS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)

                            elif metal == "Ni" : # Ni(+2) 
                                if geometry == "Tetrahedron" :
                                     spin = "IS"      
                                elif geometry == "Square"  :
                                    spin = "LS"                  
                                else :
                                    preferred_spin = "IS" 
                                    spin = get_posspin_v1 (coord_sphere, CN, posspin, preferred_spin)
            else : 
                spin = "unknown"                
        
    smul = decide_spin_multiplicity (spin, N)
    
    return smul