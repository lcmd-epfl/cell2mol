#!/usr/bin/env python

import numpy as np
from cell2mol.elementdata import ElementData
from cosymlib.shape.tools import shape_structure_references
from cell2mol.tmcharge_common import getelementcount
import ast #ast.literal_eval



elemdatabase = ElementData()

################################
def count_elec (met_label, m_ox):

    v_elec = elemdatabase.valenceelectrons[met_label]
    group = elemdatabase.elementgroup[met_label]

    if v_elec - m_ox >= 0 :
        elec = v_elec - m_ox
    elif v_elec - m_ox < 0 :
        elec = group - m_ox

    return elec

################################
def decide_spin_multiplicity (spin, N):
    
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
def guess_spin (l, CN, posspin, preferred_spin):
    
    if (l.count("C") + l.count("P")) / CN > 0.5 :
        posspin = posspin[0] # lowest spin state
    elif (l.count("F") + l.count("Cl") + l.count("Br") + l.count("I") )/ CN > 0.5 :
        posspin =  posspin[-1] # highest spin state
    else :
        posspin = preferred_spin
           
    return posspin

################################
def predict_ground_state_spin (metal, elec, coord_sphere, geometry, CN, N, nitrosyl) :
    
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
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)
                                else :
                                    spin = "HS" 

                            elif metal == "Mn" : # Mn(+3):
                                if geometry == "Octahedron":
                                    preferred_spin = "HS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)                        
                                else :
                                    spin = "HS" 

                        elif elec == 5 :
                            posspin = ["LS", "IS", "HS"]

                            if metal == "Mn": # Mn(+2)
                                if geometry == "Octahedron":
                                    preferred_spin = "HS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)                   
                                elif geometry == "Square":
                                    spin = "IS"                  
                                else :
                                    spin = "HS" 

                            elif metal == "Fe": # Fe(+3) 
                                preferred_spin = "HS" 
                                spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)                                  

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
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)
                                elif geometry == "Octahedron":
                                    preferred_spin = "LS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)         
                                else :
                                    preferred_spin = "HS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)

                            elif metal == "Co" : # Co(+3)
                                if geometry == "Square":    
                                    spin = "IS" 
                                elif geometry == "Octahedron":
                                    spin="LS"
                                else : 
                                    preferred_spin = "LS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)

                        elif elec == 7 :
                            posspin = ["LS", "IS"]
                            if metal == "Co" : # Co(+2)
                                if geometry == "Square":    
                                    spin = "LS"                
                                else :
                                    preferred_spin = "IS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)

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
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)

                            elif metal == "Ni" : # Ni(+2) 
                                if geometry == "Tetrahedron" :
                                     spin = "IS"      
                                elif geometry == "Square"  :
                                    spin = "LS"                  
                                else :
                                    preferred_spin = "IS" 
                                    spin = guess_spin (coord_sphere, CN, posspin, preferred_spin)
            else : 
                spin = "unknown"                
        
    smul = decide_spin_multiplicity (spin, N)
    
    return smul

################################
def counts_element (temp) : 
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
    count_NO = 0 
    for lig in arr :
        #print(lig)
        # if lig == ['N', 'O'] : # Nitrosyl ligand
        #     count_NO += 1
        if all(map(lambda x, y: x == y, lig, ['N', 'O'])):
            count_NO += 1
    return count_NO

################################
def count_N (mol):
    N = 0
    for atom in mol.labels:
        N += elemdatabase.elementnr[atom]
    N -= mol.totcharge
    return N

################################
def make_geom_list ():

    geom_list = {}
    count = 0
    for i in shape_structure_references.values():
    #     print(np.array(i)[:,3])
        for geom in np.array(i)[:,3]:
            geom_list[geom] = count
            count +=1
    return geom_list