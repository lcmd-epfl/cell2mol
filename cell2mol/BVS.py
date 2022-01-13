#!/usr/bin/env python

import numpy as np
import sys
import pickle
import time
import os
import re
import pandas as pd
import itertools
from cell2mol.tmcharge_common import Cell
from cell2mol.formal_charge import get_metal_poscharge

def bond_valence_sum (cell, bv_para, mode):

    for mol in cell.moleclist:

        if mol.type == "Complex":

            for metal in mol.metalist:

                if metal.coord in mol.coord:                    
                    metal_index = mol.coord.index(metal.coord)
                    sphere_index = np.where(mol.mconmat[metal_index] == 1)[0]
                    
                    print("Possible charge of Metal {}: {}".format(metal.label, get_metal_poscharge(metal.label)))
                    get_dict = {}
                    for i, charge in enumerate([1,2,3]):    # for Cu                
#                     for i, charge in enumerate(get_metal_poscharge(metal.label)):
                        
                        print("============================== OS : +{} Metal : {} ================================".format(charge, metal.label))
                        
                        S = []  # valence list for a metal–ligand bond 
                        for index in sphere_index:
                            
                            condition = (bv_para["atom_1"]== metal.label) & (bv_para["atom_1_valence"]== charge) & (bv_para["atom_2"]== mol.labels[index])
                            
                            
                            if any(condition): # If parameters exists for a given metal-ligand bond
                                R0_list = bv_para[condition]["R0"].values
                                B_list = bv_para[condition]["B"].values
                                bond_length = np.linalg.norm(np.array(metal.coord) - mol.coord[index])
#                                 print(bv_para[condition])
#                                 print("R0_list", R0_list)
#                                 print("B_list", B_list)
#                                 print("bond_length", bond_length)
                                
                                if mode == 'single':
                                    R0 = R0_list[0]
                                    B = B_list[0]
                                    S.append(np.exp ((R0 - bond_length)/B))                                
                                
                                elif mode == 'average':
                                    S_sub = [np.exp ((R0 - bond_length)/B) for R0, B in zip(R0_list, B_list)]
#                                     print("S_sub", S_sub)
                                    S.append(sum(S_sub)/len(S_sub))
#                                     print("S", S)
                                elif mode == 'smallest_delta':
                                    S_sub = [np.exp ((R0 - bond_length)/B) for R0, B in zip(R0_list, B_list)]
#                                     print("S_sub", S_sub)
                                    S.append(S_sub)
#                                     print("S", S)                                    
                                else:
                                    print("Please input proper mode.")
                                print("OS : +{} Bond : {}-{} S : {}".format(charge, metal.label, mol.labels[index], S))
                            
                            else :
                                print("OS : +{} Bond : {}-{} condition : {} ==> No parameters exist.".format(charge, metal.label, mol.labels[index], any(condition)))
#                         print ("==========Final S : {}==========". format(S))
                        
                        if len(S) == len(metal.coord_sphere) : 
                            Warning = False
                            if mode == 'smallest_delta':
                                combination = list(itertools.product(*S))
                                delta_comb = [sum(s) - charge for s in combination]
                                delta = min(delta_comb, key=abs)
                                print("Trial oxidation state (V_t) of {} is +{}".format(metal.label, delta + charge))
                                delta = abs(delta)
                            else :
                                print("Trial oxidation state (V_t) of {} is +{}".format(metal.label, sum(S)))
                                delta = abs(sum(S)-charge)
                            print("Absolute delta is {} when charge of {} is +{}".format(delta, metal.label, charge))
                            get_dict[charge] = delta
                            print(get_dict)
                        else :
                            Warning = True
                            
                final_charge = min(get_dict, key=get_dict.get)                            
                if get_dict[final_charge] > 0.5 :
                    Warning = True
                else :
                    Warning = False
                return final_charge, Warning





if __name__ == "__main__":

    pwd = os.getcwd()
    refcode = "EGITOF"
    print(refcode)


    # Import and creat DataFrame bond valence parameters 2020
    bv_para= pd.read_csv("bvparm2020.txt",delimiter="\t")

    output_dir = refcode
    
    file = open(f"{output_dir}/Cell_{refcode}.gmol",'rb')
    object_file = pickle.load(file)

    bond_valence_sum (object_file, bv_para, 'smallest_delta')
