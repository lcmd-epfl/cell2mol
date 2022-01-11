#!/usr/bin/env python

import numpy as np
import sys
import pickle
import time
import os
import re
import pandas as pd
from tmcharge_common import Cell
from formal_charge import get_metal_poscharge


isincluster = False

#########################
# Loads Element Library #
#########################
if isincluster: 
    elempath = '/home/ycho/Projects/Marvel_TM_Database/sourcefiles_v2'
else:
    elempath = '/Users/ycho/Projects/Marvel_TM_Database/sourcefiles_v2'

elempath = elempath.replace("\\","/")
sys.path.append(elempath)

from elementdata import ElementData
elemdatabase = ElementData()

#######################
### Loads Functions ###
#######################
if isincluster:
    utilspath = '/home/ycho/Projects/Marvel_TM_Database/sourcefiles_v2'
else:
    utilspath = '/Users/ycho/Projects/Marvel_TM_Database/sourcefiles_v2'
sys.path.append(utilspath)

#######################
### Loads Functions ###
#######################
if isincluster:
    utilspath = '/home/ycho/Projects/Marvel_TM_Database/sourcefiles_v2'
else:
    utilspath = '/Users/ycho/Projects/Marvel_TM_Database/sourcefiles_v2'
sys.path.append(utilspath)


def split_infofile (infofile):
    
    splitname = infofile.split(".")
    if (len(splitname) == 2): return splitname[0]    
    elif (len(splitname) == 3): return splitname[0], splitname[1]
    else:
        print("can't understand the filename you gave me")
        exit()


def bond_valence_sum (cell, bv_para, mode):
    
    # If mode = 'average' : 
    
    
    for mol in cell.moleclist[:1]:
        
        if mol.type == "Complex":

            for metal in mol.metalist:
                
                if metal.coord in mol.coord:                    
                    metal_index = mol.coord.index(metal.coord)
                    sphere_index = np.where(mol.mconmat[metal_index] == 1)[0]
                    
                    for charge in get_metal_poscharge(metal.label):
                        S = []  # valence list for a metal–ligand bond 
                        for i,index in enumerate(sphere_index):
                            
                            condition = (bv_para["atom_1"]== metal.label) & (bv_para["atom_1_valence"]== charge) & (bv_para["atom_2"]== mol.labels[index])
                            
                            
                            if any(condition): # If parameters exists for a given metal-ligand bond
                                R0_list = bv_para[condition]["R0"].values
                                B_list = bv_para[condition]["B"].values
                                bond_length = np.linalg.norm(np.array(metal.coord) - mol.coord[index])
                                
                                print("R0_list", R0_list)
                                print("B_list", B_list)
                                print("bond_length", bond_length)
                                
                                if mode == 'single':
                                    R0 = R0_list[0]
                                    B = B_list[0]
                                    S.append(np.exp ((R0 - bond_length)/B))                                
                                
                                elif mode == 'average':
                                    S_sub = [np.exp ((R0 - bond_length)/B) for R0, B in zip(R0_list, B_list)]
                                    print("S_sub", S_sub)
                                    S.append(sum(S_sub)/len(S_sub))
                                    print("S", S)
                                else:
                                    print("Please input proper mode.")
                                print("OS : {} Bond : {}-{} S : {}".format(charge, metal.label, mol.labels[index], S))
                            
                            else :
                                print("Bond : {}-{} condition : {} ==> No parameters exist.".format(metal.label, mol.labels[index], any(condition)))
                        print ("==========Final S : {}==========". format(S))
                        
                        if len(S) == len(metal.coord_sphere) : 
                            
                            print("\tTrial oxidation state (V_t) of {} is +{}".format(metal.label, sum(S)))
                            delta = abs(sum(S)-charge)
                            print("\tDelat is {} when charge of {} is +{}".format(delta, metal.label, charge))
    return min(d, key=d.get)


# Import and creat DataFrame bond valence parameters 2020
bv_para= pd.read_csv("bvparm2020.txt",delimiter="\t")


if __name__ == "__main__":
    tini = time.time()
       
    #### PREPARES TO SUBMIT MAIN ####
    pwd = os.getcwd() ##returns current working directory of a process
    pwd = pwd.replace("\\","/")
    
    if isincluster:
        pass
    else:
#         folder = '/Users/ycho/Projects/Marvel_TM_Database/Check_Manual'
        folder = '/Users/ycho/Projects/Marvel_TM_Database/BVS_Reeves'
        infofile = 'KUYHES.info'
        
        refcode = split_infofile (infofile)
        infopath = folder+'/'+infofile  
        output_dir = folder+"/"+refcode     
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)     


file = open(f"{output_dir}/Cell_{refcode}.gmol",'rb')
object_file = pickle.load(file)


bond_valence_sum (object_file, bv_para, 'single')

