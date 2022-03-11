#!/usr/bin/env python

import numpy as np
import os
import pickle
import sys
import pandas as pd
import argparse
import cell2mol
import itertools
from cell2mol.formal_charge import classify_mols
from cell2mol.tmcharge_common import Cell


def determine_charge_BVS (get_dict, thre_max):

    sort = sorted(get_dict, key=get_dict.get)

    mean_values = sum(get_dict[key] for key in get_dict.keys())/len(get_dict)

    min_delta = get_dict[sort[0]]
    second_min_delta =  get_dict[sort[1]]

    if np.isclose(mean_values, 9999):
        charge_BVS = 9999
        print(f"No valid charge by BVS")

    elif min_delta > thre_max :
        charge_BVS = 8888
        print(f"Minimum delta exceeds threshold value {thre_max}")

    elif min_delta <= thre_max and second_min_delta <= thre_max :
        print("charge of 1st minimum delta {} : {}".format(sort[0], min_delta))
        print("charge of 2nd minimum delta {} : {}".format(sort[1], second_min_delta))
        charge_BVS = 7777
        print(f"Multiple deltas are less than the threshold value")
    else:
        charge_BVS = sort[0]

    print(f"{charge_BVS=}")

    return charge_BVS
                    
def parsing_arguments_BVS():
    parser = argparse.ArgumentParser(
        prog="BVS", description="BVS get metal charge from cell object"
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="filename",
        type=str,
        required=True,
        help="Filepath of Input Cell object (.gmol)",
    )

    args = parser.parse_args()
    
    return args.filename
                    
                    
def bond_valence_sum(cell, bv_para, result, thre_max):
    
    if not any(cell.warning_list[:5]):
        print("Cell reconstruction successfully finished.\n")

    print("====================================== BVS ======================================")
    print("Classify unique species\n")
    
    
    molec_indices, ligand_indices, unique_indices, unique_species = classify_mols(cell.moleclist)
    
    print(f"{len(unique_species)} Species (Ligand or Molecules) to Characterize\n")
    
    with open(result,'a') as data:  
        for idx, spec in enumerate(unique_species):

            if spec[0] == "Metal":
                print("Metal indice of unique species:{}".format(idx))
                metal = spec[1]
                mol = spec[2]
                
                try:
                    print("metal.totcharge by cell2mol : {}\n".format(metal.totcharge))
                except AttributeError:
                    metal.totcharge = 9999
                    print("metal.totcharge by cell2mol : {}\n".format(metal.totcharge))

                if metal.coord in mol.coord:
                    metal_index = mol.coord.index(metal.coord)
                    sphere_index = np.where(mol.mconmat[metal_index] == 1)[0]

                    possible_charge_metal = list(df[df['atom_1']== metal.label].atom_1_valence.unique())

                    if 9 in possible_charge_metal: # 9 : OS didn't be specified in the original reference
                        possible_charge_metal.remove(9)
    #                 print(metal.label, possible_charge_metal)

                    get_dict = {}

                    print("Possible charges of Metal {} by BVS : {}".format(metal.label, possible_charge_metal))

                    for charge in possible_charge_metal:
                        print("============================== OS : +{} Metal : {} ================================".format(charge, metal.label))

                        S = []  # valence list for a metal–ligand bond

                        for index in sphere_index:
#                             print("index in sphere_index", index)
                            condition = (
                                (bv_para["atom_1"] == metal.label)
                                & (bv_para["atom_1_valence"] == charge)
                                & (bv_para["atom_2"] == mol.labels[index])
                            )

                            if any(
                                condition
                            ):  # If parameters exists for a given metal-ligand bond
                                R0_list = bv_para[condition]["R0"].values
                                B_list = bv_para[condition]["B"].values
                                bond_length = np.linalg.norm(
                                    np.array(metal.coord) - mol.coord[index]
                                )
#                                 print(bv_para[condition])
#                                 print("R0_list", R0_list)
#                                 print("B_list", B_list)
#                                 print("bond_length", bond_length)

                                S_sub = [
                                    np.exp((R0 - bond_length) / B)
                                    for R0, B in zip(R0_list, B_list)
                                ]
                                S.append(S_sub)
                                print(
                                    "OS : +{} Bond : {}-{} S : {} sets of parameters".format(
                                        charge, metal.label, mol.labels[index], len(S_sub)
                                    )
                                )

                            else:
                                print(
                                    "OS : +{} Bond : {}-{} condition : {} ==> No parameters exist.".format(
                                        charge,
                                        metal.label,
                                        mol.labels[index],
                                        any(condition),
                                    )
                                )

                        if len(S) == len(metal.coord_sphere):
                            combination = list(itertools.product(*S))
                            delta_comb = [sum(s) - charge for s in combination]
                            delta = min(delta_comb, key=abs)
                            delta = abs(delta)
#                             print("OS \t: \t +{}".format(charge))
#                             print("Trial OS: \t +{:.6f}".format(delta + charge))
                            print("OS : +{} \t Trial OS : {:.6f} \t Delta : {:.6f}\n".format(charge, delta + charge, delta))
                        else:
                            delta = 9999
                            print(
                                "===================================== Failed =====================================\n".format()
                            )
                        get_dict[charge] = delta

                    get_dict.update((key, round(val, 6)) for key, val in get_dict.items()) 
                    print("Charge and Delta pairs", get_dict)
                    
                    charge_BVS = determine_charge_BVS (get_dict, thre_max)
                    
                    data.write("%s\t%s\t%d\t%d\t%s\t%d\n" % (cell.refcode, metal.label, metal.totcharge, charge_BVS, get_dict, idx))
                    print(
                        "***************************************************************************************"
                    )

if __name__ == "__main__" :

    pwd = os.getcwd()
    filename = parsing_arguments_BVS()
    print(filename)
    df = pd.read_csv("bvparm2020.txt",delimiter="\t")
    result = "BVS_result.txt"    
    file = open(filename,'rb')
    cell = pickle.load(file)
    fail_reconstruct = "fail_in_reconstruct.txt"
    
    if any(cell.warning_list[:5]):
        with open(fail_reconstruct,'a') as fail: 
            fail.write("%s\n" % (cell.refcode))
    else:
        output_dir = pwd + "/" + cell.refcode
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_fname = output_dir + "/bvs_output.out"
        
        thre_max = 0.5
        
        sys.stdout = open(output_fname, "w")
        print("[Refcode]", cell.refcode)
        bond_valence_sum(cell, df, result, thre_max)
        sys.stdout.close()
