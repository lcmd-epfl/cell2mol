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


def bond_valence_sum(
    unique_species, unique_indices, bv_para, final_charge_distribution, debug=0
):

    BVS_list = []
    metal_indices_list = []

    for idx, spec in enumerate(unique_species):
        print("indice of unique species:{}".format(idx))
        if spec[0] == "Metal":
            metal = spec[1]
            mol = spec[2]
            metal_indicies = np.where(np.array(unique_indices) == idx)
            metal_indices_list.append(metal_indicies)
            possible_charge_metal = []

            for dist in np.array(final_charge_distribution):
                poscharge = list(set(dist[metal_indicies]))[0]
                possible_charge_metal.append(poscharge)

            if metal.coord in mol.coord:
                metal_index = mol.coord.index(metal.coord)
                sphere_index = np.where(mol.mconmat[metal_index] == 1)[0]

                #                 print("Possible charge of Metal {}: {}".format(metal.label, get_metal_poscharge(metal.label)))
                print(
                    "Final Possible charges of Metal {}: {}".format(
                        metal.label, possible_charge_metal
                    )
                )
                get_dict = {}
                #                 for charge in get_metal_poscharge(metal.label):
                for charge in possible_charge_metal:
                    print(
                        "============================== OS : +{} Metal : {} ================================".format(
                            charge, metal.label
                        )
                    )

                    S = []  # valence list for a metal–ligand bond

                    for index in sphere_index:
                        print("index in sphere_index", index)
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
                            #                         print(bv_para[condition])
                            #                         print("R0_list", R0_list)
                            #                         print("B_list", B_list)
                            #                         print("bond_length", bond_length)

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
                        print(
                            "==================== Trial oxidation state: +{} ====================\n".format(
                                delta + charge
                            )
                        )
                        delta = abs(delta)
                    else:
                        delta = 9999
                        print(
                            "===================================== Failed =====================================\n".format()
                        )
                    get_dict[charge] = delta

                print(get_dict)
                final_charge = min(get_dict, key=get_dict.get)
                print(f"{final_charge=}")
                print(
                    "***************************************************************************************"
                )
                BVS_list.append(get_dict)
    #     print(BVS_list)
    return BVS_list, metal_indices_list


def choose_final_dist_using_BVS(
    final_charge_distribution, BVS_list, metal_indices_list
):
    collection = []
    final_charge_distribution = np.array(final_charge_distribution)
    for dist in final_charge_distribution:
        result = [
            np.equal(dist[midx], min(BVS, key=BVS.get))
            for midx, BVS in zip(metal_indices_list, BVS_list)
        ]
        #         result = [np.equal(dist[metal_indices_list[i]], min(BVS_list[i], key=BVS_list[i].get)) for i in range(len(BVS_list))]
        #         print(result)
        result = np.array(result)
    #         print(result)
    #         print(result.all())
    if result.all():
        #         print(dist)
        return dist


if __name__ == "__main__":

    pwd = os.getcwd()
    refcode = "ISIPIJ"
    print(refcode)

    # Import and creat DataFrame bond valence parameters 2020
    bv_para = pd.read_csv("bvparm2020.txt", delimiter="\t")
