#!/usr/bin/env python

import sys
import os

# Import modules
from cell2mol.c2m_module import split_infofile, save_cell, cell2mol

if __name__ == "__main__" or __name__ == "cell2mol.c2m_driver":
    print("Running!")

    pwd = os.getcwd()
    # infofile = "YOXKUS.info" # no warning
    # infofile = "ISIPIJ.info" # final distribtuion 2, BVS worked
    infofile = "HACXOY.info" # final distribtuion 2
    refcode = split_infofile(infofile)
    print(refcode)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    infopath = dir_path + "/test/infodata/" + infofile
    output_dir = dir_path + "/test/infodata/" + refcode

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_fname = output_dir + "/output.out"

    # save output and cell object
    sys.stdout = open(output_fname, "w")
    cell = cell2mol(infopath, refcode)
    print(cell)
    if not any(cell.warning_list):
        print(cell.print_charge_assignment())

    print_types = "gmol"
    save_cell(cell, print_types, output_dir)

    sys.stdout.close()

    # save error
    res = [i for i, val in enumerate(cell.warning_list) if val]
    if len(res) == 0:
        error_code = 0
    else:
        for i in res:
            error_code = i + 1

    error_fname = output_dir + "/error_{}.out".format(error_code)
    sys.stdout = open(error_fname, "w")
    print(cell.print_Warning())
    sys.stdout.close()
