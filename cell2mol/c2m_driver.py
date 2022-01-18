#!/usr/bin/env python

import sys
import os
import pickle

# Import modules
from cell2mol.helper import parsing_arguments
from cell2mol.c2m_module import split_infofile, save_cell, cell2mol
from cell2mol.cif2info import cif_2_info

if __name__ == "__main__" or __name__ == "cell2mol.c2m_driver":
    print("Running!")

    pwd = os.getcwd()

    infofile, step = parsing_arguments()

    root, extension = os.path.splitext(infofile)
    refcode = split_infofile(infofile)

    # If infofile is a .cif file
    if extension == ".cif":
        new_info_file = "{}.info".format(refcode)
        sys.stdout = open(new_info_file, "w")
        cif_2_info(infofile)
        sys.stdout.close()
        infopath = pwd + "/" + new_info_file

    # If infofile is a .info file
    elif extension == ".info":
        infopath = pwd + "/" + infofile

    else:
        print("Wrong Input File")

    # dir_path = os.path.dirname(os.path.realpath(__file__))

    output_dir = pwd + "/" + refcode
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save output and cell object
    output_fname = output_dir + "/output.out"

    if step == 1 or step == 3:
        sys.stdout = open(output_fname, "w")
    elif step == 2:
        sys.stdout = open(output_fname, "a")
        file = open(f"{output_dir}/Cell_{refcode}.gmol", "rb")
        cell = pickle.load(file)
    else:
        print("Inproper step number")

    cell = cell2mol(infopath, refcode, output_dir, step)
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
