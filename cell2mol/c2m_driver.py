#!/usr/bin/env python

import sys
import os
import pickle
import time

# Import modules
from cell2mol.helper import parsing_arguments
from cell2mol.c2m_module import save_cell, cell2mol
from cell2mol.cif2info import cif_2_info

if __name__ == "__main__" or __name__ == "cell2mol.c2m_driver":

    pwd = os.getcwd()

    input, step = parsing_arguments()
    print(input, step)
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    root, extension = os.path.splitext(file)
    root = root.split(".")
    refcode = root[0]

    stdout = sys.stdout

    if step == None:
        step = 3
        pass
    elif step in [1, 2, 3]:
        pass
    else:
        sys.exit(1)
    print(input, step)
    if step == 2:
        infopath = None

    elif step == 1 or step == 3:
        if os.path.exists(input_path):
            if extension == ".cif":
                infopath = os.path.join(pwd, f"{refcode}.info")
                f = open(infopath, "w")
                sys.stdout = f
                # Create .info file
                cif_2_info(input_path, refcode)
                f.close()
                sys.stdout = stdout

            elif extension == ".info":
                infopath = input_path

            else:
                # print("Wrong Input File Format")
                sys.exit(1)
        else:
            # print(f"Error: The file {file} could not be found.\n")
            sys.exit(1)

    output_dir = pwd

    # output_dir = os.path.join(output_dir, refcode)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # print(f"{output_dir=}")

    if step == 2:
        cellpath = os.path.join(output_dir, "Cell_{}.gmol".format(refcode))
        if not os.path.exists(cellpath):
            # print("No Cell object")
            sys.exit(1)

    # save output and cell object
    output_fname = os.path.join(output_dir, "output.out")

    if step == 1 or step == 3:
        output = open(output_fname, "w")
    elif step == 2:
        output = open(output_fname, "a")

    sys.stdout = output
    cell = cell2mol(infopath, refcode, output_dir, step)
    print_types = "gmol"
    save_cell(cell, print_types, output_dir)
    output.close()

    sys.stdout = stdout

    # save error
    res = [i for i, val in enumerate(cell.warning_list) if val]
    if len(res) == 0:
        error_code = 0
    else:
        for i in res:
            error_code = i + 1

    error_fname = os.path.join(output_dir, f"error_{error_code}.out")
    error = open(error_fname, "w")
    sys.stdout = error
    print(cell.print_Warning())
    error.close()

    sys.stdout = stdout
else:
    sys.exit(1)
