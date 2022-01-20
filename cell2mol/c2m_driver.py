#!/usr/bin/env python

import sys
import os
import pickle
import time

# Import modules
from cell2mol.helper import parsing_arguments
from cell2mol.c2m_module import split_infofile, save_cell, cell2mol
from cell2mol.cif2info import cif_2_info

if __name__ == "__main__" or __name__ == "cell2mol.c2m_driver":

    # print("Running!")

    pwd = os.getcwd()

    infofile, step = parsing_arguments()

    if step in [1, 2, 3]:
        # print("Proper step number : {}".format(step))
        pass
    else:
        # print("Improper step number : {}".format(step))
        sys.exit(1)

    root, extension = os.path.splitext(infofile)
    refcode = split_infofile(infofile)

    if step == 2:
        infopath = None

    elif step == 1 or step == 3:
        # If infofile is a .cif file
        if extension == ".cif":
            # print("Convert .cif file to .info file")
            new_info_file = "{}.info".format(refcode)
            sys.stdout = open(new_info_file, "w")
            cif_2_info()
            sys.stdout.close()
            infopath = pwd + "/" + new_info_file
        # If infofile is a .info file
        elif extension == ".info":
            infopath = pwd + "/" + infofile
            if not os.path.exists(infopath):
                # print(f"Error: The file {infofile} could not be found.\n")
                sys.exit(1)
        else:
            # print("Wrong Input File Format")
            sys.exit(1)

    output_dir = pwd + "/" + refcode

    if step == 2:
        cellpath = output_dir + "/Cell_{}.gmol".format(refcode)
        filename = str(output_dir) + "/" + "Cell_" + str(refcode) + ".gmol"
        if os.path.exists(cellpath):
            # print("Cell object exist in output directory")
            pass
        else:
            # print("No such file exists {}".format(filename))
            sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save output and cell object
    output_fname = output_dir + "/output.out"

    if step == 1 or step == 3:
        sys.stdout = open(output_fname, "w")
    elif step == 2:
        sys.stdout = open(output_fname, "a")

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
else:
    sys.exit(1)
