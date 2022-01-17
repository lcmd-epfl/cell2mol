#!/usr/bin/env python

import os

# Import modules
from cell2mol.c2m_module import split_infofile, save_cell, cell2mol

if __name__ == "__main__" or __name__ == "cell2mol.c2m_driver":
    print("Running!")

    pwd = os.getcwd()
    infofile = "YOXKUS.info"
    # infofile = "ISIPIJ.info"
    refcode = split_infofile(infofile)
    print(refcode)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    infopath = dir_path + "/test/infodata/" + infofile
    output_dir = dir_path + "/test/infodata/" + refcode

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cell = cell2mol(infopath, refcode)

    # Print the Charges or Warnings
    if not any(cell.warning_list):
        print("Charge Assignment successfully finished.\n")
        cell.print_charge_assignment()
    else:
        print("Charge Assignment failed.\n")

    cell.print_Warning()

    print_types = "gmol"
    save_cell(cell, print_types, output_dir)
