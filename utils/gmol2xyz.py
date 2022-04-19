import pickle
import sys
import os

from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
from cell2mol.readwrite import writexyz, print_molecule

pwd = os.getcwd()
pwd = pwd.replace("\\", "/")

gmolfile = sys.argv[1]


if gmolfile.endswith(".gmol"):

    splitname=gmolfile.split(".")
    if len(splitname) == 2:
        namexyz = splitname[0]+".xyz"

    with open(pwd + "/" + gmolfile, "rb") as pickle_file:
        mol = pickle.load(pickle_file)

        if hasattr(mol, "totcharge"): 
            remainder = (mol.eleccount + mol.totcharge) % 2
        else:
            remainder = mol.eleccount % 2

        if remainder == 0: 
            spin = 1
        else:
            spin = 2
    
        #namexyz = mol.name+".xyz"
        if hasattr(mol, "totcharge") and hasattr(mol, "spin"): 
            writexyz(pwd, namexyz, mol.labels, mol.coord, mol.totcharge, mol.spin)
            print("All info available", mol.refcode)
        elif hasattr(mol, "totcharge") and not hasattr(mol, "spin"): 
            writexyz(pwd, namexyz, mol.labels, mol.coord, mol.totcharge, spin)
            print("Assuming spin for", mol.refcode)
        else:
            writexyz(pwd, namexyz, mol.labels, mol.coord, 0, spin)
            print("Assuming defaults for", mol.refcode)
else:
    print("File does not have .gmol extension")
    exit()

