import pickle
import sys
import os

from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
from cell2mol.readwrite import writexyz, print_molecule

pwd = os.getcwd()
pwd = pwd.replace("\\", "/")

gmolfile = sys.argv[1]

if gmolfile.endswith(".gmol"):
    splitname = gmolfile.split(".")
    if len(splitname) == 2:
        corename = splitname[0]
        extension = splitname[1]
    elif len(splitname) == 3:
        corename = splitname[0]
        searchname = splitname[1]
        extension = splitname[2]
    else:
        print("can't understand the filename you gave me")
        exit()
else:
    print("File does not have .gmol extension")
    exit()

with open(pwd + "/" + gmolfile, "rb") as pickle_file:
    cell = pickle.load(pickle_file)

    for idx, mol in enumerate(cell.speclist):
         
        if mol.type == "Ligand":
    
            namemol = mol.refcode+"_lig_"+str(idx)
            print_molecule(mol, namemol, "gmol", pwd)
