import pickle
import sys
import os

from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
from cell2mol.readwrite import writexyz, print_molecule

pwd = os.getcwd()
pwd = pwd.replace("\\", "/")

gmolfile = sys.argv[1]

if gmolfile.endswith(".gmol"):
    with open(pwd + "/" + gmolfile, "rb") as pickle_file:
        mol = pickle.load(pickle_file)
    
        namexyz = mol.refcode+".xyz"
        writexyz(pwd, namexyz, mol.labels, mol.coord)
else:
    print("File does not have .gmol extension")
    exit()

