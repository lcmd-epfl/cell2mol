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

    namecell = cell.refcode+"_unit_cell.xyz"

    #print_molecule(cell, namecell, "xyz", pwd)
    with open(namecell, "w") as fil:
        print(len(cell.labels), file=fil)
        print("", file=fil)
        for mol in cell.moleclist:
            #print(mol.labels, mol.natoms, len(mol.atoms))
            for a in mol.atoms:
                print("%s   %.6f   %.6f   %.6f" % (a.label, a.coord[0], a.coord[1], a.coord[2]),file=fil)
