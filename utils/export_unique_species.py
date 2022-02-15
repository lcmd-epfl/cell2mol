import pickle
import sys
import os

from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
from cell2mol.readwrite import writexyz, print_molecule

#def writexyz(output_dir, output_name, labels, pos):
#    if output_dir[-1] != "/":
#        output_dir = output_dir + "/"
#    natoms = len(labels)
#    fullname = output_dir + output_name
#    with open(fullname, "w") as fil:
#        print(natoms, file=fil)
#        print(" ", file=fil)
#        for idx, l in enumerate(labels):
#            print("%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]),file=fil)

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
        namexyz = mol.refcode+"_spec_"+str(idx)+".xyz"
        writexyz(pwd, namexyz, mol.labels, mol.coord)

        namemol = mol.refcode+"_spec_"+str(idx)
        print_molecule(mol, namemol, "gmol", pwd)
