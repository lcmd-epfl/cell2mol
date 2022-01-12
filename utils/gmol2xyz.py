import pickle
import sys
import os

utilspath = (
    "/home/velallau/Postdoc/Marvel_TM_Database/Scripts/Formal_Charge/sourcefiles"
)
sys.path.append(utilspath)

# Imports Classes
import tmcharge_common
from tmcharge_common import atom
from tmcharge_common import molecule
from tmcharge_common import ligand
from tmcharge_common import metal


def writexyz(output_dir, output_name, labels, pos):
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"
    natoms = len(labels)
    fullname = output_dir + output_name
    with open(fullname, "w") as fil:
        print(natoms, file=fil)
        print(" ", file=fil)
        for idx, l in enumerate(labels):
            print(
                "%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]),
                file=fil,
            )


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
    loadedmol = pickle.load(pickle_file)
    name = corename + ".xyz"
    writexyz(pwd, name, loadedmol.labels, loadedmol.coord)
