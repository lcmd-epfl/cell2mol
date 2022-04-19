import pickle
import sys
import os

from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
from cell2mol.readwrite import writexyz, print_molecule

########################################
def get_cell_info(fil):

    list_of_metals = []
    list_of_charges = []
    with open(fil, 'rb') as gmol:
        loaded_cell = pickle.load(gmol)

        donelist = []
        unique_metals = []
        for idx, mol in enumerate(loaded_cell.moleclist):
            if mol.type == "Complex":
                for met in mol.metalist:
                    found = False
                    for ldx, typ in enumerate(donelist):
                        if (met.coord_sphere_ID == typ).all() and not found:
                            found = True
                    if not found:
                        donelist.append(met.coord_sphere_ID)
                        unique_metals.append(met)

        for idx, met in enumerate(unique_metals):
            list_of_metals.append(met.label)
            list_of_charges.append(met.totcharge)

    return list_of_metals, list_of_charges
########################################

pwd = os.getcwd()
pwd = pwd.replace("\\", "/")

cellfile = sys.argv[1]

if cellfile.endswith(".gmol"):

    splitname=cellfile.split(".")
    splitname=splitname[0].split("_")
    refcode=splitname[1]

    list_of_metals, list_of_charges = get_cell_info(cellfile)
 
    if len(list_of_charges) == 1: print(refcode, list_of_charges[0])
    if len(list_of_charges) != 1: print(refcode, list_of_charges)
    
else:
    print("File does not have .gmol extension")
    exit()

