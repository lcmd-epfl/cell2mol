#!/usr/bin/env python

import sys
import os

# Import modules
from cell2mol.helper import parsing_arguments
from cell2mol.c2m_module import save_cell, cell2mol
from cell2mol.cif2info import cif_2_info
from cell2mol.classes import *
from cell2mol.read_write import readinfo, prefiter_cif

if __name__ != "__main__" and __name__ != "cell2mol.c2m_driver": sys.exit(1)

input, isverbose, isquiet = parsing_arguments()
current_dir     = os.getcwd()
input_path      = os.path.normpath(input)
dir, file       = os.path.split(input_path)
root, extension = os.path.splitext(file)
root = root.split(".")
name = root[0]

stdout = sys.stdout
stderr = sys.stderr

# Filenames for output and cell object
cell_fname   = os.path.join(current_dir, "Cell_{}.cell".format(name))
output_fname = os.path.join(current_dir, "cell2mol.out")

##### Deals with the parsed arguments for verbosity ######
if isverbose and not isquiet:       debug = 2
elif isverbose and isquiet:         debug = 0
elif not isverbose and isquiet:     debug = 0
elif not isverbose and not isquiet: debug = 1

##### Deals with files ######
if os.path.exists(input_path):    
    ## If the input is a .cif file, then it is converted to a .info file using cif_2_info from cif2cell
    if extension == ".cif":
        # Pre-filtering of the .cif file
        prefiter_cif(input_path)
        errorpath    = os.path.join(current_dir, "cif2cell.err")
        infopath     = os.path.join(current_dir, "{}.info".format(name))
        # if error exist : sys.exit(1)
        # Create .info file 
        cif_2_info(input_path, infopath, errorpath)
        # Checks errors in cif_2_info
        with open(errorpath, 'r') as err:
            for line in err.readlines():
                if "Error" in line: sys.exit(1)

    ## If the input is an .info file, then is used directly
    elif extension == ".info": infopath = input_path
    else:                      sys.exit(1)

output = open(output_fname, "w")
sys.stdout = output

################################
### PREPARES THE CELL OBJECT ###
################################
print(f"INITIATING cell object from input") 

# Reads reference molecules from info file, as well as labels and coordinates
labels, pos, ref_labels, ref_fracs, cellvec, cellparam = readinfo(infopath)
# Initiates cell
newcell = cell(name, labels, pos, cellvec, cellparam)
# Loads the reference molecules and checks_missing_H
# TODO : reconstruct the unit cell without using reference molecules
# TODO : reconstruct the unit cell using (only reconstruction of) reference molecules and Space group
newcell.get_reference_molecules(ref_labels, ref_fracs, debug=debug) 

######################
### CALLS CELL2MOL ###
######################
print(f"ENTERING cell2mol with debug={debug}")
cell = cell2mol(newcell, reconstruction=True, charge_assignment=True, spin_assignment=True, debug=debug)
cell.assess_errors()
cell.save(cell_fname)

output.close()
sys.stdout = stdout
