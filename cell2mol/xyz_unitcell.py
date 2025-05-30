import os
import sys
import logging
from contextlib import redirect_stdout
from ase.io import read
from cell2mol.classes import molecule

def process_unitcell_from_xyz(input_path, name, cell_para, current_dir, debug_mode):
    labels, pos, ref_labels, ref_fracs, cellvec, cellparam = readinfo(infopath)
    atoms = read(input_path)
    newcell = cell.from_positional(name, labels, pos, cellvec, cellparam)
    
    ## Get the fragments, which is the moleclist of a fragmented cell
    fragments = newcell.get_moleclist(cov_factor=cov_factor, metal_factor=metal_factor, debug=debug)