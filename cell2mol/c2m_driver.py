#!/usr/bin/env python

import sys
import os
# Import modules
from cell2mol.helper import parsing_arguments
from cell2mol.c2m_module import cell2mol
from cell2mol.cif2info import cif_2_info
from cell2mol.classes import cell
from cell2mol.read_write import readinfo, prefiter_cif, writexyz
from cell2mol.other import handle_error
import ase.io
from cell2mol.cell_operations import frac2cart_fromparam


# if __name__ != "__main__" and __name__ != "cell2mol.c2m_driver": sys.exit(1)
if __name__ == "__main__" or __name__ == "cell2mol.c2m_driver":

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
    ref_cell_fname = os.path.join(current_dir, "Ref_Cell_{}.cell".format(name))
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
    version = "2.0"
    print(f"cell2mol version {version}")
    print(f"INITIATING cell object from info path: {infopath}") 
    print(f"Debug level: {debug}")  
    # Reads reference molecules from info file, as well as labels and coordinates
    labels, pos, ref_labels, ref_fracs, cellvec, cellparam = readinfo(infopath)
    atoms = ase.io.read(input_path)

    # Get Cartesian coordinates
    cartesian_coords = atoms.get_positions()

    # Get atomic symbols (labels)
    atomic_labels = atoms.get_chemical_symbols()

    print("Checking atomic labels and coordinates")
    print("Atomic labels:", len(atomic_labels), "from ase", len(labels), "from cif2cell")
    print("Cartesian coordinates:", len(cartesian_coords),  "from ase",  len(pos), "from cif2cell")
    if len(atomic_labels)==len(labels) and len(cartesian_coords)==len(pos): 
        pass
    else: print("Atomic labels and coordinates are inconsistent")
    writexyz(current_dir, "Cell_{}_ase.xyz".format(name), atomic_labels, cartesian_coords)
    writexyz(current_dir, "Cell_{}_cif2cell.xyz".format(name), labels, pos)
    # Initiates cell
    # newcell = cell(name, labels, pos, cellvec, cellparam)
    newcell = cell(name, atomic_labels, cartesian_coords, cellvec, cellparam)
    # Loads the reference molecules and checks_missing_H
    # TODO : reconstruct the unit cell without using reference molecules
    # TODO : reconstruct the unit cell using (only reconstruction of) reference molecules and Space group
    newcell.get_reference_molecules(ref_labels, ref_fracs,cov_factor=1.3, debug=debug) 
    ref_pos = frac2cart_fromparam(ref_fracs, cellparam)
    writexyz(current_dir, "Ref_All_{}.xyz".format(name), ref_labels, ref_pos)
    if not newcell.has_isolated_H:  newcell.check_missing_H(debug=debug)                                     
    newcell.assess_errors(ref=True)
    newcell.save(ref_cell_fname)
    for idx, ref in enumerate(newcell.refmoleclist):
        writexyz(current_dir, "Ref_Molecule_{}_{}.xyz".format(name, idx), ref.labels, ref.coord)

    # sys.exit(0) 
    ######################
    ### CALLS CELL2MOL ###
    ######################
    print(f"ENTERING cell2mol with debug={debug}")
    cell = cell2mol(newcell, reconstruction=True, charge_assignment=True, spin_assignment=True, debug=debug)
    cell.assess_errors()
    print("*** Cell ***")
    cell.save(cell_fname)
    print(cell)
    print("*** Reference molecules ***")
    print(cell.refmoleclist)
    # print("*** Molecules ***")
    # for idx, mol in enumerate(cell.moleclist):
    #     if mol.iscomplex:
    #         print(f"{idx}: {mol.subtype}({mol.type}) {mol.formula} {mol.is_haptic=} {mol.totcharge=} {mol.spin=}") #\n   {mol.adjnum=}\n   {mol.madjnum=} \n   {mol.smiles=}")
    #         # print(mol.adjnum)
    #         for lig in mol.ligands:
    #             print(f"|- {lig.subtype}({lig.type}) {lig.formula} {lig.is_haptic=} {lig.denticity=} {lig.totcharge=}")# \n   {lig.smiles=}")
    #             # print(f"|- {lig.connected_idx}")
    #             # print(lig.groups)
    #             for group in lig.groups:
    #                 print(f"|-- {group.subtype} ({group.type}) {group.formula} {group.is_haptic=} {group.denticity=} {group.closest_metal.label}")
    #                 for met in group.metals:
    #                     print(f"|--- {met.label} {met.mconnec=}")
    #             print("")
    #         for metal in mol.metals:
    #             print(f"|# {metal.subtype}({metal.type}) {metal.label} {metal.coord_nr=} {metal.coord_geometry} {metal.charge=} {metal.spin=} {metal.coord_sphere_formula} {metal.mconnec=} {metal.connec=}")
    #             # print(f"|# {metal.get_coord_sphere_formula()}")
    #             # print(f"|# {metal.coord_sphere_formula}")
    #             # print(f"|# {metal.mconnec=} {metal.connec=}")
    #             # print(metal.metal_adjacency)
    #             # for bond in metal.bonds:
    #                 # print(f"|--- {bond}")
    #     else:
    #         print(f"{idx}: {mol.subtype}({mol.type}) {mol.formula} {mol.totcharge=} {mol.spin=}\n  {mol.smiles}")
    #     print("")
    
    output.close()
    sys.stdout = stdout

    # Error handling
    case = cell.error_case
    error_fname = os.path.join(current_dir, f"error_{case}.out")
    error = open(error_fname, "w")
    sys.stdout = error
    handle_error(case)
    error.close()
    sys.stdout = stdout