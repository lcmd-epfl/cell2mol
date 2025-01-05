#!/usr/bin/env python

import numpy as np
import pickle
import sys
import re
from collections import defaultdict
import os
import traceback

#######################
def get_wyckoff_positions(file_path):
    # Open and read the CIF file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flags and storage for parsing
    start_parsing = False
    data = []

    # Iterate through each line
    for line in lines:
        # Check if line contains the loop declaration for atomic sites
        if '_atom_site_fract_z' in line:
            start_parsing = True
            continue

        # Stop parsing if another loop starts or if data is done
        if 'loop_' in line and start_parsing:
            break

        # Parse the atomic position data
        if start_parsing:
            if line.strip():  # Check for non-empty line
                parts = line.split()
                if len(parts) >= 4:  # Ensure there are enough parts to parse
                    # Clean and extract the coordinates before parentheses
                    x = parts[2].split('(')[0]
                    y = parts[3].split('(')[0]
                    z = parts[4].split('(')[0]
                    # Append label, type, and cleaned fractional coordinates
                    data.append((parts[0], parts[1], float(x), float(y), float(z)))

    # Now 'data' is a list of tuples, each containing:
    # (Label, Element, Fractional_x, Fractional_y, Fractional_z)    
    # for entry in data:
    #     print(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]} {entry[4]}")

    ref_labels = [entry[1] for entry in data]
    ref_fracs = [[entry[2], entry[3], entry[4]] for entry in data]

    # print(f"{len(ref_labels)=} {ref_labels}=")
    # print(f"{len(ref_fracs)=} {ref_fracs}=")

    return ref_labels, ref_fracs

#######################
def exit_with_error_input(message):
    """Logs the error message to a file and exits the program."""
    error_log_path = os.path.join(os.getcwd(), "error_input.out")
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error: {message}\n")
    sys.exit(message)

#######################    
def exit_with_error_exception(e):
    """Logs the error details to a file and exits the program."""
    error_details = traceback.format_exc()
    error_log_path = os.path.join(os.getcwd(), f"error_{type(e).__name__}.out")
    
    # Write the full error details to the log file
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error message: {type(e).__name__} - {str(e)}\n")
        error_log.write(f"Error details:\n{error_details}")
    
    # Print the error details to the console
    print(f"An error occurred. Details have been logged to {error_log_path}")
    print(f"Error details:\n{error_details}")
    
    sys.exit(e)
#######################
def prefilter_cif(input_path):

    with open(input_path, 'r') as ciffile:
        file_content = ciffile.read()
        if 'radical' in file_content:
            exit_with_error_input("Radical found in cif file. STOPPING")                   
            return False
        elif '_atom_site_fract_x' not in file_content:
            exit_with_error_input("No fractional coordinates found in cif file. STOPPING")  
            return False
        elif '?' in file_content:
            if "_diffrn_ambient_temperature ?" not in file_content and "_chemical_melting_point ?" not in file_content:
                exit_with_error_input("Disorder found in cif file. STOPPING")
                return False
            else:
                num_greps = file_content.count('?')
                if num_greps > 1:
                    exit_with_error_input("Disorder found in cif file. STOPPING")                      
                    return False
                else:
                    return True
        else:
            print("Cif file is ready for processing")
            return True
#######################
def save_binary(variable, pathfile, backup: bool=False):
    try:
        file = open(pathfile,'wb')
        pickle.dump(variable,file)
        file.close()
    except Exception as exc:
        print("Error Saving Binary for pathfile:", pathfile)
        print(exc)

#######################
def load_binary(pathfile):
    with open(pathfile, "rb") as pickle_file:
        binary = pickle.load(pickle_file)
    return binary            

##############
def readxyz(file):
    labels = []
    pos = []
    xyz = open(file, "r")
    n_atoms = xyz.readline()
    title = xyz.readline()
    for line in xyz:
        line_data = line.split()
        if len(line_data) == 4:
            label, x, y, z = line.split()
            pos.append([float(x), float(y), float(z)])
            labels.append(label)
        else:
            print("I can't read the xyz. It has =/ than 4 columns")
    xyz.close()

    return labels, pos

################################
def printxyz(labels, pos):
    print(len(labels))
    print("")
    for idx, l in enumerate(labels):
        print("%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]))

##############
def writexyz(fdir, fname, labels, pos, charge: int=0, spin: int=1):
    if fdir[-1] != "/":
        fdir = fdir + "/"
    natoms = len(labels)
    fullname = fdir + fname
    with open(fullname, "w") as fil:
        print(natoms, file=fil)
        print(charge, spin, file=fil)
        for idx, l in enumerate(labels):
            print("%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]),file=fil)

##############
def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, "r") as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number))

    return list_of_results

##############
def readinfo(filepath):

    info = open(filepath, "r")
    lines = list(info.readlines())
    info.close()

    strings = [
        "Lattice parameters:",
        "Representative sites :",
        "OUTPUT CELL INFORMATION",
        "Bravais lattice vectors :",
        "All sites, (cartesian coordinates):",
        "Unit cell volume",
    ]
    lint = np.zeros((len(strings)))

    for l, line in enumerate(lines):
        for stx, string in enumerate(strings):
            if string in line:
                lint[stx] = l


    latparamsone = int(lint[0] + 2)
    latparamstwo = int(lint[1] - 1)
    fracstart = int(lint[1] + 2)
    fracend = int(lint[2] - 2)
    cellvecstart = int(lint[3] + 1)
    cellvecend = int(lint[4] - 1)
    coordstart = int(lint[4] + 2)
    coordend = int(lint[5] - 2)

    lfracs = []
    labels = []
    pos = []
    fracs = []
    cellvec = []
    cellparam = []

    for l, line in enumerate(lines):
        # reads lattice parameters
        if (l == latparamsone) or (l == latparamstwo):
            a, b, c = line.split()
            cellparam += [float(a), float(b), float(c)]

        # reads fractional coordinates
        if (l >= fracstart) and (l <= fracend):
            line_data = line.split()
            if len(line_data) == 4:
                label, x, y, z = line.split()
            elif len(line_data) == 5:  #sometimes, an occupation value is also given.
                label, x, y, z, occ = line.split()
            fracs.append([float(x), float(y), float(z)])
            lfracs.append(label)

        # reads cell vectors
        if (l >= cellvecstart) and (l <= cellvecend):
            v1, v2, v3 = line.split()
            cellvec.append([float(v1), float(v2), float(v3)])

        # reads cartesian coordinates
        if (l >= coordstart) and (l <= coordend):
            line_data = line.split()
            if len(line_data) == 4:
                label, x, y, z = line.split()
            elif len(line_data) == 5:  #sometimes, an occupation value is also given.
                label, x, y, z, occ = line.split()
            pos.append([float(x), float(y), float(z)])
            labels.append(label)


    return labels, pos, lfracs, fracs, cellvec, cellparam

##############
def readcif(filepath):

    info = open(filepath, "r")
    lines = list(info.readlines())
    info.close()

    strings = [
        "_journal_name_full",
        "_chemical_name_systematic",
        "_cell_volume",
        "_atom_type_radius_bond",
        "_atom_site_label",
        "_chemical_name_common",
    ]
    lint = np.zeros((len(strings)))

    for l, line in enumerate(lines):
        for stx, string in enumerate(strings):
            if string in line:
                lint[stx] = l

    common = int(lint[5])
    iscommon = False
    if common != 0:
        iscommon = True

    journalname_line = int(lint[0])

    radius_start = int(lint[3] + 1)
    radius_end = int(lint[4] - 2)

    if not iscommon:
        chemname_start = int(lint[1] + 2)
        chemname_end = int(lint[2] - 2)
    if iscommon:
        chemname_start = int(lint[1] + 2)
        chemname_end = int(lint[5] - 2)

    labels = []
    radii = []
    chemname = ""
    for l, line in enumerate(lines):
        # reads journal name
        if l == journalname_line:
            journal = line.split("'")[1]
        # reads fractional coordinates
        if (l >= chemname_start) and (l <= chemname_end):
            chemname += line.split("\n")[0]
        # reads cell vectors
        if (l >= radius_start) and (l <= radius_end):
            label, radius = line.split()
            labels.append(label)
            radii.append(radius)

    return journal, chemname, labels, radii

###########
def print_molecule(mol, name, ext, folder):
    filename = str(folder) + "/" + str(name) + "." + str(ext)

    if ext == "xyz" or ext == "txt":
        with open(filename, "w") as fil:

            # XYZ
            if ext == "xyz":
                print(mol.natoms, file=fil)
                if hasattr(mol, "totcharge") and hasattr(mol, "spin"):
                    print(mol.totcharge, mol.spin, file=fil)
                if hasattr(mol, "totcharge") and not hasattr(mol, "spin"):
                    print(mol.totcharge, "SPIN", file=fil)
                # if not hasattr(mol, 'totcharge') and not hasattr(mol, 'spin'):
                else:
                    print("", file=fil)

                for a in mol.atoms:
                    print("%s   %.6f   %.6f   %.6f" % (a.label, a.coord[0], a.coord[1], a.coord[2]),file=fil)

            # TXT
            elif ext == "txt":
                print(vars(mol), file=fil)

    elif ext == "gmol" or ext == "mol" or ext == "npy" or ext == "dict":
        with open(filename, "wb") as fil:

            # GMOL
            if ext == "gmol":
                pickle.dump(mol, fil)

            # MOL
            elif ext == "mol":
                pickle.dump(mol.object, fil)

            # NPY
            elif ext == "npy":
                np.save(filename, mol)

            # DICT
            elif ext == "dict":
                mydict = vars(mol)
                pickle.dump(mydict, fil)

    else:
        print(ext, "not found as a valid print extension in print_molecule")

#############
def savemolecules(moleclist, output_dir, print_types, option_print_repeated=True):

    # DEFAULTS
    print_xyz = True
    print_gmol = True
    print_npy = False
    print_mol = False
    print_txt = False
    print_dict = False

    if "xyz" not in print_types:
        print_xyz = False
    if "gmol" not in print_types:
        print_gmol = False
    if "npy" in print_types:
        print_npy = True
    if "mol" in print_types:
        print_mol = True
    if "txt" in print_types:
        print_txt = True
    if "dict" in print_types:
        print_dict = True

    printedmolecs = []

    for mol in moleclist:
        shalliprint = False

        if any((mol.elemcountvec == pmol.elemcountvec).all() for pmol in printedmolecs):
            shalliprint = False
        else:
            shalliprint = True
        if option_print_repeated:  # Overwrites decision if the user decides so
            shalliprint = True

        if shalliprint:
            printedmolecs.append(mol)

            if print_xyz:
                print_molecule(mol, mol.name, "xyz", output_dir)
            if print_gmol:
                print_molecule(mol, mol.name, "gmol", output_dir)
            if print_npy:
                print_molecule(mol, mol.name, "npy", output_dir)
            if print_txt:
                print_molecule(mol, mol.name, "txt", output_dir)
            if print_dict:
                print_molecule(mol, mol.name, "dict", output_dir)
            if hasattr(mol, "object"):
                if print_mol:
                    print_molecule(mol, mol.name, "mol", output_dir)

###########
def print_unit_cell(cell, output_dir):

    # Print Original cell
    print("Original_Cell.xyz")
    writexyz(output_dir, "Original_Cell.xyz", cell.labels, cell.pos)

    # Print Full Unit Cell
    cellatoms = 0
    for mol in cell.moleclist:
        cellatoms += mol.natoms

    print("{}_Full_Cell.xyz".format(cell.refcode))
    cell_fname = output_dir + "/" + cell.refcode + "_Full_Cell.xyz"

    with open(cell_fname, "w") as fil:
        print(cellatoms, file=fil)
        print(" ", file=fil)
        for mol in cell.moleclist:
            for a in mol.atoms:
                print(a.label, a.coord[0], a.coord[1], a.coord[2], file=fil)

############
def extract_chemical_formula_moiety(file_path):
    with open(file_path, 'r') as file:
        cif_content = file.read()
    
    # Find the chemical formula moiety using regex
    pattern = r"_chemical_formula_moiety\s+['\";]([^;'\"]+)['\";]"
    match = re.search(pattern, cif_content)
    if not match:
        raise ValueError("Chemical formula moiety not found.")
    formula = match.group(1).strip()
    return formula

def parse_formula_with_quantity(formula: str):
    element_pattern = r"(\d*)\(?([A-Za-z0-9\s]+)\)?(\d*)(\d+[+-]?)"
    moieties_info = []

    for match in re.findall(element_pattern, formula):
        quantity = int(match[0]) if match[0] else 1
        elements = defaultdict(int)
        for element, count in re.findall(r"([A-Z][a-z]*)(\d*)", match[1]):
            elements[element] += int(count) if count else 1

        charge_str = match[3]
        charge = int(charge_str[:-1]) * (1 if charge_str.endswith("+") else -1)

        moieties_info.append({
            "elements": dict(elements),
            "quantity": quantity,
            "charge": charge
        })
    return moieties_info

######################################################
def print_output(moleclist):
    
    for idx, mol in enumerate(moleclist):
        if mol.iscomplex:
            if hasattr(mol, "totcharge") and hasattr(mol, "spin"):
                print(f"{idx}: {mol.subtype}({mol.type}) {mol.formula} {mol.is_haptic=} {mol.totcharge=} {mol.spin=}") #\n   {mol.adjnum=}\n   {mol.madjnum=} \n   {mol.smiles=}")
            else:
                print(f"{idx}: {mol.subtype}({mol.type}) {mol.formula} {mol.is_haptic=}") # {mol.totcharge=} {mol.spin=}") #\n   {mol.adjnum=}\n   {mol.madjnum=} \n   {mol.smiles=}")
            # print(mol.adjnum)
            for lig in mol.ligands:
                if hasattr(lig, "totcharge") and hasattr(lig, "smiles"):
                    print(f"|- {lig.subtype}({lig.type}) {lig.formula} {lig.is_haptic=} {lig.denticity=} {lig.totcharge=} {lig.smiles=}")
                else :
                    print(f"|- {lig.subtype}({lig.type}) {lig.formula} {lig.is_haptic=} {lig.denticity=}")# {lig.totcharge=} {lig.smiles=}")
                # print(f"|- {lig.connected_idx}")
                # print(lig.groups)
                for group in lig.groups:
                    print(f"|-- {group.subtype} ({group.type}) {group.formula} {group.is_haptic=} {group.denticity=} {group.closest_metal.label=}")
                    for met in group.metals:
                        print(f"|--- {met.label} {met.mconnec=}")
                print("")
            for metal in mol.metals:
                if hasattr(metal, "charge") and hasattr(metal, "spin"):
                    print(f"|# {metal.subtype}({metal.type}) {metal.label} {metal.coord_nr=} {metal.coord_geometry=} {metal.coord_sphere_formula=} {metal.mconnec=} {metal.connec=} {metal.charge=} {metal.spin=}")
                else :
                    print(f"|# {metal.subtype}({metal.type}) {metal.label} {metal.coord_nr=} {metal.coord_geometry=} {metal.coord_sphere_formula=} {metal.mconnec=} {metal.connec=}") #{metal.charge=} {metal.spin=} 
                # print(f"|# {metal.get_coord_sphere_formula()}")
                # print(f"|# {metal.coord_sphere_formula}")
                # print(f"|# {metal.mconnec=} {metal.connec=}")
                # print(metal.metal_adjacency)
                # for bond in metal.bonds:
                #     print(f"|--- {bond}")
        else:
            if hasattr(mol, "totcharge") and hasattr(mol, "spin"):
                print(f"{idx}: {mol.subtype}({mol.type}) {mol.formula} {mol.totcharge=} {mol.spin=}\n  {mol.smiles}")
            else :
                print(f"{idx}: {mol.subtype}({mol.type}) {mol.formula}") #{mol.totcharge=} {mol.spin=}\n  {mol.smiles}")
        print("")

######################################################
def print_refmoleclist (cell):
    for i, ref in enumerate(cell.refmoleclist):
        if hasattr(ref, "totcharge"):
            if ref.iscomplex:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge=} (Complex)\n{ref}")
            else:
                print(f"Reference Molecule {i} : {ref.formula} {ref.smiles=} {ref.totcharge=} (Non-complex)")
        else:
            if ref.iscomplex:
                print(f"Reference Molecule {i}: {ref.formula} (Complex)")
            else:
                print(f"Reference Molecule {i} : {ref.formula} (Non-complex)")

        if ref.iscomplex:
            for met in ref.metals:
                if hasattr(met, "charge"):
                    print(f"\t{met.formula} {met.coord_sphere_formula=} {met.coord_geometry=} {met.geom_deviation=} {met.coord_nr=} {met.charge=}")
                else:
                    print(f"\t{met.formula} {met.coord_sphere_formula=}{met.coord_geometry=} {met.geom_deviation=} {met.coord_nr=}")
            for lig in ref.ligands:
                if hasattr(lig, "totcharge"):
                    print(f"\t{lig.formula} {lig.smiles=} {lig.is_haptic=} {lig.haptic_type=} {lig.denticity=} {lig.totcharge=}")
                else:
                    print(f"\t{lig.formula} {lig.is_haptic=} {lig.haptic_type=} {lig.denticity=}")
                for group in lig.groups:
                    print(f"\t|--(group){group.labels} {group.is_haptic=} {group.haptic_type=} {group.denticity=} {group.closest_metal.label=}")
                    # for met in group.metals:
                    #     print(f"\t|--(group.metals){met.label} {met.mconnec=}")
######################################################
def print_unique_species (cell):
    print(f"Unique Species in {cell.subtype}:")
    for specie in cell.unique_species:
        if specie.subtype == "metal":
            if hasattr(specie, "charge"):
                print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} {specie.charge=}")
            else:
                print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=}")
        else:
            if hasattr(specie, "totcharge"):
                print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.smiles=} {specie.totcharge=}")
            else:
                 print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype})")
######################################################
def print_moleclist (cell):                 
    for i, mol in enumerate(cell.moleclist):
        if hasattr(mol, "totcharge"):
            if mol.iscomplex:
                print(f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex)\n{mol}")
            else:
                print(f"Unitcell Molecule {i} : {mol.formula} {mol.smiles=} {mol.totcharge=} (Non-complex)")
        else:
            if mol.iscomplex:
                print(f"Unitcell Molecule {i}: {mol.formula} (Complex)")
            else:
                print(f"Unitcell Molecule {i} : {mol.formula} (Non-complex)")

        if mol.iscomplex:
            for met in mol.metals:
                if hasattr(met, "charge"):
                    print(f"\t{met.formula} {met.coord_sphere_formula=} {met.coord_geometry=} {met.geom_deviation=} {met.coord_nr=} {met.charge=}")
                else:
                    print(f"\t{met.formula} {met.coord_sphere_formula=} {met.coord_geometry=} {met.geom_deviation=} {met.coord_nr=}")
            for lig in mol.ligands:
                if hasattr(lig, "totcharge"):
                    print(f"\t{lig.formula} {lig.smiles=} {lig.is_haptic=} {lig.haptic_type=} {lig.denticity=} {lig.totcharge=}")
                else:
                    print(f"\t{lig.formula} {lig.is_haptic=} {lig.haptic_type=} {lig.denticity=}")
                for group in lig.groups:
                    print(f"\t|--(group){group.labels} {group.is_haptic=} {group.haptic_type=} {group.denticity=} {group.closest_metal.label=}")
                    # for met in group.metals:
                    #     print(f"\t|--(group.metals){met.label} {met.mconnec=}")