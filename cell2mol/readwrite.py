#!/usr/bin/env python

import numpy as np
import pickle

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
