#!/usr/bin/env python

import pickle

# Import modules
from cell2mol.cell_reconstruct import (
    getmolecs,
    indentify_frag_molec_H,
    split_complexes_reassign_type,
    fragments_reconstruct,
    get_reference_molecules,
)
from cell2mol.formal_charge import (
    get_poscharges_unique_species,
    classify_mols,
    balance_charge,
    build_bonds,
    prepare_mols,
)
from cell2mol.missingH import check_missingH
from cell2mol.tmcharge_common import Cell, getcentroid
from cell2mol.cellconversions import cart2frac, frac2cart_fromparam
from cell2mol.readwrite import readinfo, print_unit_cell


def get_refmoleclist_and_check_missingH(cell, reflabels, fracs):

    refpos = frac2cart_fromparam(fracs, cell.cellparam)

    # Get ref.molecules --> output: a valid list of ref.molecules
    (refmoleclist, covalent_factor, metal_factor, Warning) = get_reference_molecules(
        reflabels, refpos
    )
    cell.warning_list.append(Warning)

    # Check missing hydrogens in ref.molecules
    if not any(cell.warning_list):
        (Warning, ismissingH, Missing_H_in_C, Missing_H_in_CoordWater) = check_missingH(
            refmoleclist
        )
        cell.warning_list.append(Missing_H_in_C)
        cell.warning_list.append(Missing_H_in_CoordWater)

    if not any(cell.warning_list):
        for mol in refmoleclist:
            mol.refcode = cell.refcode
            mol.name = str(cell.refcode + "_Reference_" + str(refmoleclist.index(mol)))

        cell.refmoleclist = refmoleclist

    return cell, covalent_factor, metal_factor


def reconstruct(cell, reflabels, fracs):

    debug = 0
    moleclist = []

    # Get a list of ref.molecules and Check missing H in ref.molecules
    cell, covalent_factor, metal_factor = get_refmoleclist_and_check_missingH(
        cell, reflabels, fracs
    )

    # Get blocks in the unit cells constructing the adjacency matrix (A)
    if not any(cell.warning_list):
        Warning, blocklist = getmolecs(
            cell.labels, cell.pos, covalent_factor, metal_factor
        )
        cell.warning_list.append(Warning)

    # Indentify blocks and Reconstruct Fragments
    if not any(cell.warning_list):
        (moleclist, fraglist, Hlist, init_natoms) = indentify_frag_molec_H(
            blocklist, moleclist, cell.refmoleclist, cell.cellvec
        )

        moleclist, finalmols, Warning = fragments_reconstruct(
            moleclist,
            fraglist,
            Hlist,
            cell.refmoleclist,
            cell.cellvec,
            debug,
            covalent_factor,
            metal_factor,
        )
        cell.warning_list.append(Warning)

    # Check final number of atoms
    if not any(cell.warning_list):
        moleclist.extend(finalmols)
        final_natoms = 0
        for mol in moleclist:
            final_natoms += mol.natoms
        if final_natoms != init_natoms:
            Warning = True
        else:
            Warning = False
        cell.warning_list.append(Warning)

    # Split Complexes and Reassign Type
    if not any(cell.warning_list):
        cell = split_complexes_reassign_type(cell, moleclist)

    return cell


def determine_charge(cell):

    # Indentify unique chemical spicies
    (molec_indices, ligand_indices, unique_indices, unique_species) = classify_mols(
        cell.moleclist
    )
    print(f"{len(unique_species)} Species (Ligand or Molecules) to Characterize")

    unique_species, Warning = get_poscharges_unique_species(unique_species)
    cell.warning_list.append(Warning)

    # Find possible charge distribution
    if not any(cell.warning_list):
        final_charge_distribution = balance_charge(
            cell.moleclist,
            molec_indices,
            ligand_indices,
            unique_indices,
            unique_species,
        )
        print("final_charge_distribution", final_charge_distribution)
        if len(final_charge_distribution) > 1:
            Warning = True
            print(
                "More than one Possible Distribution Found:", final_charge_distribution
            )
        else:
            Warning = False
        cell.warning_list.append(Warning)

        if len(final_charge_distribution) == 0:
            Warning = True
            print("No valid Distribution Found", final_charge_distribution)
        else:
            Warning = False
        cell.warning_list.append(Warning)

    # Only one possible charge distribution -> getcharge for the repeated species
    if not any(cell.warning_list):
        print("\nFINAL Charge Distribution:", final_charge_distribution)
        print("Assigning Charges and Preparing Molecules")
        cell.moleclist, Warning = prepare_mols(
            cell.moleclist, unique_indices, unique_species, final_charge_distribution[0]
        )
        cell.warning_list.append(Warning)

    # Build Bond objects for molecules
    if not any(cell.warning_list):
        cell.moleclist = build_bonds(cell.moleclist)

    return cell


def save_cell(cell, ext, output_dir):
    print("\n[Output files]")
    # print_unit_cell (cell, output_dir)

    filename = str(output_dir) + "/" + "Cell_" + str(cell.refcode) + "." + str(ext)
    with open(filename, "wb") as fil:
        if ext == "gmol":
            print("Cell_" + str(cell.refcode) + "." + str(ext))
            pickle.dump(cell, fil)
        else:
            print(ext, "not found as a valid print extension in print_molecule")


def split_infofile(infofile):

    splitname = infofile.split(".")
    if len(splitname) == 2:
        return splitname[0]
    elif len(splitname) == 3:
        return splitname[0], splitname[1]
    else:
        print("can't understand the filename you gave me")
        exit()


def cell2mol(infopath, refcode):

    # Read reference molecules from info file
    labels, pos, reflabels, fracs, cellvec, cellparam = readinfo(infopath)

    # Initialize cell object
    warning_list = []
    cell = Cell(refcode, labels, pos, cellvec, cellparam, warning_list)
    print("[Refcode]", cell.refcode)

    # Cell Reconstruction
    cell = reconstruct(cell, reflabels, fracs)

    # Charge Assignment
    if not all(cell.warning_list):
        print("Cell reconstruction successfully finished.\n")
        cell = determine_charge(cell)
    else:
        print("Cell reconstruction failed.\n")

    return cell
