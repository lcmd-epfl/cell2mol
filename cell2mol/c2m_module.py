#!/usr/bin/env python

import pickle
import time
import sys
import os
import copy
from typing import Tuple
# Import modules
from cell2mol.cell_reconstruct import (
    getmolecs,
    identify_frag_molec_H,
    split_complexes_reassign_type,
    fragments_reconstruct,
    get_reference_molecules,
)
from cell2mol.formal_charge import (
    drive_get_poscharges,
    classify_mols,
    balance_charge,
    build_bonds,
    prepare_mols,
)
from cell2mol.missingH import check_missingH
from cell2mol.tmcharge_common import Cell
from cell2mol.cellconversions import frac2cart_fromparam
from cell2mol.readwrite import readinfo
from typing import Tuple


def get_refmoleclist_and_check_missingH(cell: object, reflabels: list, fracs: list) -> Tuple[object, float, float]:

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


def reconstruct(cell: object, reflabels: list, fracs: list) -> object:

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
        (moleclist, fraglist, Hlist, init_natoms) = identify_frag_molec_H(
            blocklist, moleclist, cell.refmoleclist, cell.cellvec
        )

        moleclist, finalmols, Warning = fragments_reconstruct(
            moleclist,
            fraglist,
            Hlist,
            cell.refmoleclist,
            cell.cellvec,
            covalent_factor,
            metal_factor,
            debug,
        )

        moleclist.extend(finalmols)
        final_natoms = 0
        for mol in moleclist:
            final_natoms += mol.natoms

        # Check final number of atoms after reconstruction
        if final_natoms != init_natoms:
            warning_num = True
            print(
                f"Final and initial atoms do not coincide. Final/Initial {final_natoms}/ {init_natoms}\n"
            )
        else:
            warning_num = False
            print(
                f"Final and initial atoms coincide. Final/Initial {final_natoms}/ {init_natoms}\n"
            )
        cell.warning_list.append(any([Warning, warning_num]))
    
    cell.warning_after_reconstruction = copy.deepcopy(cell.warning_list)
    
    # Split Complexes and Reassign Type
    if not any(cell.warning_list):
        cell = split_complexes_reassign_type(cell, moleclist)

    return cell


def determine_charge(cell: object) -> object:

    # Indentify unique chemical species
    molec_indices, ligand_indices, unique_indices, unique_species = classify_mols(cell.moleclist)
 
    # Group all unique species in a cell variable
    for spec in unique_species:            # spec is a list in which item 1 is the actual unique specie
        cell.speclist.append(spec[1])    

    print(f"{len(unique_species)} Species (Ligand or Molecules) to Characterize")
    #print("unique_indices", unique_indices)

    # drive_get_poscharges adds posible charges to the metal, ligand, and molecule objects of all species in the unit cell
    # also, it retrieves "Selected_charge_states", which is a tuple with [the actual charge state, and the protonation it belongs to] for all objects except metals
    selected_charge_states, Warning = drive_get_poscharges(unique_species)
    cell.warning_list.append(Warning)

    # Find possible charge distribution
    if not any(cell.warning_list):
        final_charge_distribution = balance_charge(unique_indices,unique_species)
        print("final_charge_distribution", final_charge_distribution)
        if len(final_charge_distribution) > 1:
            Warning = True
            print("More than one Possible Distribution Found:", final_charge_distribution)
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
        print(" ")
        print("#########################################")
        print("Assigning Charges and Preparing Molecules")
        print("#########################################")
        cell.moleclist, Warning = prepare_mols(cell.moleclist, unique_indices, unique_species, selected_charge_states, final_charge_distribution[0])
        cell.warning_list.append(Warning)

    # Build Bond objects for molecules
    if not any(cell.warning_list):
        cell.moleclist = build_bonds(cell.moleclist)

    return cell


def save_cell(cell: object, ext: str, output_dir: str):
    print("\n[Output files]")

    cellpath = os.path.join(output_dir, "Cell_{}.gmol".format(cell.refcode))
    with open(cellpath, "wb") as fil:
        if ext == "gmol":
            print("Output file path", cellpath)
            pickle.dump(cell, fil)
        else:
            print(ext, "not found as a valid print extension in print_molecule")


def load_cell_reset_charges (cellpath: str) -> object:
    
    file = open(cellpath, "rb")
    cell = pickle.load(file)
    print("[Refcode]", cell.refcode, cell)
    # print(f"{cell.moleclist=}")  
    print(f"{cell.warning_list=}")
    cell.warning_list = copy.deepcopy(cell.warning_after_reconstruction)
    print(f"{cell.warning_after_reconstruction=}")

    for mol in cell.moleclist:
        mol.poscharge = []
        mol.posatcharge = []
        mol.posobjlist = []
        mol.posspin = []
        mol.possmiles = []            
        mol.atcharge = None
        mol.totcharge = None
        mol.smiles = None
        mol.object = None
        for atom in mol.atoms:
            atom.atom_charge(None)

        if mol.type == "Complex":
            for lig in mol.ligandlist:
                lig.poscharge = []
                lig.posatcharge = []
                lig.posobjlist = []
                lig.posspin = []
                lig.possmiles = []
                lig.atcharge = None
                lig.totcharge = None
                lig.smiles = None
                lig.object = None
                for atom in lig.atoms:
                    atom.atom_charge(None)

            for met in mol.metalist :
                met.poscharge = []
                met.totcharge = None

    return cell


def cell2mol(infopath: str, refcode: str, output_dir: str, step: int=3) -> object:

    if step == 1 or step == 3:

        tini = time.time()

        # Read reference molecules from info file
        labels, pos, reflabels, fracs, cellvec, cellparam = readinfo(infopath)

        # Initialize cell object
        warning_list = []
        cell = Cell(refcode, labels, pos, cellvec, cellparam, warning_list)
        print("[Refcode]", cell.refcode)

        # Cell Reconstruction
        print(
            "===================================== step 1 : Cell reconstruction =====================================\n"
        )
        cell = reconstruct(cell, reflabels, fracs)
        tend = time.time()
        print(
            f"\nTotal execution time for Cell Reconstruction: {tend - tini:.2f} seconds"
        )

    elif step == 2:
        print("\n***Imprementing only Charge Assignment***")
        print("\nCell object loading by pickle")
        cellpath = os.path.join(output_dir, "Cell_{}.gmol".format(refcode))
        cell = load_cell_reset_charges (cellpath)
    else:
        print("Inproper step number")
        sys.exit(1)

    if not any(cell.warning_after_reconstruction):
        if step == 1 or step == 3:
            print("Cell reconstruction successfully finished.\n")
        elif step == 2:
            print("No Warnings in loaded Cell object in cell reconstruction \n")

        if step == 1:
            pass
        elif step == 2 or step == 3:
            # Charge Assignment
            tini_2 = time.time()
            print(
                "===================================== step 2 : Charge Assignment =======================================\n"
            )
            cell = determine_charge(cell)
            tend_2 = time.time()
            print(
                f"\nTotal execution time for Charge Assignment: {tend_2 - tini_2:.2f} seconds"
            )

            if not any(cell.warning_list):
                print("Charge Assignment successfully finished.\n")
                cell.print_charge_assignment()
                cell.print_Warning()
            else:
                print("Charge Assignment failed.\n")
                cell.print_Warning()
    else:
        if step == 1 or step == 3:
            print("Cell reconstruction failed.\n")
        elif step == 2:
            print("Warnings in loaded Cell object\n")
        cell.print_Warning()
        print("Cannot proceed step 2 Charge Assignment")

    return cell
