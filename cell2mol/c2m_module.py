#!/usr/bin/env python

import pickle
import time
import sys
import os
import copy
from typing import Tuple
import numpy as np
# Import modules
from cell2mol.cell_reconstruct import (getmolecs,identify_frag_molec_H,split_complexes_reassign_type,fragments_reconstruct,get_reference_molecules)#, compare_moleclist_refmoleclist)
from cell2mol.cell_reconstruct import get_reference_molecules_simple
from cell2mol.formal_charge import (drive_get_poscharges,classify_mols,balance_charge,build_bonds,prepare_mols,prepare_unresolved)
from cell2mol.missingH import check_missingH
from cell2mol.tmcharge_common import cell
from cell2mol.cellconversions import frac2cart_fromparam
from cell2mol.readwrite import readinfo
from cell2mol.spin import count_N, count_d_elec, assign_ground_state_spin_empirical, count_nitrosyl, calcualte_relative_metal_radius, calcualte_relative_metal_radius_haptic_complexes, generate_feature_vector
from typing import Tuple
import sklearn
from cell2mol import __file__
from cell2mol.elementdata import ElementData


elemdatabase = ElementData()

##################################################################################
def get_refmoleclist_and_check_missingH(cell: object, reflabels: list, fracs: list, debug: int=0) -> Tuple[object, float, float]:
    """Get a list of ref.molecules and check missing hydrogens in ref.molecules
    
    Args:
        cell (object): cell object
        reflabels (list): list of reference labels
        fracs (list): list of fractional coordinates
        debug (int, optional): debug level. Defaults to 0.

    Returns:
        Tuple[object, float, float]: cell object, covalent_factor, metal_factor        
    """

    refpos = frac2cart_fromparam(fracs, cell.cellparam)

    # Get ref.molecules --> output: a valid list of ref.molecules
    # refmoleclist, covalent_factor, metal_factor, Warning = get_reference_molecules(reflabels, refpos, debug=debug)
    refmoleclist, covalent_factor, metal_factor, Warning = get_reference_molecules_simple (reflabels, refpos, debug)
    cell.warning_list.append(Warning)

    # Check missing hydrogens in ref.molecules
    if not any(cell.warning_list):
        Warning, ismissingH, Missing_H_in_C, Missing_H_in_CoordWater = check_missingH(refmoleclist, debug)
        cell.warning_list.append(Missing_H_in_C)
        cell.warning_list.append(Missing_H_in_CoordWater)

    if not any(cell.warning_list):
        for mol in refmoleclist:
            mol.refcode = cell.refcode
            mol.name = str(cell.refcode + "_Reference_" + str(refmoleclist.index(mol)))

        cell.refmoleclist = refmoleclist

    return cell, covalent_factor, metal_factor


##################################################################################
def reconstruct(cell: object, reflabels: list, fracs: list, debug: int=0) -> object:

    # We start with an empty list of molecules
    moleclist = []
    Warning = False

    # Get a list of ref.molecules and Check missing H in ref.molecules
    cell, covalent_factor, metal_factor = get_refmoleclist_and_check_missingH(cell, reflabels, fracs, debug=debug)

    # Get blocks in the unit cells constructing the adjacency matrix (A)
    if not any(cell.warning_list):
        Warning, blocklist = getmolecs(cell.labels, cell.atom_coord, covalent_factor, metal_factor, atlist=[], debug=debug)
        cell.warning_list.append(Warning)

    # Indentify blocks and Reconstruct Fragments
    if not any(cell.warning_list):
        moleclist, fraglist, Hlist, init_natoms = identify_frag_molec_H(blocklist, moleclist, cell.refmoleclist, cell.cellvec, debug=debug)
        moleclist, finalmols, Warning = fragments_reconstruct(moleclist,fraglist,Hlist,cell.refmoleclist,cell.cellvec,covalent_factor,metal_factor,debug=debug)
        moleclist.extend(finalmols)
        
    # Split Complexes and Reassign Type
    if not any(cell.warning_list):

        cell = split_complexes_reassign_type(cell, moleclist, debug=debug)
  
        if debug >= 2: 
            print("Molecule Types assigned. These are:")
            for mol in moleclist:
                print(mol.formula, mol.type)
         
        if any(mol.type != "Complex" and mol.type != "Other" for mol in moleclist):
            Warning = True
            if debug >= 1: print(f"Fragment hasn't been fully reconstructed. Stopping")
               
    # Check final number of atoms after reconstruction   
    if not any(cell.warning_list):
        final_natoms = 0
        for mol in moleclist:
            final_natoms += mol.natoms

        if final_natoms != init_natoms:
            Warning = True
            if debug >= 1: print(f"Final and initial atoms do not coincide. Final/Initial {final_natoms}/{init_natoms}\n")

    cell.warning_list.append(Warning)
    cell.warning_after_reconstruction = copy.deepcopy(cell.warning_list)

    return cell


##################################################################################
def determine_charge(cell: object, debug: int=0) -> object:


    # Indentify unique chemical species
    molec_indices, ligand_indices, unique_indices, unique_species = classify_mols(cell.moleclist, debug=debug)
 
    # Group all unique species in a cell variable
    for spec in unique_species:            # spec is a list in which item 1 is the actual unique specie
        cell.speclist.append(spec[1])    

    if len(unique_species) == 0:
        if debug >= 1: print("Empty list of species found. Stopping")
        sys.exit()
    else:
        if debug >= 1: print(f"{len(unique_species)} Species (Ligand or Molecules) to Characterize")

    # drive_get_poscharges adds posible charges to the metal, ligand, and molecule objects of all species in the unit cell
    # also, it retrieves "Selected_charge_states", which is a tuple with [the actual charge state, and the protonation it belongs to] for all objects except metals
    selected_charge_states, Warning = drive_get_poscharges(unique_species, debug=debug)
    cell.warning_list.append(Warning)

    # Find possible charge distribution(s)
    if not any(cell.warning_list):
        final_charge_distribution = balance_charge(unique_indices,unique_species,debug=debug)

        ### DEALS WITH WARNINGS
        if debug >= 1: print("final_charge_distribution", final_charge_distribution)
        if len(final_charge_distribution) > 1:
            Warning = True
            if debug >= 1: print("More than one Possible Distribution Found:", final_charge_distribution)
        else:
            Warning = False
        cell.warning_list.append(Warning)

        if len(final_charge_distribution) == 0:
            Warning = True
            if debug >= 1: print("No valid Distribution Found", final_charge_distribution)
        else:
            Warning = False
        cell.warning_list.append(Warning)
        #######################

        if len(final_charge_distribution) > 1:
           pp_mols, pp_idx, pp_opt = prepare_unresolved(unique_indices,unique_species,final_charge_distribution, debug=debug)
           cell.data_for_postproc(pp_mols, pp_idx, pp_opt)

    # Only one possible charge distribution -> getcharge for the repeated species
    if not any(cell.warning_list):
        if debug >= 1:
            print(f"\nFINAL Charge Distribution: {final_charge_distribution}\n")
            print("#########################################")
            print("Assigning Charges and Preparing Molecules")
            print("#########################################")
        cell.moleclist, Warning = prepare_mols(cell.moleclist, unique_indices, unique_species, selected_charge_states, final_charge_distribution[0], debug=debug)
        cell.warning_list.append(Warning)

    # Build Bond objects for molecules
    if not any(cell.warning_list):
        cell.moleclist = build_bonds(cell.moleclist)

    return cell

##################################################################################
def save_cell(cell: object, ext: str, output_dir: str, debug: int=0):
    if debug >= 1: print("\n[Output files]")

    cellpath = os.path.join(output_dir, "Cell_{}.gmol".format(cell.refcode))
    with open(cellpath, "wb") as fil:
        if ext == "gmol":
            if debug >= 1: print("Output file path", cellpath)
            pickle.dump(cell, fil)
        else:
            if debug >= 1: print(ext, "not found as a valid print extension in print_molecule")

##################################################################################
def load_cell_reset_charges (cellpath: str, debug: int=0) -> object:
    
    file = open(cellpath, "rb")
    cell = pickle.load(file)
    if debug >= 1: print("[Refcode]", cell.refcode, cell)
    # if debug >= 1: print(f"{cell.moleclist=}")  
    if debug >= 1: print(f"{cell.warning_list=}")
    cell.warning_list = copy.deepcopy(cell.warning_after_reconstruction)
    if debug >= 1: print(f"{cell.warning_after_reconstruction=}")

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

##################################################################################
def assign_spin (cell: object, debug: int=0) -> object:
    """Assign spin multiplicity to molecules in the cell object
    
    Args:
        cell (object): cell object
        debug (int, optional): debug level. Defaults to 0.
    Returns:
        object: cell object with spin multiplicity assigned
    """
    
    if debug >= 1: 
        print("#########################################")
        print("Assigning spin multiplicity")
        print("#########################################")    

    for mol in cell.moleclist:
        # count number of electrons in the complex
        N = count_N(mol)        

        if mol.type == "Complex":
            if len(mol.metalist) == 1: # mono-metallic complexes
                met = mol.metalist[0]
                period = elemdatabase.elementperiod[met.label]
                d_elec = count_d_elec (met.label, met.totcharge)

                if period == 4: # 3d transition metals
                    if d_elec in [0, 1, 9, 10]:
                        if N % 2 == 0:
                            mol.magnetism(1) 
                        else:
                            mol.magnetism(2) 
                    elif d_elec in [2, 3] and met.hapticity == False :
                        if N % 2 == 0:
                            mol.magnetism(3) 
                        else:
                            mol.magnetism(4) 
                    elif d_elec in [4, 5, 6, 7, 8] or (d_elec in [2, 3] and met.hapticity == True) :
                        # Predict spin multiplicity of metal based on Random forest model
                        feature = generate_feature_vector (met)
                        path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), "total_spin_3131.pkl")
                        rf = pickle.load(open(path_rf, 'rb'))
                        predictions = rf.predict(feature)
                        spin_rf = predictions[0]
                        mol.magnetism(spin_rf)
                    else :
                        print("Error: d_elec is not in the range of 0-10", d_elec)

                    if met.hapticity == False :
                        rel = calcualte_relative_metal_radius (met)
                        met.relative_radius(rel, rel, rel)
                    else :
                        rel = calcualte_relative_metal_radius (met)
                        rel_g, rel_c = calcualte_relative_metal_radius_haptic_complexes (met)
                        met.relative_radius(rel, rel_g, rel_c)

                    for lig in mol.ligandlist:
                        if count_N(lig) %2 == 0:
                            lig.magnetism(1) 
                        else:
                            lig.magnetism(2) 

                    if debug >= 1: print(f"{mol.type=}, {mol.formula=}, {mol.spin=} {spin_rf=}")
                    if debug >= 1: print(f"{met.label=} {met.hapticity=} {met.hapttype=} {met.geometry=} {met.coordination_number=} {met.coordinating_atoms=}")
                
                #elif (period == 5 or period == 6 ) and (d_elec in [2, 3] and met.hapticity == False) :
                # TODO : Predict the ground state spin of coordination complexes with 4d or 5d transition metal (d2, d3)
                else :  # 4d or 5d transition metals     
                    if N % 2 == 0:
                        mol.magnetism(1) 
                    else:
                        mol.magnetism(2) 
                    
                    for lig in mol.ligandlist:
                        if count_N(lig) %2 == 0:
                            lig.magnetism(1) 
                        else:
                            lig.magnetism(2)                            
            
            else : # Bi- & Poly-metallic complexes
                if N % 2 == 0:
                    mol.magnetism(1) 
                else:
                    mol.magnetism(2) 
                
                for lig in mol.ligandlist:
                    if count_N(lig) %2 == 0:
                        lig.magnetism(1) 
                    else:
                        lig.magnetism(2) 

        else: # mol.type == "Other" 
            if N % 2 == 0:
                mol.magnetism(1) 
            else:
                mol.magnetism(2) 

    if debug >= 1: 
        for mol in cell.moleclist:
            if mol.type == "Complex":
                print(f"{mol.type=}, {mol.formula=}, {mol.spin=}")
                for lig in mol.ligandlist:
                    if lig.natoms != 1:
                        print(f"\t{lig.formula=}, {lig.spin=}")
                    else :
                        print(f"\t{lig.formula=}")
            else :
                if mol.natoms != 1:
                    print(f"{mol.type=}, {mol.formula=}, {mol.spin=}") 
                else :
                    print(f"{mol.type=}, {mol.formula=}")

    return cell

##################################################################################
def assign_spin_old (cell: object, debug: int=0) -> object:
    """Assign spin multiplicity to molecules in the cell object
    
    Args:
        cell (object): cell object
        debug (int, optional): debug level. Defaults to 0.
    Returns:
        object: cell object with spin multiplicity assigned
    """
    
    if debug >= 1: 
        print("#########################################")
        print("Assigning spin multiplicity")
        print("#########################################")    

    for mol in cell.moleclist:
        # count number of electrons in the complex
        N = count_N(mol)        

        if mol.type == "Complex":
            if len(mol.metalist) == 1: # mono-metallic complexes
                met = mol.metalist[0]

                # count valence electrons
                d_elec = count_d_elec(met.label, met.totcharge)

                # calculate relative metal radius
                rel = calcualte_relative_metal_radius(met)
                
                # Make a list of ligands
                arr = []
                for lig in mol.ligandlist:
                    arr.append(sorted(lig.labels))
                    if count_N(lig) %2 == 0:
                        lig.magnetism(1) 
                    else:
                        lig.magnetism(2)

                # Count nitrosyl ligands                               
                nitrosyl = count_nitrosyl(np.array(arr, dtype=object))
                if debug >= 2: print(np.array(arr, dtype=object))
                if debug >= 2: print(f"{nitrosyl=}")
                
                if met.hapticity == False: # coordination complexes
                    # Assign spin multiplicity of metal based on empirical rules
                    met.relative_radius(rel, rel, rel)
                    spin, rule, threshold  = assign_ground_state_spin_empirical(d_elec, met.totcharge, met.geometry, met.label, met.coordination_number, rel, N)
                    # spin = predict_ground_state_spin_v1 (met.label, elec, met.coordinating_atoms, met.geometry, met.coordination_number, N, nitrosyl)

                    # Predict spin multiplicity of metal based on Random forest model
                    feature = generate_feature_vector (met)
                    print(feature)
                    path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), "TM-GSspin_RandomForest.pkl")

                    #print(path_rf)
                    rf = pickle.load(open(path_rf, 'rb'))
                    predictions = rf.predict(feature)
                    spin_rf = predictions[0]
                    #if debug >= 1: print(f"{spin=} {spin_rf=}")
                    mol.ml_prediction(spin_rf)

                    if spin == 0 : # unknown spin state
                        if N % 2 == 0:
                            mol.magnetism(1) 
                        else:
                            mol.magnetism(2) 
                    else:
                        mol.magnetism(spin_rf)
                        #mol.magnetism(spin)

                    if debug >= 1: print(f"{mol.type=}, {mol.formula=}, {mol.spin=} {mol.spin_rf=}")
                    if debug >= 1: print(f"{met.label=} {met.hapticity=} {met.geometry=} {met.coordination_number=} {met.coordinating_atoms=}")
                    if debug >= 1: print(f"met_OS={met.totcharge} {d_elec=} {N=} {nitrosyl=} {met.rel=}\n")

                else: # hapticity == True 
                    rel_g, rel_c = calcualte_relative_metal_radius_haptic_complexes(met)
                    #if debug >= 1: print(f"{rel_g=} {rel_c=}")
                    met.relative_radius(rel, rel_g, rel_c)
                    
                    if N % 2 == 0:
                        mol.magnetism(1) # spin multiplicity = 1 Singlet
                    else:
                        mol.magnetism(2) # spin multiplicity = 2 Doublet

                    if debug >= 1: print(f"{mol.type=}, {mol.formula=}, {mol.spin=}")
                    if debug >= 1: print(f"{met.label=} {met.hapticity=} {met.geometry=} {met.coordination_number=} {met.coordinating_atoms=}")
                    if debug >= 1: print(f"met_OS={met.totcharge} {d_elec=} {N=} {nitrosyl=} {met.rel=} {met.rel_g=} {met.rel_c=}\n")

            else : # Bi- & Poly-metallic complexes
                if N % 2 == 0:
                    mol.magnetism(1) 
                else:
                    mol.magnetism(2) 
                
                for lig in mol.ligandlist:
                    if count_N(lig) %2 == 0:
                        lig.magnetism(1) 
                    else:
                        lig.magnetism(2) 

        else: # mol.type == "Other" or "Ligand"
            if N % 2 == 0:
                mol.magnetism(1) 
            else:
                mol.magnetism(2) 

    if debug >= 1: 
        for mol in cell.moleclist:
            if mol.type == "Complex":
                print(f"{mol.type=}, {mol.formula=}, {mol.spin=}")
                for lig in mol.ligandlist:
                    if lig.natoms != 1:
                        print(f"\t{lig.formula=}, {lig.spin=}")
                    else :
                        print(f"\t{lig.formula=}")
            else :
                if mol.natoms != 1:
                    print(f"{mol.type=}, {mol.formula=}, {mol.spin=}") 
                else :
                    print(f"{mol.type=}, {mol.formula=}")

    return cell

##################################################################################
################################## MAIN ##########################################
##################################################################################
def cell2mol(infopath: str, refcode: str, output_dir: str, step: int=3, debug: int=1) -> object:

    if step == 1 or step == 3:

        tini = time.time()

        # Read reference molecules from info file
        labels, pos, ref_labels, ref_fracs, cellvec, cellparam = readinfo(infopath)

        # Initialize cell object
        warning_list = []
        newcell = cell(refcode, labels, pos, cellvec, cellparam, warning_list)
        if debug >= 1: print("[Refcode]", newcell.refcode)

        # Cell Reconstruction
        if debug >= 1: print("===================================== step 1 : Cell reconstruction =====================================\n")
        newcell = reconstruct(newcell, ref_labels, ref_fracs, debug=debug)
        tend = time.time()
        if debug >= 1: print(f"\nTotal execution time for Cell Reconstruction: {tend - tini:.2f} seconds")

    elif step == 2:
        if debug >= 1: print("\n***Runing only Charge Assignment***")
        if debug >= 1: print("\nCell object loaded with pickle")
        cellpath = os.path.join(output_dir, "Cell_{}.gmol".format(refcode))
        newcell = load_cell_reset_charges(cellpath)
    else:
        if debug >= 1: print("Step number is incorrect. Only values 1, 2 or 3 are accepted")
        sys.exit(1)

    if not any(newcell.warning_after_reconstruction):
        if step == 1 or step == 3:
            if debug >= 1: print("Cell reconstruction successfully finished.\n")
        elif step == 2:
            if debug >= 1: print("No Warnings in loaded Cell object in cell reconstruction \n")

        if step == 1:
            pass
        elif step == 2 or step == 3:
            # Charge Assignment
            tini_2 = time.time()
            if debug >= 1: print("===================================== step 2 : Charge Assignment =======================================\n")
            newcell = determine_charge(newcell, debug=debug)
            tend_2 = time.time()
            if debug >= 1: print(f"\nTotal execution time for Charge Assignment: {tend_2 - tini_2:.2f} seconds")

            if not any(newcell.warning_list):
                if debug >= 1: print("Charge Assignment successfully finished.\n")
                # TODO : Compare assigned charges with ML predicted charges

                # Spin state assignmentc
                newcell = assign_spin(newcell, debug=debug)

                if debug >= 1: newcell.print_charge_assignment()
                if debug >= 1: newcell.print_Warning()
            else:
                if debug >= 1: print("Charge Assignment failed.\n")
                if debug >= 1: newcell.print_Warning()
    else:
        if step == 1 or step == 3:
            if debug >= 1: print("Cell reconstruction failed.\n")
        elif step == 2:
            if debug >= 1: print("Warnings in loaded Cell object\n")
        if debug >= 1: newcell.print_Warning()
        if debug >= 1: print("Cannot proceed step 2 Charge Assignment")
   
    if not any(newcell.warning_list): newcell.arrange_cell_coord()

    return newcell
