#!/usr/bin/env python

import numpy as np  
from cell2mol.elementdata import ElementData
from cell2mol.connectivity import *
from collections import defaultdict
import itertools
import sys
from cell2mol.hungarian import reorder
from cell2mol.xyz2mol import xyz2mol

from cell2mol.classes import specie, molecule, ligand, atom, metal, group, bond

elemdatabase = ElementData()

#############################
### Loads Rdkit & xyz2mol ###
#############################

from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions  # Only needed if modifying defaults
DrawingOptions.bondLineWidth = 2.2

# IPythonConsole.ipython_useSVG = False
from rdkit import rdBase
if "ipykernel" in sys.modules:
    try:
        from rdkit.Chem.Draw import IPythonConsole
    except ModuleNotFoundError:
        pass
# print("RDKIT Version:", rdBase.rdkitVersion)
rdBase.DisableLog("rdApp.*")

###############################################################################
def identify_unique_species(moleclist: list, debug: int=0):
    
    unique_species = []
    unique_indices = []
    
    typelist_mols = []
    typelist_ligs = []
    typelist_mets = []
    
    specs_found = -1    
    for idx, mol in enumerate(moleclist):
        if debug >= 2: print(f"Molecule {idx} formula={mol.formula}")
        if not mol.iscomplex:
            found = False
            for ldx, typ in enumerate(typelist_mols):   # Molecules
                issame = compare_species(mol, typ[0], debug=1)
                if issame : 
                    found = True ; kdx = typ[1] 
                    if debug >= 2: print(f"Molecule {idx} is the same with {ldx} in typelist")      
            if not found:
                specs_found += 1 ; kdx = specs_found
                typelist_mols.append(list([mol, kdx]))
                unique_species.append(mol)
                if debug >= 2: print(f"New molecule found with: formula={mol.formula} and added in position {kdx}")
            unique_indices.append(kdx)
        
        else:
            for jdx, lig in enumerate(mol.ligands):     # ligands
                found = False
                for ldx, typ in enumerate(typelist_ligs):
                    issame = compare_species(lig, typ[0], debug=1)
                    if issame : 
                        found = True ; kdx = typ[1] 
                        if debug >= 2: print(f"ligand {jdx} is the same with {ldx} in typelist")    
                if not found:
                    specs_found += 1 ; kdx = specs_found
                    typelist_ligs.append(list([lig, kdx]))
                    unique_species.append(lig)
                    if debug >= 2: print(f"New ligand found with: formula {lig.formula} and denticity={lig.denticity}, and added in position {kdx}")
                unique_indices.append(kdx)
        
            for jdx, met in enumerate(mol.metals):      #  metals
                found = False
                for ldx, typ in enumerate(typelist_mets):
                    issame = compare_metals(met, typ[0], debug=1)
                    if issame : 
                        found = True ; kdx = typ[1] 
                        if debug >= 2: print(f"Metal {jdx} is the same with {ldx} in typelist")     
                if not found:
                    specs_found += 1 ; kdx = specs_found
                    typelist_mets.append(list([met, kdx]))
                    unique_species.append(met)
                    if debug >= 2: print(f"New Metal Center found with: labels {met.label} and added in position {kdx}")
                unique_indices.append(kdx)
            
    return unique_species, unique_indices

#######################################################
def get_poscharges(spec: object, debug: int=0): 
    if not hasattr(spec,"protonation_states"): spec.get_protonation_states(debug=debug)
    if spec.protonation_states is None:                                             return None
    if spec.subtype == "group" or (spec.subtype == 'molecule' and spec.is_complex): return None

    selected_charge_states = []
    ##############################
    #### Evaluates possible charges 
    ##############################
    for prot in spec.protonation_states:
        charge_states = []
        target_charges = get_list_of_charges_to_try(spec, prot)
        if debug >= 2: print(f"    POSCHARGE will try charges {target_charges}") 

        for ich in target_charges:
            ch_state = get_charge(ich, prot)
            ch_state.correction(prot)
            charge_states.append(ch_state)
            #list_of_protonations_for_each_state.append(prot)
            if debug >= 2: print(f"    POSCHARGE: charge 0 with smiles {ch_state.smiles}")

        if spec.subtype == "ligand":
            if spec.is_nitrosyl:
                if   spec.NO_type == "Linear": best_charge_state = charge_states[2]      ## When Nitrosyl, we sistematically get the correct charge_distribution in [2] and [0] for Linear and Bent respectively
                elif spec.NO_type == "Bent":   best_charge_state = charge_states[0]
            else: best_charge_state = select_charge_distr(charge_states, debug=debug)     ## For ligands other than nitrosyl
        else:     best_charge_state = select_charge_distr(charge_states, debug=debug)     ## For organic molecules

        #############################
        # For all protonations, it adds the resulting states to selected_charge_states
        #############################
        found = False
        for cs in spec.possible_cs:
            if cs.corr_total_charge == best_charge_state.corr_total_charge: found = True  
            if not found:                                                       ## IMPORTANT. We only add possible states if the possible charge is already not considered for the specie.  
                selected_charge_states.append(best_charge_state)                                      ## That is, we only consider one possible connectivity for each possible charge

    ### HERE IS HAS FINISHED WITH ALL PROTONATIONS
    if len(selected_charge_states) == 0:    return None
    else:                                   return selected_charge_states

#######################################################
def select_charge_distr(charge_states: list, debug: int=0) -> list:
    # This function selects the best charge_distribuion among the ones generated by the function "getcharge"
    # It does so based, in general, on the number of charges in each connectivity.
    #:return goodlist: list of acceptable charge distributions.
    # goodlist contains the indices of those distributions as the enter this function

    nlists = len(charge_states)
    uncorr_total = []
    uncorr_abs_total = []
    uncorr_abs_atcharge = []
    uncorr_zwitt = []
    coincide = []
    for chs in charge_states:
        uncorr_total.append(chs.uncorr_total_charge)
        uncorr_abs_total.append(chs.uncorr_abstotal)
        uncorr_abs_atcharge.append(chs.uncorr_abs_atcharge)
        uncorr_zwitt.append(chs.uncorr_zwitt)
        coincide.append(chs.coincide)

    if debug >= 2: print(f"    NEW SELECT FUNCTION: uncorr_total: {uncorr_total}")
    if debug >= 2: print(f"    NEW SELECT FUNCTION: uncorr_abs_total: {uncorr_abs_total}")
    if debug >= 2: print(f"    NEW SELECT FUNCTION: uncorr_abs_atcharge: {uncorr_abs_atcharge}")
    if debug >= 2: print(f"    NEW SELECT FUNCTION: uncorr_zwitt: {uncorr_zwitt}")
    if debug >= 2: print(f"    NEW SELECT FUNCTION: coincide: {coincide}")

    minoftot = np.min(uncorr_abs_total)
    minofabs = np.min(uncorr_abs_atcharge)
    listofmintot = [i for i, x in enumerate(uncorr_abs_total) if x == minoftot]
    listofminabs = [i for i, x in enumerate(uncorr_abs_atcharge) if x == minofabs]
    if debug >= 2: print(f"    NEW SELECT FUNCTION: listofmintot: {listofmintot}")
    if debug >= 2: print(f"    NEW SELECT FUNCTION: listofminabs: {listofminabs}")
    # Searches for entries that have the smallest total charge(appear in listofmintot),
    # and smallest number of charges(appear in listofminabs)

    ####################
    # building tmplist #
    ####################
    tmplist = []
    for idx in range(0, nlists):
        if (idx in listofminabs) and (idx in listofmintot) and coincide[idx]:
            tmplist.append(idx)

    # IF listofminabs and listofmintot do not have any value in common. Then we select from minima, coincide, and zwitt
    if len(tmplist) == 0:
        if debug >= 2: print("    NEW SELECT FUNCTION: No entry in initial tmplist. We now select from minima, coincide and zwitt:")
        for idx in range(0, nlists):
            if ((idx in listofminabs) or (idx in listofmintot)) and coincide[idx] and not uncorr_zwitt[idx]:
                tmplist.append(idx)

    # IF no values yet, we relax the criterion 
    if len(tmplist) == 0: 
        if debug >= 2: print("    NEW SELECT FUNCTION: No entry in initial tmplist yet. We now select from minima and coincide:")
        for idx in range(0, nlists):
            if ((idx in listofminabs) or (idx in listofmintot)) and coincide[idx]: 
                tmplist.append(idx)

    # IF no values yet, we relax the criterion even more 
    if len(tmplist) == 0: 
        if debug >= 2: print("    NEW SELECT FUNCTION: No entry in initial tmplist yet. We now select from minima:")
        for idx in range(0, nlists):
            if ((idx in listofminabs) or (idx in listofmintot)): 
                tmplist.append(idx)
 
    ####################
    # tmplist is built #
    ####################
    if debug >= 2: print(f"    NEW SELECT FUNCTION: tmplist: {tmplist}, including:")
    for idx in range(0, nlists):
        if idx in tmplist: 
            if debug >= 2: print(f"    NEW SELECT FUNCTION: Corr_charge={charge_states[idx].corr_total_charge}")
            if debug >= 2: print(f"    NEW SELECT FUNCTION: Smiles={charge_states[idx].smiles}")

    corr_charges = []
    for idx in range(0, nlists):
        if idx in tmplist: 
            if charge_states[idx].corr_total_charge not in corr_charges: 
                corr_charges.append(charge_states[idx].corr_total_charge)
    if debug >= 2: print(f"    NEW SELECT FUNCTION: found corr_charges={corr_charges}")
            
    good_idx = []
    for jdx, tgt_charge in enumerate(corr_charges): 
        if debug >= 2: print(f"    NEW SELECT FUNCTION: doing tgt_charge={tgt_charge}")
        list_for_tgt_charge = []
        for idx in tmplist: 
            if charge_states[idx].corr_total_charge == tgt_charge: 
                list_for_tgt_charge.append(idx)
                if debug >= 2: print(f"    NEW SELECT FUNCTION: charge_state added")
             
        # CASE 1, IF only one distribution meets the requirement. Then it is chosen
        if len(list_for_tgt_charge) == 1:
            good_idx.append(list_for_tgt_charge[0])
            if debug >= 2: print(f"    NEW SELECT FUNCTION: Case 1, only one entry for {tgt_charge} in tmplist")
 
        # CASE 2, IF more than one charge_state is found for a given final charge
        elif len(list_for_tgt_charge) > 1:
            good_idx.append(list_for_tgt_charge[0])
            if debug >= 2: print(f"    NEW SELECT FUNCTION: Case 2, more than one entry for {tgt_charge} in tmplist. Taking first")

    return good_idx

#######################################################
def get_protonation_states(specie: object, debug: int=0) -> list:
    ##############################
    #### Creates protonation states. That is, geometries in which atoms have been added to the original molecule
    ##############################
    if   specie.type != "specie":                                   return None
    if   specie.subtype == "group":                                 return None
    elif specie.subtype == "molecule" and specie.iscomplex == True: return None
    elif specie.subtype == "molecule" and specie.iscomplex == False: 
        if debug >= 2: print(f"    POSCHARGE: doing empty PROTONATION for this specie")
        empty_protonation = protonation(specie.labels, specie.coord, specie.cov_factor, int(0), [], [], [], [], typ="Empty", parent=specie)
        return list(empty_protonation)

    ## If specie.subtype == "ligand": 
    ligand = specie      ## Change the variable name as it is easier to follow
    protonation_states = []

    natoms   = ligand.natoms
    newlab   = ligand.labels.copy()
    newcoord = ligand.coord.copy()

    # Variables that control how many atoms have been added.
    tmp_added_atoms = 0
    added_atoms = 0

    # Boolean that decides whether a non-local approach is needed
    non_local_groups = 0
    needs_nonlocal = False

    # Initialization of the variables
    addedlist       = np.zeros((natoms)).astype(int)
    block           = np.zeros((natoms)).astype(int)
    metal_electrons = np.zeros((natoms)).astype(int)  # It will remain as such
    elemlist        = np.empty((natoms)).astype(str)

    # Program runs sequentially for each group of the ligand
    for g in ligand.groups:

        ########################
        # Cases with Hapticity #
        ########################
        if g.is_haptic:
            Selected_Hapticity = False
            if debug >= 2: print("        GET_PROTONATION_STATES: addressing group with hapticity:", g.haptic_type)
            if debug >= 2: print("        GET_PROTONATION_STATES: and parent indices:", g.parent_indices)

            if "h5-Cp" in g.haptic_type and not Selected_Hapticity:
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else: block[idx] = 1

            elif "h7-Cicloheptatrienyl" in g.haptic_type and not Selected_Hapticity:
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else: block[idx] = 1

            elif "h5-AsCp" in g.haptic_type and not Selected_Hapticity:
                Selected_Hapticity = True

                # Rules change depending on whether the ring is substituted or not 
                issubstituted = False
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        for jdx in a.adjacency:
                            if ligand.labels[jdx] != "As":
                                issubstituted = True
                if issubstituted:  tobeadded = 0
                else:              tobeadded = 1

                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif "h5-Pentaphosphole" in g.haptic_type and not Selected_Hapticity: ## Case of IMUCAX
                Selected_Hapticity = True

                # Rules change depending on whether the ring is substituted or not 
                issubstituted = False
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        for jdx in a.adjacency:
                            if ligand.labels[jdx] != "P":
                                issubstituted = True
                if issubstituted: 
                    tobeadded = 0
                else: 
                    tobeadded = 1

                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif ("h3-Allyl" in g.haptic_type or "h3-Cp" in g.haptic_type) and not Selected_Hapticity:
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif ("h4-Benzene" in g.haptic_type or "h4-Butadiene" in g.haptic_type) and not Selected_Hapticity:
                if debug >= 2: print("        GET_PROTONATION_STATES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        block[idx] = 1

            elif ("h2-Benzene" in g.haptic_type or "h2-Butadiene" or "h2-ethylene" in g.haptic_type) and not Selected_Hapticity:
                if debug >= 2: print("        GET_PROTONATION_STATES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        block[idx] = 1

            elif "h4-Enone" in g.haptic_type and not Selected_Hapticity:
                if debug >= 2: print("        GET_PROTONATION_STATES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        block[idx] = 1

            # If the group hapticity type is not recognized -or instructions are not defined-, nothing is done
            if not Selected_Hapticity:
                if debug >= 2: print(f"        GET_PROTONATION_STATES: {g.haptic_type} not recognized or new rules are necessary")

        else:  # cases without hapticity
            ions = ["F", "Cl","Br","I","As"]  # Atoms for which an H atom is always added
            ###########################
            # Cases with No Hapticity #
            ###########################
            # An initial attempt to add elements based on the adjacency of the connected atom
            for idx in g.parent_indices:
                a = ligand.atoms[idx]
                if debug >= 2: print(f"        GET_PROTONATION_STATES: evaluating non-haptic group with index {idx} and label {a.label}")
                # Simple Ionic Case
                if a.label in ions:
                    if a.connec == 0:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    elif a.connec >= 1:
                        block[idx] = 1
                # Oxygen
                elif a.label == "O" :
                    if a.connec == 1:
                        needs_nonlocal = True
                        non_local_groups += 1
                        if debug >= 2: print(f"        GET_PROTONATION_STATES: will be sent to nonlocal due to {a.label} atom")
                    elif a.connec > 1:
                        block[idx] = 1
                # Sulfur and Selenium
                elif a.label == "S" or a.label == "Se":
                    if a.connec == 1:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    elif a.connec > 1:
                        block[idx] = 1
                # Hydrides
                elif a.label == "H":
                    if a.connec == 0:
                        elemlist[idx] = "Cl"
                        addedlist[idx] = 1
                    else:
                        block[idx] = 1
                # Nitrogen
                elif a.label == "N":
                    # Nitrosyl
                    if ligand.natoms == 2 and ligand.is_nitrosyl:
                        if ligand.NO_type == "Linear":
                            if debug >= 2: print("        GET_PROTONATION_STATES: Found Linear Nitrosyl")
                            elemlist[idx] = "O"
                            addedlist[idx] = 2
                            metal_electrons[idx] = 1
                        elif ligand.NO_type == "Bent":
                            if debug >= 2: print("        GET_PROTONATION_STATES: Found Bent Nitrosyl")
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                    else:
                        # nitrogen with at least 3 adjacencies doesnt need H
                        if a.connec >= 3: block[idx] = 1
                        else:
                            # Checks for adjacent Atoms
                            list_of_adj_atoms = []
                            for i in a.adjacency:
                                list_of_adj_atoms.append(ligand.labels[i])
                            numN = list_of_adj_atoms.count("N")
                            if numN == 2:  # triazole or tetrazole
                                elemlist[idx] = "H"
                                addedlist[idx] = 1
                            else:
                                needs_nonlocal = True
                                non_local_groups += 1
                                if debug >= 2: print(f"        GET_PROTONATION_STATES: will be sent to nonlocal due to {a.label} atom")
                # Phosphorous
                elif (a.connec == 3) and a.label == "P": block[idx] = 1
                # Case of Carbon (Simple CX vs. Carbenes)
                elif a.label == "C":
                    if ligand.natoms == 2:
                        # CN
                        if "N" in ligand.labels:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                        # CO
                        if "O" in ligand.labels: block[idx] = 1
                    # Added for amides
                    elif (any(ligand.labels[i] == "O" for i in a.adjacency) and any(ligand.labels[i] == "N" for i in a.adjacency) and a.connec == 2 ):
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        iscarbene, tmp_element, tmp_added, tmp_metal = check_carbenes(a, ligand, ligand.parent)
                        if debug >= 2: print(f"        GET_PROTONATION_STATES: Evaluating as carbene and {iscarbene}")
                        if iscarbene:
                            # Carbene identified
                            elemlist[idx] = tmp_element
                            addedlist[idx] = tmp_added
                            metal_electrons[idx] = tmp_metal
                        else:
                            needs_nonlocal = True
                            non_local_groups += 1
                            if debug >= 2: print(f"        GET_PROTONATION_STATES: will be sent to nonlocal due to {a.label} atom")
                # Silicon
                elif a.label == "Si":
                    if a.connec < 4:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else: block[idx]
                # Boron
                elif a.label == "B":
                    if a.connec < 4:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else: block[idx]
                # None of the previous options
                else:
                    if not needs_nonlocal:
                        needs_nonlocal = True
                        non_local_groups += 1
                        if debug >= 2: print(f"        GET_PROTONATION_STATES: will be sent to nonlocal due to {a.label} atom with no rules")

        # If, at this stage, we have found that any atom must be added, this is done before entering the non_local part.
        # The block variable makes that more atoms cannot be added to these connected atoms
        for idx, a in enumerate(ligand.atoms):
            if addedlist[idx] != 0 and block[idx] == 0:
                isadded, newlab, newcoord = add_atom(newlab, newcoord, idx, ligand, ligand.molecule.metals, elemlist[idx])
                if isadded:
                    added_atoms += addedlist[idx]
                    block[idx] = 1  # No more elements will be added to those atoms
                    if debug >= 2: print(f"        GET_PROTONATION_STATES: Added {elemlist[idx]} to atom {idx} with: a.mconnec={a.mconnec} and label={a.label}")
                else:
                    addedlist[idx] = 0 
                    block[idx] = 1  # No more elements will be added to those atoms
                   
    ############################
    ###### NON-LOCAL PART ######
    ############################
    
    if not needs_nonlocal:
        new_prot = protonation(newlab, newcoord, ligand.cov_factor, added_atoms, addedlist, block, metal_electrons, elemlist, parent=specie) 
        protonation_states.append(new_prot)
    else:
        # Generate the new adjacency matrix after local elements have been added to be sent to xyz2mol
        local_labels = newlab.copy()
        local_coords  = newcoord.copy()
        local_radii = get_radii(local_labels)
        local_natoms = len(local_labels)
        #local_atnums = [int_atom(label) for label in local_labels]  # from xyz2mol.py
        dummy, local_adjmat, local_adjnum = get_adjmatrix(local_labels, local_coords, ligand.cov_factor, local_radii)

        local_addedlist = addedlist.copy()
        local_block = block.copy()
        local_added_atoms = added_atoms

        # Initiate variables
        avoid = ["Si", "P"]

        if debug >= 2: print(" ")
        if debug >= 2: print(f"        GET_PROTONATION_STATES: Enters non-local with:")
        if debug >= 2: print(f"        GET_PROTONATION_STATES: block: {block}")
        if debug >= 2: print(f"        GET_PROTONATION_STATES: addedlist: {addedlist}")
        if debug >= 2: print(f"        GET_PROTONATION_STATES: {non_local_groups} non_local_groups groups found") 

        # CREATES ALL COMBINATIONS OF PROTONATION STATES# 
        # Creates [0,1] tuples for each non_local protonation site
        tmp = []
        for kdx in range(0,non_local_groups):
            tmp.append([0,1])
        
        if len(tmp) > 1:
            combinations = list(itertools.product(*tmp))
            combinations.sort(key=sum)
        else:
            combinations = [0,1]

        for com in combinations:
            newlab = local_labels.copy()
            newcoord = local_coords.copy()
            if debug >= 2:  print(f" ") 
            if debug >= 2:  print(f"        GET_PROTONATION_STATES: doing combination {com}") 
            metal_electrons = np.zeros((local_natoms)).astype(int)  ## Electrons Contributed to the Metal
            elemlist = np.empty((local_natoms)).astype(str)
            # block and addedlist are inherited from LOCAL
            addedlist = local_addedlist.copy()
            block = local_block.copy()
            added_atoms = local_added_atoms
            non_local_added_atoms = 0

            os = np.sum(com)
            toallocate = int(0)
            for jdx, a in enumerate(ligand.atoms):
                if a.mconnec >= 1 and a.label not in avoid and block[jdx] == 0:
                    if non_local_groups > 1:
                        if com[toallocate] == 1:
                            elemlist[jdx] = "H"
                            addedlist[jdx] = 1
                            isadded, newlab, newcoord = add_atom(newlab, newcoord, jdx, ligand, ligand.parent.metals, elemlist[jdx])
                            if isadded:
                                added_atoms += addedlist[jdx]
                                if debug >= 2: print(f"        GET_PROTONATION_STATES: Added {elemlist[jdx]} to atom {jdx} with: a.mconnec={a.mconnec} and label={a.label}")
                            else:
                                addedlist[idx] = 0 
                                block[idx] = 1  # No more elements will be added to those atoms
                    elif non_local_groups == 1:
                        if com == 1:
                            elemlist[jdx] = "H"
                            addedlist[jdx] = 1
                            isadded, newlab, newcoord = add_atom(newlab, newcoord, jdx, ligand, ligand.parent.metals, elemlist[jdx])
                            if isadded:
                                added_atoms += addedlist[jdx]
                                if debug >= 2: print(f"        GET_PROTONATION_STATES: Added {elemlist[jdx]} to atom {jdx} with: a.mconnec={a.mconnec} and label={a.label}")
                            else:
                                addedlist[idx] = 0 
                                block[idx] = 1  # No more elements will be added to those atoms
                    #in any case, moves index
                    toallocate += 1

            smi = " "
        
            new_prot = protonation(newlab, newcoord, ligand.cov_factor, added_atoms, addedlist, block, metal_electrons, elemlist, smi, os, typ="Non-local", parent=specie) 
            if new_prot.status == 1 and new_prot.added_atoms == os+local_added_atoms:
                protonation_states.append(new_prot)
                if debug >= 2:  print(f"        GET_PROTONATION_STATES: Protonation SAVED with {added_atoms} atoms added to ligand. status={new_prot.status}")
            else:
                if debug >= 2:  print(f"        GET_PROTONATION_STATES: Protonation DISCARDED. Steric Clashes found when adding atoms. status={new_prot.status}")
                
    return protonation_states 

#######################################################
def get_charge(ich: int, prot: object, allow: bool=True, debug: int=0): 
    ## Generates the connectivity of a molecule given a desired charge (ich).
    # The molecule is described by a protonation states that has labels, and the atomic cartesian coordinates "coords"
    # The adjacency matrix is also provided in the protonation state(adjmat)
    #:return charge_state which is an object with the necessary information for other functions to handle the result

    pt = Chem.GetPeriodicTable()  # needed to retrieve the default valences in the 2nd and 3rd checks
    natoms = prot.natoms
    atnums = prot.atnums

    ##########################
    # xyz2mol is called here #
    ##########################
    # use_graph is called for a faster generation
    # allow_charged_fragments is necessary for non-neutral molecules
    # embed_chiral shouldn't ideally be necessary, but it runs a sanity check that improves the proposed connectivity
    # use_huckel false means that the xyz2mol adjacency will be generated based on atom distances and vdw radii.
    # instead of use_huckel, we provide the adjacency matrix 

    mols = xyz2mol(atnums,prot.coords,prot.adjmat,prot.cov_factor,charge=ich,use_graph=True,allow_charged_fragments=allow,embed_chiral=True,use_huckel=False)
    if len(mols) > 1: print("WARNING: More than 1 mol received from xyz2mol for initcharge:", ich)

    # Smiles are generated with rdkit
    smiles = Chem.MolToSmiles(mols[0])

    # Gets the resulting charges
    atom_charge = []
    total_charge = 0
    for i in range(natoms):
        a = mols[0].GetAtomWithIdx(i)  # Returns a particular Atom
        atom_charge.append(a.GetFormalCharge())
        total_charge += a.GetFormalCharge()

    # Connectivity is checked
    iscorrect = check_rdkit_mol_connectivity(mols[0], prot.natoms, debug=debug)

    # Charge_state is initiated
    ch_state = charge_state(iscorrect, total_charge, atom_charge, mols[0], smiles, ich, allow)

    return ch_state

def check_rdkit_mol_connectivity(mol: object, natoms: int, debug: int=0): 
    # Here, the atom charge is retrieved, and the connectivity of each atom goes through 3 checks.
    # The variable iscorrect will track whether the overall generated structure is meaningful
    iscorrect = True
    for i in range(natoms):
        a = mols[0].GetAtomWithIdx(i)  # Returns a particular Atom
        valence = a.GetTotalValence()  # Valence of the atom in the mol object
        bonds = 0
        countaromatic = 0
        for b in a.GetBonds():  # Returns a read-only sequence containing all of the molecule' bonds
            bonds += b.GetBondTypeAsDouble()
            # total number of bonds (weighted by bond order) of the atom in the mol object
            # Returns the type of the bond as a double (i.e. 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE)
            if b.GetBondTypeAsDouble() == 1.5:
                countaromatic += 1
        if countaromatic % 2 != 0:
            bonds -= 0.5

        lonepairs = (elemdatabase.valenceelectrons[a.GetSymbol()] - a.GetFormalCharge() - valence) / 2
        totalvalenceelectrons = int(bonds) + int(lonepairs) * 2 + a.GetFormalCharge()

        # Checks the quality of the resulting smiles
        # First check, the number of lonepairs is computed and should make sense
        if lonepairs != 0 and lonepairs != 1 and lonepairs != 2 and lonepairs != 3 and lonepairs != 4:
            if debug >= 2: print("GETCHARGE: 2nd Check-lonepairs=", i, a.GetSymbol(), lonepairs)
            iscorrect = False

        # RDKIT has some troubles assigning the valence for atoms with aromatic bonds.
        # So the 2nd and 3rd Check applies only for countaromatic==0
        if countaromatic == 0:
            # Second check, the number of bonds should coincide with the valence.
            # I know it should be the same, but in bad SMILES they often do not coincide
            if bonds != valence:
                if debug >= 2: print("GETCHARGE: 1st Check-bonds/valence:",i,a.GetSymbol(),bonds,valence); iscorrect = False
                if debug >= 2:
                    for b in a.GetBonds():
                        print(b.GetBondTypeAsDouble(),b.GetBeginAtomIdx(),b.GetEndAtomIdx())

            # Third check, using the totalvalenceelectrons
            if totalvalenceelectrons != elemdatabase.valenceelectrons[a.GetSymbol()]:
                if debug >= 2: print("GETCHARGE: 3rd Check: Valence gives false for atom",i,a.GetSymbol(),"with:",totalvalenceelectrons,elemdatabase.valenceelectrons[a.GetSymbol()])
                iscorrect = False

    if debug >= 2 and i == 0: print("ich, atom idx, label, charge, pt.GetDefaultValence(a.GetAtomicNum()), valence, num bonds, num lonepairs, iscorrect")
    if debug >= 2:            print(ich,i,a.GetSymbol(),a.GetFormalCharge(),pt.GetDefaultValence(a.GetAtomicNum()),valence,int(bonds),int(lonepairs),iscorrect)

    return iscorrect

#######################################################
def get_list_of_charges_to_try(prot: object, debug: int=0) -> list:
    ### Determines which charges are worth trying for a given specie and a protonation state
    lchar = []
    spec = prot.parent

    #### Educated Guess on the Maximum Charge one can expect from the spec[1]
    if   spec.subtype == "molecule":  maxcharge = 3
    elif spec.subtype == "ligand":  
        count_non_connected_O = 0
        for a in spec.atoms:
            if a.label == "O" and a.mconnec == 0 and a.connec == 1:
                count_non_connected_O += 1
        if not spec.is_haptic:
            maxcharge = spec.denticity + count_non_connected_O - prot.added_atoms
            if debug >= 2: print(f"MAXCHARGE: maxcharge set at {maxcharge} with {spec.denticity}+{count_non_connected_O}-{prot.added_atoms}")
        else: maxcharge = 2

        # Cases of same atom being connected to more than one metal
        if any(a.mconnec >= 2 for a in spec.atoms): pass
        else:                                       if maxcharge > spec.natoms: maxcharge = spec.natoms
        if maxcharge > 4: maxcharge = 4  ## At most, we try range(-4,5,1)
        if maxcharge < 2: maxcharge = 2  ## At leaest, we try range(-2,3,1)
    if debug >= 2: print(f"MAXCHARGE: maxcharge set at {maxcharge}")
    
    # Defines list of charges that will try
    for magn in range(0, int(maxcharge + 1)):
        if magn == 0:    signlist = [1]
        elif magn != 0:  signlist = [-1, 1]
        for sign in signlist:
            ich = int(magn * sign)
            lchar.append(ich)
    return lchar

#######################################################
def eval_chargelist(atom_charges: list, debug: int=0) -> Tuple[np.ndarray, np.ndarray, bool]:
    abstotal = np.abs(np.sum(atom_charges))
    abs_atlist = []
    for a in atom_charges:
        abs_atlist.append(abs(a))
    abs_atcharge = np.sum(abs_atlist)
    if any(b > 0 for b in atom_charges) and any(b < 0 for b in atom_charges):
        zwitt = True
    else:
        zwitt = False
    return abstotal, abs_atcharge, zwitt

#######################################################
def check_carbenes(atom: object, ligand: object, molecule: object, debug: int=0) -> Tuple[bool, str, int, int]:
    # Function that determines whether a given connected "atom" of a "ligand" of a "molecule" is a carbene
    # This function is in progress. Ideally, should be able to identify Fischer, Schrock and N-Heterocyclic Carbenes
    # The former two cases probably require rules that involve other ligands in the molecule, hence why the "molecule" is provided
    #:return iscarbene: Boolean variable. True/False
    #:return element:   Type of element that will be later added in the "add_atom" function below
    #:return addedlist: List of integers which track in which atom of the ligand we're adding "elements"
    #:return metal_electrons: List of integers, similar to addedlist, which track in which atom of the ligand we're counting on metal_electrons.

    # about Metal electrons: This variable is a way to contemplate cases in which the metal atom is actually contributing with electrons to the metal-ligand bond.
    # about Metal electrons: In reality, I'm not sure about how to use it correctly, and now is used without much chemical sense

    iscarbene = False
    element = "H"
    addedlist = 0
    metal_electrons = 0

    # Initial attempt with Carbenes, but they are much more complex
    # Looks for Neighbouring N atoms
    list_of_coord_atoms = []
    for i in atom.adjacency:
        list_of_coord_atoms.append(ligand.labels[i])
    numN = list_of_coord_atoms.count("N")

    if numN == 2:  # it is an N-Heterocyclic carbenes
        iscarbene = True
        element = "H"
        addedlist = 1

    return iscarbene, element, addedlist, metal_electrons

#######################################################
def get_metal_poscharges(metal: object, debug: int=0) -> list:
    ## Retrieves plausible oxidation states for a given metal
    # Data Obtained from:
    # Venkataraman, D.; Du, Y.; Wilson, S. R.; Hirsch, K. A.; Zhang, P.; Moore, J. S. A
    # Coordination Geometry Table of the D-Block Elements and Their Ions.
    # J. Chem. Educ. 1997, 74, 915.
   
    atnum = elemdatabase.elementnr[metal.label]

    at_charge = defaultdict(list)
    # 1st-row transition metals.
    at_charge[21] = [3]  # Sc
    at_charge[22] = [2, 3, 4]  # Ti
    at_charge[23] = [1, 2, 3, 4, 5]  # V
    at_charge[24] = [0, 2, 3] # Cr ; including 5 leads to worse results
    at_charge[25] = [1, 2, 3]  # Mn
    at_charge[26] = [2, 3]  # Fe
    at_charge[27] = [1, 2, 3]  # Co
    at_charge[28] = [2, 3]  # Ni
    at_charge[29] = [1, 2]  # Cu
    at_charge[30] = [2]  # Zn
    # 2nd-row transition metals.
    at_charge[39] = [3]  # Y
    at_charge[40] = [2, 3, 4]  # Zr
    at_charge[41] = [1, 3, 4, 5]  # Nb
    at_charge[42] = [0, 2, 4, 5, 6]  # Mo
    at_charge[43] = [1, 2, 3, 4, 5]  # Tc
    at_charge[44] = [2, 3, 4]  # Ru
    at_charge[45] = [1, 2, 3]  # Rh
    at_charge[46] = [0, 2]  # Pd
    at_charge[47] = [1]  # Ag
    at_charge[48] = [2]  # Cd
    # 3rd-row transition metals.
    at_charge[57] = []  # La
    at_charge[72] = [4]  # Hf
    at_charge[73] = [2, 3, 4, 5]  # Ta
    at_charge[74] = [0, 2, 4, 5, 6]  # W
    at_charge[75] = [1, 2, 3, 4, 5, 7]  # Re
    at_charge[76] = [2, 3, 4, 5, 6]  # Os
    at_charge[77] = [1, 3]  # Ir
    at_charge[78] = [0, 2, 4]  # Pt
    at_charge[79] = [1, 3]  # Au
    at_charge[80] = [2]  # Hg

    poscharges = at_charge[atnum]

    list_of_zero_OS = ["Fe", "Ni", "Ru"]
    if metal.label in list_of_zero_OS:
        # In some cases, it adds 0 as possible metal charge
        # -if it has CO ligands
        if any((lig.natoms == 2 and "C" in lig.labels and "O" in lig.labels) for lig in metal.parent.ligands):
            if int(0) not in poscharges:
                poscharges.append(int(0))
        # -if it has any ligand with hapticity
        if any((lig.is_haptic) for lig in metal.parent.ligands):
            if int(0) not in poscharges:
                poscharges.append(int(0))
    
    return poscharges

#######################################################
def balance_charge(unique_indices: list, unique_species: list, debug: int=0) -> list:

    # Function to Select the Best Charge Distribution for the unique species.
    # It accepts multiple charge options for each molecule/ligand/metal (poscharge, etc...).
    # NO: It should select the best one depending on whether the final metal charge makes sense or not.
    # In some cases, can accept metal oxidation state = 0, if no other makes sense

    iserror = False
    iterlist = []
    for idx, spec_tuple in enumerate(unique_species):
        spec = spec_tuple[1]
        toadd = []
        if len(spec.possible_cs) == 1:
            toadd.append(spec.possible_cs[0])
        elif len(spec.possible_cs) > 1:
            for tch in spec.possible_cs:
                toadd.append(tch)
        elif len(spec.possible_cs) == 0:
            iserror = True
            toadd.append("-")
        iterlist.append(toadd)

    if debug >= 2: print("BALANCE: iterlist", iterlist)
    if debug >= 2: print("BALANCE: unique_indices", unique_indices)

    if not iserror:
        tmpdistr = list(itertools.product(*iterlist))
        if debug >= 2: print("BALANCE: tmpdistr", tmpdistr)

        # Expands tmpdistr to include same species, generating alldistr:
        alldistr = []
        for distr in tmpdistr:
            tmp = []
            for u in unique_indices:
                tmp.append(distr[u])
            alldistr.append(tmp)
            if debug >= 2: print("BALANCE: alldistr added:", tmp)

            final_charge_distribution = []
            for idx, d in enumerate(alldistr):
                final_charge = np.sum(d)
                if final_charge == 0:
                    final_charge_distribution.append(d)
    elif iserror:
        if debug >= 1: print("Error found in BALANCE: one species has no possible charges")
        final_charge_distribution = []

    return final_charge_distribution

#######################################################
def prepare_unresolved(unique_indices: list, unique_species: list, distributions: list, debug: int=0):
    list_molecules = [] 
    list_indices = []
    list_options = []
    if debug >= 2: print("")

    # spec_tuple[0] is the subtype of the specie
    # spec_tuple[1] is the specie object
    # spec_tuple[2] is the molecule object to which the specie belongs
    for idx, spec_tuple in enumerate(unique_species):  
        if spec_tuple[0] == "metal": 
            position = [jdx for jdx, uni in enumerate(unique_indices) if uni == idx]
            if debug >= 2: print(f"UNRESOLVED: found metal in positions={position} of the distribution")
            values = [distr[position[0]] for distr in distributions]
            options = list(set(values))
            if debug >= 2: print(f"UNRESOLVED: list of values={values}\n")
            if debug >= 2: print(f"UNRESOLVED: options={options}\n")

            if len(options) > 1:
                list_molecules.append(spec_tuple[2])
                list_indices.append(spec_tuple[1].parent_index)
                list_options.append(options)

    return list_molecules, list_indices, list_options

#######################################################
def prepare_mols(moleclist: list, unique_indices: list, unique_species: list, selected_charge_states: list, final_charge_distribution: list, debug: int=0) -> Tuple[list, bool]:
    # The charge and connectivity of a given specie in the unit cell is only determined for one representative case. i
    # For instance, if four molecules "A" are in the unit cell, only one is evaluated in the rest of the code. 
    # This function ensures that all other "A" molecules in the unit cell end up having the same interpretation (charge and connectivity).
    # In some cases, this might be a difficult job, since the connectivity (i.e. Lewis Structure) often depends on the atom ordering, which might change
    # Thus, an Hungarian ordering is implemented.
    
    Warning = False
    idxtoallocate = 0
    # prints SUMMARY at start
    if debug >= 1: 
        print(f"PREPARE: {len(selected_charge_states)} selected_charge_states received")
        print("")
        print(f"PREPARE: {len(moleclist)} molecules to prepare, of types")
        for idx, mol in enumerate(moleclist):
            print(f"PREPARE: Molecule {idx} is a {mol.subtype} with formula {mol.formula}"),
        print("")

    for idx, sel in enumerate(selected_charge_states):
        for jdx, opt in enumerate(sel):
            chstate = opt[0]
            prot = opt[1]
            if debug >= 2: print(f"PREPARE: State {idx} and option {jdx}. Target state and protonation received with {chstate.corr_total_charge} and {prot.added_atoms}")
 
    for idx, mol in enumerate(moleclist):
        if debug >= 2: print("")
        if debug >= 2: print(f"PREPARE: Molecule {idx} is a {mol.subtype} with formula {mol.formula}"),
    
        ###################################
        ### FOR SOLVENT AND COUNTERIONS ###
        ###################################
        if mol.subtype == "molecule" and mol.iscomplex == False :
            specie = unique_indices[idxtoallocate]
            spec_object = unique_species[specie][1]
            if debug >= 2: print(f"PREPARE: Specie number={specie} with molecule poscharges: {spec_object.possible_cs}")
            if debug >= 2: print(f"PREPARE: Doing molecule {idx} with idxtoallocate: {idxtoallocate}")
    
            allocated = False
            for jdx, ch in enumerate(spec_object.possible_cs):
                if final_charge_distribution[idxtoallocate] == ch and not allocated:   # If the charge in poscharges coincides with the one for this entry in final_distribution
    
                    tgt_charge_state = selected_charge_states[specie][jdx][0]
                    tgt_protonation = selected_charge_states[specie][jdx][1]
                    if debug >= 2: print(f"PREPARE: target state and protonation loaded, with {tgt_charge_state.corr_total_charge} and {tgt_protonation.added_atoms}")
                    allocated = True
                    ch_state = get_charge(mol.labels, mol.coord, mol.conmat, ch, mol.cov_factor)
                    ch_state.correction(tgt_protonation.addedlist, tgt_protonation.metal_electrons, tgt_protonation.elemlist)
    
                    if ch_state.corr_total_charge == tgt_charge_state.corr_total_charge:
                        mol.charge(ch_state.corr_atom_charges, ch_state.corr_total_charge, ch_state.rdkit_mol, ch_state.smiles)
                        if debug >= 2: print(f"PREPARE: Success doing molecule {idx}. Created Charge State with total_charge={ch_state.corr_total_charge}") 
                    else:
                        if debug >= 2: print(f"PREPARE: Error doing molecule {idx}. Created Charge State is different than Target")
                        Warning = True
                    
            if allocated: 
                idxtoallocate += 1
            else:
                idxtoallocate += 1
                if debug >= 2: print(f"PREPARE: Warning allocating molecule {idx} with {final_charge_distribution[idxtoallocate]} as target charge") 
                Warning = True
    
        ###########################
        ######  FOR LIGANDS  ######
        ###########################
        elif mol.iscomplex :
            if debug >= 2: print(f"PREPARE: Molecule {moleclist.index(mol)} has {len(mol.ligands)} ligands")
    
            for kdx, lig in enumerate(mol.ligands):
                #while not Warning:
                #try:
                allocated = False
                specie = unique_indices[idxtoallocate]
                spec_object = unique_species[specie][1]
                if debug >= 2: print("")
                if debug >= 2: print(f"PREPARE: Ligand {kdx}, formula: {lig.formula} is specie {specie}")
                if debug >= 2: print(f"PREPARE: Ligand poscharges: {spec_object.possible_cs}")
                if debug >= 2: print(f"PREPARE: Doing ligand {kdx} with idxtoallocate {idxtoallocate}")
        
                for jdx, ch in enumerate(spec_object.possible_cs):
                    if debug >= 2: print(f"PREPARE: Doing {ch} of options {spec_object.possible_cs}. jdx={jdx}")
                    tgt_charge_state = selected_charge_states[specie][jdx][0]
                    tgt_protonation = selected_charge_states[specie][jdx][1]
                    if debug >= 2: print(f"PREPARE: Found Target Prot State with {tgt_protonation.added_atoms} added atoms and {tgt_protonation.os} OS") 
                    if debug >= 2: print(f"PREPARE: addedlist of Target Prot State: {tgt_protonation.addedlist}")
        
                    if final_charge_distribution[idxtoallocate] == ch and not allocated:
                        allocated = True
        
                        # RE-RUNS the Charge assignation for same-type molecules in the cell
                        list_of_protonations = get_protonation_states(lig, debug=debug)
                        # list_of_protonations = define_sites(lig, mol, debug=1)
                        found_prot = False
                        
                        # Hungarian sort
                        issorted = False
                        if not lig.hapticity:
                            # Adding connectivity data to labels to improve the hungarian sort
                            ligand_data = []
                            ref_data = []
                            for a in lig.atoms:
                                ligand_data.append(a.label+str(a.mconnec))
                            for a in spec_object.atoms:
                                ref_data.append(a.label+str(a.mconnec))
                            dummy1, dummy2, map12 = reorder(ref_data, ligand_data, spec_object.coord, lig.coord)

                            issorted = True
                            ###############                      
                        
                        for p in list_of_protonations:
                            if debug >= 2: print(f"PREPARE: evaluating prot state with added_atoms={p.added_atoms}")#, addedlist={p.addedlist}")
                            if p.os == tgt_protonation.os and p.added_atoms == tgt_protonation.added_atoms and not found_prot:
                                if issorted:
                                    tmp_addedlist = list(np.array(p.addedlist)[map12])
                                else:
                                    tmp_addedlist = p.addedlist
                                if debug >= 2: print(f"PREPARE: tmp_addedlist={tmp_addedlist}")
                                if all(tmp_addedlist[idx] == tgt_protonation.addedlist[idx] for idx in range(len(p.addedlist))):
                                    if debug >= 2: print(f"PREPARE: found match in protonation with tmpsmiles:{p.tmpsmiles}")
                                    prot = p
                                    found_prot = True

                        #### Evaluates possible charges except if the ligand is a nitrosyl
                        if found_prot:
                            if lig.is_nitrosyl:
                                if lig.NO_type == "Linear": NOcharge = 1   #NOcharge is the charge with which I need to run getcharge to make it work
                                if lig.NO_type == "Bent": NOcharge = 0
        
                                ch_state = get_charge(prot.labels, prot.coords, prot.conmat, NOcharge, prot.cov_factor)
                                ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
        
                                if debug >= 2: print(f"PREPARE: Found Nitrosyl of type= {lig.NO_type}")
                                if debug >= 2: print(f"PREPARE: Wanted charge {ch}, obtained: {ch_state.corr_total_charge}")
                                if debug >= 2: print(f"PREPARE: smiles: {ch_state.smiles}")
                            else:
                                if debug >= 2: print(f"PREPARE: Sending getcharge with prot.added_atoms={prot.added_atoms} to obtain charge {ch}")
                                ch_state = get_charge(prot.labels, prot.coords, prot.conmat, ch+prot.added_atoms, prot.cov_factor, tgt_charge_state.allow)
                                ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
        
                                if debug >= 2: print(f"PREPARE: Wanted charge {ch}, obtained: {ch_state.corr_total_charge}")
                                if debug >= 2: print(f"PREPARE: smiles: {ch_state.smiles}")
        
                            if ch_state.corr_total_charge != ch:
                                if debug >= 1: print(f"PREPARE: WARNING: total charge obtained without correction {ch_state.corr_total_charge} while it should be {ch}")
                                Warning = True
                            else:
                                lig.charge(ch_state.corr_atom_charges, ch_state.corr_total_charge, ch_state.rdkit_mol, ch_state.smiles)
                                if debug >= 1: print(f"PREPARE: Success doing ligand {kdx}. Created Charge State with total_charge={ch_state.corr_total_charge}") 

                        else:
                            if debug >= 1: print(f"PREPARE: WARNING, I Could not identify the protonation state. I'll try to obtain the desired result")
                            found_charge_state = False
                            for prot in list_of_protonations:
                                list_of_charge_states = []
                                list_of_protonations_for_each_state = []
                                 
                                tmpobject = ["Ligand", lig, mol]
                                chargestried = get_list_of_charges_to_try(spec, prot)
                                for ich in chargestried:
                                    ch_state = get_charge(prot.labels, prot.coords, prot.conmat, ich, prot.cov_factor)
                                    ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
                                    list_of_charge_states.append(ch_state)
                                    list_of_protonations_for_each_state.append(prot)
                                    if debug >= 1: print(f"    POSCHARGE: charge 0 with smiles {ch_state.smiles}") 

                            if len(list_of_charge_states) > 0:
                                best_charge_distr_idx = select_charge_distr(list_of_charge_states, debug=debug)
                            else:
                                if debug >= 1: print(f"    POSCHARGE. found EMPTY best_charge_distr_idx for PROTONATION state")
                                best_charge_distr_idx = []

                            if debug >= 2: print(f"    POSCHARGE. best_charge_distr_idx={best_charge_distr_idx}")
                            for idx in best_charge_distr_idx:
                                c = list_of_charge_states[idx]
                                p = list_of_protonations_for_each_state[idx]
                                if debug >= 2: print(f"    POSCHARGE. {c.corr_total_charge}={ch}, {p.added_atoms}={tgt_protonation.added_atoms}")
                                if c.corr_total_charge == ch and p.added_atoms == tgt_protonation.added_atoms:
                                    lig.charge(c.corr_atom_charges, c.corr_total_charge, c.rdkit_mol, c.smiles)
                                    if debug >= 1: print(f"PREPARE: Success doing ligand {kdx}. Created Charge State with total_charge={c.corr_total_charge}") 
                                    found_charge_state = True
 
                            if not found_charge_state: Warning = True
                if allocated: 
                    idxtoallocate += 1
                else:
                    idxtoallocate += 1
                    if debug >= 1: print(f"PREPARE: Warning allocating molecule {idx} with {final_charge_distribution[idxtoallocate]} as target charge") 
                    Warning = True
    
            for kdx, met in enumerate(mol.metals):
                specie = unique_indices[idxtoallocate]
                spec_object = unique_species[specie][1]
                allocated = False
                if debug >= 2: print("")
                if debug >= 2: print(f"PREPARE: Metal {kdx}, label {met.label} is specie {specie}")
                if debug >= 2: print(f"PREPARE: Metal possible_css: {spec_object.possible_cs}")
                for jdx, ch in enumerate(spec_object.possible_cs):
                    if final_charge_distribution[idxtoallocate] == ch and not allocated:
                        allocated = True
                        met.charge(ch)
                if allocated:
                    idxtoallocate += 1
    
            if not Warning:
                # Now builds the Charge Data for the final molecule. Smiles is a list with all ligand smiles separately.
                if debug >= 2: print(f"PREPARE: Building Molecule {idx} From Ligand&Metal Information")
                tmp_atcharge = np.zeros((mol.natoms))
                tmp_smiles = []
                for lig in mol.ligands:
                    tmp_smiles.append(lig.smiles)
                    for kdx, a in enumerate(lig.parent_indices):
                        tmp_atcharge[a] = lig.atcharge[kdx]
                for met in mol.metals:
                    tmp_atcharge[met.atlist] = met.totcharge
    
                mol.charge(tmp_atcharge, int(sum(tmp_atcharge)), [], tmp_smiles)

    return moleclist, Warning

#######################################################
def build_bonds(moleclist: list, debug: int=0) -> list:
    ## Builds bond data for all molecules
    ## Now that charges are known, we use the rdkit-objects with the correct charge to do that
    ## Bond entries are defined in the mol and lig objects

    #######
    # First Part. Creates Bonds for Non-Complex Molecules
    #######
    if debug >= 2: print("")
    if debug >= 2: print("BUILD_BONDS: Doing 1st Part")
    if debug >= 2: print("###########################")
    for mol in moleclist:
        if not mol.iscomplex:
            if debug >= 2: print(f"BUILD BONDS: doing mol with mol.natoms={mol.natoms}")
            # Checks that the gmol and rdkit-mol objects have same order
            for idx, a in enumerate(mol.atoms):
                # Security Check. Confirms that the labels are the same
                #if debug >= 2: print("BUILD BONDS: atom", idx, a.label)
                rdkitatom = mol.rdkit_mol.GetAtomWithIdx(idx)
                tmp = rdkitatom.GetSymbol()
                if a.label != tmp: print("Error in Build_Bonds. Atom labels do not coincide. GMOL vs. MOL:", a.label, tmp)
                else:
                    # First part. Creates bond information
                    for b in rdkitatom.GetBonds():
                        bond_startatom = b.GetBeginAtomIdx()
                        bond_endatom   = b.GetEndAtomIdx()
                        bond_order     = b.GetBondTypeAsDouble()
                        if mol.atoms[bond_endatom].label != mol.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol():
                            if debug >= 1: print("Error with Bond EndAtom",mol.atoms[bond_endatom].label,mol.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol())
                        else:
                            if bond_endatom == idx:
                                start = bond_endatom
                                end   = bond_startatom
                            elif bond_startatom == idx:
                                start = bond_startatom
                                end   = bond_endatom
                            new_bond = bond(mol.atoms[start], mol.atoms[end], bond_order) ## This has changed. Now there is a bond object, and we send the atom objects, not only the index
                        a.add_bond(new_bond)

    if debug >= 2: print("")
    if debug >= 2: print("BUILD_BONDS: Doing 2nd Part")
    if debug >= 2: print("###########################")

    #######
    # 2nd Part. Creates Ligand Information
    #######
    for mol in moleclist:
        if debug >= 2: print(f"BUILD BONDS: doing mol {mol.formula} with Natoms {mol.natoms}")
        if mol.iscomplex:
            for lig in mol.ligands:
                if debug >= 2: print(f"BUILD BONDS: doing ligand {lig.formula}")

                for idx, a in enumerate(lig.atoms):
                    # Security Check. Confirms that the labels are the same
                    rdkitatom = lig.rdkit_mol.GetAtomWithIdx(idx)
                    tmp = rdkitatom.GetSymbol()
                    if a.label != tmp: print(f"Error in Build_Bonds. Atom labels do not coincide. GMOL vs. MOL: {a.label} {tmp}")
                    else:
                        # First part. Creates bond information
                        for b in rdkitatom.GetBonds():
                            bond_startatom = b.GetBeginAtomIdx()
                            bond_endatom = b.GetEndAtomIdx()
                            bond_order = b.GetBondTypeAsDouble()
                            if bond_startatom >= lig.natoms or bond_endatom >= lig.natoms:
                                continue
                            else:
                                if lig.atoms[bond_endatom].label != lig.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol():
                                    if debug >= 1: print( "Error with Bond EndAtom",lig.atoms[bond_endatom].label,lig.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol())
                                else:
                                    if bond_endatom == idx:
                                        start = bond_endatom
                                        end   = bond_startatom
                                    elif bond_startatom == idx:
                                        start = bond_startatom
                                        end   = bond_endatom
                                    new_bond = bond(mol.atoms[start], mol.atoms[end], bond_order) ## This has changed. Now there is a bond object, and we send the atom objects, not only the index
                                a.add_bond(new_bond)

    if debug >= 2: print("")
    if debug >= 2: print("BUILD_BONDS: Doing 3rd Part")
    if debug >= 2: print("###########################")

    ## SERGI: I need to work on that, but I'll do something first
     
    ########
    ## 3rd Part. Merges Ligand Information into Molecule Object using the atlists
    ########
    #for mol in moleclist:
    #    if debug >= 2: print("BUILD BONDS: doing mol", mol.formula, "with Natoms", mol.natoms)
    #    if mol.iscomplex:
    #        allstarts = []
    #        allends = []
    #        allorders = []
    #        # Adds atoms within ligands
    #        for lig in mol.ligandlist:
    #            for a in lig.atoms:
    #                for b in a.bond:
    #                    allstarts.append(lig.atlist[b[0]])
    #                    allends.append(lig.atlist[b[1]])
    #                    allorders.append(b[2])

    #        # Adds Metal-Ligand Bonds, with an arbitrary 0.5 order:
    #        for idx, row in enumerate(mol.mconmat):
    #            # if debug >= 2: print(row)
    #            for jdx, val in enumerate(row):
    #                if val > 0:
    #                    # if debug >= 2: print(idx, jdx, val)
    #                    allstarts.append(idx)
    #                    allends.append(jdx)
    #                    allorders.append(0.5)

    #        # I sould work to add Metal-Metal Bonds. Would need to work on the Metal class:
    #        # Finally, puts everything together, and creates bonds for MOLECULE atom objects
    #        for idx, a in enumerate(mol.atoms):
    #            starts = []
    #            ends = []
    #            orders = []
    #            group = []
    #            for entry in zip(allstarts, allends, allorders):
    #                if entry[0] == idx or entry[1] == idx:
    #                    if entry not in group and (entry[1], entry[0], entry[2]) not in group:
    #                        starts.append(entry[0])
    #                        ends.append(entry[1])
    #                        orders.append(entry[2])
    #                        group.append(entry)

    #            a.bonds(starts, ends, orders)

    #######
    # 4th Part. Corrects Ligand Smiles to Remove Added H atoms, the old smiles is stored in "lig.smiles_with_H" as it can still be useful
    #######
    for mol in moleclist:
        if debug >= 2: print("BUILD BONDS: doing mol", mol.formula, "with Natoms", mol.natoms)
        if mol.iscomplex :
            mol.smiles_with_H = [lig.smiles for lig in mol.ligands]
            mol.smiles = [correct_smiles_ligand(lig)[0] for lig in mol.ligands]
            # for lig in mol.ligands:
            #     lig.smiles_with_H = lig.smiles  
            #     lig.smiles, lig.rdkit_mol = correct_smiles_ligand(lig)
            #     mol.smiles.append(lig.smiles)
            #     mol.smiles_with_H.append(lig.smiles_with_H)
        
    return moleclist

#######################################################
def correct_smiles_ligand(ligand: object):
    ## Receives a ligand class object and constructs the smiles and the rdkit_mol object from scratch, using atoms and bond information

    Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
    #### Creates an empty editable molecule
    rwlig = Chem.RWMol()    
 
    # Adds atoms with their formal charge 
    for jdx, atom in enumerate(ligand.atoms):        
        rdkit_atom = Chem.Atom(atom.atnum)
        rdkit_atom.SetFormalCharge(int(atom.charge))
        rdkit_atom.SetNoImplicit(True)
        rwlig.AddAtom(rdkit_atom)
                       
    # Sets bond information and hybridization
    for jdx, atom in enumerate(ligand.atoms):
        nbonds = 0
        for bond in atom.bonds:
            nbonds += 1
            if bond.order== 1.0: btype = Chem.BondType.SINGLE
            elif bond.order == 2.0: btype = Chem.BondType.DOUBLE
            elif bond.order == 3.0: btype = Chem.BondType.TRIPLE
            elif bond.order == 1.5: 
                btype = Chem.BondType.AROMATIC
                rdkit_atom.SetIsAromatic(True)
            if bond.atom1 == jdx and bond.atom2 > jdx: rwlig.AddBond(bond.atom1, bond.atom2, btype)

        if nbonds == 1: hyb = Chem.HybridizationType.S
        elif nbonds == 2: hyb = Chem.HybridizationType.SP
        elif nbonds == 3: hyb = Chem.HybridizationType.SP2
        elif nbonds == 4: hyb = Chem.HybridizationType.SP3
        else: hyb = Chem.HybridizationType.UNSPECIFIED
        rdkit_atom.SetHybridization(hyb)
            
    # Creates Molecule
    obj = rwlig.GetMol()
    smiles = Chem.MolToSmiles(obj)
    
    Chem.SanitizeMol(obj)
    Chem.DetectBondStereochemistry(obj, -1)
    Chem.AssignStereochemistry(obj, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(obj, -1)
    
    return smiles, obj

#######################################################

###################
### NEW OBJECTS ###
###################
class protonation(object):
    def __init__(self, labels, cov_factor, added_atoms, addedlist, block, metal_electrons, elemlist, tmpsmiles=" ", os=int(0), typ="Local", parent: object=None):
        self.labels = labels
        self.natoms = len(labels)
        self.coords = coords
        self.added_atoms = added_atoms
        self.addedlist = addedlist
        self.block = block
        self.metal_electrons = metal_electrons 
        self.elemlist = elemlist
        self.typ = typ
        self.cov_factor = cov_factor
        self.os = os
        self.tmpsmiles = tmpsmiles
        self.atnums = [elemdatabase.elementnr[l] for l in labels]  # from xyz2mol
        self.parent = parent

        self.radii = get_radii(labels)
        self.status, self.adjmat, self.adjnum = get_adjmatrix(self.labels, self.coords, self.cov_factor, self.radii)

#######################################################

class charge_state(object):
    def __init__(self, status, uncorr_total_charge, uncorr_atom_charges, rdkit_mol, smiles, charge_tried, allow):
        self.status = status
        self.uncorr_total_charge = uncorr_total_charge
        self.uncorr_atom_charges = uncorr_atom_charges
        self.rdkit_mol = rdkit_mol
        self.smiles = smiles
        self.charge_tried = charge_tried
        self.allow = allow
        self.uncorr_abstotal, self.uncorr_abs_atcharge, self.uncorr_zwitt = eval_chargelist(uncorr_atom_charges)
        
        if uncorr_total_charge == charge_tried:  
            self.coincide = True
        else:
            self.coincide = False

    def correction(self, addedlist, metal_electrons, elemlist):
        self.addedlist = addedlist
        self.metal_electrons = metal_electrons
        self.elemlist = elemlist
        self.corr_total_charge = int(0)
        self.corr_atom_charges = []

        # Corrects the Charge of atoms with addedH
        count = 0 
        if len(addedlist) > 0:
            for idx, add in enumerate(addedlist):  # Iterates over the original number of ligand atoms, thus without the added H
                if add != 0:
                    count += 1 
                    corrected = self.uncorr_atom_charges[idx] - addedlist[idx] + metal_electrons[idx] - self.uncorr_atom_charges[len(addedlist)-1+count]
                    self.corr_atom_charges.append(corrected)
                    # last term corrects for cases in which a charge has been assigned to the added atom
                else:
                    self.corr_atom_charges.append(self.uncorr_atom_charges[idx])
            self.corr_total_charge = int(np.sum(self.corr_atom_charges))
        else:
            self.corr_total_charge = self.uncorr_total_charge
            self.corr_atom_charges = self.uncorr_atom_charges.copy()

        self.corr_abstotal, self.corr_abs_atcharge, self.corr_zwitt = eval_chargelist(self.corr_atom_charges)


#######################################################
