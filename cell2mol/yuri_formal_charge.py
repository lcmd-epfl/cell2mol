#!/usr/bin/env python

import numpy as np  
from cell2mol.elementdata import ElementData
from cell2mol.connectivity import *
from collections import defaultdict
import itertools
import sys

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
    typelist_mets =[]
    
    specs_found = -1    
    for idx, mol in enumerate(moleclist):
        if debug >= 2: print(f"Molecule {idx} formula={mol.formula}")
        if not mol.iscomplex:
            found = False
            for ldx, typ in enumerate(typelist_mols):   # Molecules
                issame = compare_species (mol, typ[0], debug=1)
                if issame : 
                    found = True ; kdx = typ[1] 
                    if debug >= 2: print(f"Molecule {idx} is the same with {ldx} in typelist")      
            if not found:
                specs_found += 1 ; kdx = specs_found
                typelist_mols.append(list([mol, kdx]))
                unique_species.append(list([mol.subtype, mol]))
                if debug >= 2: print(f"New molecule found with: formula={mol.formula} and added in position {kdx}")
            unique_indices.append(kdx)
        
        else:
            for jdx, lig in enumerate(mol.ligands):     # ligands
                found = False
                for ldx, typ in enumerate(typelist_ligs):
                    issame = compare_species (lig, typ[0], debug=1)
                    if issame : 
                        found = True ; kdx = typ[1] 
                        if debug >= 2: print(f"ligand {jdx} is the same with {ldx} in typelist")    
                if not found:
                    specs_found += 1 ; kdx = specs_found
                    typelist_ligs.append(list([lig, kdx]))
                    unique_species.append(list([lig.subtype, lig, mol]))
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
                    unique_species.append(list([met.subtype, met, mol]))
                    if debug >= 2: print(f"New Metal Center found with: labels {met.label} and added in position {kdx}")
                unique_indices.append(kdx)
            
    return unique_species, unique_indices

###############################################################################
def drive_get_poscharges(unique_species: list, debug: int=0):

    Warning = False
    # For each specie, a plausible_charge_state is generated. 
    # These include the actual charge state, and the protonation it belongs to
    selected_charge_states = []
    for idx, spec in enumerate(unique_species):
        spec[1].poscharge = []
        spec[1].posatcharge = []
        spec[1].posobjlist = []
        spec[1].posspin = []
        spec[1].possmiles = []

        # Obtains charge for Organic Molecules
        if debug >= 1: print("")
        if spec[0] == "molecule" and spec[1].iscomplex == False:
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("    #### NON-Complex ####")
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("          ", spec[1].natoms, spec[1].formula)
        elif spec[0] == "ligand":
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("    #### Ligand ####")
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("          ", spec[1].natoms, spec[1].formula, spec[1].denticity)
        elif spec[0] == "metal":
            coordinating_atoms = labels2formula ([atom.label for atom in spec[1].coord_sphere])
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("    #### Metal ####")
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("          ", spec[1].label, "coordinating_atoms\t",  coordinating_atoms)


        # Gets possible Charges for Ligands and Non-complex molecules
        if (spec[0] == "molecule" and spec[1].iscomplex == False) or spec[0] == "ligand":
            tmp, Warning = get_poscharges(spec, debug=debug)
            if Warning: 
                if debug >= 1: print(f"Empty possible charges received for molecule {spec[1].formula}")
            else: 
                if debug >= 1: print("Charge state and protonation received for molecule", len(tmp)) 
                selected_charge_states.append(tmp)

        # Gets possible Charges for Metals
        elif spec[0] == "metal":
            met = spec[1]
            mol = spec[2]
            met.poscharges = get_metal_poscharges(met, mol, debug=debug)
            if debug >= 1: print("Possible charges received for metal:", met.poscharges)
            selected_charge_states.append([])

    return selected_charge_states, Warning
    
#######################################################
def get_metal_poscharges(metal: object, molecule: object, debug: int=0) -> list:
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
        if any((lig.natoms == 2 and "C" in lig.labels and "O" in lig.labels) for lig in molecule.ligands):
            if int(0) not in poscharges:
                poscharges.append(int(0))
        # -if it has any ligand with hapticity
        if any((lig.hapticity) for lig in molecule.ligands):
            if int(0) not in poscharges:
                poscharges.append(int(0))
    
    return poscharges

#######################################################
def get_list_of_charges_to_try(spec: list, prot: object, debug: int=0) -> list:
    lchar = []

    #### Educated Guess on the Maximum Charge one can expect from the spec[1]
    if spec[0] == "molecule":  maxcharge = 3
    elif spec[0] == "ligand":
        count_non_connected_O = 0
        for a in spec[1].atoms:
            if a.label == "O" and a.mconnec == 0 and a.connec == 1:
                count_non_connected_O += 1
        if not spec[1].hapticity:
            maxcharge = spec[1].denticity + count_non_connected_O - prot.added_atoms
            if debug >= 2: print(f"MAXCHARGE: maxcharge set at {maxcharge} with {spec[1].denticity}+{count_non_connected_O}-{prot.added_atoms}")
        else:
            maxcharge = 2
        # Cases of same atom being connected to more than one metal
        if any(a.mconnec >= 2 for a in spec[1].atoms):
            pass
        else: 
            if maxcharge > spec[1].natoms: maxcharge = spec[1].natoms
        if maxcharge > 4: maxcharge = 4
        if maxcharge < 2: maxcharge = 2
    if debug >= 2: print(f"MAXCHARGE: maxcharge set at {maxcharge}")
    
    # Defines list of charges that will try
    for magn in range(0, int(maxcharge + 1)):
        if magn == 0:
            signlist = [1]
        elif magn != 0:
            signlist = [-1, 1]
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
def get_poscharges(spec: list, debug: int=0) -> Tuple[list, bool]:
    # This function drives the detrmination of the charge for a "ligand=lig" of a given "molecule=mol"
    # The whole process is done by the functions:
    # 1) define_sites, which determines which atoms must have added elements (see above)
    # 2) once any atom is added, the function "getcharge" generates one connectivity for a set of charges
    # 3) select_charge_distr chooses the best connectivity among the generated ones.

    # Basically, this function connects these other three functions,
    # while managing some key information for those
    # Initiates variables
    Warning = False
    selected_charge_states = []
    # Finds maximum plausible charge for the specie

    ##############################
    #### Creates protonation states. That is, geometries in which atoms have been added to the original molecule
    ##############################
    if spec[0] == "ligand":
        list_of_protonations = define_sites(spec[1], spec[2], debug)
    elif spec[0] == "molecule" and spec[1].iscomplex == False:
        empty_protonation = protonation(spec[1].labels, spec[1].coord, spec[1].cov_factor, int(0), [], [], [], [], typ="Empty")
        list_of_protonations = [] 
        list_of_protonations.append(empty_protonation)
        if debug >= 2: print(f"    POSCHARGE: doing empty PROTONATION for this specie")
    if debug >= 2: print(f"    POSCHARGE: received {len(list_of_protonations)} protonations for this specie")

    ##############################
    #### Evaluates possible charges except if the ligand is a nitrosyl
    ##############################
    if spec[0] == "ligand" and spec[1].natoms == 2 and "N" in spec[1].labels and "O" in spec[1].labels:
        for prot in list_of_protonations: 
            list_of_charge_states = []
            list_of_protonations_for_each_state = [] 
            chargestried = get_list_of_charges_to_try(spec, prot)
            if debug >= 2: print(f"    POSCHARGE will try charges {chargestried}") 

            for ich in chargestried:
                ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, ich, prot.cov_factor)
                ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
                list_of_charge_states.append(ch_state)
                list_of_protonations_for_each_state.append(prot)
                if debug >= 2: print(f"    POSCHARGE: charge 0 with smiles {ch_state.smiles}")

            if spec[1].NO_type == "Linear":
                best_charge_distr_idx = [2]
            elif spec[1].NO_type == "Bent":
                best_charge_distr_idx = [0]

        #############################
        # For all protonations, it adds the resulting states to selected_charge_states
        #############################
        for idx in best_charge_distr_idx:
            c = list_of_charge_states[idx]
            p = list_of_protonations_for_each_state[idx] 
            if c.corr_total_charge not in spec[1].poscharge:
                spec[1].poscharge.append(c.corr_total_charge)
                spec[1].posatcharge.append(c.corr_atom_charges)
                spec[1].posobjlist.append(c.rdkit_mol)
                spec[1].possmiles.append(c.smiles)
                # Selected_charge_states is a tuple with the actual charge state, and the protonation it belongs to
                selected_charge_states.append([c,p])
                if debug >= 2: print(f"    POSCHARGE. poscharge added with corrected charge: {c.corr_total_charge} and uncorrected: {c.uncorr_total_charge}")
                if debug >= 2: print(f"    POSCHARGE. poscharge added with smiles: {c.smiles}") 

    ##############################
    #### Evaluates possible charges except if the ligand is a azide (N3-) ion
    ##############################               
    elif spec[0] == "ligand" and spec[1].natoms == 3 and len(np.unique(spec[1].labels)) == 1 and "N" in np.unique(spec[1].labels):
        poscharge = -1 
        # metal-coordinating N charge +1
        # The other two N atoms charge -1
    ##############################
    #### Evaluates possible charges except if the ligand is a triiodide (I3-) ion
    ##############################               
    elif spec[0] == "ligand" and spec[1].natoms == 3 and len(np.unique(spec[1].labels)) == 1 and "I" in np.unique(spec[1].labels):
        poscharge = -1 
        # metal-coordinating I charge -1
        # The other two I atoms charge 0

    ##############################
    # If not a Nitrosyl ligand, choose among the charge_states for this protonation
    ##############################
    else:
        list_of_charge_states = []
        list_of_protonations_for_each_state = []
        for prot in list_of_protonations: 
            if debug >= 2: print(" ")
            if debug >= 2: print(f"    POSCHARGE: doing PROTONATION with added atoms: {prot.added_atoms}")
            chargestried = get_list_of_charges_to_try(spec, prot)
            if debug >= 2: print(f"    POSCHARGE will try charges {chargestried}") 

            for ich in chargestried:
                try:
                    ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, ich, prot.cov_factor)
                    ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
                    list_of_charge_states.append(ch_state)
                    list_of_protonations_for_each_state.append(prot)
                    if debug >= 2: print(f"    POSCHARGE: charge {ich} with smiles {ch_state.smiles}")
                except Exception as exc:
                    if debug >= 1: print(f"    POSCHARGE: EXCEPTION in get_poscharges: {exc}")

        #############################
        # Selects the best distribution(s) for all protonations. We retrieve the indices first, and then move the objects to the list
        #############################
        if len(list_of_charge_states) > 0:
            best_charge_distr_idx = select_charge_distr(list_of_charge_states, debug=debug)
        else:
            if debug >= 2: print(f"    POSCHARGE. found EMPTY best_charge_distr_idx for PROTONATION state")
            best_charge_distr_idx = []

        #############################
        # For all protonations, it adds the resulting states to selected_charge_states
        #############################
        for idx in best_charge_distr_idx:
            c = list_of_charge_states[idx]
            p = list_of_protonations_for_each_state[idx] 
            if c.corr_total_charge not in spec[1].poscharge:
                spec[1].poscharge.append(c.corr_total_charge)
                spec[1].posatcharge.append(c.corr_atom_charges)
                spec[1].posobjlist.append(c.rdkit_mol)
                spec[1].possmiles.append(c.smiles)
                # Selected_charge_states is a tuple with the actual charge state, and the protonation it belongs to
                selected_charge_states.append([c,p])
                if debug >= 2: print(f"    POSCHARGE. poscharge added with corrected charge: {c.corr_total_charge} and uncorrected: {c.uncorr_total_charge}")
                if debug >= 2: print(f"    POSCHARGE. poscharge added with smiles: {c.smiles}") 

    ### HERE IS HAS FINISHED WITH ALL PROTONATIONS
    if len(selected_charge_states) == 0:  # Means that no good option has been found
        Warning = True
    else:  
        Warning = False

    return selected_charge_states, Warning

#######################################################
def define_sites(ligand: object, molecule: object, debug: int=0) -> list:

    list_of_protonations = []

    natoms = len(ligand.labels)
    newlab = ligand.labels.copy()
    newcoord = ligand.coord.copy()

    # Variables that control how many atoms have been added.
    tmp_added_atoms = 0
    added_atoms = 0

    # Boolean that decides whether a non-local approach is needed
    non_local_groups = 0
    needs_nonlocal = False

    # Initialization of the variables
    addedlist = np.zeros((natoms)).astype(int)
    block = np.zeros((natoms)).astype(int)
    metal_electrons = np.zeros((natoms)).astype(int)  # It will remain as such
    elemlist = np.empty((natoms)).astype(str)

    # Program runs sequentially for each group of the ligand
    for g in ligand.groups:

        ########################
        # Cases with Hapticity #
        ########################
        if len(g.haptic_type) > 0:  # then the group has hapticity
            Selected_Hapticity = False
            if debug >= 2: print("        DEFINE_SITES: addressing group with hapticity:", g.haptic_type)

            if "h5-Cp" in g.haptic_type and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h5-Cp with atlist:", g.parent_indices)
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    # print(idx, a.label, a.mconnec)
                    if idx in g.parent_indices and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                            # print("added")
                        else:
                            block[idx] = 1

            elif "h7-Cicloheptatrienyl" in g.haptic_type and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h7-Cicloheptatrienyl")
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

            elif "h5-AsCp" in g.haptic_type and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h5-AsCp")
                Selected_Hapticity = True

                # Rules change depending on whether the ring is substituted or not 
                issubstituted = False
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        for jdx in a.adjacency:
                            if ligand.labels[jdx] != "As":
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

            elif "h5-Pentaphosphole" in g.haptic_type and not Selected_Hapticity: ## Case of IMUCAX
                if debug >= 2: print("        DEFINE_SITES: It is an h5-Pentaphosphole")
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
                # if "h3-Allyl" or "h3-Cp" in g.haptic_type:
                if debug >= 2: print("        DEFINE_SITES: It is either an h3-Allyl or an h3-Cp")
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
                if debug >= 2: print("        DEFINE_SITES: It is either an h4-Benzene or an h4-Butadiene")
                if debug >= 2: print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        block[idx] = 1

            elif ("h2-Benzene" in g.haptic_type or "h2-Butadiene" or "h2-ethylene" in g.haptic_type) and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is either an h2-Benzene, an h2-ethylene or an h2-Butadiene")
                if debug >= 2: print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        block[idx] = 1

            elif "h4-Enone" in g.haptic_type and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h4-Enone")
                if debug >= 2: print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.parent_indices and a.mconnec == 1:
                        block[idx] = 1

            # If the group hapticity type is not recognized -or instructions are not defined-, nothing is done
            if not Selected_Hapticity:
                if debug >= 2: print(f"        DEFINE_SITES: {g.haptic_type} not recognized or new rules are necessary")

        else:  # cases without hapticity
            ions = ["F", "Cl","Br","I","As"]  # Atoms for which an H atom is always added
            ###########################
            # Cases with No Hapticity #
            ###########################
            # An initial attempt to add elements based on the adjacency of the connected atom
            for idx in g.parent_indices:
                a = ligand.atoms[idx]
                if debug >= 2: print(f"        DEFINE_SITES: evaluating non-haptic group with index {idx} and label {a.label}")
                # Simple Ionic Case
                if a.label in ions:
                    if a.connec == 0:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    elif a.connec >= 1:
                        block[idx] = 1
                # Oxygen
                elif a.label == "O" or a.label == "S" or a.label == "Se":
                    if a.connec == 1:
                        needs_nonlocal = True
                        non_local_groups += 1
                        if debug >= 2: print(f"        DEFINE_SITES: will be sent to nonlocal due to {a.label} atom")
                    elif a.connec > 1:
                        block[idx] = 1
                # SERGI: I'm considering a different one with S and Se
                #                 elif a.label == "S" or a.label == "Se":
                #                     if a.connec == 1:
                #                         elemlist[idx] = "H"
                #                         addedlist[idx] = 1
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
                    if ligand.natoms == 2 and "O" in ligand.labels:
                        NO_type = get_nitrosyl_geom(ligand)
                        if NO_type == "Linear":
                            if debug >= 2: print("        DEFINE_SITES: Found Linear Nitrosyl")
                            elemlist[idx] = "O"
                            addedlist[idx] = 2
                            metal_electrons[idx] = 1
                        elif NO_type == "Bent":
                            if debug >= 2: print("        DEFINE_SITES: Found Bent Nitrosyl")
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                    else:
                        # nitrogen with at least 3 adjacencies doesnt need H
                        if a.connec >= 3:
                            block[idx] = 1
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
                                if debug >= 2: print(f"        DEFINE_SITES: will be sent to nonlocal due to {a.label} atom")
                # Phosphorous
                elif (a.connec == 3) and a.label == "P":
                    block[idx] = 1
                # Case of Carbon (Simple CX vs. Carbenes)
                elif a.label == "C":
                    if ligand.natoms == 2:
                        # CN
                        if "N" in ligand.labels:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                        # CO
                        if "O" in ligand.labels:
                            block[idx] = 1
                    # Added for amides
                    elif (any(ligand.labels[i] == "O" for i in a.adjacency) and any(ligand.labels[i] == "N" for i in a.adjacency) and a.connec == 2 ):
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        iscarbene, tmp_element, tmp_added, tmp_metal = check_carbenes(a, ligand, molecule)
                        if debug >= 2: print(f"        DEFINE_SITES: Evaluating as carbene and {iscarbene}")
                        if iscarbene:
                            # Carbene identified
                            elemlist[idx] = tmp_element
                            addedlist[idx] = tmp_added
                            metal_electrons[idx] = tmp_metal
                        else:
                            needs_nonlocal = True
                            non_local_groups += 1
                            if debug >= 2: print(f"        DEFINE_SITES: will be sent to nonlocal due to {a.label} atom")
                # Silicon
                elif a.label == "Si":
                    if a.connec < 4:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        block[idx]
                # Boron
                elif a.label == "B":
                    if a.connec < 4:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        block[idx]
                # None of the previous options
                else:
                    if not needs_nonlocal:
                        needs_nonlocal = True
                        non_local_groups += 1
                        if debug >= 2: print(f"        DEFINE_SITES: will be sent to nonlocal due to {a.label} atom with no rules")

        # If, at this stage, we have found that any atom must be added, this is done before entering the non_local part.
        # The block variable makes that more atoms cannot be added to these connected atoms
        for idx, a in enumerate(ligand.atoms):
            if addedlist[idx] != 0 and block[idx] == 0:
                isadded, newlab, newcoord = add_atom(newlab, newcoord, idx, ligand, molecule.metalist, elemlist[idx])
                if isadded:
                    added_atoms += addedlist[idx]
                    block[idx] = 1  # No more elements will be added to those atoms
                    if debug >= 2: print(f"        DEFINE_SITES: Added {elemlist[idx]} to atom {idx} with: a.mconnec={a.mconnec} and label={a.label}")
                else:
                    addedlist[idx] = 0 
                    block[idx] = 1  # No more elements will be added to those atoms
                   
    ############################
    ###### NON-LOCAL PART ######
    ############################
    
    if not needs_nonlocal:
        new_prot = protonation(newlab, newcoord, ligand.cov_factor, added_atoms, addedlist, block, metal_electrons, elemlist) 
        list_of_protonations.append(new_prot)
    else:
        # Generate the new adjacency matrix after local elements have been added to be sent to xyz2mol
        local_labels = newlab.copy()
        local_coords  = newcoord.copy()
        local_radii = get_radii(local_labels)
        local_natoms = len(local_labels)
        local_atnums = [int_atom(label) for label in local_labels]  # from xyz2mol.py
        dummy, local_conmat, local_connec, dummy, dummy = getconec(local_labels, local_coords, ligand.cov_factor, local_radii)

        local_addedlist = addedlist.copy()
        local_block = block.copy()
        local_added_atoms = added_atoms

        # Initiate variables
        avoid = ["Si", "P"]

        if debug >= 2: print(" ")
        if debug >= 2: print(f"        DEFINE_SITES: Enters non-local with:")
        if debug >= 2: print(f"        DEFINE_SITES: block: {block}")
        if debug >= 2: print(f"        DEFINE_SITES: addedlist: {addedlist}")
        if debug >= 2: print(f"        DEFINE_SITES: {non_local_groups} non_local_groups groups found") 

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
            if debug >= 2:  print(f"        DEFINE_SITES: doing combination {com}") 
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
                            isadded, newlab, newcoord = add_atom(newlab, newcoord, jdx, ligand, molecule.metalist, elemlist[jdx])
                            if isadded:
                                added_atoms += addedlist[jdx]
                                if debug >= 2: print(f"        DEFINE_SITES: Added {elemlist[jdx]} to atom {jdx} with: a.mconnec={a.mconnec} and label={a.label}")
                            else:
                                addedlist[idx] = 0 
                                block[idx] = 1  # No more elements will be added to those atoms
                    elif non_local_groups == 1:
                        if com == 1:
                            elemlist[jdx] = "H"
                            addedlist[jdx] = 1
                            isadded, newlab, newcoord = add_atom(newlab, newcoord, jdx, ligand, molecule.metalist, elemlist[jdx])
                            if isadded:
                                added_atoms += addedlist[jdx]
                                if debug >= 2: print(f"        DEFINE_SITES: Added {elemlist[jdx]} to atom {jdx} with: a.mconnec={a.mconnec} and label={a.label}")
                            else:
                                addedlist[idx] = 0 
                                block[idx] = 1  # No more elements will be added to those atoms
                    #in any case, moves index
                    toallocate += 1

            smi = " "
        
            new_prot = protonation(newlab, newcoord, ligand.cov_factor, added_atoms, addedlist, block, metal_electrons, elemlist, smi, os, typ="Non-local") 
            if new_prot.status == 1 and new_prot.added_atoms == os+local_added_atoms:
                list_of_protonations.append(new_prot)
                if debug >= 2:  print(f"        DEFINE_SITES: Protonation SAVED with {added_atoms} atoms added to ligand. status={new_prot.status}")
            else:
                if debug >= 2:  print(f"        DEFINE_SITES: Protonation DISCARDED. Steric Clashes found when adding atoms. status={new_prot.status}")
                
    return list_of_protonations 

#######################################################
def getcharge(labels: list, pos: list, conmat: np.ndarray, ich: int, cov_factor: float=1.3, allow: bool=True, debug: int=0)  -> list:
    ## Generates the connectivity of a molecule given a charge.
    # The molecule is described by the labels, and the atomic cartesian coordinates "pos"
    # The adjacency matrix is also provided (conmat)
    #:return iscorrect: boolean variable with a notion of whether the function delivered a good=True or bad=False connectivity
    #:return total_charge: total charge associated with the connectivity
    #:return atom_charge: atomic charge for each atom of the molecule
    #:return mols: rdkit molecule object
    #:return smiles: smiles representation of the molecule

    pt = Chem.GetPeriodicTable()  # needed to retrieve the default valences in the 2nd and 3rd checks
    natoms = len(labels)
    atnums = [elemdatabase.elementnr[label] for label in labels]  # from xyz2mol

    ##########################
    # xyz2mol is called here #
    ##########################
    # use_graph is called for a faster generation
    # allow_charged_fragments is necessary for non-neutral molecules
    # embed_chiral shouldn't ideally be necessary, but it runs a sanity check that improves the proposed connectivity
    # use_huckel false means that the xyz2mol adjacency will be generated based on atom distances and vdw radii.
    # Ideally, the adjacency matrix could be provided

    mols = xyz2mol(atnums,pos,conmat,cov_factor,charge=ich,use_graph=True,allow_charged_fragments=allow,embed_chiral=True,use_huckel=False)
    if len(mols) > 1: print("WARNING: More than 1 mol received from xyz2mol for initcharge:", ich)

    # smiles is generated with rdkit
    smiles = Chem.MolToSmiles(mols[0])

    # Here, the atom charge is retrieved, and the connectivity of each atom goes through 3 checks.
    # The variable iscorrect will track whether the overall generated structure is meaningful
    iscorrect = True
    atom_charge = []
    total_charge = 0
    for i in range(natoms):
        a = mols[0].GetAtomWithIdx(i)  # Returns a particular Atom
        atom_charge.append(a.GetFormalCharge())

        valence = a.GetTotalValence()  # Valence of the atom in the mol object
        bonds = 0
        countaromatic = 0
        for b in a.GetBonds():  # Returns a read-only sequence containing all of the molecule’s Bonds
            bonds += b.GetBondTypeAsDouble()
            # total number of bonds (weighted by bond order) of the atom in the mol object
            # Returns the type of the bond as a double (i.e. 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE)
            if b.GetBondTypeAsDouble() == 1.5:
                countaromatic += 1
        if countaromatic % 2 != 0:
            bonds -= 0.5

        total_charge += a.GetFormalCharge()
        lonepairs = (elemdatabase.valenceelectrons[a.GetSymbol()] - a.GetFormalCharge() - valence) / 2
        totalvalenceelectrons = int(bonds) + int(lonepairs) * 2 + a.GetFormalCharge()

        # Checks the quality of the resulting smiles
        # First check, the number of lonepairs is computed and should make sense
        if lonepairs != 0 and lonepairs != 1 and lonepairs != 2 and lonepairs != 3 and lonepairs != 4:
            if debug >= 2:
                print("GETCHARGE: 2nd Check-lonepairs=", i, a.GetSymbol(), lonepairs)
            iscorrect = False

        # RDKIT has some troubles assigning the valence for atoms with aromatic bonds.
        # So the 2nd and 3rd Check applies only for countaromatic==0
        if countaromatic == 0:
            # Second check, the number of bonds should coincide with the valence.
            # I know it should be the same, but in bad SMILES they often do not coincide
            if bonds != valence:
                if debug >= 2:
                    print("GETCHARGE: 1st Check-bonds/valence:",i,a.GetSymbol(),bonds,valence)
                iscorrect = False
                if debug >= 2:
                    for b in a.GetBonds():
                        print(b.GetBondTypeAsDouble(),b.GetBeginAtomIdx(),b.GetEndAtomIdx())

            # Third check, using the totalvalenceelectrons
            if totalvalenceelectrons != elemdatabase.valenceelectrons[a.GetSymbol()]:
                if debug >= 2: print("GETCHARGE: 3rd Check: Valence gives false for atom",i,a.GetSymbol(),"with:",totalvalenceelectrons,elemdatabase.valenceelectrons[a.GetSymbol()])
                iscorrect = False

        if debug >= 2 and i == 0:
            print("ich, atom idx, label, charge, pt.GetDefaultValence(a.GetAtomicNum()), valence, num bonds, num lonepairs, iscorrect")
        if debug >= 2: print(ich,i,a.GetSymbol(),a.GetFormalCharge(),pt.GetDefaultValence(a.GetAtomicNum()),valence,int(bonds),int(lonepairs),iscorrect)

    # Creates the charge_state
    try: 
        ch_state = charge_state(iscorrect, total_charge, atom_charge, mols[0], smiles, ich, allow)
    except Exception as exc:
        if debug >= 1: print(f"    GETCHARGE: EXCEPTION in charge_state creation: {exc}")

    return ch_state


#######################################################
###################
### NEW OBJECTS ###
###################

class protonation(object):
    def __init__(self, labels, coordinates, cov_factor, added_atoms, addedlist, block, metal_electrons, elemlist, tmpsmiles=" ", os=int(0), typ="Local"):
        self.labels = labels
        self.coordinates = coordinates
        self.added_atoms = added_atoms
        self.addedlist = addedlist
        self.block = block
        self.metal_electrons = metal_electrons 
        self.elemlist = elemlist
        self.typ = typ
        self.cov_factor = cov_factor
        self.os = os
        self.tmpsmiles = tmpsmiles

        self.radii = get_radii(labels)
        status, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(self.labels, self.coordinates, self.cov_factor, self.radii)

        #if status == 0:
        #    print("PROTONATION WITH STATUS=0, meaning probable steric clashes")
        #    for idx in range(len(self.labels)):
        #        print("%s  %.6f  %.6f  %.6f" % (self.labels[idx], self.coordinates[idx][0], self.coordinates[idx][1], self.coordinates[idx][2]))

        self.status = status     # 1 when correct, 0 when steric clashes
        self.conmat = tmpconmat
        self.connec = tmpconnec
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