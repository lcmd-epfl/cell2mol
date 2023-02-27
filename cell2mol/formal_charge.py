#!/usr/bin/env python

# fmt: off

import numpy as np
import itertools
import sys
import time
from typing import Tuple
from collections import defaultdict

from cell2mol.tmcharge_common import getradii, getconec, find_closest_metal
from cell2mol.xyz2mol import int_atom, xyz2mol
from cell2mol.missingH import getangle
from cell2mol.hungarian import reorder
from cell2mol.elementdata import ElementData
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


#######################################################
def classify_mols(moleclist: list, debug: int=0) -> Tuple[list, list, list, list]:

    # This subroutine reads all molecules and
    # identifies those that are identical based on a sort of ID number, which is
    # the variable "elemcountvec". This is a vector of occurrences for all elements in the elementdatabase

    molec_indices = []  # molecule index
    ligand_indices = []  # ligand index
    metal_ion_indices = []
    unique_indices = []  # Unique Specie (molecule or ligand) Index in Typelist
    unique_species = []  # List of Unique Species
    typelist_mols = []  # List of ID codes for Molecules
    typelist_ligs = []  # List of ID codes for Ligands
    typelist_mets = []

    # The reason for splitting typelist in two is that
    # some species might appear as ligands and as counterions, (eg. Cl-)
    # in the same unit cell. It is cleaner if these cases are treated separately

    # Note that both "typelists", and "unique_species" will be a list of tuples.
    specs_found = -1
    # Non-Complexes
    for idx, mol in enumerate(moleclist):
        if mol.type != "Complex":
            found = False
            for ldx, typ in enumerate(typelist_mols):
                if (mol.elemcountvec == typ[0]).all() and not found:
                    if (mol.adjtypes == typ[1]).all():
                        if mol.totmconnec == typ[2]:
                            found = True
                            kdx = typ[3]
                            if debug >= 2: print(f"Molecule {idx} is the same than {ldx} in typelist")
            if not found:
                specs_found += 1
                kdx = specs_found
                if debug >= 2: print(f"New molecule found with: formula={mol.formula}, totmconnec={mol.totmconnec}, and added in position {kdx}")
                typ_comparison = [mol.elemcountvec, mol.adjtypes, mol.totmconnec, kdx]
                typelist_mols.append(typ_comparison)
                unique_species.append(list([mol.type, mol]))

            jdx = "-"
            molec_indices.append(idx)
            ligand_indices.append(jdx)
            unique_indices.append(kdx)

    # Complexes
        elif mol.type == "Complex":
            for jdx, lig in enumerate(mol.ligandlist):
                found = False
    # Ligands
                for ldx, typ in enumerate(typelist_ligs):
                    if (lig.elemcountvec == typ[0]).all() and not found:
                        if (lig.adjtypes == typ[1]).all():
                            if lig.totmconnec == typ[2]:
                                found = True
                                kdx = typ[3]
                                if debug >= 2: print(f"Ligand {jdx} is the same than {ldx} in typelist")
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    if debug >= 2: print(f"New ligand found with: formula {lig.formula} and totmconnec={lig.totmconnec}, and added in position {kdx}")
                    typ_comparison = [lig.elemcountvec, lig.adjtypes, lig.totmconnec, kdx]
                    typelist_ligs.append(typ_comparison)
                    unique_species.append(list([lig.type, lig, mol]))

                molec_indices.append(idx)
                ligand_indices.append(jdx)
                unique_indices.append(kdx)

    # Metals
            for jdx, met in enumerate(mol.metalist):
                found = False
                for ldx, typ in enumerate(typelist_mets):
                    if (met.coord_sphere_ID == typ[0]).all() and not found:
                        found = True
                        kdx = typelist_mets[ldx][1]
                        if debug >= 2: print(f"Metal {jdx} is the same than {ldx} in typelist")
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    if debug >= 2: print(f"New Metal Center found with: labels {met.label} and added in position {kdx}")
                    typ_comparison = list([met.coord_sphere_ID, kdx])
                    typelist_mets.append(typ_comparison)
                    unique_species.append(list([met.type, met, mol]))

                molec_indices.append(idx)
                ligand_indices.append(jdx)
                unique_indices.append(kdx)

    if debug >= 2: print("CLASSIFY: molec_indices", molec_indices)
    if debug >= 2: print("CLASSIFY: ligand_indices", ligand_indices)
    if debug >= 2: print("CLASSIFY: unique_indices", unique_indices)

    nspecs = len(unique_species)
    for idx in range(0, nspecs):
        count = unique_indices.count(idx)
        if debug >= 2: print(f"CLASSIFY: specie {idx} appears {count} times, with type: {unique_species[idx][0]}")
        #unique_species[idx][1].occurrence = count

    return molec_indices, ligand_indices, unique_indices, unique_species


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
    atnums = [int_atom(label) for label in labels]  # from xyz2mol

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
def drive_get_poscharges(unique_species: list, debug: int=0) -> Tuple[list, bool]:

    Warning = False
    # For each specie, a plausible_charge_state is generated. These include the actual charge state, and the protonation it belongs to
    selected_charge_states = []
    for idx, spec in enumerate(unique_species):

        spec[1].poscharge = []
        spec[1].posatcharge = []
        spec[1].posobjlist = []
        spec[1].posspin = []
        spec[1].possmiles = []

        # Obtains charge for Organic Molecules
        if debug >= 1: print("")
        if spec[0] == "Other":
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("    #### NON-Complex ####")
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("          ", spec[1].natoms, spec[1].formula)
        elif spec[0] == "Ligand":
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("    #### Ligand ####")
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("          ", spec[1].natoms, spec[1].formula, spec[1].totmconnec)
        elif spec[0] == "Metal":
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("    #### Metal ####")
            if debug >= 1: print("    ---------------")
            if debug >= 1: print("          ", spec[1].label, "coordination_sphere\t", spec[1].coord_sphere)
            if debug >= 1: print("          ", spec[1].label, "coordinating_atoms\t", spec[1].coordinating_atoms)


        # Gets possible Charges for Ligands and Other
        if spec[0] == "Other" or spec[0] == "Ligand":
            tmp, Warning = get_poscharges(spec, debug=debug)
            if Warning: 
                if debug >= 1: print(f"Empty possible charges received for molecule {spec[1].formula}")
            else: 
                if debug >= 1: print("Charge state and protonation received for molecule", len(tmp)) 
                selected_charge_states.append(tmp)

        # Gets possible Charges for Metals
        elif spec[0] == "Metal":
            met = spec[1]
            mol = spec[2]
            met.poscharge = get_metal_poscharge(met, mol, debug=debug)
            if debug >= 1: print("Possible charges received for metal:", met.poscharge)
            selected_charge_states.append([])

    return selected_charge_states, Warning

#######################################################
def get_list_of_charges_to_try(spec: list, prot: object, debug: int=0) -> list:
    lchar = []

    #### Educated Guess on the Maximum Charge one can expect from the spec[1]
    if spec[0] == "Other":  maxcharge = 3
    elif spec[0] == "Ligand":
        count_non_connected_O = 0
        for a in spec[1].atoms:
            if a.label == "O" and a.mconnec == 0 and a.connec == 1:
                count_non_connected_O += 1
        if not spec[1].hapticity:
            maxcharge = spec[1].totmconnec + count_non_connected_O - prot.added_atoms
            if debug >= 2: print(f"MAXCHARGE: maxcharge set at {maxcharge} with {spec[1].totmconnec}+{count_non_connected_O}-{prot.added_atoms}")
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
    if spec[0] == "Ligand":
        list_of_protonations = define_sites(spec[1], spec[2], debug)
    elif spec[0] == "Other":
        empty_protonation = protonation(spec[1].labels, spec[1].coord, spec[1].factor, int(0), [], [], [], [], typ="Empty")
        list_of_protonations = [] 
        list_of_protonations.append(empty_protonation)
        if debug >= 2: print(f"    POSCHARGE: doing empty PROTONATION for this specie")
    if debug >= 2: print(f"    POSCHARGE: received {len(list_of_protonations)} protonations for this specie")

    ##############################
    #### Evaluates possible charges except if the ligand is a nitrosyl
    ##############################
    if spec[0] == "Ligand" and spec[1].natoms == 2 and "N" in spec[1].labels and "O" in spec[1].labels:
        for prot in list_of_protonations: 
            list_of_charge_states = []
            list_of_protonations_for_each_state = [] 
            chargestried = get_list_of_charges_to_try(spec, prot)
            if debug >= 2: print(f"    POSCHARGE will try charges {chargestried}") 

            for ich in chargestried:
                ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, ich, prot.factor)
                ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
                list_of_charge_states.append(ch_state)
                list_of_protonations_for_each_state.append(prot)
                if debug >= 2: print(f"    POSCHARGE: charge 0 with smiles {ch_state.smiles}")

            NO_type = get_nitrosyl_geom(spec[1])
            if NO_type == "Linear":
                best_charge_distr_idx = [2]
            elif NO_type == "Bent":
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
    elif spec[0] == "Ligand" and spec[1].natoms == 3 and len(np.unique(spec[1].labels)) == 1 and "N" in np.unique(spec[1].labels):
        poscharge = -1 
        # metal-coordinating N charge +1
        # The other two N atoms charge -1
    ##############################
    #### Evaluates possible charges except if the ligand is a triiodide (I3-) ion
    ##############################               
    elif spec[0] == "Ligand" and spec[1].natoms == 3 and len(np.unique(spec[1].labels)) == 1 and "I" in np.unique(spec[1].labels):
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
                    ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, ich, prot.factor)
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
def get_nitrosyl_geom(ligand: object, debug: int=0) -> str:
    # Function that determines whether the M-N-O angle of a Nitrosyl "ligand" is "Bent" or "Linear"
    # Each case is treated differently
    #:return NO_type: "Linear" or "Bent"

    for idx, a in enumerate(ligand.atoms):
        if a.label == "N":
            central = a.coord.copy()
        if a.label == "O":
            extreme = a.coord.copy()

    dist = []
    for idx, met in enumerate(ligand.metalatoms):
        metal = np.array(met.coord)
        dist.append(np.linalg.norm(central - metal))
    tgt = np.argmin(dist)

    metal = ligand.metalatoms[tgt].coord.copy()
    if debug >= 2: print("NITRO coords:", central, extreme, metal)

    vector1 = np.subtract(np.array(central), np.array(extreme))
    vector2 = np.subtract(np.array(central), np.array(metal))
    if debug >= 2: print("NITRO Vectors:", vector1, vector2)

    angle = getangle(vector1, vector2)
    if debug >= 2: print("NITRO ANGLE:", angle, np.degrees(angle))

    if np.degrees(angle) > float(160): NO_type = "Linear"
    else: NO_type = "Bent"

    return str(NO_type)

#######################################################
def get_metal_poscharge(metal: object, molecule: object, debug: int=0) -> list:
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
    at_charge[23] = [1, 2, 3, 4]  # V
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
    at_charge[41] = [5, 4, 3]  # Nb
    at_charge[42] = [0, 2, 3, 4, 5, 6]  # Mo
    at_charge[43] = [1, 2, 3, 4]  # Tc
    at_charge[44] = [2, 3]  # Ru
    at_charge[45] = [1, 2, 3]  # Rh
    at_charge[46] = [2]  # Pd
    at_charge[47] = [1]  # Ag
    at_charge[48] = [2]  # Cd
    # 3rd-row transition metals.
    at_charge[57] = []  # La
    at_charge[72] = [4]  # Hf
    at_charge[73] = [2, 3, 4]  # Ta
    at_charge[74] = [0, 2, 3, 4, 5, 6]  # W
    at_charge[75] = [1, 2, 3, 4, 5, 7]  # Re
    at_charge[76] = [0, 2, 3, 4, 5]  # Os
    at_charge[77] = [1, 3]  # Ir
    at_charge[78] = [2, 4]  # Pt
    at_charge[79] = [1, 3]  # Au
    at_charge[80] = [2]  # Hg

    poscharge = at_charge[atnum]

    list_of_zero_OS = ["Fe", "Ni", "Ru"]
    if metal.label in list_of_zero_OS:
        # In some cases, it adds 0 as possible metal charge
        # -if it has CO ligands
        if any((lig.natoms == 2 and "C" in lig.labels and "O" in lig.labels) for lig in molecule.ligandlist):
            if int(0) not in poscharge:
                poscharge.append(int(0))
        # -if it has any ligand with hapticity
        if any((lig.hapticity) for lig in molecule.ligandlist):
            if int(0) not in poscharge:
                poscharge.append(int(0))
    
    return poscharge


#######################################################
def prepare_unresolved(unique_indices: list, unique_species: list, distributions: list, debug: int=0):

    list_molecules = [] 
    list_indices = []
    list_options = []
    if debug >= 2: print("")
    for idx, spec_tuple in enumerate(unique_species): 
        if spec_tuple[0] == "Metal":
            pos = [jdx for jdx, uni in enumerate(unique_indices) if uni == idx]
            if debug >= 2: print(f"UNRESOLVED: found metal in positions={pos} of the distribution")
            values = []
            for distr in distributions:
                values.append(distr[pos[0]])
            options = list(set(values))
            if debug >= 2: print(f"UNRESOLVED: list of values={values}\n")
            if debug >= 2: print(f"UNRESOLVED: options={options}\n")

            if len(options) > 1:
                list_molecules.append(spec_tuple[2])
                list_indices.append(spec_tuple[1].atlist)
                list_options.append(options)

    return list_molecules, list_indices, list_options

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
        if len(spec.poscharge) == 1:
            toadd.append(spec.poscharge[0])
        elif len(spec.poscharge) > 1:
            for tch in spec.poscharge:
                toadd.append(tch)
        elif len(spec.poscharge) == 0:
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
def define_sites(ligand: object, molecule: object, debug: int=0) -> list:
    # This function is the heart of the program.
    # It decides whether a connected atom must be temporarily added an extra element
    # when running the getcharge function
    # It does so in order to generate a meaningful connectivity for the ligand,
    # without any additional difficulty due to the missing M-L bond(s)

    # The function works on all "groups" of a ligand,
    # which are groups of atoms that are all connected among themselves and to the metal
    # For instance, in a Cp ligand, all C atoms belong to the same group.
    # Also, for the ligand CN, C is the only group

    # Groups are identified in the part of the code that does the cell reconstruction, and have properties.
    # One of this properties is whether they form haptic bonds.
    # Haptic groups require a catalogue of rules, since they behave differently from regular groups (i.e. non-haptic)
    # These rules are defined below.

    # If a group is not haptic. Then rules are applied depending on the type of the connected atom.
    # Ideally, these rules can be defined depending on the adjacency of the atom.
    # In some cases, this is not the case, and so we resort to an actual connectivity,
    # in what I call a non-local approach.

    # The non-local approach consists of generating a tentative connectivity using xyz2mol, assuming a neutral charge.
    # This is not necessarily true, but hopefully the connectivity of the connected atom will be correct.

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
    for g in ligand.grouplist:

        ########################
        # Cases with Hapticity #
        ########################
        if len(g.hapttype) > 0:  # then the group has hapticity
            Selected_Hapticity = False
            if debug >= 2: print("        DEFINE_SITES: addressing group with hapticity:", g.hapttype)

            if "h5-Cp" in g.hapttype and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h5-Cp with atlist:", g.atlist)
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    # print(idx, a.label, a.mconnec)
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                            # print("added")
                        else:
                            block[idx] = 1

            elif "h7-Cicloheptatrienyl" in g.hapttype and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h7-Cicloheptatrienyl")
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif "h5-AsCp" in g.hapttype and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h5-AsCp")
                Selected_Hapticity = True

                # Rules change depending on whether the ring is substituted or not 
                issubstituted = False
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        for jdx in a.adjacency:
                            if ligand.labels[jdx] != "As":
                                issubstituted = True
                if issubstituted: 
                    tobeadded = 0
                else: 
                    tobeadded = 1

                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif "h5-Pentaphosphole" in g.hapttype and not Selected_Hapticity: ## Case of IMUCAX
                if debug >= 2: print("        DEFINE_SITES: It is an h5-Pentaphosphole")
                Selected_Hapticity = True

                # Rules change depending on whether the ring is substituted or not 
                issubstituted = False
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        for jdx in a.adjacency:
                            if ligand.labels[jdx] != "P":
                                issubstituted = True
                if issubstituted: 
                    tobeadded = 0
                else: 
                    tobeadded = 1

                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif ("h3-Allyl" in g.hapttype or "h3-Cp" in g.hapttype) and not Selected_Hapticity:
                # if "h3-Allyl" or "h3-Cp" in g.hapttype:
                if debug >= 2: print("        DEFINE_SITES: It is either an h3-Allyl or an h3-Cp")
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx] = 1

            elif ("h4-Benzene" in g.hapttype or "h4-Butadiene" in g.hapttype) and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is either an h4-Benzene or an h4-Butadiene")
                if debug >= 2: print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        block[idx] = 1

            elif ("h2-Benzene" in g.hapttype or "h2-Butadiene" or "h2-ethylene" in g.hapttype) and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is either an h2-Benzene, an h2-ethylene or an h2-Butadiene")
                if debug >= 2: print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        block[idx] = 1

            elif "h4-Enone" in g.hapttype and not Selected_Hapticity:
                if debug >= 2: print("        DEFINE_SITES: It is an h4-Enone")
                if debug >= 2: print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        block[idx] = 1

            # If the group hapticity type is not recognized -or instructions are not defined-, nothing is done
            if not Selected_Hapticity:
                if debug >= 2: print(f"        DEFINE_SITES: {g.hapttype} not recognized or new rules are necessary")

        else:  # cases without hapticity
            ions = ["F", "Cl","Br","I","As"]  # Atoms for which an H atom is always added
            ###########################
            # Cases with No Hapticity #
            ###########################
            # An initial attempt to add elements based on the adjacency of the connected atom
            for idx in g.atlist:
                a = ligand.atoms[idx]
                if debug >= 2: print(f"        DEFINE_SITES: evaluating non-haptic group with index {idx} and label {a.label}")
                # Simple Ionic Case
                if a.label in ions:
                    if a.connec == 0:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    elif a.connec >= 1:
                        block[idx] = 1
                # Oxigen
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
        new_prot = protonation(newlab, newcoord, ligand.factor, added_atoms, addedlist, block, metal_electrons, elemlist) 
        list_of_protonations.append(new_prot)
    else:
        # Generate the new adjacency matrix after local elements have been added to be sent to xyz2mol
        local_labels = newlab.copy()
        local_coords  = newcoord.copy()
        local_radii = getradii(local_labels)
        local_natoms = len(local_labels)
        local_atnums = [int_atom(label) for label in local_labels]  # from xyz2mol.py
        dummy, local_conmat, local_connec, dummy, dummy = getconec(local_labels, local_coords, ligand.factor, local_radii)

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
        
            new_prot = protonation(newlab, newcoord, ligand.factor, added_atoms, addedlist, block, metal_electrons, elemlist, smi, os, typ="Non-local") 
            if new_prot.status == 1 and new_prot.added_atoms == os+local_added_atoms:
                list_of_protonations.append(new_prot)
                if debug >= 2:  print(f"        DEFINE_SITES: Protonation SAVED with {added_atoms} atoms added to ligand. status={new_prot.status}")
            else:
                if debug >= 2:  print(f"        DEFINE_SITES: Protonation DISCARDED. Steric Clashes found when adding atoms. status={new_prot.status}")
                
    return list_of_protonations 

#######################################################
def add_atom(labels: list, coords: list, site: int, ligand: object, metalist: list, element: str="H", debug: int=0) -> Tuple[bool, list, list]:
    # This function adds one atom of a given "element" to a given "site=atom index" of a "ligand".
    # It does so at the position of the closest "metal" atom to the "site"
    #:return newlab: labels of the original ligand, plus the label of the new element
    #:return newcoord: same as above but for coordinates

    # Original labels and coordinates are copied
    isadded = False
    posadded = len(labels)
    newlab = labels.copy()
    newcoord = coords.copy()
    newlab.append(str(element))  # One H atom will be added

    if debug >= 2: print("        ADD_ATOM: Metalist length", len(metalist))
    # It is adding the element (H, O, or whatever) at the vector formed by the closest TM atom and the "site"
    for idx, a in enumerate(ligand.atoms):
        if idx == site:
            tgt, apos, dist = find_closest_metal(a, metalist)
            idealdist = a.radii + elemdatabase.CovalentRadius2[element]
            addedHcoords = apos + (metalist[tgt].coord - apos) * (idealdist / dist)  # the factor idealdist/dist[tgt] controls the distance
            newcoord.append([addedHcoords[0], addedHcoords[1], addedHcoords[2]])  # adds H at the position of the closest Metal Atom

            # Evaluates the new adjacency matrix.
            tmpradii = getradii(newlab)
            dummy, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(newlab, newcoord, ligand.factor, tmpradii)
            # If no undesired adjacencies have been created, the coordinates are kept
            if tmpconnec[posadded] <= 1:
                isadded = True
                if debug >= 2: print(f"        ADD_ATOM: Chosen {tgt} Metal atom. {element} is added at site {site}")
            # Otherwise, coordinates are reset
            else:
                if debug >= 1: print(f"        ADD_ATOM: Chosen {tgt} Metal atom. {element} was added at site {site} but RESET due to connec={tmpconnec[posadded]}")
                isadded = False
                newlab = labels.copy()
                newcoord = coords.copy()
    
    return isadded, newlab, newcoord


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
            print(f"PREPARE: Molecule {idx} is a {mol.type} with formula {mol.formula}"),
        print("")

    for idx, sel in enumerate(selected_charge_states):
        for jdx, opt in enumerate(sel):
            chstate = opt[0]
            prot = opt[1]
            if debug >= 2: print(f"PREPARE: State {idx} and option {jdx}. Target state and protonation received with {chstate.corr_total_charge} and {prot.added_atoms}")
 
    for idx, mol in enumerate(moleclist):
        if debug >= 2: print("")
        if debug >= 2: print(f"PREPARE: Molecule {idx} is a {mol.type} with formula {mol.formula}"),
    
        ###################################
        ### FOR SOLVENT AND COUNTERIONS ###
        ###################################
        if mol.type == "Other":
            specie = unique_indices[idxtoallocate]
            spec_object = unique_species[specie][1]
            if debug >= 2: print(f"PREPARE: Specie number={specie} with molecule poscharges: {spec_object.poscharge}")
            if debug >= 2: print(f"PREPARE: Doing molecule {idx} with idxtoallocate: {idxtoallocate}")
    
            allocated = False
            for jdx, ch in enumerate(spec_object.poscharge):
                if final_charge_distribution[idxtoallocate] == ch and not allocated:   # If the charge in poscharges coincides with the one for this entry in final_distribution
    
                    tgt_charge_state = selected_charge_states[specie][jdx][0]
                    tgt_protonation = selected_charge_states[specie][jdx][1]
                    if debug >= 2: print(f"PREPARE: target state and protonation loaded, with {tgt_charge_state.corr_total_charge} and {tgt_protonation.added_atoms}")
                    allocated = True
                    ch_state = getcharge(mol.labels, mol.coord, mol.conmat, ch, mol.factor)
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
        elif mol.type == "Complex":
            if debug >= 2: print(f"PREPARE: Molecule {moleclist.index(mol)} has {len(mol.ligandlist)} ligands")
    
            for kdx, lig in enumerate(mol.ligandlist):
                #while not Warning:
                #try:
                allocated = False
                specie = unique_indices[idxtoallocate]
                spec_object = unique_species[specie][1]
                if debug >= 2: print("")
                if debug >= 2: print(f"PREPARE: Ligand {kdx}, formula: {lig.formula} is specie {specie}")
                if debug >= 2: print(f"PREPARE: Ligand poscharges: {spec_object.poscharge}")
                if debug >= 2: print(f"PREPARE: Doing ligand {kdx} with idxtoallocate {idxtoallocate}")
        
                if lig.natoms == 2 and "N" in lig.labels and "O" in lig.labels:
                    isnitrosyl = True
                else:
                    isnitrosyl = False
        
                for jdx, ch in enumerate(spec_object.poscharge):
                    if debug >= 2: print(f"PREPARE: Doing {ch} of options {spec_object.poscharge}. jdx={jdx}")
                    tgt_charge_state = selected_charge_states[specie][jdx][0]
                    tgt_protonation = selected_charge_states[specie][jdx][1]
                    if debug >= 2: print(f"PREPARE: Found Target Prot State with {tgt_protonation.added_atoms} added atoms and {tgt_protonation.os} OS") 
                    if debug >= 2: print(f"PREPARE: addedlist of Target Prot State: {tgt_protonation.addedlist}")
        
                    if final_charge_distribution[idxtoallocate] == ch and not allocated:
                        allocated = True
        
                        # RE-RUNS the Charge assignation for same-type molecules in the cell
                        list_of_protonations = define_sites(lig, mol, debug=1)
                        found_prot = False
                        
                        # Hungarian sort
                        issorted = False
                        if not lig.hapticity:
                            tini_hun = time.time()

                            # Adding connectivity data to labels to improve the hungarian sort
                            ligand_data = []
                            ref_data = []
                            for a in lig.atoms:
                                ligand_data.append(a.label+str(a.mconnec))
                            for a in spec_object.atoms:
                                ref_data.append(a.label+str(a.mconnec))
                            dummy1, dummy2, map12 = reorder(ref_data, ligand_data, spec_object.coord, lig.coord)

                            issorted = True
                            tend_hun = time.time()
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
                            if isnitrosyl:
                                NO_type = get_nitrosyl_geom(lig)
                                if NO_type == "Linear": NOcharge = 1   #NOcharge is the charge with which I need to run getcharge to make it work
                                if NO_type == "Bent": NOcharge = 0
        
                                ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, NOcharge, prot.factor)
                                ch_state.correction(prot.addedlist, prot.metal_electrons, prot.elemlist)
        
                                if debug >= 2: print(f"PREPARE: Found Nitrosyl of type= {NO_type}")
                                if debug >= 2: print(f"PREPARE: Wanted charge {ch}, obtained: {ch_state.corr_total_charge}")
                                if debug >= 2: print(f"PREPARE: smiles: {ch_state.smiles}")
                            else:
                                if debug >= 2: print(f"PREPARE: Sending getcharge with prot.added_atoms={prot.added_atoms} to obtain charge {ch}")
                                ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, ch+prot.added_atoms, prot.factor, tgt_charge_state.allow)
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
                                chargestried = get_list_of_charges_to_try(tmpobject, prot)
                                for ich in chargestried:
                                    ch_state = getcharge(prot.labels, prot.coordinates, prot.conmat, ich, prot.factor)
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
    
            for kdx, met in enumerate(mol.metalist):
                specie = unique_indices[idxtoallocate]
                spec_object = unique_species[specie][1]
                allocated = False
                if debug >= 2: print("")
                if debug >= 2: print(f"PREPARE: Metal {kdx}, label {met.label} is specie {specie}")
                if debug >= 2: print(f"PREPARE: Metal poscharges: {spec_object.poscharge}")
                for jdx, ch in enumerate(spec_object.poscharge):
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
                for lig in mol.ligandlist:
                    tmp_smiles.append(lig.smiles)
                    for kdx, a in enumerate(lig.atlist):
                        tmp_atcharge[a] = lig.atcharge[kdx]
                for met in mol.metalist:
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
        if mol.type != "Complex":
            if debug >= 2: print(f"BUILD BONDS: doing mol with Natoms {mol.natoms}")
            # Checks that the gmol and rdkit-mol objects have same order
            for idx, a in enumerate(mol.atoms):

                # Security Check. Confirms that the labels are the same
                #if debug >= 2: print("BUILD BONDS: atom", idx, a.label)
                rdkitatom = mol.rdkit_mol.GetAtomWithIdx(idx)
                tmp = rdkitatom.GetSymbol()
                if a.label != tmp: 
                    if debug >= 1: print("Error in Build_Bonds. Atom labels do not coincide. GMOL vs. MOL:",a.label,tmp)
                else:
                    # First part. Creates bond information
                    starts = []
                    ends = []
                    orders = []
                    for b in rdkitatom.GetBonds():
                        bond_startatom = b.GetBeginAtomIdx()
                        bond_endatom = b.GetEndAtomIdx()
                        bond_order = b.GetBondTypeAsDouble()
                        # if debug >= 1: print(bond_endatom, bond_order)
                        if mol.atoms[bond_endatom].label != mol.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol():
                            if debug >= 1: print("Error with Bond EndAtom",mol.atoms[bond_endatom].label,mol.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol())
                        else:
                            if bond_endatom == idx:
                                starts.append(bond_endatom)
                                ends.append(bond_startatom)
                                orders.append(bond_order)
                            elif bond_startatom == idx:
                                starts.append(bond_startatom)
                                ends.append(bond_endatom)
                                orders.append(bond_order)
                            else:
                                if debug >= 1: print("Warning BUILD_BONDS: Index atom is neither start nor end bond")
                    a.bonds(starts, ends, orders)

    if debug >= 2: print("")
    if debug >= 2: print("BUILD_BONDS: Doing 2nd Part")
    if debug >= 2: print("###########################")

    #######
    # 2nd Part. Creates Ligand Information
    #######
    for mol in moleclist:
        if debug >= 2: print(f"BUILD BONDS: doing mol {mol.formula} with Natoms {mol.natoms}")
        if mol.type == "Complex":
            for lig in mol.ligandlist:
                if debug >= 2: print(f"BUILD BONDS: doing ligand {lig.formula}")

                for idx, a in enumerate(lig.atoms):
                    # Security Check. Confirms that the labels are the same
                    rdkitatom = lig.rdkit_mol.GetAtomWithIdx(idx)
                    tmp = rdkitatom.GetSymbol()
                    if a.label != tmp:
                        if debug >= 1: print(f"Error in Build_Bonds. Atom labels do not coincide. GMOL vs. MOL: {a.label} {tmp}")
                        if debug >= 1: print("BUILD BONDS DEBUG:")
                        if debug >= 1: print(f"Ligand; {lig.labels}")
                        if debug >= 1: print("Atoms of RDKIT-Object")
                        if debug >= 1: 
                            for kdx, a in enumerate(lig.rdkit_mol.GetAtoms()):
                                print(kdx, a.GetSymbol())
                            print("Atoms of GMOL-Object")
                            for kdx, a in enumerate(lig.atoms):
                                print(kdx, a.label)
                    else:
                        # First part. Creates bond information
                        starts = []
                        ends = []
                        orders = []
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
                                        starts.append(bond_endatom)
                                        ends.append(bond_startatom)
                                        orders.append(bond_order)
                                    elif bond_startatom == idx:
                                        starts.append(bond_startatom)
                                        ends.append(bond_endatom)
                                        orders.append(bond_order)
                        a.bonds(starts, ends, orders)

    if debug >= 2: print("")
    if debug >= 2: print("BUILD_BONDS: Doing 3rd Part")
    if debug >= 2: print("###########################")

    #######
    # 3rd Part. Merges Ligand Information into Molecule Object using the atlists
    #######
    for mol in moleclist:
        if debug >= 2: print("BUILD BONDS: doing mol", mol.formula, "with Natoms", mol.natoms)
        if mol.type == "Complex":
            allstarts = []
            allends = []
            allorders = []
            # Adds atoms within ligands
            for lig in mol.ligandlist:
                for a in lig.atoms:
                    for b in a.bond:
                        allstarts.append(lig.atlist[b[0]])
                        allends.append(lig.atlist[b[1]])
                        allorders.append(b[2])

            # Adds Metal-Ligand Bonds, with an arbitrary 0.5 order:
            for idx, row in enumerate(mol.mconmat):
                # if debug >= 2: print(row)
                for jdx, val in enumerate(row):
                    if val > 0:
                        # if debug >= 2: print(idx, jdx, val)
                        allstarts.append(idx)
                        allends.append(jdx)
                        allorders.append(0.5)

            # I sould work to add Metal-Metal Bonds. Would need to work on the Metal class:
            # Finally, puts everything together, and creates bonds for MOLECULE atom objects
            for idx, a in enumerate(mol.atoms):
                starts = []
                ends = []
                orders = []
                group = []
                for entry in zip(allstarts, allends, allorders):
                    if entry[0] == idx or entry[1] == idx:
                        if entry not in group and (entry[1], entry[0], entry[2]) not in group:
                            starts.append(entry[0])
                            ends.append(entry[1])
                            orders.append(entry[2])
                            group.append(entry)

                a.bonds(starts, ends, orders)

    #######
    # 4th Part. Corrects Ligand Smiles to Remove Added H atoms, the old smiles is stored in "lig.smiles_with_H" as it can still be useful
    #######
    for mol in moleclist:
        if debug >= 2: print("BUILD BONDS: doing mol", mol.formula, "with Natoms", mol.natoms)
        if mol.type == "Complex":
            mol.smiles = []
            mol.smiles_with_H = []
            for lig in mol.ligandlist:
                lig.smiles_with_H = lig.smiles  
                lig.smiles, lig.rdkit_mol = correct_smiles_ligand(lig)
                mol.smiles.append(lig.smiles)
                mol.smiles_with_H.append(lig.smiles_with_H)
        
    return moleclist


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

############
def correct_smiles_ligand(lig: object):
    ## Receives a ligand class object and constructs the smiles and the rdkit_mol object from scratch, using atoms and bond information

    Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
    #### Creates an empty editable molecule
    rwlig = Chem.RWMol()    
 
    # Adds atoms with their formal charge 
    for jdx, a in enumerate(lig.atoms):        
        rdkit_atom = Chem.Atom(lig.atnums[jdx])
        rdkit_atom.SetFormalCharge(int(a.charge))
        rdkit_atom.SetNoImplicit(True)
        rwlig.AddAtom(rdkit_atom)
                       
    # Sets bond information and hybridization
    for jdx, a in enumerate(lig.atoms):
        nbonds = 0
        for bond in a.bond:
            nbonds += 1
            isaromatic = False
            if bond[2] == 1.0: btype = Chem.BondType.SINGLE
            elif bond[2] == 2.0: btype = Chem.BondType.DOUBLE
            elif bond[2] == 3.0: btype = Chem.BondType.TRIPLE
            elif bond[2] == 1.5: 
                btype = Chem.BondType.AROMATIC
                rdkit_atom.SetIsAromatic(True)
            if bond[0] == jdx and bond[1] > jdx: rwlig.AddBond(bond[0], bond[1], btype)

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

###################
### NEW OBJECTS ###
###################

class protonation(object):
    def __init__(self, labels, coordinates, factor, added_atoms, addedlist, block, metal_electrons, elemlist, tmpsmiles=" ", os=int(0), typ="Local"):
        self.labels = labels
        self.coordinates = coordinates
        self.added_atoms = added_atoms
        self.addedlist = addedlist
        self.block = block
        self.metal_electrons = metal_electrons 
        self.elemlist = elemlist
        self.typ = typ
        self.factor = factor
        self.os = os
        self.tmpsmiles = tmpsmiles

        self.radii = getradii(labels)
        status, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(self.labels, self.coordinates, self.factor, self.radii)

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

# fmt: on
