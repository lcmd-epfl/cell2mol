import numpy as np
from cell2mol.hungarian import reorder
import copy
from cell2mol.xyz2mol import xyz2mol
from cell2mol.charge_assignment import check_rdkit_obj_connectivity, arrange_data_for_reorder, charge_state, protonation
from rdkit import Chem
import itertools

#######################################################
def balance_charge(unique_indices: list, unique_species: list, debug: int=0) -> list:

    # Function to Select the Best Charge Distribution for the unique species.
    # It accepts multiple charge options for each molecule/ligand/metal (poscharge, etc...).
    # NO: It should select the best one depending on whether the final metal charge makes sense or not.
    # In some cases, can accept metal oxidation state = 0, if no other makes sense

    iserror = False
    iterlist = []
    for idx, spec in enumerate(unique_species):
        toadd = []
        if spec.subtype == "metal":
            for tch in spec.possible_cs:
                toadd.append(tch)
        else :   
            if len(spec.possible_cs) == 1:
                toadd.append(spec.possible_cs[0].corr_total_charge)
            elif len(spec.possible_cs) > 1:
                for tch in spec.possible_cs:
                    toadd.append(tch.corr_total_charge)   
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
        final_charges= []
        for distr in tmpdistr:
            tmp = []
            for u in unique_indices:
                tmp.append(distr[u])
            alldistr.append(tmp)
            if debug >= 2: print("BALANCE: alldistr added:", tmp)

            final_charge_distribution = []
            for idx, d in enumerate(alldistr):
                if debug >= 2: print(f"BALANCE: distribution={d}")
                charges_sum = np.sum(d)
                if charges_sum == 0:
                    final_charge_distribution.append(d)
                    final_charges.append(distr)
    elif iserror:
        if debug >= 1: print("Error found in BALANCE: one species has no possible charges")
        final_charge_distribution = []

    return final_charge_distribution, final_charges
######################################################

def assign_charge_state_for_unique_species(unique_species, final_charges_tuple, debug: int=0):
    
    for specie, final_charge in zip(unique_species, final_charges_tuple):
        print(specie.unique_index, specie.formula)
        if (specie.subtype == "molecule" and specie.iscomplex == False) or (specie.subtype == "ligand"):
            charge_list = [cs.corr_total_charge for cs in specie.possible_cs]
            idx = charge_list.index(final_charge)
            cs = specie.possible_cs[idx]
            specie.charge_state = cs
            # print(specie.charge_state.protonation)
            specie.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
        elif specie.subtype == "metal" :
            charge_list = specie.possible_cs   
            idx = charge_list.index(final_charge)
            cs = specie.possible_cs[idx]
            specie.set_charge(cs) 
    for specie in unique_species:
        if (specie.subtype == "molecule" and specie.iscomplex == False) or (specie.subtype == "ligand"):
            print(specie.formula, specie.charge_state, specie.totcharge, specie.smiles)
    return unique_species
######################################################
def get_unique_indices(newcell, reference_species_list, debug: int=0):
    
    newcell.unique_indices = []
    newcell.species_list = []
    for mol in newcell.moleclist:
        if not mol.iscomplex:
            for ref in reference_species_list:
                if (ref.subtype == "molecule") and not ref.iscomplex:
                    issame = compare_molecules(ref, mol, debug=debug)
                    if issame:
                        mol.unique_index = ref.unique_index 
                        if debug >= 1: print(f"Matched {mol.formula} {ref.formula} {mol.unique_index} {ref.unique_index}")
                        newcell.unique_indices.append(mol.unique_index)
                        newcell.species_list.append(mol)
        else:
            for ref in reference_species_list:
                if ref.subtype == "ligand":
                    for lig in mol.ligands:
                        issame = compare_molecules(ref, lig, debug=debug)
                        if issame: 
                            lig.unique_index = ref.unique_index
                            if debug >= 1: print(f"Matched {lig.formula} {ref.formula} {lig.unique_index} {ref.unique_index}")
                            newcell.unique_indices.append(lig.unique_index)
                            newcell.species_list.append(lig)
                if ref.subtype == "metal":
                    for met in mol.metals:
                        if ref.get_parent_index("reference") == met.get_parent_index("reference"):
                            met.unique_index = ref.unique_index
                            if debug >= 1: print(f"Matched {met.formula} {ref.formula} {met.unique_index} {ref.unique_index}")
                            newcell.unique_indices.append(met.unique_index)
                            newcell.species_list.append(met)

    return newcell
######################################################
def get_unique_indices_old (newcell, refcell, debug: int=0):
    
    newcell.unique_indices = []
    for mol in newcell.moleclist:
        if not mol.iscomplex:
            for ref in newcell.refmoleclist:
                if not ref.iscomplex:
                    issame = compare_molecules(ref, mol, debug=debug)
                    if issame:
                        mol.unique_index = ref.unique_index 
                        if debug >= 1: print(f"Matched {mol.formula} {ref.formula} {ref.unique_index}")
                        newcell.unique_indices.append(mol.unique_index)
        else:
            for ref in newcell.refmoleclist:
                if ref.iscomplex:
                    for lig in mol.ligands:
                        for ref_lig in ref.ligands:
                            issame = compare_molecules(ref_lig, lig, debug=debug)
                            if issame: 
                                lig.unique_index = ref_lig.unique_index
                                if debug >= 1: print(f"Matched {lig.formula} {ref_lig.formula} {ref_lig.unique_index}")
                                newcell.unique_indices.append(lig.unique_index)

                    for met in mol.metals:
                        for ref_met in ref.metals:
                            if ref_met.get_parent_index("reference") == met.get_parent_index("reference"):
                                met.unique_index = ref_met.unique_index
                                if debug >= 1: print(f"Matched {met.formula} {ref_met.formula} {ref_met.unique_index}")
                                newcell.unique_indices.append(met.unique_index)

    return newcell
######################################################
def compare_molecules(ref, mol, debug: int=0):
    if (ref.natoms == mol.natoms) & (ref.formula == mol.formula):
        if (sorted(ref.get_parent_indices("reference")) == sorted(mol.get_parent_indices("reference"))):
            print("Matched", mol.formula, ref.formula, ref.get_parent_indices("reference"), mol.get_parent_indices("reference"))
            issame = True
            # mol.unique_index = ref.unique_index
            # set_charge_state_simple(ref, mol, debug=debug)
            # set_charge_state(ref, mol, mode=2, debug=debug) 
        else:
            issame = False
    else : 
        issame = False
    return issame

######################################################
def set_charge_state_simple (reference, target, debug: int=0):

    final_charge = reference.totcharge
    print("SET_CHARGE_STATE:", reference.charge_state)

    if target.subtype == "molecule" and target.iscomplex == False:
        if debug >=1 : print(f"({target.subtype}) {target.formula} {final_charge=} Create Empty PROTONATION for this specie")
        empty_list = [int(0)]*len(target.labels)
        empty_prot = protonation(target.labels, target.coord, target.cov_factor, 
                                int(0), empty_list, empty_list, empty_list, empty_list, typ="Empty", parent=target)
        cs = get_charge(final_charge, empty_prot)
    
    elif target.subtype == "ligand":
        if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Ligand")
        target.get_protonation_states(debug=debug)
        prot = target.protonation_states[0]
        cs = get_charge(reference.charge_state.uncorr_total_charge, prot)

        if len(target.protonation_states) != 1 :
            if debug >=1 : print("WARNING:", target.protonation_states)

    target.charge_state = cs
    if final_charge != cs.corr_total_charge:
        print(f"WARNING: {target.formula=} {final_charge=} {cs.corr_total_charge=} final_charge != cs.corr_total_charge")

    target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
    print(f"SET_CHARGE_STATE. {target.formula=} {target.totcharge=} {target.smiles=}")

######################################################
def set_charge_state(reference, target, mode, debug: int=0):

    final_charge = reference.totcharge
    print("SET_CHARGE_STATE:", reference.charge_state)

    if target.subtype == "molecule" and target.iscomplex == False:
        if debug >=1 : print(f"({target.subtype}) {target.formula} {final_charge=} Create Empty PROTONATION for this specie")
        empty_list = [int(0)]*len(target.labels)
        empty_prot = protonation(target.labels, target.coord, target.cov_factor, 
                                int(0), empty_list, empty_list, empty_list, empty_list, typ="Empty", parent=target)
        cs = get_charge(final_charge, empty_prot)
    
    elif target.subtype == "ligand":
        if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Ligand")
        prot = reference.charge_state.protonation
        temp_prot = copy.deepcopy(prot)
        temp_prot.parent = target
        
        if mode == 1:
            # For "reference" cell
            # target.get_protonation_states(debug=debug)
            # prot = target.protonation_states[0]
            # cs = get_charge(reference.charge_state.uncorr_total_charge, prot)

            # if len(target.protonation_states) != 1 :
            #     if debug >=1 : print(f"WARNING: {target.protonation_states=}")
            
            ref_data, target_data = arrange_data_for_reorder(reference, target)
            if debug >=2 : print(ref_data, target_data)
            dummy1, dummy2, map12 = reorder(ref_data, target_data, reference.coord, target.coord)
        
            if np.array_equal(map12, np.arange(len(target_data))):
                if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} No need to reorder")
                # temp_prot.coords = target.coord
                cs = get_charge(reference.charge_state.uncorr_total_charge, temp_prot)               
            else:
                reordered_prot = temp_prot.reorder(map12)
                # reordered_prot.coords = target.coord
                if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Reordered {map12=}")
                cs = get_charge(reference.charge_state.uncorr_total_charge, reordered_prot)
        
        elif mode == 2:
            print(f"{temp_prot.labels=} {len(temp_prot.labels)=} {len(temp_prot.coords)=} {len(temp_prot.block)=} {temp_prot.added_atoms=}")
            # For "unit" cell
            ref_data = reference.get_parent_indices("reference")
            target_data = target.get_parent_indices("reference")
            index_map = {value: index for index, value in enumerate(target_data)}
            sorted_indices = sorted(range(len(ref_data)), key=lambda i: index_map[ref_data[i]])            
            
            reordered_prot = temp_prot.reorder(sorted_indices)
            # reordered_prot.coords = target.coord
            if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Reordered {sorted_indices=}")
            cs = get_charge(reference.charge_state.uncorr_total_charge , reordered_prot)

    target.charge_state = cs
    if final_charge != cs.corr_total_charge:
        print(f"WARNING: {target.formula=} {final_charge=} {cs.corr_total_charge=} final_charge != cs.corr_total_charge")
    target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
    print(f"{target.formula=} {target.totcharge=} {target.smiles=}")

######################################################
def get_charge(charge: int, prot: object, allow: bool=True, debug: int=0): 
    ## Generates the connectivity of a molecule given a desired charge (charge).
    # The molecule is described by a protonation states that has labels, and the atomic cartesian coordinates "coords"
    # The adjacency matrix is also provided in the protonation state(adjmat)
    #:return charge_state which is an object with the necessary information for other functions to handle the result

    natoms = prot.natoms
    atnums = prot.atnums

    # prot.coords and prot.cov_factor will not be used
    mols = xyz2mol(atnums, prot.coords, prot.adjmat, prot.cov_factor, charge=charge, allow_charged_fragments=allow)
    print(f"GET_CHARGE.{len(mols)=} received from xyz2mol with charge {charge}")
    
    if len(mols) > 1: print("WARNING: More than 1 mol received from xyz2mol for initcharge:", charge)

    # Smiles are generated with rdkit
    smiles = Chem.MolToSmiles(mols[0])
    if debug >= 2: print(f"GET_CHARGE. {smiles=}")
    # Gets the resulting charges
    atom_charge = []
    total_charge = 0
    for i in range(natoms):
        a = mols[0].GetAtomWithIdx(i)  # Returns a particular Atom
        atom_charge.append(a.GetFormalCharge())
        total_charge += a.GetFormalCharge()

    # Connectivity is checked
    iscorrect = check_rdkit_obj_connectivity(mols[0], prot.natoms, charge, debug=debug)

    # Charge_state is initiated
    ch_state = charge_state(iscorrect, total_charge, atom_charge, mols[0], smiles, charge, allow, prot)

    return ch_state
    
######################################################
def prepare_mol (mol):
    tmp_atcharge = np.zeros((mol.natoms))
    tmp_smiles = []
    
    for lig in mol.ligands: 
        tmp_smiles.append(lig.smiles)
        parent_indices = lig.get_parent_indices("molecule")
        for kdx, a in enumerate(parent_indices):
            tmp_atcharge[a] = lig.atomic_charges[kdx]
    
    for met in mol.metals:  
        parent_index = met.get_parent_index("molecule")
        tmp_atcharge[parent_index] = met.charge  
        
    mol.set_charges(int(sum(tmp_atcharge)), atomic_charges=tmp_atcharge, smiles=tmp_smiles)
