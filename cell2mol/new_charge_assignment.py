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
            # idx = charge_list.index(final_charge)
            cs = specie.possible_cs[idx]
            specie.set_charge(cs) 
    for specie in unique_species:
        if (specie.subtype == "molecule" and specie.iscomplex == False) or (specie.subtype == "ligand"):
            print(specie.formula, specie.charge_state, specie.totcharge, specie.smiles)
    return unique_species

######################################################
def print_possible_and_selected_cs (newcell, refcell, debug: int=0):
    if debug >= 1:
        print(f"\npossible charge states and charge of selected charge state for unique species")
        for idx, (specie, select) in enumerate(zip(refcell.unique_species, refcell.selected_cs)):
            print(f"Unique {specie.unique_index=} {specie.formula=}")
            print(f"charge of selected charge state={select}\n{specie.possible_cs=}\n")
        
        print(f"species list in the reference and their unique indices")
        for specie, idx in zip(refcell.species_list, refcell.unique_indices):
            print(f"\t{specie.formula=} {specie.unique_index=}")
            if idx != specie.unique_index:
                print(f"WARNING: {specie.formula=} {specie.unique_index=} {idx=} from refcell unique indices")
        
        print(f"species list in the unit cell and their unique indices")
        for specie, idx in zip(newcell.species_list, newcell.unique_indices):
            print(f"\t{specie.formula=} {specie.unique_index=}")
            if idx != specie.unique_index:
                print(f"WARNING: {specie.formula=} {specie.unique_index=} {idx=} from newcell unique indices")

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
        if debug >=1 : print(f"SET_CHARGE_STATE:({target.subtype}) {target.formula} {final_charge=} Create Empty PROTONATION for this specie")
        empty_list = [int(0)]*len(target.labels)
        empty_prot = protonation(target.labels, target.coord, target.cov_factor, 
                                int(0), empty_list, empty_list, empty_list, empty_list, typ="Empty", parent=target)
        cs = get_charge(final_charge, empty_prot)
    
    elif target.subtype == "ligand":
        if debug >=1 : print(f"SET_CHARGE_STATE:({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Ligand")
        prot = reference.charge_state.protonation
        temp_prot = copy.deepcopy(prot)
        temp_prot.parent = target
        
        if mode == 1:
            # For "reference" cell
            target.get_protonation_states(debug=debug)
            target.get_possible_cs(debug=debug)
            charge_list = [cs.corr_total_charge for cs in target.possible_cs]
            idx = charge_list.index(final_charge)
            cs = target.possible_cs[idx]
            # if len(target.protonation_states) != 1 :
            #     if debug >=1 : print(f"WARNING: {target.protonation_states=}")
            
            # ref_data, target_data = arrange_data_for_reorder(reference, target)
            # if debug >=2 : print(ref_data, target_data)
            # dummy1, dummy2, map12 = reorder(ref_data, target_data, reference.coord, target.coord)
        
            # if np.array_equal(map12, np.arange(len(target_data))):
            #     if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} No need to reorder")
            #     # temp_prot.coords = target.coord
            #     cs = get_charge(reference.charge_state.uncorr_total_charge, temp_prot)               
            # else:
            #     reordered_prot = temp_prot.reorder(map12)
            #     # reordered_prot.coords = target.coord
            #     if debug >=1 : print(f"({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Reordered {map12=}")
            #     cs = get_charge(reference.charge_state.uncorr_total_charge, reordered_prot)
        
        elif mode == 2:
            print(f"SET_CHARGE_STATE:{temp_prot.labels=} {len(temp_prot.labels)=} {len(temp_prot.coords)=} {len(temp_prot.block)=} {temp_prot.added_atoms=}")
            # For "unit" cell
            ref_data = reference.get_parent_indices("reference")
            target_data = target.get_parent_indices("reference")
            index_map = {value: index for index, value in enumerate(target_data)}
            sorted_indices = sorted(range(len(ref_data)), key=lambda i: index_map[ref_data[i]])            
            
            reordered_prot = temp_prot.reorder(sorted_indices)
            # reordered_prot.coords = target.coord
            if debug >=1 : print(f"SET_CHARGE_STATE:({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Reordered {sorted_indices=}")
            cs = get_charge(reference.charge_state.uncorr_total_charge , reordered_prot)

    target.charge_state = cs
    if final_charge != cs.corr_total_charge:
        print(f"SET_CHARGE_STATE: WARNING!!! {target.formula=} {final_charge=} {cs.corr_total_charge=} final_charge != cs.corr_total_charge")
    target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
    print(f"SET_CHARGE_STATE:{target.formula=} {target.totcharge=} {target.smiles=}")

######################################################
def get_charge(charge: int, prot: object, allow: bool=True, embed_chiral: bool=True, debug: int=0): 
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

######################################################
def create_bonds_specie (specie, debug: int=0):
    from cell2mol.classes import bond
    if debug >= 1: print(f"CREATE_bonds_specie: {specie.formula=}, {specie.subtype=} {specie.smiles=}")
    n_atoms = specie.natoms # e.g. 9 
    n_atoms_rdkit = specie.rdkit_obj.GetNumAtoms() # e.g.10 
    if debug >= 1: print(f"CREATE_bonds_specie: {specie.formula=}, {specie.subtype=}")

    if n_atoms == n_atoms_rdkit:
        if debug >= 2: print(f"\tNumber of atoms in {specie.subtype} object and RDKit object are equal: {n_atoms} {n_atoms_rdkit}")
        for idx, rdkit_atom in enumerate(specie.rdkit_obj.GetAtoms()): # e.g. idx 0, 1, 2, 3, 4, 5, 6, 7, 8
            if debug >= 2: print(f"\t{idx=}", rdkit_atom.GetSymbol(), "Number of bonds :", len(rdkit_atom.GetBonds()))
            if len(rdkit_atom.GetBonds()) == 0:
                if debug >= 1: print(f"\tNO BONDS CREATED for {specie.atoms[idx].label} due to no bonds in {specie.subtype} RDKit object")
            else:
                for b in rdkit_atom.GetBonds():
                    bond_startatom = b.GetBeginAtomIdx()
                    bond_endatom   = b.GetEndAtomIdx()
                    bond_order     = b.GetBondTypeAsDouble()
                    if specie.atoms[bond_endatom].label != specie.rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol():
                        if debug >= 1: print(f"\tError with Bond EndAtom", specie.atoms[bond_endatom].label, specie.rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol())
                    else:
                        if bond_endatom == idx:
                            start = bond_endatom
                            end   = bond_startatom
                        elif bond_startatom == idx:
                            start = bond_startatom
                            end   = bond_endatom      

                        # create new bond object
                        if debug >=2: print(f"\tBOND CREATED", idx, start, end, bond_order, specie.atoms[start].label, specie.atoms[end].label)
                        new_bond = bond(specie.atoms[start], specie.atoms[end], bond_order)
                        specie.atoms[idx].add_bond(new_bond)
                
                if hasattr(specie.atoms[idx], "bonds"):
                    if debug >=1 : 
                        print(f"\tBONDS", [(bd.atom1.label, bd.atom2.label, bd.order, round(bd.distance,3)) for bd in specie.atoms[idx].bonds])
                else:
                    if specie.natoms == 1:
                        if debug >=1: print(f"\tNO BONDS CREATED for {specie.atoms[idx].label} because it is the only atom in {specie.subtype} object")
                        pass
                    else:
                        if debug >=1: print(f"\tNO BONDS for {specie.atoms[idx].label} with {specie.subtype} RDKit object index {idx}. Please check the RDKit object.")
                        return False # return False if no bonds are created
    else:
        if debug >= 1: print(f"\tNumber of atoms in {specie.subtype} object and RDKit object are different: {n_atoms} {n_atoms_rdkit}")
        if debug >= 2: print(f"\t{[(i, atom.label) for i, atom in enumerate(specie.atoms)]}")
        if debug >= 2: print(f"\t{[(i, atom.GetSymbol()) for i, atom in enumerate(specie.rdkit_obj.GetAtoms())]}")       
        non_bonded_atoms = list(range(0, n_atoms_rdkit))[n_atoms:]
        if debug >= 2: print(f"\tNON_BONDED_ATOMS", non_bonded_atoms)

        for idx, rdkit_atom in enumerate(specie.rdkit_obj.GetAtoms()): # e.g. idx 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            if debug >= 2: print(f"\t{idx=}", rdkit_atom.GetSymbol(), "Number of bonds :", len(rdkit_atom.GetBonds()))
            if len(rdkit_atom.GetBonds()) == 0:
                if debug >= 1: print(f"\tNO BONDS CREATED for {rdkit_atom.GetSymbol()} due to no bonds in {specie.subtype} RDKit object")
            else:
                for b in rdkit_atom.GetBonds():
                    bond_startatom = b.GetBeginAtomIdx()
                    bond_endatom   = b.GetEndAtomIdx()
                    bond_order     = b.GetBondTypeAsDouble()
  
                    if bond_startatom in non_bonded_atoms or bond_endatom in non_bonded_atoms:
                        if debug >= 2: print(f"\tNO BOND CREATED {bond_startatom=} or {bond_endatom=} is not in the specie.atoms. It belongs to {non_bonded_atoms=}.")
                    else :
                        if bond_endatom == idx:
                            start = bond_endatom
                            end   = bond_startatom
                        elif bond_startatom == idx:
                            start = bond_startatom
                            end   = bond_endatom   

                        # create new bond object
                        if debug >=2: print(f"\tBOND CREATED", idx, start, end, bond_order, specie.atoms[start].label, specie.atoms[end].label)
                        new_bond = bond(specie.atoms[start], specie.atoms[end], bond_order)
                        specie.atoms[idx].add_bond(new_bond)
                
                if idx not in non_bonded_atoms:
                    if hasattr(specie.atoms[idx], "bonds"):
                        if debug >=2: 
                            print(f"\tBONDS", [(bd.atom1.label, bd.atom2.label, bd.order, round(bd.distance,3)) for bd in specie.atoms[idx].bonds])
                    else:
                        if specie.natoms == 1:
                            if debug >=1: print(f"\tNO BONDS CREATED for {specie.atoms[idx].label} because it is the only atom in {specie.subtype} object")
                            pass
                        else:
                            if debug >=1: print(f"\tNO BONDS for {specie.atoms[idx].label} with {specie.subtype} RDKit object index {idx}. Please check the RDKit object.")
                            return False # return False if no bonds are created
                else :
                    if debug >=1: print(f"\tNO BONDS for {rdkit_atom.GetSymbol()} with {specie.subtype} RDKit object index {idx} because it is an added atom")

    return True                    
######################################################
def create_metal_ligand_bonds (mol, debug: int=0):
    # Third Part. Adds Metal-Ligand Bonds, with a zero order:
    from cell2mol.classes import bond
    if mol.iscomplex:
        for lig in mol.ligands:
            for at in lig.atoms:
                count = 0
                for met in mol.metals: 
                    isconnected = at.check_connectivity(met, debug=debug)
                    if isconnected:
                        index_1 = at.get_parent_index("molecule")
                        index_2 = met.get_parent_index("molecule")
                        if index_1 < index_2 : 
                            bond_startatom = at
                            bond_endatom   = met
                        else:
                            bond_startatom = met
                            bond_endatom   = at
                        newbond = bond(bond_startatom, bond_endatom, 0)
                        at.add_bond(newbond)
                        met.add_bond(newbond)
                        count += 1 
                if count != at.mconnec: 
                    if debug >= 1: print(f"CELL.CREATE_BONDS: error creating bonds for atom: \n{at}\n of ligand: \n{lig}\n")
                    if debug >= 1: print(f"CELL.CREATE_BONDS: count differs from atom.mconnec: {count}, {at.mconnec}")

######################################################
def create_metal_metal_bonds (mol, debug: int=0):
    from cell2mol.classes import bond
    # Adds Metal-Metal Bonds, with a zero order:
    if mol.iscomplex:
        if len(mol.metals) > 1 :
            if debug >= 1: print(f"CELL.CREATE_BONDS: Creating Metal-Metal Bonds for molecule {mol.formula}")
            if debug >= 2: print(f"CELL.CREATE_BONDS: Metals: {mol.metals}")
            for idx, met1 in enumerate(mol.metals):
                for jdx, met2 in enumerate(mol.metals):
                    if idx <= jdx: continue
                    isconnected = met1.check_connectivity(met2, debug=debug)
                    if isconnected:
                        index_1 = met1.get_parent_index("molecule")
                        index_2 = met2.get_parent_index("molecule")
                        if index_1 < index_2 : 
                            bond_startatom = met1
                            bond_endatom   = met2
                        else:
                            bond_startatom = met2
                            bond_endatom   = met1
                        newbond = bond(bond_startatom, bond_endatom, 0)
                        met1.add_bond(newbond) 
                        met2.add_bond(newbond) 
######################################################
