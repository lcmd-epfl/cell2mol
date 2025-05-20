import numpy as np
import copy
from cell2mol.charge_assignment import protonation, get_charge, get_charge_manual, charge_state, check_rdkit_obj_connectivity, aromatic_info
import itertools
import os
from cell2mol import __file__
from cell2mol.spin import generate_feature_vector
import pickle
from rdkit import Chem
manual_assign = ["O4-Cl", "N3", "I3", "N2", "N-O"]
#######################################################
def balance_charge(unique_indices: list, unique_species: list, input_charge: int=0, rare: bool=False, predict: bool=False, aromatic: bool=False, debug: int=0) -> list:
    """Function to Select the Best Charge Distribution for the unique species.
    It accepts multiple charge options for each molecule/ligand/metal (poscharge, etc...).
    NO: It should select the best one depending on whether the final metal charge makes sense or not.
    In some cases, can accept metal oxidation state = 0, if no other makes sense
    """
    all_possible_m_ox = [0, 1, 2, 3, 4, 5, 6, 7]
    iserror = False
    iterlist = []
    for idx, spec in enumerate(unique_species):
        toadd = []

        if spec.subtype == "metal":
            if rare == True:
                rare_m_ox = [x for x in all_possible_m_ox if x not in spec.possible_cs]
                print(f"RARE METAL OXIDATION STATES: {spec.formula} {rare_m_ox}")
                for tch in rare_m_ox:
                    toadd.append(tch)
            elif predict == True :
                tch = predict_metal_ox(spec, debug=debug)
                toadd.append(tch)
            else:
                for tch in spec.possible_cs:
                    toadd.append(tch)                
        else :   
            if len(spec.possible_cs) == 1:
                toadd.append(spec.possible_cs[0].corr_total_charge)
            elif len(spec.possible_cs) > 1:
                if not aromatic:
                    for tch in spec.possible_cs:
                        toadd.append(tch.corr_total_charge)
                else:
                    aromatic_counts = []
                    for cs in spec.possible_cs:
                        aromatic_dict = aromatic_info(cs.rdkit_obj)
                        aromatic_counts.append(aromatic_dict["Aromatic atoms"])

                    print(f"aromatic_counts: {aromatic_counts}")

                    if len(set(aromatic_counts)) == 1 and aromatic_counts[0] == 0:
                        for tch in spec.possible_cs:
                            toadd.append(tch.corr_total_charge)
                        pass  # all values are 0 — skip
                    else:
                        max_aromatic = max(aromatic_counts)
                        max_indices = [i for i, val in enumerate(aromatic_counts) if val == max_aromatic]

                        if len(max_indices) == 1:
                            idx = max_indices[0]
                            cs = spec.possible_cs[idx]
                            print(f"   Unique most aromatic cs at index {idx}: {cs.smiles} (aromatic atoms: {max_aromatic})")
                            toadd.append(cs.corr_total_charge)
                        else:
                            print(f"   Multiple cs with same max aromatic atoms ({max_aromatic}), appending all")
                            for idx in max_indices:
                                cs = spec.possible_cs[idx]
                                print(f"    - Index {idx}: {cs.smiles}")
                                toadd.append(cs.corr_total_charge)
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
                if charges_sum == input_charge:
                    final_charge_distribution.append(d)
                    final_charges.append(distr)
    elif iserror:
        if debug >= 1: print("Error found in BALANCE: one species has no possible charges")
        final_charge_distribution = []

    if debug:
        print(f"Final Charge Distribution: {final_charge_distribution}")
        print(f"Final Charges: {final_charges}")

    return final_charge_distribution, final_charges

######################################################
def assign_charge_state_for_unique_species(unique_species, final_charge_tuple, debug: int=0):

    for specie, final_charge in zip(unique_species, final_charge_tuple):
        print(specie.unique_index, specie.formula)
        if (specie.subtype == "molecule" and not specie.iscomplex and not specie.has_IA_IIA) or (specie.subtype == "ligand"):
            charge_list = [cs.corr_total_charge for cs in specie.possible_cs]
            idx = charge_list.index(final_charge)
            cs = specie.possible_cs[idx]
            specie.charge_state = cs
            # print(specie.charge_state.protonation)
            specie.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
        elif specie.subtype == "metal" :
            # charge_list = specie.possible_cs   
            # idx = charge_list.index(final_charge)
            # cs = specie.possible_cs[idx]
            specie.set_charge(final_charge) 
    for specie in unique_species:
        if (specie.subtype == "molecule" and not specie.iscomplex and not specie.has_IA_IIA) or (specie.subtype == "ligand"):
            print(specie.formula, specie.charge_state, specie.totcharge, specie.smiles)
    return unique_species

######################################################
def print_possible_and_selected_cs (newcell, refcell, debug: int=0):
    """Print possible charge states and selected charge state for unique species."""
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

#######################################################
def get_reordered_protonation (refcell: object, reference: object, target: object):

    temp_prot = copy.deepcopy(reference.charge_state.protonation)
    temp_prot.parent = target

    ref_indices = reference.get_parent_indices("reference")
    target_indices = target.get_parent_indices("reference")

    ref_data = [refcell.atom_site_labels[idx] for idx in ref_indices]
    target_data = [refcell.atom_site_labels[idx] for idx in target_indices]

    index_map = {value: index for index, value in enumerate(target_data)}
    sorted_indices = sorted(range(len(ref_data)), key=lambda i: index_map[ref_data[i]])            

    reordered_prot = temp_prot.reorder(sorted_indices)

    return reordered_prot
#######################################################
def reorder_rdkit_atoms(ref_mol, ref_labels, target_labels):
    # 1. Create label → atom index map from ref_data
    label_to_index = {label: idx for idx, label in enumerate(ref_labels)}

    # 2. Create old index → new index map
    old_to_new = {label_to_index[label]: i for i, label in enumerate(target_labels)}
    new_to_old = {v: k for k, v in old_to_new.items()}

    # 3. Create editable mol
    new_mol = Chem.RWMol()

    # 4. Add atoms in new order
    for new_idx in range(len(target_labels)):
        old_idx = new_to_old[new_idx]
        atom = ref_mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_mol.AddAtom(new_atom)

    # 5. Add bonds based on original molecule
    for bond in ref_mol.GetBonds():
        begin_old = bond.GetBeginAtomIdx()
        end_old = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # remap to new indices
        begin_new = old_to_new[begin_old]
        end_new = old_to_new[end_old]
        new_mol.AddBond(begin_new, end_new, bond_type)
    
    return new_mol.GetMol()
######################################################
def set_charge_state(reference, target, mode, debug: int=0):

    final_charge = reference.totcharge
    print("SET_CHARGE_STATE:", reference.charge_state)
    refcell = reference.get_parent("reference")

    if mode == 1 : # For "reference" cell. Their possible charge states are already calculated
        if target.formula in manual_assign:
            cs = get_charge_manual(target, debug=debug)
        else :
            if not hasattr(target, "possible_cs"): 
                target.get_possible_cs(debug=debug)
            else:
                if debug >= 1: print("SET_CHARGE_STATE: possible_cs of reference already exists")
            print(f"SET_CHARGE_STATE: {mode=} {target.formula=} {target.possible_cs=}")
            charge_list = [cs.corr_total_charge for cs in target.possible_cs]
            print(charge_list, final_charge, target.possible_cs)
            idx = charge_list.index(final_charge)
            cs = target.possible_cs[idx]
    
        target.charge_state = cs
        if final_charge != cs.corr_total_charge:
            print(f"SET_CHARGE_STATE: WARNING!!! {target.formula=} {final_charge=} {cs.corr_total_charge=} final_charge != cs.corr_total_charge")
        print("SET_CHARGE_STATE!!!!", f"{mode=}", cs, cs.smiles)
        target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
        print(f"SET_CHARGE_STATE:{target.formula=} {target.totcharge=} {target.smiles=}")   
    
    elif mode == 2 : # For "unit" cell. Their charge state are not calculated
        if (target.subtype == "ligand") or (target.subtype == "molecule" and not target.iscomplex and not target.has_IA_IIA):
            
            rdkit_obj = reorder_rdkit_atoms(reference.rdkit_obj, reference.atom_site_labels, target.atom_site_labels)
            smiles = Chem.MolToSmiles(rdkit_obj)
            atom_charges = []
            total_charge = 0
            
            for i in range(target.natoms):
                a = rdkit_obj.GetAtomWithIdx(i)  # Returns a particular Atom
                atom_charges.append(a.GetFormalCharge())
                total_charge += a.GetFormalCharge()
            if total_charge != final_charge:
                print(f"SET_CHARGE_STATE: WARNING!!! {target.formula=} {total_charge=} {final_charge=}")

            print(f"SET_CHARGE_STATE: {smiles=} {final_charge=}")
            target.set_charges(total_charge, atom_charges, smiles, rdkit_obj)
            print(f"SET_CHARGE_STATE:{target.formula=} {target.totcharge=} {target.smiles=}")

    elif mode == 3 : # For "unit" cell. Their charge state are not calculated
        if target.formula in ["O4-Cl", "N3", "I3"]:
            # TODO  : This is temporary solution. It should be refined
            cs = get_charge_manual(target, debug=debug)

        elif (target.subtype == "molecule" and not target.iscomplex and not target.has_IA_IIA) :
            if debug >=1 : print(f"SET_CHARGE_STATE:({target.subtype}) {target.formula} {final_charge=} Create Empty PROTONATION for this specie")
            empty_list = [int(0)]*len(target.labels)
            empty_prot = protonation(target.labels, target.coord, target.cov_factor, 
                                    int(0), empty_list, empty_list, empty_list, empty_list, typ="Empty", parent=target)
            cs = get_charge(final_charge, empty_prot)

        elif target.subtype == "ligand":
            if debug >=1 : print(f"SET_CHARGE_STATE:({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Ligand")
            prot = reference.charge_state.protonation
            if debug >=1 : print(f"SET_CHARGE_STATE:{prot=}")

            temp_uncorr_atom_charges = reference.charge_state.uncorr_atom_charges
            abs_charge = sum([abs(chg) for chg in temp_uncorr_atom_charges])
            if debug >=1 : print(f"SET_CHARGE_STATE: {reference.charge_state.uncorr_atom_charges=} {len(reference.charge_state.uncorr_atom_charges)} {abs_charge=}")
            temp_prot = copy.deepcopy(prot)
            temp_prot.parent = target

            print(f"SET_CHARGE_STATE:{temp_prot.labels=} {len(temp_prot.labels)=} {len(temp_prot.coords)=} {len(temp_prot.block)=} {temp_prot.added_atoms=} ")
            
            # For "unit" cell
            ref_indices = reference.get_parent_indices("reference")
            target_indices = target.get_parent_indices("reference")

            ref_data = [refcell.atom_site_labels[idx] for idx in ref_indices]
            target_data = [refcell.atom_site_labels[idx] for idx in target_indices]

            index_map = {value: index for index, value in enumerate(target_data)}
            sorted_indices = sorted(range(len(ref_data)), key=lambda i: index_map[ref_data[i]])            

            if debug >=1 : print(f"SET_CHARGE_STATE:{ref_data=}")
            if debug >=1 : print(f"SET_CHARGE_STATE:{target_data=}")
            if debug >=1 : print(f"SET_CHARGE_STATE:{sorted_indices=}")

            reordered_prot = temp_prot.reorder(sorted_indices)
            # reordered_prot.coords = target.coord
            if debug >=1 : print(f"SET_CHARGE_STATE: {reordered_prot=}")

            if len(temp_uncorr_atom_charges) == len(sorted_indices):
                reordered_uncorr_atom_charges = [temp_uncorr_atom_charges[idx] for idx in sorted_indices]
            else:
                reordered_uncorr_atom_charges = [temp_uncorr_atom_charges[idx] for idx in sorted_indices]
                dummy_charges_added_atoms = [temp_uncorr_atom_charges[idx] for idx in range(len(sorted_indices), len(temp_uncorr_atom_charges))]
                reordered_uncorr_atom_charges.extend(dummy_charges_added_atoms)
            
            if debug >=1 :print(f"SET_CHARGE_STATE: {reordered_uncorr_atom_charges=} {len(reordered_uncorr_atom_charges)=}")

            if debug >=1 : print(f"SET_CHARGE_STATE:({target.subtype}) {target.formula} {reference.charge_state.uncorr_total_charge=} Reordered {sorted_indices=}")
            cs = get_charge(reference.charge_state.uncorr_total_charge , reordered_prot, ref_uncorr_atom_charges=reordered_uncorr_atom_charges)

        target.charge_state = cs
        if final_charge != cs.corr_total_charge:
            print(f"SET_CHARGE_STATE: WARNING!!! {target.formula=} {final_charge=} {cs.corr_total_charge=} final_charge != cs.corr_total_charge")
        print("SET_CHARGE_STATE!!!!", f"{mode=}", cs, cs.smiles)
        target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
        print(f"SET_CHARGE_STATE:{target.formula=} {target.totcharge=} {target.smiles=}")
    
######################################################
def prepare_mol (mol):
    tmp_atcharge = np.zeros((mol.natoms), dtype=int)
    tmp_smiles = []
    
    for lig in mol.ligands: 
        if hasattr(lig, "smiles"):
            print(f"prepare_mol: {lig.formula=} {lig.smiles=}")
        else:
            print(f"prepare_mol: {lig.formula=}")
        tmp_smiles.append(lig.smiles)
        parent_indices = lig.get_parent_indices("molecule")
        for kdx, a in enumerate(parent_indices):
            tmp_atcharge[a] = lig.atomic_charges[kdx]
    
    for met in mol.metals:  
        parent_index = met.get_parent_index("molecule")
        tmp_atcharge[parent_index] = met.charge
        
    mol.set_charges(int(sum(tmp_atcharge)), atomic_charges=tmp_atcharge, smiles=tmp_smiles)

######################################################
def create_bonds_specie (specie, rdkit_obj: object=None, debug: int=0):
    from cell2mol.classes import bond
    if debug >= 1: print(f"CREATE_bonds_specie: {specie.formula=}, {specie.subtype=} {specie.smiles=}")
    n_atoms = specie.natoms # e.g. 9 
    if rdkit_obj is not None:
        rdkit_obj = rdkit_obj
    else :
        rdkit_obj = specie.rdkit_obj

    n_atoms_rdkit = rdkit_obj.GetNumAtoms() # e.g.10 
    if debug >= 1: print(f"CREATE_bonds_specie: {specie.formula=}, {specie.subtype=}")

    if n_atoms == n_atoms_rdkit:
        if debug >= 2: print(f"\tNumber of atoms in {specie.subtype} object and RDKit object are equal: {n_atoms} {n_atoms_rdkit}")

        for idx, rdkit_atom in enumerate(rdkit_obj.GetAtoms()): # e.g. idx 0, 1, 2, 3, 4, 5, 6, 7, 8
            if debug >= 2: print(f"\t{idx=}", rdkit_atom.GetSymbol(), "Number of bonds :", len(rdkit_atom.GetBonds()))
            if len(rdkit_atom.GetBonds()) == 0:
                if debug >= 1: print(f"\tNO BONDS CREATED for {specie.atoms[idx].label} due to no bonds in {specie.subtype} RDKit object")
            else:
                for b in rdkit_atom.GetBonds():
                    bond_startatom = b.GetBeginAtomIdx()
                    bond_endatom   = b.GetEndAtomIdx()
                    bond_order     = b.GetBondTypeAsDouble()
                    if specie.atoms[bond_endatom].label == "D" and rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol() == "H":
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

                    elif specie.atoms[bond_endatom].label != rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol():
                        if debug >= 1: print(f"\tError with Bond EndAtom", specie.atoms[bond_endatom].label, rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol())
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
                        print(f"\tBONDS", [(bd.atom1.label, bd.atom2.label, bd.order, np.round(bd.distance,3)) for bd in specie.atoms[idx].bonds])
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
        if debug >= 2: print(f"\t{[(i, atom.GetSymbol()) for i, atom in enumerate(rdkit_obj.GetAtoms())]}")       
        non_bonded_atoms = list(range(0, n_atoms_rdkit))[n_atoms:]
        if debug >= 2: print(f"\tNON_BONDED_ATOMS", non_bonded_atoms)

        for idx, rdkit_atom in enumerate(rdkit_obj.GetAtoms()): # e.g. idx 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
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
                            print(f"\tBONDS", [(bd.atom1.label, bd.atom2.label, bd.order, np.round(bd.distance,3)) for bd in specie.atoms[idx].bonds])
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
    if mol.iscomplex or mol.has_IA_IIA:
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
                        # Chem.BondType.DATIVE
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
    if mol.iscomplex or mol.has_IA_IIA:
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

def assign_charge_to_specie(specie, final_charge, debug: int=0):    
    """Assign the charge to a specific species based on its type."""
    if debug >= 1: print(f"ASSIGN_CHARGE_TO_SPECIE: Unique Species final charges {specie.formula=} {final_charge=}")
    if (specie.subtype == "molecule" and not specie.iscomplex and not specie.has_IA_IIA) or (specie.subtype == "ligand"):
        idx = [cs.corr_total_charge for cs in specie.possible_cs].index(final_charge)
        cs = specie.possible_cs[idx]
        specie.charge_state = cs
        specie.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
        if debug >= 1: print(specie.unique_index, specie.formula, specie.totcharge, specie.smiles)
    
    elif specie.subtype == "metal":
        specie.set_charge(final_charge)  
        if debug >= 1: print(specie.unique_index, specie.formula, specie.charge)

######################################################

def validate_reference_molecules(self, debug):
    """Validate reference molecules by checking ligand and metal charges."""
    for idx, ref in enumerate(self.refmoleclist):
        if ref.iscomplex or ref.has_IA_IIA:
            if debug >= 1: print(f"VALIDATE_REFERENCE_MOLECULES: {ref.formula=}")
            self.validate_complex_ligands(ref, idx, debug)
            self.validate_complex_metals(ref, debug)
            prepare_mol(ref)
        else:
            self.validate_non_complex_molecule(ref, debug)

######################################################
def predict_metal_ox (metal:object, debug: int=0) -> None:
    model = "Fe_mono_m_ox_5486.pkl"
    feature = generate_feature_vector (metal, target_prop = "m_ox", debug=debug)
    path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), model)
    ramdom_forest = pickle.load(open(path_rf, 'rb'))
    predictions = ramdom_forest.predict(feature)
    m_ox_rf = predictions[0]
    print(f"PREDICT_METAL_OS: metal OS of the metal {metal.label} is predicted as {m_ox_rf} using Random Forest model")

    return m_ox_rf
######################################################