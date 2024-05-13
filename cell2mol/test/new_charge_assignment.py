import numpy as np
from cell2mol.hungarian import reorder
import copy
from cell2mol.xyz2mol import xyz2mol
from cell2mol.connectivity import compare_species, compare_metals
from cell2mol.charge_assignment import check_rdkit_obj_connectivity, arrange_data_for_reorder
from cell2mol.hungarian import reorder
from rdkit import Chem

######################################################
def set_charge_state(unique_specie, target, debug: int=0):
    
    prot = unique_specie.charge_state.protonation
    final_charge = unique_specie.totcharge

    # For "reference" cell
    ref_data, target_data = arrange_data_for_reorder(unique_specie, target)
    if debug >=2 : print(ref_data, target_data)
    dummy1, dummy2, map12 = reorder(ref_data, target_data, unique_specie.coord, target.coord)
    
    if np.array_equal(map12, np.arange(len(target_data))):
        if debug >=1 : print(f"({target.subtype}) {target.formula} {final_charge=} No need to reorder")
        cs = get_charge(final_charge, prot)
        target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)
               
    else:
        temp_prot = copy.deepcopy(prot)
        reordered_prot = temp_prot.reorder(map12)
        # print(f"{reordered_prot=}")
        # print(f"{temp_prot=}")
        # print(f"{prot=}")
        # print(map12)
        if debug >=1 : print(f"({target.subtype}) {target.formula} {final_charge=} Reordered {map12=}")
        cs = get_charge(final_charge, reordered_prot)
        target.set_charges(cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj)

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