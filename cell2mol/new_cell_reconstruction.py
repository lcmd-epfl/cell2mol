import numpy as np
from ase import Atoms
from cell2mol.classes import molecule
from cell2mol.cell_reconstruction import tmatgenerator
from cell2mol.other import get_dist, extract_from_list
from cell2mol.connectivity import split_species, count_species, compare_reference_indices
from cell2mol.cell_operations import translate
from itertools import combinations
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()
import os
from cell2mol.read_write import writexyz

######################################################
def modify_cov_factor_due_to_H (refcell, debug: int=0):
    cov_factor = refcell.refmoleclist[0].cov_factor
    if not refcell.has_isolated_H:  
        refcell.check_missing_H(debug=debug)                                     
    else:
        if debug >= 1: print(f"Initial covalent factor: {cov_factor=} before increasing")
        while refcell.has_isolated_H and cov_factor < 1.5:
            # Increase covalent factor for H atoms
            cov_factor += 0.05
            refcell.get_reference_molecules(refcell.labels, refcell.frac_coord, cov_factor=cov_factor, debug=0)
            if debug >= 1: print(f"Covalent factor increases: {cov_factor=}")
        refcell.check_missing_H(debug=debug)
    refcell.assess_errors(mode="hydrogens")
    return refcell

######################################################
def modify_cov_factor_due_to_possible_charges (refcell, debug: int=0):

    cov_factor = refcell.refmoleclist[0].cov_factor
    print(f"Initial ovalent factor: {cov_factor=}")
    
    temp_selection = []
    while (len(temp_selection) != len(refcell.species_list) or (refcell.has_isolated_H or refcell.has_missing_H) ) and (cov_factor > 1.15):
        for specie in refcell.species_list:
            tmp = specie.get_possible_cs(debug=debug)
            if tmp is None:
                cov_factor -= 0.05
                refcell.get_reference_molecules(refcell.labels, refcell.frac_coord, cov_factor=cov_factor, debug=0)
                if not refcell.has_isolated_H : refcell.check_missing_H(debug=debug)
                if not refcell.has_missing_H :  refcell.get_unique_species(debug=debug)
                temp_selection = []
                break
            elif specie.subtype != "metal":
                temp_selection.append(list([cs.corr_total_charge for cs in specie.possible_cs]))
            else :
                temp_selection.append(specie.possible_cs)    

    if debug >= 1: print(f"Covalent factor : {cov_factor=}")
    refcell.assess_errors(mode="hydrogens")
    if refcell.error_case == 0:
        if debug >= 1: print(f"OK with decreasing cov_factor {cov_factor=}")
        return refcell
    else:
        if debug >= 1: print(f"Error with decreasing cov_factor {cov_factor=}")
        refcell = modify_cov_factor_due_to_H(refcell, debug=debug)
        return refcell

######################################################
def apply_symmetry_operations_reference (refcell, cell_vector, sym_ops, normalize: bool=True, pbc: bool=True):
    
    new_structures = []
    ref_labels = refcell.labels
    fractional_coords = np.array(refcell.frac_coord)

    if "D" in ref_labels:
        numbers = [elemdatabase.elementnr[elem] for elem in ref_labels] # Atoms object cannot handle Deuterium in the symbols

    for rot, trans in zip(sym_ops[0], sym_ops[1]):
        transformed_positions = np.dot(fractional_coords, rot.T)
        transformed_positions += np.array(trans)       
        if normalize:
            transformed_positions = np.remainder(transformed_positions, 1)
        if "D" in ref_labels:
            new = Atoms(scaled_positions=transformed_positions, numbers=numbers, cell=cell_vector)
        else:
            new = Atoms(symbols=ref_labels, scaled_positions=transformed_positions, cell=cell_vector)

        new_structures.append(new)
    
    return new_structures

######################################################
def find_row_indices(source, target):
    # cell_pos, new.positions
    # List to store the indices of found rows
    found_indices = []
    found_rows = []
    remaining_indices = []
    remaining_rows = []
    
    # Iterate over each row in the source array with enumeration to track the index
    for index, row in enumerate(source):
        # Check if any row in the target array matches the current row
        if any(np.allclose(row, target_row, atol=1e-4, rtol=1e-2) for target_row in target):
            found_indices.append(index)
            found_rows.append(row)
        else :
            remaining_indices.append(index)
            remaining_rows.append(row)
    return found_indices, found_rows

######################################################
def find_row_index_from_matrix (matrix, query_row):
    
    # Convert the inputs to NumPy arrays if they aren't already
    matrix = np.array(matrix)
    query_row = np.array(query_row)
    
    # Check each row for equality with the query_row
    for index, row in enumerate(matrix):
        if np.allclose(row, query_row, atol=1e-4, rtol=1e-2):
            return index
    return -1

######################################################
def get_fragments (newcell, updated, indices_in_ref, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
                   
    updated_labels  = extract_from_list(updated, newcell.labels, dimension=1)
    updated_coord   = extract_from_list(updated, newcell.coord, dimension=1)
    updated_fracs = extract_from_list(updated, newcell.frac_coord, dimension=1)
    
    blocklist = split_species(updated_labels, updated_coord, debug=debug)
    
    fragments = []
    
    for b in blocklist:
        if debug > 2 : print(f"GET_FRAGMENTS: doing block={b}")
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord  = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord  = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        
        # Creates Molecule Object
        newmolec    = molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
        
        # For debugging
        newmolec.origin = "cell.get_fragments"
        
        # Adds cell as parent of the molecule, with indices
        newmolec.add_parent(newcell, indices=cell_indices)        
        newmolec.set_fractional_coord(mol_frac_coord)
        newmolec.set_adjacency_parameters(cov_factor=cov_factor, metal_factor=metal_factor)
        newmolec.set_atoms(create_adjacencies=True, debug=debug)
        newmolec.ref_indices = ref_indices
        newmolec.cell_indices = cell_indices
        fragments.append(newmolec)
        
    return fragments

######################################################

def get_fragments_from_moiety (newcell, updated, indices_in_ref, refcell, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
    
    atom_site_labels = refcell.atom_site_labels
    geom_bond_cif = refcell.geom_bond_cif
    moiety_indices = refcell.moiety_indices
    
    if debug >= 2 : print(f"get_fragments: {updated=}")
    if debug >= 2 : print(f"get_fragments: {indices_in_ref=}")
    if debug >= 2 : print(f"get_fragments: {moiety_indices=}")

    updated_labels  = extract_from_list(updated, newcell.labels, dimension=1)
    updated_coord   = extract_from_list(updated, newcell.coord, dimension=1)
    updated_fracs = extract_from_list(updated, newcell.frac_coord, dimension=1)
    updated_moieties_list = []
    updated_moieties_indices_in_ref_list = []
    for sublist in moiety_indices:
        new_sublist = []
        new_sublist_indices = []
        for i in sublist:
            if i in indices_in_ref:
                new_sublist.append(updated[indices_in_ref.index(i)])
                new_sublist_indices.append(i)
        updated_moieties_list.append(new_sublist)
        updated_moieties_indices_in_ref_list.append(new_sublist_indices)
    # updated_moieties_list = [[updated[indices_in_ref.index(i)] for i in sublist if i in indices_in_ref] for sublist in moiety_indices]
    # updated_moieties_indices_in_ref = [[indices_in_ref.index(i) for i in sublist if i in indices_in_ref] for sublist in moiety_indices]      
    # updated_moieties_list = [[updated[idx] for idx in sublist] for sublist in updated_moieties_indices_in_ref]
    # if debug >= 2 : print(f"get_fragments: updated_moieties_indices_in_ref", updated_moieties_indices_in_ref)
    if debug >= 2 : print(f"get_fragments: updated_moieties_list", len(updated_moieties_list),  updated_moieties_list)
    if debug >= 2 : print(f"get_fragments: updated_moieties_indices_in_ref_list", len(updated_moieties_indices_in_ref_list), updated_moieties_indices_in_ref_list)

    tmp_blocklist=[]
    for updated_moieties, updated_moieties_indices_in_ref in zip(updated_moieties_list, updated_moieties_indices_in_ref_list):
        if len(updated_moieties) == 0: continue
        if debug >= 2 : print("get_fragments: updated_moieties", updated_moieties)
        updated_moieties_labels  = extract_from_list(updated_moieties, newcell.labels, dimension=1)
        updated_moieties_coord   = extract_from_list(updated_moieties, newcell.coord, dimension=1)
        updated_moieties_atom_site_labels = [atom_site_labels[i] for i in updated_moieties_indices_in_ref]
        if debug >= 2 : print(f"get_fragments: updated_moieties_labels", updated_moieties_labels)
        if debug >= 2 : print(f"get_fragments: updated_moieties_atom_site_labels", updated_moieties_atom_site_labels)
        block = split_species(updated_moieties_labels, 
                              updated_moieties_coord, 
                              indices=updated_moieties, 
                              atom_site_labels=updated_moieties_atom_site_labels, 
                              geom_bond_cif=geom_bond_cif,
                              debug=debug)
        tmp_blocklist.extend(block)
        
    if debug >= 2 : print("get_fragments: tmp_blocklist", tmp_blocklist)

    value_to_index = {val: idx for idx, val in enumerate(updated)}
    blocklist = [
        [value_to_index[val] for val in sublist if val in value_to_index]
        for sublist in tmp_blocklist
    ]
    if debug >= 2 : print("get_fragments: blocklist", blocklist)
    
    fragments = []  
    for b in blocklist:
        if debug > 2 : print(f"get_fragments: doing block={b}")
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord  = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord  = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        mol_atom_site_labels = [atom_site_labels[idx] for idx in ref_indices]
        # print(f"get_fragments: {cell_indices=}")
        # print(f"get_fragments: {ref_indices=}")
        # print(f"get_fragments: {mol_atom_site_labels=}")
 
        # Creates Molecule Object
        newmolec    = molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
        
        # For debugging
        newmolec.origin = "cell.get_fragments"
        
        # Adds cell as parent of the molecule, with indices
        newmolec.add_parent(newcell, indices=cell_indices)        
        newmolec.set_fractional_coord(mol_frac_coord)
        newmolec.set_adjacency_parameters(cov_factor=cov_factor, metal_factor=metal_factor)
        newmolec.set_atoms(create_adjacencies=True, atom_site_labels=mol_atom_site_labels, geom_bond_cif=geom_bond_cif, debug=debug)
        newmolec.ref_indices = ref_indices
        newmolec.cell_indices = cell_indices
        fragments.append(newmolec)
        
    return fragments

######################################################
def get_fragments_new (newcell, updated, indices_in_ref, refcell, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
    
    atom_site_labels = refcell.atom_site_labels
    geom_bond_cif = refcell.geom_bond_cif
    moiety_indices = refcell.moiety_indices
    
    if debug >= 2 : print(f"get_fragments: {updated=}")
    if debug >= 2 : print(f"get_fragments: {indices_in_ref=}")
    if debug >= 2 : print(f"get_fragments: {moiety_indices=}")
    
    updated_labels  = extract_from_list(updated, newcell.labels, dimension=1)
    updated_coord   = extract_from_list(updated, newcell.coord, dimension=1)
    updated_fracs = extract_from_list(updated, newcell.frac_coord, dimension=1)
    if not refcell.exist_cif_bond_moiety:
        blocklist = split_species(updated_labels, updated_coord, debug=debug)
    else:
        updated_moieties_list = [[updated[indices_in_ref.index(i)] for i in sublist if i in indices_in_ref] for sublist in moiety_indices]
        if debug >= 2 : print(f"get_fragments: updated_moieties_list", updated_moieties_list)
        tmp_blocklist=[]
        for updated_moieties in updated_moieties_list:
            if len(updated_moieties) == 0: continue
            updated_moieties_labels  = extract_from_list(updated_moieties, newcell.labels, dimension=1)
            updated_moieties_coord   = extract_from_list(updated_moieties, newcell.coord, dimension=1)
            block = split_species(updated_moieties_labels, updated_moieties_coord, indices=updated_moieties, debug=debug)
            tmp_blocklist.extend(block)
            if debug >= 2 : print("get_fragments: updated_moieties", updated_moieties)
        if debug >= 2 : print("get_fragments: tmp_blocklist", tmp_blocklist)

        value_to_index = {val: idx for idx, val in enumerate(updated)}
        blocklist = [
            [value_to_index[val] for val in sublist if val in value_to_index]
            for sublist in tmp_blocklist
        ]
    if debug >= 2 : print("get_fragments: blocklist", blocklist)
    
    fragments = []  
    for b in blocklist:
        if debug > 2 : print(f"get_fragments: doing block={b}")
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord  = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord  = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        mol_atom_site_labels = [atom_site_labels[idx] for idx in ref_indices]
        # print(f"get_fragments: {cell_indices=}")
        # print(f"get_fragments: {ref_indices=}")
        # print(f"get_fragments: {mol_atom_site_labels=}")
 
        # Creates Molecule Object
        newmolec    = molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
        
        # For debugging
        newmolec.origin = "cell.get_fragments"
        
        # Adds cell as parent of the molecule, with indices
        newmolec.add_parent(newcell, indices=cell_indices)        
        newmolec.set_fractional_coord(mol_frac_coord)
        newmolec.set_adjacency_parameters(cov_factor=cov_factor, metal_factor=metal_factor)
        newmolec.set_atoms(create_adjacencies=True, atom_site_labels=mol_atom_site_labels, geom_bond_cif=geom_bond_cif, debug=debug)
        newmolec.ref_indices = ref_indices
        newmolec.cell_indices = cell_indices
        fragments.append(newmolec)
        
    return fragments

######################################################


def classify_fragments (fragments, newcell, debug: int=0):
    
    molecules = []
    remaining_fragments = []
    hydrogens = []
    for frag in fragments:
        found = False
        for idx, ref in enumerate(newcell.refmoleclist):
            if (ref.natoms == frag.natoms) & (ref.formula == frag.formula):
                if (sorted(ref.get_parent_indices("reference")) == sorted(frag.ref_indices)):
                    if debug > 2 : 
                        print(frag.formula, frag.ref_indices, frag.frac_coord, \
                              f"equivalent to Ref {idx} {ref.formula}")
                    frag.subtype = "molecule"
                    frag.origin = "cell.classify_fragments"
                    molecules.append(frag)
                    found = True
        
        if found == False:
            frag.subtype = "fragment"
            frag.origin = "cell.classify_fragments"
            if (frag.natoms == 1) and (frag.set_element_count()[4] + frag.set_element_count()[3] == 1): 
                hydrogens.append(frag) # # Hydrogen or Deuterium 
            else:    
                remaining_fragments.append(frag)

    rem_size = np.array([rem.natoms for rem in remaining_fragments])
    order = np.argsort(rem_size)
    descending_order = order[::-1]
    remaining_fragments = [remaining_fragments[i] for i in descending_order]
    for rem in remaining_fragments + hydrogens:
        rem.get_centroid()
        
    if debug >=2 :
        print("Remaining_fragments:", [rem.formula for rem in remaining_fragments])
        print("Remaining_fragments:", [rem.natoms for rem in remaining_fragments])
        print("Remaining_fragments:", [get_dist(rem.frac_centroid, [0.5, 0.5, 0.5]) for rem in remaining_fragments])    
        print("Hydrogens:", [h.formula for h in hydrogens])
    # return molecules, remaining_fragments, hydrogens
    return molecules, remaining_fragments + hydrogens

######################################################
def grouping_smaller_lists_for_target_sets(target_sets, smaller_lists, debug: int=0):
    
    # target_sets : Define the larger target lists as sets for fast lookup
    # List of smaller lists to be grouped
    
    # Group smaller lists into their respective larger list
    grouped_lists = [[] for _ in range(len(target_sets))]
    grouped_lists_idx = [[] for _ in range(len(target_sets))]
    target_idx_lists = [[] for _ in range(len(target_sets))]
    for j, small_list in enumerate(smaller_lists):
        small_set = set(small_list)
        for i, target_set in enumerate(target_sets):
            if small_set.issubset(target_set):
                if debug >=2 : print(f"grouped_lists {i} add {small_set}")
                grouped_lists[i].extend(small_list)
                grouped_lists_idx[i].append(j)
                target_idx_lists[i].append(i)
                
                if set(grouped_lists[i]) == target_set:
                    if debug >=2 :print(f"grouped_lists {i} is same with {target_set} {grouped_lists_idx[i]}")
                else :
                    if debug >=2 :print(f"continue for {i} {grouped_lists_idx[i]}")
                break
    
    if debug >= 2 :
        # Print the grouped lists
        for i, group in enumerate(grouped_lists):
            print(f"Group {i}: {group}")
            
    return grouped_lists, grouped_lists_idx, target_idx_lists

######################################################
def grouping_remaining_fragments(remaining_fragments, not_found_list, newcell, debug: int=0):

    smaller_lists = [rem_frag.ref_indices for rem_frag in remaining_fragments]
    smaller_lists.sort(key=len, reverse=True)

    target_sets = []
    for j in not_found_list:
        try:
            ref_indices = newcell.refmoleclist[j].get_parent_indices("reference")
            target_sets.append(set(ref_indices))
        except AttributeError as e:
            print(f"Error accessing or calling get_parent_indices on refmoleclist {j}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with element {j}: {e}")

    if debug >= 2: 
        print(f"{target_sets=}")
        print(f"{smaller_lists=}")
        
    grouped_rem_frags, indices_of_rem_frags, target_idx_lists = grouping_smaller_lists_for_target_sets(target_sets, smaller_lists, debug=debug)
    
    not_found_refmoleclist_indices = [ list(set(sublist))[0] if sublist else None for sublist in target_idx_lists ]

    indices_of_target_ref = []
    for t in not_found_refmoleclist_indices:
        indices_of_target_ref.append( newcell.refmoleclist[t].get_parent_indices("reference") )

    return grouped_rem_frags, indices_of_rem_frags, indices_of_target_ref
    
######################################################
def merge_fragments (frags: list, cell_vector: list, refcell: object, cov_factor: float=1.3, metal_factor: float=1.0, full: bool=False, final_merge: bool=False, debug: int=0):

    #finds biggest fragment and keeps it in the original cell
    sizes = []
    for f in frags:
        size = f.natoms
        sizes.append(size)
    keep_idx = np.argmax(sizes)
    if   keep_idx == 0: move_idx = 1
    elif keep_idx == 1: move_idx = 0
    keep_frag = frags[keep_idx]
    move_frag = frags[move_idx]
    if debug >= 2: print("MERGE_FRAGMENTS: keep_idx", keep_idx)
    if debug >= 2: print("MERGE_FRAGMENTS: move_idx", move_idx)

    move_frag.get_centroid()
    
    if move_frag.natoms == 1 and (move_frag.set_element_count()[4] + move_frag.set_element_count()[3] == 1):
        full=True

    tmatrix = tmatgenerator(move_frag.frac_centroid, full=full)

    if len(tmatrix) == 0: return None

    for t in tmatrix:
        if debug >= 2: print("MERGE_FRAGMENTS: translation", t)
        ## Applies Translations and each time, it checks if a bigger molecule is formed
        ## meaning that the translation was successful
        reclabels = []
        reclabels.extend(keep_frag.labels)
        reclabels.extend(move_frag.labels)
        reccoord = []
        reccoord.extend(keep_frag.coord)
        if t == (0, 0, 0): reccoord.extend(move_frag.coord)
        else:              reccoord.extend(translate(t, move_frag.coord, cell_vector))
        
        rec_ref_indices = []
        rec_ref_indices.extend(keep_frag.ref_indices)
        rec_ref_indices.extend(move_frag.ref_indices)

        rec_cell_indices = []
        rec_cell_indices.extend(keep_frag.cell_indices)
        rec_cell_indices.extend(move_frag.cell_indices)

        recfracs = []
        recfracs.extend(keep_frag.frac_coord)
        recfracs.extend(move_frag.frac_coord)

        rec_ref_atom_site_labels = [refcell.atom_site_labels[idx] for idx in rec_ref_indices]
        if final_merge :
            numspecs  = count_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        else:
            numspecs  = count_species(reclabels, reccoord, atom_site_labels=rec_ref_atom_site_labels, geom_bond_cif=refcell.geom_bond_cif, cov_factor=cov_factor, debug=debug)
        
        if debug >= 0 : print("MERGE_FRAGMENTS: count_species found", numspecs)
        if numspecs != 1 : continue

        if final_merge :
            blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        else:
            if refcell.exist_cif_bond_moiety and refcell.geom_bond_cif is not None:
                blocklist = split_species(reclabels, reccoord, atom_site_labels=rec_ref_atom_site_labels, geom_bond_cif=refcell.geom_bond_cif, cov_factor=cov_factor, debug=debug)
            else :
                blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        if debug >= 0 : print("MERGE_FRAGMENTS: split_species found", len(blocklist), f"{blocklist=}")

        if blocklist is None: continue
        else:
            if len(blocklist) != 1: continue
            if len(blocklist) == 1: 
                newmolec = molecule.from_positional(reclabels, reccoord, recfracs)
                newmolec.origin = "cell.reconstruct"
                newmolec.ref_indices = rec_ref_indices
                newmolec.cell_indices = rec_cell_indices
                newmolec.atom_site_labels = rec_ref_atom_site_labels
                newmolec.set_adjacency_parameters(cov_factor, metal_factor)
                newmolec.set_element_count()
                newmolec.get_centroid()
                newmolec.get_adjmatrix(geom_bond_cif=refcell.geom_bond_cif)
                newmolec.get_metal_adjmatrix(geom_bond_cif=refcell.geom_bond_cif)
                newmolec.set_adj_types()
                return newmolec
    return None
######################################################
def merge_elements(fragments, target_ref, cell_vector, refcell, cov_factor: float=1.3, metal_factor: float=1.0, full: bool=False,final_merge: bool=False, debug: int=0):

    while True:
        # Create an index list for the current state of fragments
        idx_list = list(range(len(fragments)))
        # Keep track of whether any items were merged during this pass
        merged = False

        # Generate all possible combinations of indices for current fragments
        for comb, idx_comb in zip(combinations(fragments, 2), combinations(idx_list, 2)):
            # print(comb, idx_comb)  # Optional: for debugging to see the pairs being processed    
            if comb[0].formula == "H" and comb[1].formula == "H" :
                pass
            elif comb[0].natoms + comb[1].natoms > len(target_ref):
                pass
            elif comb[0].subtype == "Rec. Molecule" or comb[1].subtype == "Rec. Molecule":
                pass
            elif set(comb[0].ref_indices).intersection(set(comb[1].ref_indices)):
                # print(comb[0].ref_indices)
                # print(comb[1].ref_indices)
                pass
            else:
                if debug >= 2: print("Fragments TO BE MERGED", [k.formula for k in comb], [k.subtype for k in comb], idx_comb)  
                
                newmolec = merge_fragments(comb, cell_vector, refcell, cov_factor=cov_factor, metal_factor=metal_factor, full=full, final_merge=final_merge,debug=debug)

                if newmolec is None: 
                    if debug >= 2: print(f"\tNOT MERGED {[k.formula for k in comb]}")
                else :
                    # print(comb[0].ref_indices)
                    # print(comb[1].ref_indices)
                    if debug >= 2: print(f"\tMERGED {newmolec.formula} from {[k.formula for k in comb]} at indices {idx_comb}")
                    small_set = set(newmolec.ref_indices)
                    if debug >= 2: print(f"{newmolec.formula} {newmolec.natoms=} {len(small_set)=} {small_set=}")
                    if small_set.issubset(target_ref): 
                        if sorted(small_set) == target_ref:
                            newmolec.subtype = "Rec. Molecule"
                            if debug >= 2: print("Molecule found", newmolec.formula)
                        else:                    
                            newmolec.subtype = "Rec. Fragment"
                            if debug >= 2: print("Bigger fragment found", newmolec.formula)
                    fragments[idx_comb[0]] = newmolec
                    fragments.pop(idx_comb[1])
                    merged = True
                    break
        
        if not merged:
            break

    return fragments         

######################################################
def fragments_reconstruct (subset_remaining_fragments, target_ref, cell_vector, refcell, cov_factor: float=1.3, metal_factor: float=1.0, full: bool=False, final_merge: bool=False,debug: int=0):

    list_of_found_molecules = []
    list_of_bigger_fragments = []    
    remaining_frag = subset_remaining_fragments.copy()

    fragments = merge_elements(remaining_frag, target_ref, cell_vector, refcell, cov_factor=cov_factor, metal_factor=metal_factor, full=full, final_merge=final_merge, debug=debug)

    for newmolec in fragments:
        small_set = set(newmolec.ref_indices)
        if small_set.issubset(target_ref): 
            if sorted(small_set) == target_ref:
                list_of_found_molecules.append(newmolec)
            elif newmolec.natoms == 1 and (newmolec.set_element_count()[4] + newmolec.set_element_count()[3] == 1):
                newmolec.subtype = "fragment"
                if debug > 2: print("Hydrogen found", newmolec.formula)
                list_of_bigger_fragments.append(newmolec)
            else :
                list_of_bigger_fragments.append(newmolec)            

    if debug > 2: print(f"{len(list_of_found_molecules)=}")
    if debug > 2: print(f"{len(list_of_bigger_fragments)=}")
    
    return list_of_found_molecules, list_of_bigger_fragments

######################################################
def get_updated_indices(sp_idx, ref_labels, new, cell_labels, cell_pos, cell_fracs, debug: int=0):
    """
    sp_idx : index of the symmetry operation
    new : ase atoms object by applying symmetry operations to the reference structure
    cell_labels : list of chemical symbols of the atoms in the unit cell
    cell_pos : list of cartesian coordinates of the atoms in the unit cell
    cell_fracs : list of fractional coordinates of the atoms in the unit cell
    """
    indices_lists = []
    # new_labels =  new.get_chemical_symbols()
    new_pos = new.get_positions()    
    new_fracs = new.get_scaled_positions()

    for jdx, (n_l, n_p, n_f) in enumerate(zip(ref_labels, new_pos, new_fracs)):
        for kdx, (l, p, f) in enumerate(zip(cell_labels, cell_pos, cell_fracs)):
            if n_l == l and np.allclose(n_p, p, atol=1e-4, rtol=1e-2):
                if np.allclose(np.remainder(n_f, 1), np.remainder(f, 1), atol=1e-4, rtol=1e-2):
                    if debug > 2: 
                        print(f"symmtry operation {sp_idx}:", f"atom of new (index: {jdx})", n_l, n_p, n_f, f"is the same as the atom of the unit cell (index: {kdx})", l, p, f)
                indices_lists.append((jdx, kdx)) # jdx is the index of the atom in the new structure, kdx is the index of the atom in the unit cell
    return indices_lists

######################################################
def get_updated_indices_old (ref_labels, new, cell_labels, cell_pos, cell_fracs, all_found, debug: int=0):

    # new structure : ase atoms object by applying symmetry operations to the reference structure
    # Find the indices of the atoms in the unit cell that have the same cartisian coordinates as the atoms in the new structure
    
    # Cartesian coordinate and fractional coordinate of the atoms in the new structure agree with atoms in the unit cell
    found_indices, found_rows = find_row_indices(cell_pos, new.positions)
    print(len(found_indices), len(found_rows))

    updated = [i for i in found_indices if i not in all_found]

    new_fracs = new.get_scaled_positions()
    # new_labels =  new.get_chemical_symbols()
    new_labels = ref_labels

    indices_in_ref = [ find_row_index_from_matrix(new_fracs, cell_fracs[u]) for u in updated ]
    if debug >= 2:
        print("Get indices in reference")
        print(f"{len(updated)=} {updated=}")
        print(f"{len(indices_in_ref)=} {indices_in_ref=}")
    
    if -1 in indices_in_ref:
        if debug >= 2: print("Remove -1 indices")
        for j, i in enumerate(indices_in_ref):
            if i == -1:
                    print(f"Cannot find the {j}th atom of the new structure based on fractional coordinates from the unit cell.")
                    print(f"{j} {new_labels[j]=} {new.positions[j]=} {new_fracs[j]=}")
                    print(f"Its fractional coord disagrees with the fractional coord of the {updated[j]}th atom of the unit cell.")
                    print(f"{updated[j]} {cell_labels[updated[j]]=} {cell_pos[updated[j]]=} {cell_fracs[updated[j]]=}")
            # else :
                # print(new_labels[i], np.allclose(new.positions[i], cell_pos[updated[j]]), np.allclose(new_fracs[i], cell_fracs[updated[j]]))

        indices_to_remove = [i for i, x in enumerate(indices_in_ref) if x == -1]
        indices_in_ref = [x for x in indices_in_ref if x != -1]
        if debug >= 2: print(f"{indices_to_remove=}")

        # Remove corresponding elements from 'updated' starting from the highest index
        for index in sorted(indices_to_remove, reverse=True):
            if index < len(updated):  # Check if index is valid for 'updated'
                updated.pop(index)
        
        if debug >= 2:
            print("After removing -1 indices")
            print(f"{len(updated)=} {updated=}")
            print(f"{len(indices_in_ref)=} {indices_in_ref=}")
    
    return updated, indices_in_ref
######################################################
def sort_remaining_fragments_list (original_remaining_fragments):

    remaining_fragments = []
    hydrogens = []
    for frag in original_remaining_fragments:
        if (frag.natoms == 1) and (frag.set_element_count()[4] + frag.set_element_count()[3] == 1): 
            hydrogens.append(frag) # # Hydrogen or Deuterium 
        else:    
            remaining_fragments.append(frag)

    rem_size = np.array([rem.natoms for rem in remaining_fragments])
    order = np.argsort(rem_size)
    descending_order = order[::-1]
    remaining_fragments = [remaining_fragments[i] for i in descending_order]
    remaining_fragments += hydrogens
    for rem in remaining_fragments:
        rem.get_centroid()

    return remaining_fragments
######################################################
def determine_wrap_keywords_pbc (atoms,refcell, wrap_keywords, debug: int=0):
    
    new_wrap_keywords = wrap_keywords.copy()
    print(f"wrap_keywords: {wrap_keywords}")
    print(f"new_wrap_keywords: {new_wrap_keywords}")
    cell_labels = atoms.get_chemical_symbols()
    cell_pos = atoms.get_positions(wrap=True, **new_wrap_keywords)
    cell_fracs = atoms.get_scaled_positions()
    cell_vector = atoms.cell.array
    space_group = atoms.info['spacegroup']
    sym_ops = space_group.get_op()
    new_structures = apply_symmetry_operations_reference(refcell, cell_vector, sym_ops)
    print(f"Number of symmetry operations: {len(new_structures)}")

    all_found = []
    for idx, new in enumerate(new_structures):
        if debug >=2 : print(f"\nApplying symmetry operations to reference {idx}")
        indices_lists = get_updated_indices(idx, new, cell_labels, cell_pos, cell_fracs, debug=debug)
        if debug >=2 : print(f"{len(indices_lists)=}")

        updated_lists = [i for i in indices_lists if i[1] not in all_found]   
        updated_ref_indices = [i[0] for i in updated_lists]
        updated_cell = [i[1] for i in updated_lists]
        if debug >=2 : print(f"{len(updated_cell)=} {updated_cell=}")
        if debug >=2 : print(f"{len(updated_ref_indices)=} {updated_ref_indices=}")

        all_found.extend(updated_cell)
        if debug >=2 : print(len(all_found))
        if debug >=2 : print(len(all_found) == len(cell_pos))

    if len(all_found) == len(cell_pos):
        need_to_change = False
    else:
        print("Error in getting fragments and reconstruction!!")
        for i, pos in enumerate(cell_pos):
            if i not in all_found:
                print(f"Cannot find the {i}th atom of the unit cell based on cartesian coordinates from the new structure.  ")
                print(f"{i} {cell_labels[i]=} {cell_pos[i]=} {cell_fracs[i]=}")
        print("Change wrap_keywords 'pbc' from", new_wrap_keywords["pbc"], "to", (not new_wrap_keywords["pbc"]))
        need_to_change = True
        new_wrap_keywords["pbc"] = (not wrap_keywords["pbc"])
        
    return need_to_change, new_wrap_keywords

######################################################
def reconstruct (refcell, newcell, sym_ops, debug: int=0):
    
    cell_labels = newcell.labels
    print(f"{cell_labels=}")
    if "D" in cell_labels:
        print("Deuterium is in the cell")
    cell_pos = newcell.coord
    cell_fracs = newcell.frac_coord
    cell_vector = newcell.cell_vector
    
    ref_labels = refcell.labels
    atom_site_labels = refcell.atom_site_labels

    print(f"{ref_labels=}")
    print(f"{atom_site_labels=}")

    if "D" in ref_labels:
        print("Deuterium is in the reference")
    cov_factor = refcell.refmoleclist[0].cov_factor
    metal_factor = refcell.refmoleclist[0].metal_factor
    

    new_structures = apply_symmetry_operations_reference(refcell, cell_vector, sym_ops)
    print(f"Number of symmetry operations: {len(new_structures)}")

    all_found = []
    all_molecules = []
    reconstructed_molecules = []
    remaining_fragments = [[] for _ in range(len(newcell.refmoleclist))]

    for idx, new in enumerate(new_structures):
        if debug >=2 : print(f"\nApplying symmetry operations to reference {idx}")
        indices_lists = get_updated_indices(idx, ref_labels, new, cell_labels, cell_pos, cell_fracs, debug=debug)
        if debug >=2 : print(f"{len(indices_lists)=}")

        updated_lists = [i for i in indices_lists if i[1] not in all_found]   
        updated_ref_indices = [i[0] for i in updated_lists]
        updated_cell = [i[1] for i in updated_lists]
        if debug >=2 : print(f"{len(updated_cell)=} {updated_cell=}")
        if debug >=2 : print(f"{len(updated_ref_indices)=} {updated_ref_indices=}")

        all_found.extend(updated_cell)
        if debug >=2 : print(len(all_found))
        if debug >=2 : print(len(all_found) == len(cell_pos))

        if len(updated_cell) > 0 :
            # #### make blocks and get fragments ####
            if refcell.exist_cif_bond_moiety:
                initial_fragments = get_fragments_from_moiety (newcell, updated_cell, updated_ref_indices, refcell, cov_factor=cov_factor, metal_factor=metal_factor, debug=2)    
            else :
                initial_fragments = get_fragments_new (newcell, updated_cell, updated_ref_indices, refcell, cov_factor=cov_factor, metal_factor=metal_factor, debug=2)    

            molecules, fragments = classify_fragments(initial_fragments, newcell, debug=2)
            all_molecules.extend(molecules)
            
            # Grouping fragments for each reference molecule
            fragments_of_new = [[] for _ in range(len(newcell.refmoleclist))]

            for i, ref in enumerate(newcell.refmoleclist):
                target_set = set(ref.get_parent_indices("reference"))
                for j, frag in enumerate(fragments):
                    small_set = set(frag.ref_indices)
                    if small_set.issubset(target_set):
                        if debug >= 2 : print(f"{j} {frag.formula} is a subset of target_set {i} {ref.formula}")
                        fragments_of_new[i].append(frag)
            
            if debug >=2 : print(f"symmetry operations: {idx}")
            for i, (ref, frag_list) in enumerate(zip(newcell.refmoleclist, fragments_of_new)):
                if debug >=2 : print(f"Reference {i}: {ref.formula} target : {ref.get_parent_indices('reference')}")
                if debug >=2 : print(f"\tFragments formula    : {[frag.formula for frag in frag_list]}")
                if debug >=2 : print(f"\tFragments ref_indices: {[frag.ref_indices for frag in frag_list]}")

            # Reconstructing fragments within one new structure
            for i, frag_list in enumerate(fragments_of_new):
                if len(frag_list) > 1:
                    if debug >=2 : print(f"target_ref: {newcell.refmoleclist[i].formula}")
                    if debug >=2 : print(f"Fragments formula {i}: {[frag.formula for frag in frag_list]}")
                    target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
                    list_of_found_molecules, remaining_frag = fragments_reconstruct (frag_list, target_ref, cell_vector, refcell, cov_factor=cov_factor, metal_factor=metal_factor, debug=0)
                    if debug >=2 : print(f"symmetry operations: {idx} {[mol.formula for mol in list_of_found_molecules]=}")
                    if debug >=2 : print(f"symmetry operations: {idx} {[frag.formula for frag in remaining_frag]=}")
                    if len(list_of_found_molecules) > 0:
                        reconstructed_molecules.extend(list_of_found_molecules)
                    if len(remaining_frag) > 0:
                        remaining_fragments[i].extend(remaining_frag)
                        if debug >=2 : print(f"symmetry operations: {idx} with ref{i} {[frag.formula for frag in remaining_fragments[i]]=}")
                elif len(frag_list) == 1:
                    if debug >=2 : print("only one fragment", frag_list[0].formula, "is found")
                    remaining_fragments[i].extend(frag_list)
                    if debug >=2 : print(f"symmetry operations: {idx} with ref{i} {[frag.formula for frag in remaining_fragments[i]]=}")

    if debug >=1 : print("complete molecules :", f"counts={len(all_molecules)}", [mol.formula for mol in all_molecules])
    if debug >=1 : print("reconstructed molecules :", f"counts={len(reconstructed_molecules)}", [mol.formula for mol in reconstructed_molecules])

    if len(all_found) == len(cell_pos):
        newcell.error_get_fragments = False
        num_rem_frags = 0
        for rem_frag_list in remaining_fragments:
            num_rem_frags += len(rem_frag_list) 
        
        if num_rem_frags == 0 :
            print("All fragments are reconstructed successfully.")
            newcell.error_reconstruction = False
        else:
            print(f"There are {num_rem_frags} remaining fragments.")
            final_remaining_fragments, reconstructed_molecules = final_remaining_reconstruction(remaining_fragments, newcell, cell_vector, reconstructed_molecules, refcell, cov_factor=cov_factor, metal_factor=metal_factor, debug=0)
            if len(final_remaining_fragments) == 0:
                print("All fragments are reconstructed successfully.")
                newcell.error_reconstruction = False
            else :
                print("Error in reconstruction!!")
                newcell.error_reconstruction = True
                print("final remaining fragments", len(final_remaining_fragments), [mol.formula for mol in final_remaining_fragments]) 
                for j, rem in enumerate(final_remaining_fragments):
                    writexyz(os.getcwd(), f"{newcell.name}_Frag_{i}_{rem.formula}_{j}.xyz", rem.labels, rem.coord) 
                # final_remaining_fragments_v2, reconstructed_molecules = final_remaining_reconstruction_v2(remaining_fragments, newcell, cell_vector, reconstructed_molecules, refcell, cov_factor=cov_factor, metal_factor=metal_factor, debug=0)
                # if len(final_remaining_fragments_v2) == 0:
                #     print("GOOD!! All fragments are reconstructed successfully.")
                #     newcell.error_reconstruction = False
                # else :
                #     print("AGAIN Error in reconstruction!!")
                #     newcell.error_reconstruction = True
                #     print("final remaining fragments", len(final_remaining_fragments_v2), [mol.formula for mol in final_remaining_fragments_v2]) 
                #     for j, rem in enumerate(final_remaining_fragments_v2):
                #         writexyz(os.getcwd(), f"{newcell.name}_Frag_{i}_{rem.formula}_{j}.xyz", rem.labels, rem.coord)                

    else:
        print("Error in getting fragments and reconstruction!!")
        newcell.error_get_fragments = True
        newcell.error_reconstruction = True          
        for i, pos in enumerate(cell_pos):
            if i not in all_found:
                print(f"Cannot find the {i}th atom of the unit cell based on cartesian coordinates from the new structure.")
                print(f"{i} {cell_labels[i]=} {cell_pos[i]=} {cell_fracs[i]=}")

    return all_molecules, reconstructed_molecules

######################################################
def final_remaining_reconstruction(remaining_fragments, newcell, cell_vector, reconstructed_molecules, refcell, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
    # Reconstructing remaining fragments within the whole new cell
    final_remaining_fragments = []
    for i, rem_frag_list in enumerate(remaining_fragments):
        if len(rem_frag_list) > 1:
            # for j, rem in enumerate(rem_frag_list):
            #     writexyz(os.getcwd(), f"{newcell.name}_Ref_{i}_{rem.formula}_{j}.xyz", rem.labels, rem.coord)
            print(f"target_ref: {newcell.refmoleclist[i].formula}")
            print(f"Fragments formula {i}: {[rem.formula for rem in rem_frag_list]}")
            target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
            list_of_found_molecules, final_remaining = fragments_reconstruct(rem_frag_list, target_ref, cell_vector, refcell, cov_factor=cov_factor, metal_factor=metal_factor, full=True, debug=debug)
            print(f"{[mol.formula for mol in list_of_found_molecules]=}")
            print(f"{[frag.formula for frag in final_remaining]=}")
            if len(list_of_found_molecules) > 0:
                reconstructed_molecules.extend(list_of_found_molecules)
            if len(final_remaining) > 0:
                final_remaining_fragments.extend(final_remaining)
        elif len(rem_frag_list) == 1:
            final_remaining_fragments.extend(rem_frag_list)

    return final_remaining_fragments, reconstructed_molecules
########################################################
def final_remaining_reconstruction_v2(remaining_fragments, newcell, cell_vector, reconstructed_molecules, refcell, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
    # Reconstructing remaining fragments within the whole new cell
    final_remaining_fragments = []
    for i, rem_frag_list in enumerate(remaining_fragments):
        if len(rem_frag_list) > 1:
            # for j, rem in enumerate(rem_frag_list):
            #     writexyz(os.getcwd(), f"{newcell.name}_Ref_{i}_{rem.formula}_{j}.xyz", rem.labels, rem.coord)
            print(f"target_ref: {newcell.refmoleclist[i].formula}")
            print(f"Fragments formula {i}: {[rem.formula for rem in rem_frag_list]}")
            target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
            list_of_found_molecules, final_remaining = fragments_reconstruct(rem_frag_list, target_ref, cell_vector, refcell, cov_factor=cov_factor, metal_factor=metal_factor, full=True, final_merge=True, debug=debug)
            print(f"{[mol.formula for mol in list_of_found_molecules]=}")
            print(f"{[frag.formula for frag in final_remaining]=}")
            if len(list_of_found_molecules) > 0:
                reconstructed_molecules.extend(list_of_found_molecules)
            if len(final_remaining) > 0:
                final_remaining_fragments.extend(final_remaining)
        elif len(rem_frag_list) == 1:
            final_remaining_fragments.extend(rem_frag_list)

    return final_remaining_fragments, reconstructed_molecules
######################################################
def get_moleclist (newcell, refcell, all_molecules, debug: int=0):
    cov_factor = refcell.refmoleclist[0].cov_factor
    metal_factor = refcell.refmoleclist[0].metal_factor

    # Get moleclist for the unit cell
    newcell.moleclist = []

    for mol in all_molecules:
        newmolec = molecule.from_positional(mol.labels, mol.coord, mol.frac_coord)
        mol_atom_site_labels = [refcell.atom_site_labels[idx] for idx in mol.ref_indices]
        if debug >=2 : 
            print("GET_MOLECLIST: ", mol.formula)
            print("GET_MOLECLIST: ", mol.ref_indices)
            print("GET_MOLECLIST: ", mol.labels)
            print("GET_MOLECLIST: ", mol_atom_site_labels)
        
        newmolec.origin = "cell.reconstruct"
        newmolec.set_adjacency_parameters(cov_factor, metal_factor)
        newmolec.set_atoms(create_adjacencies=True, 
                           atom_site_labels=mol_atom_site_labels, 
                           geom_bond_cif=refcell.geom_bond_cif, debug=debug)
        newmolec.add_parent(newcell, mol.cell_indices)
        newmolec.add_parent(refcell, mol.ref_indices) 
        for atom, idx in zip(newmolec.atoms, mol.cell_indices):
            atom.add_parent(newcell, index=idx)  
        for atom, idx in zip(newmolec.atoms, mol.ref_indices):
            atom.add_parent(refcell, index=idx)  
        if newmolec.iscomplex: 
            newmolec.split_complex()
        elif newmolec.has_IA_IIA:
            newmolec.split_IA_IIA()
        else:
            newmolec.add_parent(newmolec, indices=[*range(0,newmolec.natoms,1)])
        newcell.moleclist.append(newmolec)  

    for mol in newcell.moleclist:
        if mol.iscomplex: 
            if debug >=1 : print(f"GET_MOLECLIST: working with {mol.formula} with transition metals")
            mol.get_hapticity(debug=debug)
            if len(mol.ligands) == 0 :
                if debug >=1 : print(f"GET_MOLECLIST: {mol.formula} is a metal cluster")
            else:
                for lig in mol.ligands:
                    lig.get_denticity(debug=debug)
            for met in mol.metals:
                met.get_connected_metals(debug=debug)                         
                met.get_coordination_geometry(debug=debug)
                met.get_coord_sphere_formula(debug=debug)
        elif mol.has_IA_IIA:
            if debug >=1 : print(f"GET_MOLECLIST: working with {mol.formula} with alkali or alkali earth metals")
            if len(mol.ligands) == 0 :
                pass
            else:
                for lig in mol.ligands:
                    lig.get_denticity(debug=debug)
            for met in mol.metals: 
                met.get_connected_metals(debug=debug)                         
                met.get_coordination_geometry(debug=debug)
                met.get_coord_sphere_formula(debug=debug)

    return newcell

######################################################
def get_unique_indices(newcell, reference_species_list, debug: int=0):
    
    newcell.unique_indices = []
    newcell.species_list = []
    for mol in newcell.moleclist:
        if not mol.iscomplex and not mol.has_IA_IIA:
            for ref in reference_species_list:
                if (ref.subtype == "molecule") and not ref.iscomplex and not ref.has_IA_IIA:
                    issame = compare_reference_indices(ref, mol, debug=debug)
                    if issame:
                        mol.unique_index = ref.unique_index 
                        if debug >= 2: print(f"Matched {mol.formula} {ref.formula} {mol.unique_index} {ref.unique_index}")
                        newcell.unique_indices.append(mol.unique_index)
                        newcell.species_list.append(mol)
        else:
            for ref in reference_species_list:
                if ref.subtype == "ligand":
                    for lig in mol.ligands:
                        issame = compare_reference_indices(ref, lig, debug=debug)
                        if issame: 
                            lig.unique_index = ref.unique_index
                            if debug >= 2: print(f"Matched {lig.formula} {ref.formula} {lig.unique_index} {ref.unique_index}")
                            newcell.unique_indices.append(lig.unique_index)
                            newcell.species_list.append(lig)
                if ref.subtype == "metal":
                    for met in mol.metals:
                        if ref.get_parent_index("reference") == met.get_parent_index("reference"):
                            met.unique_index = ref.unique_index
                            if debug >= 2: print(f"Matched {met.formula} {ref.formula} {met.unique_index} {ref.unique_index}")
                            newcell.unique_indices.append(met.unique_index)
                            newcell.species_list.append(met)

    return newcell

# ######################################################
# def assign_equivalent_indices (refcell, unique_species, species_list, unique_indices, debug: int=0):
    
#     for idx, specie in enumerate(refcell.species_list):
#         specie.equivalent_index = idx
#     for idx, specie in enumerate(species_list):
#         found = False
#         for unique_spec in unique_species:
#             if (specie.subtype == unique_spec.subtype) and (specie.unique_index == unique_spec.unique_index):
#                 issame = compare_reference_indices(specie, unique_spec, debug=debug)
#                 if issame:
#                     found = True
#                     specie.equivalent_index = idx
#                 if found :
#                     specie.equivalent_index = idx
                