import numpy as np
from cell2mol.connectivity import split_species
from cell2mol.other import extract_from_list
from cell2mol.classes import molecule
from cell2mol.cell_reconstruction import tmatgenerator
from cell2mol.other import additem, absolute_value
from cell2mol.connectivity import count_species
from cell2mol.cell_operations import translate
import itertools
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase import Atoms

######################################################
def find_row_indices(source, target):
    
    # List to store the indices of found rows
    found_indices = []
    found_rows = []
    remaining_indices = []
    remaining_rows = []
    
    # Iterate over each row in the source array with enumeration to track the index
    for index, row in enumerate(source):
        # Check if any row in the target array matches the current row
        if any(np.allclose(row, target_row) for target_row in target):
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
        if np.allclose(row, query_row):
            return index
    return -1

######################################################
def get_fragments (newcell, updated, indices_in_ref, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
                   
    updated_labels  = extract_from_list(updated, newcell.labels, dimension=1)
    updated_coord   = extract_from_list(updated, newcell.coord, dimension=1)
    updated_fracs = extract_from_list(updated, newcell.frac_coord, dimension=1)
    
    blocklist = split_species(updated_labels, updated_coord, debug=debug)
    
    fragments = []
    list_ref_indices = []
    
    for b in blocklist:
        if debug > 0: print(f"CELL.MOLECLIST: doing block={b}")
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord  = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord  = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        
        # Creates Molecule Object
        newmolec    = molecule(mol_labels, mol_coord)
        
        # For debugging
        newmolec.origin = "cell.get_fragments"
        
        # Adds cell as parent of the molecule, with indices
        newmolec.add_parent(newcell, indices=cell_indices)        
        newmolec.set_fractional_coord(mol_frac_coord)
        newmolec.set_adjacency_parameters(cov_factor=cov_factor, metal_factor=metal_factor)
        newmolec.set_atoms(create_adjacencies=True, debug=debug)
        newmolec.ref_indices = ref_indices
        fragments.append(newmolec)
        
    return fragments

######################################################
def classify_fragments (fragments, newcell, debug: int=0):
    
    molecules = []
    found_list = []
    remaining_fragments = []

    for frag in fragments:
        found = False
        for idx, ref in enumerate(newcell.refmoleclist):
            if (ref.natoms == frag.natoms) & (ref.formula == frag.formula):
                if (sorted(ref.get_parent_indices("cell")) == sorted(frag.ref_indices)):
                    if debug >=1 : 
                        print(frag.formula, frag.ref_indices, frag.frac_coord, \
                              f"equivalent to Ref {idx} {ref.formula}")
                    frag.subtype = "molecule"
                    frag.origin = "cell.classify_fragments"
                    molecules.append(frag)
                    found = True
                    found_list.append(idx)
        
        if found == False:
            frag.subtype = "fragment"
            frag.origin = "cell.classify_fragments"
            remaining_fragments.append(frag)
            
    not_found_list = [i for i in range(len(newcell.refmoleclist)) if i not in found_list]  
    
    return molecules, found_list, remaining_fragments, not_found_list

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
                if debug >=1 : print(f"grouped_lists {i} add {small_set}")
                grouped_lists[i].extend(small_list)
                grouped_lists_idx[i].append(j)
                target_idx_lists[i].append(i)
                
                if set(grouped_lists[i]) == target_set:
                    if debug >=1 :print(f"grouped_lists {i} is same with {target_set} {grouped_lists_idx[i]}")
                else :
                    if debug >=1 :print(f"continue for {i} {grouped_lists_idx[i]}")
                break
    
    if debug >=1 :
        # Print the grouped lists
        for i, group in enumerate(grouped_lists):
            print(f"Group {i+1}: {group}")
            
    return grouped_lists, grouped_lists_idx, target_idx_lists

######################################################
def grouping_remaining_fragments(remaining_fragments, not_found_list, newcell, debug: int=0):

    smaller_lists = [rem_frag.ref_indices for rem_frag in remaining_fragments]
    smaller_lists.sort(key=len, reverse=True)

    target_sets = []
    for j in not_found_list:
        try:
            ref_indices = newcell.refmoleclist[j].get_parent_indices("cell")
            target_sets.append(set(ref_indices))
        except AttributeError as e:
            print(f"Error accessing or calling get_parent_indices on refmoleclist {j}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with element {j}: {e}")

    if debug >= 1: 
        print(f"{target_sets=}")
        print(f"{smaller_lists=}")
        
    grouped_rem_frags, indices_of_rem_frags, target_idx_lists = grouping_smaller_lists_for_target_sets(target_sets, smaller_lists, debug=debug)
    
    not_found_refmoleclist_indices = [ list(set(sublist))[0] if sublist else None for sublist in target_idx_lists ]

    indices_of_target_ref = []
    for t in not_found_refmoleclist_indices:
        indices_of_target_ref.append( newcell.refmoleclist[t].get_parent_indices("cell") )

    return grouped_rem_frags, indices_of_rem_frags, indices_of_target_ref
######################################################
def merge_fragments(frags: list, cell_vector: list, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):

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
    if debug > 0: print("MERGE_FRAGMENTS: keep_idx", keep_idx)
    if debug > 0: print("MERGE_FRAGMENTS: move_idx", move_idx)

    move_frag.get_centroid()
    move_frag.tmatrix = tmatgenerator(move_frag.frac_centroid)
    
    if len(move_frag.tmatrix) == 0: return None

    for t in move_frag.tmatrix:
        if debug > 0: print("MERGE_FRAGMENTS: translation", t)
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
        
        numspecs  = count_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        if debug > 0: print("MERGE_FRAGMENTS: count_species found", numspecs)
        if numspecs != 1: continue
        blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        if blocklist is None: continue
        else:
            if len(blocklist) != 1: continue
            if len(blocklist) == 1: 
                newmolec = molecule(reclabels, reccoord)
                newmolec.ref_indices = rec_ref_indices
                newmolec.set_adjacency_parameters(cov_factor, metal_factor)
                newmolec.set_adj_types()
                newmolec.set_element_count()
                newmolec.get_adjmatrix()
                newmolec.get_centroid()
                newmolec.get_metal_adjmatrix()
                return newmolec
    return None

######################################################
# Function to generate combinations and the rest of the elements
def combinations_and_rest(data, n):
    for combination in itertools.combinations(data, n):
        # Create a set from the combination for fast removal
        remaining = [x for x in data if x not in combination]
        yield combination, remaining

######################################################
# def find_molecules_from_remaining_fragments(remaining_fragments, indices_from_rem_frags, indices_of_target_ref):
def find_molecule_from_subset_rem_frags(subset_remaining_fragments, target_ref, cell_vector, debug: int=0):
    list_of_found_molecules = []    
    
    final_remaining = subset_remaining_fragments.copy()
    while (len(final_remaining) > 0):
        print("final_remaining", [k.formula for k in final_remaining], [k.subtype for k in final_remaining])
        results = list(combinations_and_rest(final_remaining, 2))
    
        for combo, rest in results:
            print("Fragments TO BE MERGED", [k.formula for k in combo], [k.subtype for k in combo])
            print("Rest fragments",[k.formula for k in rest],[k.subtype for k in rest])
            found_molecule =[]
            bigger_fragment = []
            not_merged = []        
            newmolec = merge_fragments(combo, cell_vector, debug=debug)
            print(newmolec)
            if newmolec is None: 
                print("NOT MERGED", combo[0].formula, combo[1].formula)
                not_merged.append(combo[0])
                not_merged.append(combo[1])
            else :
                print("MERGED", newmolec.formula, newmolec.ref_indices)
                small_set = set(sorted(newmolec.ref_indices))
                print(f"{small_set=}")
                if small_set.issubset(target_ref): 
                    if sorted(small_set) == target_ref:
                        newmolec.subtype = "molecule"
                        newmolec.origin = "cell.reconstruct"
                        found_molecule.append(newmolec)
                    else :
                        newmolec.subtype = "Rec. Fragment"
                        newmolec.origin = "cell.reconstruct"
                        bigger_fragment.append(newmolec)
                        
            print(f"{len(found_molecule)=}")
            print(f"{len(bigger_fragment)=}")
            print(f"{len(not_merged)=}")
            
            if len(not_merged) == 2:
                #continue the process
                pass    
            elif len(found_molecule) == 1 :
                list_of_found_molecules.append(newmolec)
                final_remaining = []
                final_remaining.extend(rest)
            elif len(bigger_fragment) == 1 :
                final_remaining = []
                final_remaining.append(newmolec)
                final_remaining.extend(rest)
            
            if len(final_remaining) == 0:
                break
    return list_of_found_molecules, final_remaining