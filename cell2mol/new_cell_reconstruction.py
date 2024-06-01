import ase
import numpy as np
import itertools
import ase
from ase import Atoms
from cell2mol.classes import molecule
from cell2mol.cell_reconstruction import tmatgenerator
from cell2mol.other import additem, absolute_value, get_dist, extract_from_list
from cell2mol.connectivity import split_species, count_species
from cell2mol.cell_operations import translate

######################################################
def apply_symmetry_operations_reference (reference, cell_vector, sym_ops, normalize=True):
    
    # reference : ase atoms object
    
    new_structures = []
    ref_labels = reference.get_chemical_symbols()
    fractional_coords = reference.get_scaled_positions()
    
    for rot, trans in zip(sym_ops[0], sym_ops[1]):
        transformed_positions = np.dot(fractional_coords, rot.T)
        transformed_positions += np.array(trans)       
        if normalize:
            transformed_positions = np.remainder(transformed_positions, 1)
        new = Atoms(symbols=ref_labels, scaled_positions=transformed_positions, cell=cell_vector)    
        new_structures.append(new)
    
    return new_structures

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
    
    for b in blocklist:
        if debug > 0: print(f"CELL.MOLECLIST: doing block={b}")
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord  = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord  = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        
        # Creates Molecule Object
        newmolec    = molecule(mol_labels, mol_coord, mol_frac_coord)
        
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
def classify_fragments (fragments, newcell, debug: int=0):
    
    molecules = []
    remaining_fragments = []
    hydrogens = []
    for frag in fragments:
        found = False
        for idx, ref in enumerate(newcell.refmoleclist):
            if (ref.natoms == frag.natoms) & (ref.formula == frag.formula):
                if (sorted(ref.get_parent_indices("reference")) == sorted(frag.ref_indices)):
                    if debug >=1 : 
                        print(frag.formula, frag.ref_indices, frag.frac_coord, \
                              f"equivalent to Ref {idx} {ref.formula}")
                    frag.subtype = "molecule"
                    frag.origin = "cell.classify_fragments"
                    molecules.append(frag)
                    found = True
        
        if found == False:
            frag.subtype = "fragment"
            frag.origin = "cell.classify_fragments"
            # if (frag.natoms == 1) and (frag.set_element_count()[4] + frag.set_element_count()[3] == 1): 
            #     hydrogens.append(frag) # # Hydrogen or Deuterium 
            # else:    
            #     remaining_fragments.append(frag)
            remaining_fragments.append(frag)
    rem_size = np.array([rem.natoms for rem in remaining_fragments])
    order = np.argsort(rem_size)
    descending_order = order[::-1]
    remaining_fragments = [remaining_fragments[i] for i in descending_order]
    for rem in remaining_fragments:
        rem.get_centroid()
        
    if debug >=1 :
        print("Remaining_fragments:", [rem.formula for rem in remaining_fragments])
        print("Remaining_fragments:", [rem.natoms for rem in remaining_fragments])
        print("Remaining_fragments:", [get_dist(rem.frac_centroid, [0.5, 0.5, 0.5]) for rem in remaining_fragments])    
        print("Hydrogens:", [h.formula for h in hydrogens])
    # return molecules, remaining_fragments, hydrogens
    return molecules, remaining_fragments

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

    if debug >= 1: 
        print(f"{target_sets=}")
        print(f"{smaller_lists=}")
        
    grouped_rem_frags, indices_of_rem_frags, target_idx_lists = grouping_smaller_lists_for_target_sets(target_sets, smaller_lists, debug=debug)
    
    not_found_refmoleclist_indices = [ list(set(sublist))[0] if sublist else None for sublist in target_idx_lists ]

    indices_of_target_ref = []
    for t in not_found_refmoleclist_indices:
        indices_of_target_ref.append( newcell.refmoleclist[t].get_parent_indices("reference") )

    return grouped_rem_frags, indices_of_rem_frags, indices_of_target_ref
    
######################################################
def merge_fragments (frags: list, cell_vector: list, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):

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
        
        rec_cell_indices = []
        rec_cell_indices.extend(keep_frag.cell_indices)
        rec_cell_indices.extend(move_frag.cell_indices)

        recfracs = []
        recfracs.extend(keep_frag.frac_coord)
        recfracs.extend(move_frag.frac_coord)
        
        numspecs  = count_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        
        if debug > 0: print("MERGE_FRAGMENTS: count_species found", numspecs)
        if numspecs != 1: continue
        blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor, debug=debug)
        if blocklist is None: continue
        else:
            if len(blocklist) != 1: continue
            if len(blocklist) == 1: 
                newmolec = molecule(reclabels, reccoord, recfracs)
                newmolec.origin = "cell.reconstruct"
                newmolec.ref_indices = rec_ref_indices
                newmolec.cell_indices = rec_cell_indices
                newmolec.set_adjacency_parameters(cov_factor, metal_factor)
                newmolec.set_adj_types()
                newmolec.set_element_count()
                newmolec.get_adjmatrix()
                newmolec.get_centroid()
                newmolec.get_metal_adjmatrix()
                return newmolec
    return None

######################################################
def fragments_reconstruct (subset_remaining_fragments, target_ref, cell_vector, debug: int=0):
    
    list_of_found_molecules = []    
    remaining_frag = subset_remaining_fragments.copy()
    
    count = 0
    while (len(remaining_frag) > 0):
        print("remaining_frag", [k.formula for k in remaining_frag], [k.subtype for k in remaining_frag])
        
        tobemerged = remaining_frag[:2]
        rest = remaining_frag[2:]
            
        print("Fragments TO BE MERGED", [k.formula for k in tobemerged], [k.subtype for k in tobemerged])
        print("Rest fragments",[k.formula for k in rest],[k.subtype for k in rest])
        
        found_molecule =[]
        bigger_fragment = []
        not_merged = []        
        newmolec = merge_fragments(tobemerged, cell_vector, debug=debug)

        if newmolec is None: 
            print("NOT MERGED", tobemerged[0].formula, tobemerged[1].formula)
            not_merged.append(tobemerged[0])
            not_merged.append(tobemerged[1])

        else :
            print("MERGED", newmolec.formula)
            small_set = set(newmolec.ref_indices)
            print(f"{small_set=}")
            if small_set.issubset(target_ref): 
                if sorted(small_set) == target_ref:
                    newmolec.subtype = "molecule"
                    found_molecule.append(newmolec)
                else :
                    newmolec.subtype = "Rec. Fragment"
                    bigger_fragment.append(newmolec)
                    
        print(f"{len(found_molecule)=}")
        print(f"{len(bigger_fragment)=}")
        print(f"{len(not_merged)=}")
        
        if len(not_merged) == 2:
            remaining_frag = []
            remaining_frag.append(tobemerged[0])
            remaining_frag.extend(rest)
            remaining_frag.append(tobemerged[1])
                        
        elif len(found_molecule) == 1 :
            list_of_found_molecules.append(newmolec)
            remaining_frag = []
            remaining_frag.extend(rest)

            if len(remaining_frag) == 0:
                print("All fragments are reconstructed successfully", [k.formula for k in list_of_found_molecules])
                print(f"{count=}")
                break

        elif len(bigger_fragment) == 1 :
            remaining_frag = []
            remaining_frag.append(newmolec)
            remaining_frag.extend(rest)
            if len(remaining_frag) == 1:
                print("Bigger fragments are remained", [k.formula for k in remaining_frag])
                print(f"{count=}")
                break        

        count +=1
        
            
    return list_of_found_molecules, remaining_frag


######################################################
def get_updated_indices (new, cell_pos, cell_fracs, all_found, debug: int=0):

    # new structure : ase atoms object by applying symmetry operations to the reference structure
    # Find the indices of the atoms in the unit cell that have the same cartisian coordinates as the atoms in the new structure
    
    found_indices, found_rows = find_row_indices(cell_pos, new.positions)
    print(len(found_indices), len(found_rows))

    updated = [i for i in found_indices if i not in all_found]

    new_fracs = new.get_scaled_positions()
    new_labels =  new.get_chemical_symbols()
    
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
                    print(f"Its fractional coord disagrees with the fractional coord of the {updated[j]}th atom of the unit cell.")
                    # print(f"{cell_pos[updated[j]]=} {cell_fracs[updated[j]]=}")
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
def reconstuct (reference, newcell, cell_pos, cell_fracs, cell_vector, sym_ops, debug: int=0):

    new_structures = apply_symmetry_operations_reference(reference, cell_vector, sym_ops)
    print(f"Number of symmetry operations: {len(new_structures)}")

    all_found = []
    all_molecules = []
    reconstructed_molecules = []

    remaining_fragments = [[] for _ in range(len(newcell.refmoleclist))]

    for idx, new in enumerate(new_structures):
        print(f"Applying symmetry operations to reference {idx}")

        updated, indices_in_ref = get_updated_indices(new, cell_pos, cell_fracs, all_found, debug=debug)
        all_found.extend(updated)
        print(len(all_found))
        print(len(all_found) == len(cell_pos))

        if len(updated) > 0 :
            # #### make blocks and get fragments ####
            initial_fragments = get_fragments (newcell, updated, indices_in_ref, debug=0)    
            molecules, fragments = classify_fragments(initial_fragments, newcell, debug=0)
            all_molecules.extend(molecules)
            
            # Grouping fragments for each reference molecule
            fragments_of_new = [[] for _ in range(len(newcell.refmoleclist))]

            for i, ref in enumerate(newcell.refmoleclist):
                target_set = set(ref.get_parent_indices("reference"))
                for j, frag in enumerate(fragments):
                    small_set = set(frag.ref_indices)
                    if small_set.issubset(target_set):
                        if debug >=1 : print(f"{j} {frag.formula} is a subset of target_set {i} {ref.formula}")
                        fragments_of_new[i].append(frag)
            
            print(f"symmetry operations: {idx}")
            for i, (ref, frag_list) in enumerate(zip(newcell.refmoleclist, fragments_of_new)):
                print(f"Reference {i}: {ref.formula} target : {ref.get_parent_indices('reference')}")
                print(f"\tFragments formula    : {[frag.formula for frag in frag_list]}")
                print(f"\tFragments ref_indices: {[frag.ref_indices for frag in frag_list]}")

            # Reconstructing fragments within one new structure
            for i, frag_list in enumerate(fragments_of_new):
                if len(frag_list) > 1:
                    print(f"target_ref: {newcell.refmoleclist[i].formula}")
                    print(f"Fragments formula {i}: {[frag.formula for frag in frag_list]}")
                    target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
                    list_of_found_molecules, remaining_frag = fragments_reconstruct(frag_list, target_ref, cell_vector, debug=0)
                    print(f"symmetry operations: {idx} {list_of_found_molecules=}")
                    print(f"symmetry operations: {idx} {remaining_frag=}")
                    if len(list_of_found_molecules) > 0:
                        reconstructed_molecules.extend(list_of_found_molecules)
                    if len(remaining_frag) > 0:
                        remaining_fragments[i].extend(remaining_frag)
                        print(f"symmetry operations: {idx} {remaining_fragments[i]=}")

    final_remaining_fragments = []
    if len(all_found) == len(cell_pos):
        for i, rem_frag_list in enumerate(remaining_fragments):
            if len(rem_frag_list) > 1:
                print(f"target_ref: {newcell.refmoleclist[i].formula}")
                print(f"Fragments formula {i}: {[rem.formula for rem in rem_frag_list]}")
                target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
                list_of_found_molecules, final_remaining = fragments_reconstruct(rem_frag_list, target_ref, cell_vector, debug=0)
                print(f"{list_of_found_molecules=}")
                print(f"{final_remaining=}")
                if len(list_of_found_molecules) > 0:
                    reconstructed_molecules.extend(list_of_found_molecules)
                if len(final_remaining) > 0:
                    final_remaining_fragments.extend(final_remaining)
        
        if len(final_remaining_fragments) == 0:
            newcell.is_fragmented = False
            newcell.error_reconstruction = False
            print("All fragments are reconstructed successfully")
        else:
            newcell.is_fragmented = True        
            newcell.error_reconstruction = True
            print("Final Remaining Fragments")
            for rem in final_remaining_fragments:
                print(rem.formula)
    else:
        print("Error in reconstruction!!")
        newcell.is_fragmented = True
        newcell.error_reconstruction = True

    return all_molecules, reconstructed_molecules, final_remaining_fragments

######################################################
def get_moleclist (newcell, refcell, all_molecules, debug):
    # Get moleclist for the unit cell
    newcell.moleclist = []

    for mol in all_molecules:
        newmolec = molecule(mol.labels, mol.coord, mol.frac_coord)
        newmolec.origin = "cell.reconstruct"
        newmolec.set_atoms(create_adjacencies=True, debug=debug)
        newmolec.add_parent(newcell, mol.cell_indices)
        newmolec.add_parent(refcell, mol.ref_indices) 
        for atom, idx in zip(newmolec.atoms, mol.cell_indices):
            atom.add_parent(newcell, index=idx)  
        for atom, idx in zip(newmolec.atoms, mol.ref_indices):
            atom.add_parent(refcell, index=idx)  
        if newmolec.iscomplex: newmolec.split_complex()
        newcell.moleclist.append(newmolec)  

    for mol in newcell.moleclist:
        if mol.iscomplex: 
            mol.get_hapticity(debug=debug)
            for lig in mol.ligands:
                lig.get_denticity(debug=debug)
            for met in mol.metals:                         
                met.get_coordination_geometry(debug=debug)
    
    return newcell