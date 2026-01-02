import numpy as np
import logging
from ase import Atoms
from cell2mol.classes import Molecule
from cell2mol.compare import compare_reference_indices
from cell2mol.connectivity import split_species
from cell2mol.operations import translate, tmatgenerator, extract_from_list
from itertools import combinations
from cell2mol.elementdata import ElementData

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def apply_symmetry_operations(refcell, cell_vector, sym_ops, normalize: bool = True):
    """Applies symmetry operations to a reference cell.

    This function generates a set of new atomic structures by applying a list of
    symmetry operations (rotations and translations) to the fractional coordinates
    of a reference structure.

    Args:
        refcell (object): An object representing the reference cell, which must
            have `labels` and `frac_coord` attributes.
        cell_vector (np.ndarray): The cell vectors for the new `ase.Atoms`
            objects.
        sym_ops (tuple): A tuple containing two lists: a list of rotation
            matrices and a list of translation vectors.
        normalize (bool, optional): If True, the transformed fractional
            coordinates are wrapped back into the unit cell (0 to 1).
            Defaults to True.

    Returns:
        list: A list of `ase.Atoms` objects, where each object represents the
              reference structure after one symmetry operation has been applied.
    """
    new_structures = []
    ref_labels = refcell.labels
    fractional_coords = np.array(refcell.frac_coord)

    if "D" in ref_labels:
        # Atoms object cannot handle Deuterium in the symbols
        numbers = [elemdatabase.elementnr[elem] for elem in ref_labels]

    for rot, trans in zip(sym_ops[0], sym_ops[1]):
        transformed_positions = np.dot(fractional_coords, rot.T)
        transformed_positions += np.array(trans)
        if normalize:
            transformed_positions = np.remainder(transformed_positions, 1)
        if "D" in ref_labels:
            new = Atoms(
                scaled_positions=transformed_positions,
                numbers=numbers,
                cell=cell_vector,
            )
        else:
            new = Atoms(
                symbols=ref_labels,
                scaled_positions=transformed_positions,
                cell=cell_vector,
            )

        new_structures.append(new)

    return new_structures


def get_fragments_from_moiety(
    newcell,
    updated,
    indices_in_ref,
    refcell,
    cov_factor: float = 1.0,
    metal_factor: float = 1.0,
):
    """
    Get molecular fragments from a moiety in the new cell.
    Args:
        newcell (object): The new cell object containing reference molecules.
        updated (list): List of updated atom indices in the new cell.
        indices_in_ref (list): List of atom indices in the reference cell.
        refcell (object): The reference cell object containing reference molecules.
        cov_factor (float): Covalent factor for adjacency determination.
        metal_factor (float): Metal factor for adjacency determination.
    Returns:
        fragments (list): List of molecular fragments
    """
    atom_site_labels = refcell.atom_site_labels
    geom_bond_cif = refcell.geom_bond_cif
    moiety_indices = refcell.moiety_indices

    # logger.debug(f"{updated=}")
    # logger.debug(f"{indices_in_ref=}")
    # logger.debug(f"{moiety_indices=}")

    updated_labels = extract_from_list(updated, newcell.labels, dimension=1)
    updated_coord = extract_from_list(updated, newcell.coord, dimension=1)
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

    # logger.debug(
    #     "updated_moieties_list %d %s",
    #     len(updated_moieties_list),
    #     updated_moieties_list,
    # )

    # logger.debug(
    #     "updated_moieties_indices_in_ref_list %d %s",
    #     len(updated_moieties_indices_in_ref_list),
    #     updated_moieties_indices_in_ref_list,
    # )
    tmp_blocklist = []
    for updated_moieties, updated_moieties_indices_in_ref in zip(
        updated_moieties_list, updated_moieties_indices_in_ref_list
    ):
        if len(updated_moieties) == 0:
            continue
        # logger.debug("updated_moieties %s", updated_moieties)
        updated_moieties_labels = extract_from_list(
            updated_moieties, newcell.labels, dimension=1
        )
        updated_moieties_coord = extract_from_list(
            updated_moieties, newcell.coord, dimension=1
        )
        updated_moieties_atom_site_labels = [
            atom_site_labels[i] for i in updated_moieties_indices_in_ref
        ]
        # logger.debug("updated_moieties_labels %s", updated_moieties_labels)
        # logger.debug(
        #     "updated_moieties_atom_site_labels %s",
        #     updated_moieties_atom_site_labels,
        # )
        block = split_species(
            updated_moieties_labels,
            updated_moieties_coord,
            indices=updated_moieties,
            atom_site_labels=updated_moieties_atom_site_labels,
            geom_bond_cif=geom_bond_cif,
        )
        tmp_blocklist.extend(block)

    # logger.debug("tmp_blocklist %s", tmp_blocklist)

    value_to_index = {val: idx for idx, val in enumerate(updated)}
    blocklist = [
        [value_to_index[val] for val in sublist if val in value_to_index]
        for sublist in tmp_blocklist
    ]
    # logger.debug("blocklist %s", blocklist)

    fragments = []
    for b in blocklist:
        # logger.debug("doing block=%s", b)
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        mol_atom_site_labels = [atom_site_labels[idx] for idx in ref_indices]

        # Creates Molecule Object
        newmolec = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)

        # For debugging
        newmolec.set_origin("cell.get_fragments")

        # Adds cell as parent of the molecule, with indices
        newmolec.add_parent(newcell, indices=cell_indices)
        newmolec.add_parent(refcell, indices=ref_indices)
        newmolec.set_fractional_coord(mol_frac_coord)
        newmolec.set_adjacency_parameters(
            cov_factor=cov_factor, metal_factor=metal_factor
        )
        newmolec.set_atoms(
            create_adjacencies=True,
            atom_site_labels=mol_atom_site_labels,
            geom_bond_cif=geom_bond_cif,
        )
        newmolec.ref_indices = ref_indices
        newmolec.cell_indices = cell_indices
        fragments.append(newmolec)

    return fragments


def get_fragments(
    newcell,
    updated,
    indices_in_ref,
    refcell,
    cov_factor: float = 1.0,
    metal_factor: float = 1.0,
):
    """
    Get molecular fragments from the new cell structure.
    Args:
        newcell (object): New cell object containing the updated structure.
        updated (list): List of updated atom indices in the new structure.
        indices_in_ref (list): List of atom indices in the reference structure.
        refcell (object): Reference cell object containing reference information.
        cov_factor (float): Covalent factor for adjacency determination.
        metal_factor (float): Metal factor for adjacency determination.
    Returns:
        fragments (list): List of molecular fragment objects.
    """
    atom_site_labels = refcell.atom_site_labels
    geom_bond_cif = refcell.geom_bond_cif
    moiety_indices = refcell.moiety_indices

    # logger.debug(f"{updated=}")
    # logger.debug(f"{indices_in_ref=}")
    # logger.debug(f"{moiety_indices=}")

    updated_labels = extract_from_list(updated, newcell.labels, dimension=1)
    updated_coord = extract_from_list(updated, newcell.coord, dimension=1)
    updated_fracs = extract_from_list(updated, newcell.frac_coord, dimension=1)
    if not refcell.exist_cif_bond_moiety:
        blocklist = split_species(updated_labels, updated_coord)
    else:
        updated_moieties_list = [
            [updated[indices_in_ref.index(i)] for i in sublist if i in indices_in_ref]
            for sublist in moiety_indices
        ]
        # logger.debug("updated_moieties_list %s", updated_moieties_list)
        tmp_blocklist = []
        for updated_moieties in updated_moieties_list:
            if len(updated_moieties) == 0:
                continue
            updated_moieties_labels = extract_from_list(
                updated_moieties, newcell.labels, dimension=1
            )
            updated_moieties_coord = extract_from_list(
                updated_moieties, newcell.coord, dimension=1
            )
            block = split_species(
                updated_moieties_labels,
                updated_moieties_coord,
                indices=updated_moieties,
            )
            tmp_blocklist.extend(block)
            # logger.debug("updated_moieties %s", updated_moieties)
        # logger.debug("tmp_blocklist %s", tmp_blocklist)

        value_to_index = {val: idx for idx, val in enumerate(updated)}
        blocklist = [
            [value_to_index[val] for val in sublist if val in value_to_index]
            for sublist in tmp_blocklist
        ]
    # logger.debug("blocklist %s", blocklist)
    if blocklist is None:
        return []

    fragments = []
    for b in blocklist:
        # logger.debug("doing block=%s", b)
        mol_labels = extract_from_list(b, updated_labels, dimension=1)
        mol_coord = extract_from_list(b, updated_coord, dimension=1)
        mol_frac_coord = extract_from_list(b, updated_fracs, dimension=1)

        cell_indices = extract_from_list(b, updated, dimension=1)
        ref_indices = extract_from_list(b, indices_in_ref, dimension=1)
        mol_atom_site_labels = [atom_site_labels[idx] for idx in ref_indices]

        # Creates Molecule Object
        newmolec = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)

        # For debugging
        newmolec.set_origin("cell.get_fragments")

        # Adds cell as parent of the molecule, with indices
        newmolec.add_parent(newcell, indices=cell_indices)
        newmolec.add_parent(refcell, indices=ref_indices)
        newmolec.set_fractional_coord(mol_frac_coord)
        newmolec.set_adjacency_parameters(
            cov_factor=cov_factor, metal_factor=metal_factor
        )
        newmolec.set_atoms(
            create_adjacencies=True,
            atom_site_labels=mol_atom_site_labels,
            geom_bond_cif=geom_bond_cif,
        )
        newmolec.ref_indices = ref_indices
        newmolec.cell_indices = cell_indices
        fragments.append(newmolec)

    return fragments


def classify_fragments(fragments, refmoleclist):
    """
    Classify fragments into complete molecules and remaining fragments.

    Args:
        fragments (list): List of molecular fragments to classify.
        refmoleclist (list): List of reference molecules.
    Returns:
        molecules (list): List of complete molecules matched to reference molecules.
        remaining_fragments (list): List of remaining fragments and hydrogens.
    """
    molecules = []
    remaining_fragments = []
    hydrogens = []
    for frag in fragments:
        found = False
        for idx, ref in enumerate(refmoleclist):
            if (ref.natoms == frag.natoms) & (ref.formula == frag.formula):
                if sorted(ref.get_parent_indices("reference")) == sorted(
                    frag.ref_indices
                ):
                    logger.debug(
                        f"Fragment {frag.formula} matched to reference molecule {idx} {ref.formula}"
                    )
                    frag.set_subtype("molecule")
                    frag.set_origin("cell.classify_fragments")
                    molecules.append(frag)
                    found = True

        if not found:
            frag.set_subtype("fragment")
            frag.set_origin("cell.classify_fragments")
            # Check if the fragment is a single hydrogen or deuterium atom
            if (frag.natoms == 1) and (
                frag.set_element_count()[4] + frag.set_element_count()[3] == 1
            ):
                hydrogens.append(frag)
            else:
                remaining_fragments.append(frag)

    rem_size = np.array([rem.natoms for rem in remaining_fragments])
    order = np.argsort(rem_size)
    descending_order = order[::-1]
    remaining_fragments = [remaining_fragments[i] for i in descending_order]
    for rem in remaining_fragments + hydrogens:
        rem.get_centroid()

    logger.debug(
        "Remaining_fragments: %s", [rem.formula for rem in remaining_fragments]
    )
    logger.debug("Hydrogens: %s", [h.formula for h in hydrogens])
    return molecules, remaining_fragments + hydrogens


def merge_fragment_pair(
    frags: list,
    cell_vector: list,
    refcell: object,
    cov_factor: float = 1.0,
    metal_factor: float = 1.0,
    full: bool = False,
    final_merge: bool = False,
):
    """
    Attempt to merge two fragments by translating one fragment across the unit cell.

    Returns
    -------
    Molecule or None
        A merged structure, which may be an intermediate fragment or a complete
        molecule. Returns None if no valid merge is found.
    """
    # finds biggest fragment and keeps it in the original cell
    sizes = []
    for f in frags:
        size = f.natoms
        sizes.append(size)
    keep_idx = np.argmax(sizes)
    if keep_idx == 0:
        move_idx = 1
    elif keep_idx == 1:
        move_idx = 0
    keep_frag = frags[keep_idx]
    move_frag = frags[move_idx]
    move_frag.get_centroid()

    # Check if the moving fragment is a single hydrogen or deuterium atom
    if move_frag.natoms == 1 and (
        move_frag.set_element_count()[4] + move_frag.set_element_count()[3] == 1
    ):
        full = True

    tmatrix = tmatgenerator(move_frag.frac_centroid, full=full)

    if len(tmatrix) == 0:
        return None

    for t in tmatrix:
        ## Applies Translations and each time, it checks if a bigger molecule is formed
        ## meaning that the translation was successful
        reclabels = []
        reclabels.extend(keep_frag.labels)
        reclabels.extend(move_frag.labels)
        reccoord = []
        reccoord.extend(keep_frag.coord)
        if t == (0, 0, 0):
            reccoord.extend(move_frag.coord)
        else:
            reccoord.extend(translate(t, move_frag.coord, cell_vector))

        rec_ref_indices = []
        rec_ref_indices.extend(keep_frag.ref_indices)
        rec_ref_indices.extend(move_frag.ref_indices)

        rec_cell_indices = []
        rec_cell_indices.extend(keep_frag.cell_indices)
        rec_cell_indices.extend(move_frag.cell_indices)

        recfracs = []
        recfracs.extend(keep_frag.frac_coord)
        recfracs.extend(move_frag.frac_coord)

        rec_ref_atom_site_labels = [
            refcell.atom_site_labels[idx] for idx in rec_ref_indices
        ]
        if final_merge:
            blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor)
            numspecs = len(blocklist)
            # numspecs = count_species(reclabels, reccoord, cov_factor=cov_factor)
        else:
            # numspecs = count_species(
            blocklist = split_species(
                reclabels,
                reccoord,
                atom_site_labels=rec_ref_atom_site_labels,
                geom_bond_cif=refcell.geom_bond_cif,
                cov_factor=cov_factor,
            )
            numspecs = len(blocklist)
        if numspecs != 1:
            continue

        if final_merge:
            blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor)
        else:
            if refcell.exist_cif_bond_moiety and refcell.geom_bond_cif is not None:
                blocklist = split_species(
                    reclabels,
                    reccoord,
                    atom_site_labels=rec_ref_atom_site_labels,
                    geom_bond_cif=refcell.geom_bond_cif,
                    cov_factor=cov_factor,
                )
            else:
                blocklist = split_species(reclabels, reccoord, cov_factor=cov_factor)

        if blocklist is None:
            continue
        else:
            if len(blocklist) != 1:
                continue
            if len(blocklist) == 1:
                newmolec = Molecule.from_positional(reclabels, reccoord, recfracs)
                newmolec.set_origin("cell.reconstruct")
                newmolec.add_parent(refcell, indices=rec_ref_indices)
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


def merge_fragments_iterative(
    fragments,
    target_ref,
    cell_vector,
    refcell,
    cov_factor: float = 1.0,
    metal_factor: float = 1.0,
    full: bool = False,
    final_merge: bool = False,
):
    """
    Iteratively merges fragments until no further merges are possible.
    Args:
        fragments (list): List of molecular fragments to merge.
        target_ref (list): List of target reference atom indices.
        cell_vector (np.ndarray): Cell vectors for the new structure.
        refcell (object): Reference cell object.
        cov_factor (float): Covalent factor for adjacency determination.
        metal_factor (float): Metal factor for adjacency determination.
        full (bool): If True, generates all translation vectors for merging.
        final_merge (bool): If True, performs a final merge without adjacency checks.
    Returns:
        list: List of merged molecular fragments.
    """
    while True:
        # Create an index list for the current state of fragments
        idx_list = list(range(len(fragments)))
        # Keep track of whether any items were merged during this pass
        merged = False

        # Generate all possible combinations of indices for current fragments
        for comb, idx_comb in zip(
            combinations(fragments, 2), combinations(idx_list, 2)
        ):
            if comb[0].formula == "H" and comb[1].formula == "H":
                pass
            elif comb[0].natoms + comb[1].natoms > len(target_ref):
                pass
            elif (
                comb[0].subtype == "Rec. Molecule" or comb[1].subtype == "Rec. Molecule"
            ):
                pass
            elif set(comb[0].ref_indices).intersection(set(comb[1].ref_indices)):
                # print(comb[0].ref_indices)
                # print(comb[1].ref_indices)
                pass
            else:
                logger.debug(
                    "Fragments TO BE MERGED %s %s %s",
                    [k.formula for k in comb],
                    [k.subtype for k in comb],
                    idx_comb,
                )

                newmolec = merge_fragment_pair(
                    comb,
                    cell_vector,
                    refcell,
                    cov_factor=cov_factor,
                    metal_factor=metal_factor,
                    full=full,
                    final_merge=final_merge,
                )

                if newmolec is None:
                    logger.debug("NOT MERGED %s", [k.formula for k in comb])
                else:
                    small_set = set(newmolec.ref_indices)

                    if small_set.issubset(target_ref):
                        if sorted(small_set) == target_ref:
                            newmolec.set_subtype("Rec. Molecule")
                            logger.debug("Molecule found %s", newmolec.formula)
                        else:
                            newmolec.set_subtype("Rec. Fragment")
                            logger.debug("Bigger fragment found %s", newmolec.formula)
                    fragments[idx_comb[0]] = newmolec
                    fragments.pop(idx_comb[1])
                    merged = True
                    break

        if not merged:
            break

    return fragments


######################################################
def reconstruct_fragments(
    subset_remaining_fragments,
    target_ref,
    cell_vector,
    refcell,
    cov_factor: float = 1.0,
    metal_factor: float = 1.0,
    full: bool = False,
    final_merge: bool = False,
):
    list_of_found_molecules = []
    list_of_bigger_fragments = []
    remaining_frag = subset_remaining_fragments.copy()

    fragments = merge_fragments_iterative(
        remaining_frag,
        target_ref,
        cell_vector,
        refcell,
        cov_factor=cov_factor,
        metal_factor=metal_factor,
        full=full,
        final_merge=final_merge,
    )

    for newmolec in fragments:
        small_set = set(newmolec.ref_indices)
        if small_set.issubset(target_ref):
            if sorted(small_set) == target_ref:
                list_of_found_molecules.append(newmolec)
            elif newmolec.natoms == 1 and (
                newmolec.set_element_count()[4] + newmolec.set_element_count()[3] == 1
            ):
                newmolec.set_subtype("fragment")
                logger.debug("Hydrogen found", newmolec.formula)
                list_of_bigger_fragments.append(newmolec)
            else:
                list_of_bigger_fragments.append(newmolec)

    return list_of_found_molecules, list_of_bigger_fragments


def get_updated_indices(sp_idx, ref_labels, new, cell_labels, cell_pos, cell_fracs):
    """
    Get the updated indices of atoms in the new structure that match those in the unit cell.
    Args:
        sp_idx : index of the symmetry operation
        ref_labels : list of chemical symbols of the atoms in the reference structure
        new : ase atoms object by applying symmetry operations to the reference structure
        cell_labels : list of chemical symbols of the atoms in the unit cell
        cell_pos : list of cartesian coordinates of the atoms in the unit cell
        cell_fracs : list of fractional coordinates of the atoms in the unit cell
    Returns:
        indices_lists : list of tuples, where each tuple contains the index of the atom
                        in the new structure and the index of the atom in the unit cell
    """
    indices_lists = []
    new_pos = new.get_positions()
    new_fracs = new.get_scaled_positions()

    for jdx, (n_l, n_p, n_f) in enumerate(zip(ref_labels, new_pos, new_fracs)):
        for kdx, (label, p, f) in enumerate(zip(cell_labels, cell_pos, cell_fracs)):
            if n_l == label and np.allclose(n_p, p, atol=1e-4, rtol=1e-2):
                # jdx is the index of the atom in the new structure
                # kdx is the index of the atom in the unit cell
                indices_lists.append((jdx, kdx))
    return indices_lists


def reconstruct(refcell, newcell, sym_ops):
    """
    Reconstructs a unit cell based on a reference cell and symmetry operations.
    Args:
        refcell (object): The reference cell object containing reference molecules.
        newcell (object): The new cell object to be reconstructed.
        sym_ops (tuple): A tuple containing two lists: a list of rotation
                            matrices and a list of translation vectors.
    Returns:
        molecules (list): List of found molecules.
        reconstructed_molecules (list): List of molecules reconstructed from fragments.
    """

    ref_labels = refcell.labels
    cov_factor = refcell.refmoleclist[0].cov_factor
    metal_factor = refcell.refmoleclist[0].metal_factor

    cell_labels = newcell.labels
    cell_pos = newcell.coord
    cell_fracs = newcell.frac_coord
    cell_vector = newcell.cell_vector

    if "D" in ref_labels:
        logger.debug("Deuterium is in the reference")
    if "D" in cell_labels:
        logger.debug("Deuterium is in the cell")

    new_structures = apply_symmetry_operations(refcell, cell_vector, sym_ops)
    logger.debug("Number of symmetry operations: %d", len(new_structures))

    all_found = []
    all_molecules = []
    reconstructed_molecules = []
    remaining_fragments = [[] for _ in range(len(newcell.refmoleclist))]

    for idx, new in enumerate(new_structures):
        logger.debug("Applying symmetry operations %d", idx)
        indices_lists = get_updated_indices(
            idx, ref_labels, new, cell_labels, cell_pos, cell_fracs
        )

        updated_lists = [i for i in indices_lists if i[1] not in all_found]
        updated_ref_indices = [i[0] for i in updated_lists]
        updated_cell = [i[1] for i in updated_lists]

        all_found.extend(updated_cell)

        logger.debug("Number of atoms found so far: %d", len(all_found))
        logger.debug(
            "All atoms found matches cell atoms: %s", len(all_found) == len(cell_pos)
        )

        if len(updated_cell) > 0:
            # make blocks and get fragments
            if refcell.exist_cif_bond_moiety:
                initial_fragments = get_fragments_from_moiety(
                    newcell,
                    updated_cell,
                    updated_ref_indices,
                    refcell,
                    cov_factor=cov_factor,
                    metal_factor=metal_factor,
                )
            else:
                initial_fragments = get_fragments(
                    newcell,
                    updated_cell,
                    updated_ref_indices,
                    refcell,
                    cov_factor=cov_factor,
                    metal_factor=metal_factor,
                )

            molecules, fragments = classify_fragments(
                initial_fragments, newcell.refmoleclist
            )
            all_molecules.extend(molecules)

            # Grouping fragments for each reference molecule
            fragments_of_new = [[] for _ in range(len(newcell.refmoleclist))]

            for i, ref in enumerate(newcell.refmoleclist):
                target_set = set(ref.get_parent_indices("reference"))
                for j, frag in enumerate(fragments):
                    small_set = set(frag.ref_indices)
                    fragments_of_new[i].append(frag)
                    if small_set.issubset(target_set):
                        logger.debug(
                            "%d %s is a subset of target_set %d %s",
                            j,
                            frag.formula,
                            i,
                            ref.formula,
                        )

            for i, (ref, frag_list) in enumerate(
                zip(newcell.refmoleclist, fragments_of_new)
            ):
                logger.debug("Reference %d: %s", i, ref.formula)
                logger.debug(
                    "Fragments formula : %s",
                    [frag.formula for frag in frag_list],
                )

            # Reconstructing fragments within one new structure
            for i, frag_list in enumerate(fragments_of_new):
                if len(frag_list) > 1:
                    logger.debug("Multiple fragments found for reference %d", i)
                    logger.debug("target_ref: %s", newcell.refmoleclist[i].formula)
                    logger.debug("fragments: %s", [frag.formula for frag in frag_list])
                    target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
                    list_of_found_molecules, remaining_frag = reconstruct_fragments(
                        frag_list,
                        target_ref,
                        cell_vector,
                        refcell,
                        cov_factor=cov_factor,
                        metal_factor=metal_factor,
                    )

                    if len(list_of_found_molecules) > 0:
                        reconstructed_molecules.extend(list_of_found_molecules)
                    if len(remaining_frag) > 0:
                        remaining_fragments[i].extend(remaining_frag)

                elif len(frag_list) == 1:
                    logger.debug("only one fragment %s is found", frag_list[0].formula)
                    remaining_fragments[i].extend(frag_list)

    logger.debug(
        "%d Complete molecules: molecules=%s",
        len(all_molecules),
        [mol.formula for mol in all_molecules],
    )

    logger.debug(
        "%d Reconstructed molecules: molecules=%s",
        len(reconstructed_molecules),
        [mol.formula for mol in reconstructed_molecules],
    )

    if len(all_found) == len(cell_pos):
        newcell.error_get_fragments = False
        num_rem_frags = 0
        for rem_frag_list in remaining_fragments:
            num_rem_frags += len(rem_frag_list)

        if num_rem_frags == 0:
            logger.info("All fragments are reconstructed successfully.")
            newcell.error_reconstruction = False
        else:
            logger.info(f"There are {num_rem_frags} remaining fragments.")
            final_remaining_fragments, reconstructed_molecules = final_reconstruct(
                remaining_fragments,
                newcell,
                cell_vector,
                reconstructed_molecules,
                refcell,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
            )
            if len(final_remaining_fragments) == 0:
                logger.info("All fragments are reconstructed successfully.")
                newcell.error_reconstruction = False
            else:
                logger.error("Error in reconstruction!!")
                newcell.error_reconstruction = True
                logger.info(
                    "final remaining fragments",
                    len(final_remaining_fragments),
                    [mol.formula for mol in final_remaining_fragments],
                )
    else:
        logger.error("Error in getting fragments and reconstruction!!")
        newcell.error_get_fragments = True
        newcell.error_reconstruction = True
        for i, pos in enumerate(cell_pos):
            if i not in all_found:
                logger.error(
                    f"Cannot find the {i}th atom of the unit cell based on cartesian coordinates from the new structure."
                )

    return all_molecules, reconstructed_molecules


######################################################
def final_reconstruct(
    remaining_fragments,
    newcell,
    cell_vector,
    reconstructed_molecules,
    refcell,
    cov_factor: float = 1.0,
    metal_factor: float = 1.0,
):
    # Reconstructing remaining fragments within the whole new cell
    final_remaining_fragments = []
    for i, rem_frag_list in enumerate(remaining_fragments):
        if len(rem_frag_list) > 1:
            logger.debug("target_ref: %s", newcell.refmoleclist[i].formula)
            logger.debug(
                "Fragments formula %d: %s", i, [rem.formula for rem in rem_frag_list]
            )
            target_ref = newcell.refmoleclist[i].get_parent_indices("reference")
            list_of_found_molecules, final_remaining = reconstruct_fragments(
                rem_frag_list,
                target_ref,
                cell_vector,
                refcell,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
                full=True,
            )
            logger.debug(
                "list_of_found_molecules: %s",
                [mol.formula for mol in list_of_found_molecules],
            )
            logger.debug(
                "final_remaining: %s", [frag.formula for frag in final_remaining]
            )
            if len(list_of_found_molecules) > 0:
                reconstructed_molecules.extend(list_of_found_molecules)
            if len(final_remaining) > 0:
                final_remaining_fragments.extend(final_remaining)
        elif len(rem_frag_list) == 1:
            final_remaining_fragments.extend(rem_frag_list)

    return final_remaining_fragments, reconstructed_molecules


######################################################
def get_moleclist(newcell, refcell, all_molecules):
    """
    Build the molecular list of a reconstructed unit cell
    """
    cov_factor = refcell.refmoleclist[0].cov_factor
    metal_factor = refcell.refmoleclist[0].metal_factor

    # Get moleclist for the unit cell
    newcell.moleclist = []

    for mol in all_molecules:
        newmolec = Molecule.from_positional(mol.labels, mol.coord, mol.frac_coord)
        mol_atom_site_labels = [
            refcell.atom_site_labels[idx] for idx in mol.ref_indices
        ]

        newmolec.set_origin("cell.reconstruct")
        newmolec.add_parent(newcell, mol.cell_indices)
        newmolec.add_parent(refcell, mol.ref_indices)
        newmolec.set_adjacency_parameters(cov_factor, metal_factor)
        newmolec.set_atoms(
            create_adjacencies=True,
            atom_site_labels=mol_atom_site_labels,
            geom_bond_cif=refcell.geom_bond_cif,
        )
        for atom, idx in zip(newmolec.atoms, mol.cell_indices):
            atom.add_parent(newcell, index=idx)
        for atom, idx in zip(newmolec.atoms, mol.ref_indices):
            atom.add_parent(refcell, index=idx)
        if newmolec.iscomplex:
            newmolec.split_complex()
        elif newmolec.has_IA_IIA:
            newmolec.split_IA_IIA()
        elif newmolec.has_post_transition_metal:
            newmolec.split_post_transition_metal()
        else:
            newmolec.add_parent(newmolec, indices=[*range(0, newmolec.natoms, 1)])
        newcell.moleclist.append(newmolec)

    for mol in newcell.moleclist:
        if mol.iscomplex:
            logger.debug("Working with transition metals: %s", mol.formula)
            mol.get_hapticity()
            if len(mol.ligands) == 0:
                logger.debug("%s is a metal cluster", mol.formula)
            else:
                for lig in mol.ligands:
                    lig.get_denticity()
            for met in mol.metals:
                met.get_connected_metals()
                met.get_coordination_geometry()
                met.get_coord_sphere_formula()
        elif mol.has_IA_IIA:
            logger.debug("Working with alkali or alkali earth metals: %s", mol.formula)
            if len(mol.ligands) == 0:
                pass
            else:
                for lig in mol.ligands:
                    lig.get_denticity()
            for met in mol.metals:
                met.get_connected_metals()
                met.get_coordination_geometry()
                met.get_coord_sphere_formula()
        elif mol.has_post_transition_metal:
            logger.debug("Working with post transition metals: %s", mol.formula)
            if len(mol.ligands) == 0:
                pass
            else:
                for lig in mol.ligands:
                    lig.get_denticity()
            for met in mol.metals:
                met.get_connected_metals()
                met.get_coordination_geometry()
                met.get_coord_sphere_formula()

    return newcell


def get_unique_indices(newcell, reference_species_list):
    """
    Match reconstructed species to reference species and assign unique indices
    from the reference species list.
    Args:
        newcell : The reconstructed unit cell containing the molecular list.
        reference_species_list : List of reference species to match against.
    Returns:
        newcell : Updated unit cell with unique indices and species list.
    """
    newcell.unique_indices = []
    newcell.species_list = []
    for mol in newcell.moleclist:
        if (
            not mol.iscomplex
            and not mol.has_IA_IIA
            and not mol.has_post_transition_metal
        ):
            for ref in reference_species_list:
                if (
                    (ref.subtype == "molecule")
                    and not ref.iscomplex
                    and not ref.has_IA_IIA
                    and not ref.has_post_transition_metal
                ):
                    issame = compare_reference_indices(ref, mol)
                    if issame:
                        mol.unique_index = ref.unique_index
                        newcell.unique_indices.append(mol.unique_index)
                        newcell.species_list.append(mol)
        else:
            for ref in reference_species_list:
                if ref.subtype == "ligand":
                    for lig in mol.ligands:
                        issame = compare_reference_indices(ref, lig)
                        if issame:
                            lig.unique_index = ref.unique_index
                            newcell.unique_indices.append(lig.unique_index)
                            newcell.species_list.append(lig)
                if ref.subtype == "metal":
                    ref_parent_index = ref.get_parent_index("reference")
                    for met in mol.metals:
                        if ref_parent_index == met.get_parent_index("reference"):
                            met.unique_index = ref.unique_index
                            newcell.unique_indices.append(met.unique_index)
                            newcell.species_list.append(met)

    return newcell
