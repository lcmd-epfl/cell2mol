import numpy as np
import logging
from typing import Any, cast
from ase import Atoms
from cell2mol.classes import Molecule
from cell2mol.compare import compare_reference_indices
from cell2mol.connectivity import split_species
from cell2mol.element_utils import get_radii
from cell2mol.operations import translate, extract_from_list
from itertools import combinations
from cell2mol.elementdata import ElementData
from cell2mol.utils import config

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

# Atoms touch within the sum of their radii plus this. Only shortlists shifts.
_CONTACT_MARGIN = 1.0

# A symmetry-generated atom this close to a unit-cell atom is that atom.
_SITE_MATCH_TOLERANCE = 0.05

# Closer than this is one site written twice, usually CIF disorder.
_MIN_SITE_SEPARATION = 0.3


def construct_unitcell(refcell, unitcell, sym_ops):
    """
    Construct a unit cell based on a reference cell and symmetry operations.

    Args:
        refcell (object):
            Reference cell containing reference molecules.
        unitcell (object):
            Unit cell to be constructed.
        sym_ops (tuple):
            Symmetry operations (rotations, translations).
    """
    logger.info("=" * 40)
    logger.info(" Constructing unit cell from reference ")
    logger.info("=" * 40)

    refmoleclist = refcell.refmoleclist
    sym_atoms_list = apply_symmetry_operations(refcell, sym_ops)

    # Atom indices already matched to symmetry structures
    found_atom_indices = []
    # Molecules identified directly (no reconstruction needed)
    direct_molecules = []
    # Molecules reconstructed from fragments
    reconstructed_molecules = []
    # Fragments not yet reconstructed, grouped by reference molecule
    remaining_fragments_by_reference = [[] for _ in range(len(refmoleclist))]

    for idx, sym_atoms in enumerate(sym_atoms_list):
        initial_fragments = generate_initial_fragments(
            idx, refcell, unitcell, sym_atoms, found_atom_indices
        )
        molecules, incomplete_fragments = classify_fragments_and_molecules(
            initial_fragments, refmoleclist
        )
        direct_molecules.extend(molecules)

        reconstruct_fragments_by_reference(
            incomplete_fragments,
            refcell,
            reconstructed_molecules,
            remaining_fragments_by_reference,
        )

        # Every unit-cell atom is matched -- remaining symmetry operations can
        # only re-match already-found atoms (the matcher skips those), so they
        # add nothing. Stop early. This also skips ASE's duplicate P-1 ops.
        if len(found_atom_indices) == len(unitcell.coord):
            logger.debug(
                "All %d atoms matched after %d symmetry operation(s); "
                "skipping the remaining %d.",
                len(unitcell.coord),
                idx + 1,
                len(sym_atoms_list) - (idx + 1),
            )
            break

    all_molecules = direct_molecules + reconstructed_molecules
    # --- early validation ---
    if is_reconstruction_complete(
        unitcell,
        found_atom_indices,
        remaining_fragments_by_reference,
    ):
        generate_unitcell_molecules(unitcell, refcell, all_molecules)
        assign_unitcell_species(unitcell, refcell.species_list)
        return unitcell

    # --- final reconstruction ---
    final_remaining_fragments = final_reconstruct(
        refcell,
        reconstructed_molecules,
        remaining_fragments_by_reference,
    )

    all_molecules = direct_molecules + reconstructed_molecules
    # --- final validation ---
    if is_reconstruction_complete(
        unitcell,
        found_atom_indices,
        final_remaining_fragments,
    ):
        generate_unitcell_molecules(unitcell, refcell, all_molecules)
        assign_unitcell_species(unitcell, refcell.species_list)
        return unitcell

    # Reconstruction failed, unitcell already has error flags. Say why when the
    # cell itself, rather than the merging, is the reason.
    diagnose_failed_reconstruction(unitcell, refcell, sym_atoms_list)
    return unitcell


def apply_symmetry_operations(refcell, sym_ops, normalize: bool = True):
    """Applies symmetry operations to a reference cell.

    This function generates a set of new atomic structures by applying a list of
    symmetry operations (rotations and translations) to the fractional coordinates
    of a reference structure.

    Args:
        refcell (object):
            Reference cell containing reference molecules.
        sym_ops (tuple):
            Symmetry operations (rotations, translations).
        normalize (bool, optional):
            If True, the transformed fractional coordinates are wrapped back
            into the unit cell (0 to 1). Defaults to True.

    Returns:
        list: A list of `ase.Atoms` objects, where each object represents the
              reference structure after one symmetry operation has been applied.
    """
    sym_atoms_list = []

    ref_labels = refcell.labels
    fractional_coords = np.array(refcell.frac_coord)
    cell_vector = refcell.cell_vector

    numbers: list[int] = []
    if "D" in ref_labels:
        logger.debug("Deuterium is in the reference")
        # Atoms object cannot handle Deuterium in the symbols
        numbers = [elemdatabase.elementnr[elem] for elem in ref_labels]

    for rot, trans in zip(sym_ops[0], sym_ops[1]):
        transformed_positions = np.dot(fractional_coords, rot.T)
        transformed_positions += np.array(trans)
        if normalize:
            transformed_positions = np.remainder(transformed_positions, 1)
        if "D" in ref_labels:
            sym_atoms = Atoms(
                scaled_positions=transformed_positions,
                numbers=numbers,
                cell=cell_vector,
                pbc=True,
            )
        else:
            sym_atoms = Atoms(
                symbols=ref_labels,
                scaled_positions=transformed_positions,
                cell=cell_vector,
                pbc=True,
            )
        sym_atoms_list.append(sym_atoms)
    logger.info("Number of symmetry operations: %d", len(sym_atoms_list))
    return sym_atoms_list


def generate_initial_fragments(
    symop_idx, refcell, unitcell, sym_atoms, matched_atom_indices
):
    """
    Generate initial molecular fragments from a symmetry-transformed structure.

    For a given symmetry operation, this function matches atoms from the
    symmetry-transformed structure to the unit cell, updates the list of
    already matched atom indices, and generates initial molecular fragments
    from newly matched atoms.

    Args:
        symop_idx (int):
            Index of the symmetry operation.
        refcell (object):
            Reference cell object containing the reference structure.
        unitcell (object):
            Cell object containing the unit cell information.
        sym_atoms (ase.Atoms):
            Symmetry-transformed atomic structure.
        matched_atom_indices (list):
            List of unit-cell atom indices that have already been matched.
            This list is updated in place.

    Returns:
        list:
            List of initial molecular fragments generated from newly matched
            atoms. Returns an empty list if no new atoms are found.
    """
    indices_lists = _get_updated_indices(symop_idx, refcell, unitcell, sym_atoms)

    updated_lists = [i for i in indices_lists if i[1] not in matched_atom_indices]
    updated_ref_indices = [i[0] for i in updated_lists]
    updated_cell_indices = [i[1] for i in updated_lists]
    matched_atom_indices.extend(updated_cell_indices)

    logger.debug("Number of matched atoms so far: %d", len(matched_atom_indices))
    if len(matched_atom_indices) == len(unitcell.coord):
        logger.debug("All atoms have been matched in the cell.")

    if not updated_cell_indices:
        return []

    blocklist = _generate_blocklist_in_updated(
        refcell=refcell,
        unitcell=unitcell,
        indices_in_ref=updated_ref_indices,
        indices_in_updated=updated_cell_indices,
    )
    initial_fragments = _build_fragments_from_blocklist(
        blocklist=blocklist,
        refcell=refcell,
        unitcell=unitcell,
        indices_in_ref=updated_ref_indices,
        indices_in_updated=updated_cell_indices,
    )

    return initial_fragments


def classify_fragments_and_molecules(initial_fragments, refmoleclist):
    """
    Classify initial fragments into complete molecules and incomplete fragments
    based on reference molecules.

    Each fragment is compared against the reference molecule list. A fragment is
    classified as a complete molecule if:
      - the number of atoms matches,
      - the chemical formula matches,
      - the reference parent indices exactly match the fragment reference indices.

    Fragments that do not satisfy these conditions are classified as incomplete
    fragments. Single-atom hydrogen or deuterium fragments are treated as a
    special case and appended at the end of the incomplete fragment list.

    Args:
        initial_fragments (list):
            List of initial molecular fragments to classify.
        refmoleclist (list):
            List of reference molecule objects used for classification.

    Returns:
        molecules (list):
            List of fragments classified as complete molecules.
        incomplete_fragments (list):
            List of incomplete fragments, sorted by descending fragment size,
            with single-atom hydrogen/deuterium fragments appended at the end.
    """
    molecules = []
    incomplete_fragments = []
    hydrogens = []

    for frag in initial_fragments:
        matched = False

        for idx, ref in enumerate(refmoleclist):
            if ref.natoms == frag.natoms and ref.formula == frag.formula:
                ref_parent_indices = ref.get_parent_indices("reference")
                if sorted(ref_parent_indices) == sorted(frag.ref_indices):
                    logger.debug(
                        "Fragment %s matched to reference molecule %d %s",
                        frag.formula,
                        idx,
                        ref.formula,
                    )
                    frag.set_subtype("molecule")
                    frag.set_origin("classify_fragments_and_molecules")
                    molecules.append(frag)
                    matched = True
                    break

        if not matched:
            frag.set_subtype("fragment")
            frag.set_origin("classify_fragments_and_molecules")

            # Single-atom hydrogen or deuterium fragments
            if frag.natoms == 1 and frag.labels[0] in ["H", "D"]:
                hydrogens.append(frag)
            else:
                incomplete_fragments.append(frag)

    # Sort incomplete fragments by size (descending)
    if incomplete_fragments:
        frag_sizes = np.array([frag.natoms for frag in incomplete_fragments])
        order = np.argsort(frag_sizes)[::-1]
        incomplete_fragments = [incomplete_fragments[i] for i in order]

    for frag in incomplete_fragments + hydrogens:
        frag.get_centroid()

    logger.debug(
        "Incomplete fragments: %s",
        [frag.formula for frag in incomplete_fragments],
    )
    logger.debug(
        "Hydrogen fragments: %s",
        [frag.formula for frag in hydrogens],
    )

    incomplete_fragments.extend(hydrogens)

    return molecules, incomplete_fragments


def reconstruct_fragments_by_reference(
    incomplete_fragments,
    refcell,
    reconstructed_molecules,
    remaining_fragments_by_reference,
):
    """
    Reconstruct molecules from fragments grouped by reference molecule.

    For each reference molecule, this function attempts to reconstruct complete
    molecules from the associated fragment list. Reconstruction is delegated to
    `reconstruct_fragments`, which returns successfully reconstructed molecules
    and fragments that remain incomplete.

    This function updates the provided containers in place.

    Args:
        incomplete_fragments (list):
            List of incomplete molecular fragments.
        refcell (object):
            Reference cell containing reference molecules.
        reconstructed_molecules (list):
            Container to collect successfully reconstructed molecules.
            Updated in place.
        remaining_fragments_by_reference (list of list):
            Container for fragments that could not be reconstructed,
            grouped by reference molecule. Updated in place.

    Returns:
        None
    """
    refmoleclist = refcell.refmoleclist

    # Group fragments by reference molecule
    fragments_by_reference = _group_fragments_by_reference(
        incomplete_fragments, refmoleclist
    )

    for ref_idx, fragments_for_ref in enumerate(fragments_by_reference):
        if not fragments_for_ref:
            continue

        ref = refmoleclist[ref_idx]
        target_ref_indices = ref.get_parent_indices("reference")

        logger.debug(
            "Fragments for reference %d (%s): %s",
            ref_idx,
            ref.formula,
            [frag.formula for frag in fragments_for_ref],
        )

        reconstructed, remaining_fragments = reconstruct_fragments(
            fragments_for_ref,
            target_ref_indices,
            refcell,
        )

        if reconstructed:
            reconstructed_molecules.extend(reconstructed)
        if remaining_fragments:
            remaining_fragments_by_reference[ref_idx].extend(remaining_fragments)


def reconstruct_fragments(
    fragments,
    target_ref,
    refcell,
):
    reconstructed_molecules = []
    remaining_fragments = []

    merged_candidates = _merge_fragments_iterative(
        fragments.copy(),
        target_ref,
        refcell,
    )

    for merged in merged_candidates:
        small_set = set(merged.ref_indices)
        if small_set.issubset(target_ref):
            if sorted(small_set) == target_ref:
                merged.set_subtype("molecule")
                reconstructed_molecules.append(merged)
            elif merged.natoms == 1 and (merged.labels[0] in ["H", "D"]):
                merged.set_subtype("fragment")
                logger.debug("Hydrogen found: %s", merged.formula)
                remaining_fragments.append(merged)
            else:
                merged.set_subtype("fragment")
                remaining_fragments.append(merged)

    return reconstructed_molecules, remaining_fragments


def final_reconstruct(
    refcell,
    reconstructed_molecules,
    remaining_fragments_by_reference,
):
    """
    Perform a final reconstruction pass over remaining fragments.

    This function attempts a last reconstruction step for fragments that could
    not be merged during symmetry-based reconstruction. Fragments are pooled per
    reference molecule, so leftovers from different symmetry operations can meet
    here, unlike the per-operation pass that precedes it.

    Successfully reconstructed molecules are appended to
    `reconstructed_molecules`. Fragments that still cannot be reconstructed
    are collected and discarded at this stage.

    Args:
        refcell (object):
            Reference cell containing reference molecules.
        reconstructed_molecules (list):
            List of already reconstructed molecules. Updated in place.
        remaining_fragments_by_reference (list of list):
            Remaining fragments grouped by reference molecule.

    Returns:
        list:
            Updated list of reconstructed molecules.
    """
    refmoleclist = refcell.refmoleclist

    final_remaining_fragments = []

    for ref_idx, ref_fragments in enumerate(remaining_fragments_by_reference):
        if not ref_fragments:
            continue

        ref = refmoleclist[ref_idx]
        target_ref_indices = ref.get_parent_indices("reference")

        logger.debug("Final reconstruction for reference %d (%s)", ref_idx, ref.formula)
        logger.debug(
            "Remaining fragments: %s",
            [frag.formula for frag in ref_fragments],
        )

        reconstructed, remaining = reconstruct_fragments(
            ref_fragments,
            target_ref_indices,
            refcell,
        )

        if reconstructed:
            reconstructed_molecules.extend(reconstructed)
        if remaining:
            final_remaining_fragments.extend(remaining)

    if final_remaining_fragments:
        logger.error(
            "Unreconstructed fragments after final pass: %s",
            [frag.formula for frag in final_remaining_fragments],
        )

    return final_remaining_fragments


def diagnose_failed_reconstruction(unitcell, refcell, sym_atoms_list):
    """Log why an input could never have been reconstructed.

    The cell must be a whole number of copies of each reference molecule.
    Duplicate sites and partial occupancy break that before any merging starts.
    Neither gets its own error code: the reconstruction error already reports the
    failure, this only says why, so it goes to the log alone.

    Copies are counted within a molecule only: two moieties of one structure may
    legitimately differ, an anion on a general position next to a complex on a
    two-fold axis being the ordinary case.
    """
    overlapping = _overlapping_sites(unitcell)
    if overlapping:
        first, second, distance = overlapping[0]
        logger.error(
            "%d atom pair(s) closer than %.2f A, the closest %s%d and %s%d at "
            "%.3f A. The cell lists the same site more than once.",
            len(overlapping),
            _MIN_SITE_SEPARATION,
            unitcell.labels[first],
            first,
            unitcell.labels[second],
            second,
            distance,
        )

    for ref_idx, counts in _uneven_copy_counts(unitcell, refcell, sym_atoms_list):
        logger.error(
            "Reference molecule %d (%s) is not a whole number of copies: its "
            "atoms appear %s times. Partial occupancy or a disordered site.",
            ref_idx,
            refcell.refmoleclist[ref_idx].formula,
            " and ".join(str(count) for count in sorted(counts)),
        )


def _overlapping_sites(unitcell):
    """Unit-cell atom pairs closer than ``_MIN_SITE_SEPARATION``, closest first."""
    cell_vector = np.asarray(unitcell.cell_vector)
    fracs = np.asarray(unitcell.frac_coord)

    overlapping = []
    for idx in range(len(fracs) - 1):
        separation = fracs[idx + 1 :] - fracs[idx]
        separation -= np.round(separation)
        distances = np.linalg.norm(separation @ cell_vector, axis=1)
        for offset in np.flatnonzero(distances < _MIN_SITE_SEPARATION):
            overlapping.append((idx, idx + 1 + int(offset), float(distances[offset])))

    return sorted(overlapping, key=lambda pair: pair[2])


def _uneven_copy_counts(unitcell, refcell, sym_atoms_list):
    """Reference molecules whose atoms are not all present the same number of times.

    Returns ``(reference index, {counts seen})`` for each offending molecule.
    """
    # Which reference atom each unit-cell atom is a copy of. The first symmetry
    # operation to claim an atom owns it, as in the reconstruction itself.
    owner: dict[int, int] = {}
    for symop_idx, sym_atoms in enumerate(sym_atoms_list):
        for ref_idx, cell_idx in _get_updated_indices(
            symop_idx, refcell, unitcell, sym_atoms
        ):
            owner.setdefault(cell_idx, ref_idx)

    copies_per_ref_atom: dict[int, int] = {}
    for ref_idx in owner.values():
        copies_per_ref_atom[ref_idx] = copies_per_ref_atom.get(ref_idx, 0) + 1

    uneven = []
    for idx, ref in enumerate(refcell.refmoleclist):
        counts = {
            copies_per_ref_atom.get(ref_atom, 0)
            for ref_atom in ref.get_parent_indices("reference")
        }
        if len(counts) > 1:
            uneven.append((idx, counts))
    return uneven


def is_reconstruction_complete(
    unitcell,
    found_atom_indices,
    remaining_fragments_by_reference,
):
    """
    Check whether unit cell reconstruction is complete.
    """
    if len(found_atom_indices) != len(unitcell.coord):
        unitcell.error_get_fragments = True
        unitcell.error_reconstruction = True
        return False

    unitcell.error_get_fragments = False

    num_remaining = 0
    for item in remaining_fragments_by_reference:
        if isinstance(item, list):
            # Normal case: it's a list of fragments
            num_remaining += len(item)
        elif item is not None:
            # Error case: it's a single Molecule object (causes the original crash)
            # We count this as 1 (or more) remaining item(s)
            num_remaining += 1

    if num_remaining == 0:
        logger.info("All fragments are reconstructed successfully.")
        unitcell.error_reconstruction = False
        return True

    unitcell.error_reconstruction = True
    return False


# ===============================================================
# If unit cell is reconstructed successfully, generate moleclist
# of the unit cell, assign unique indices and species list
# ===============================================================
def generate_unitcell_molecules(
    unitcell, refcell, all_molecules, use_bond_info: bool | None = None
):
    """
    Build the molecular list of a reconstructed unit cell
    """
    atom_site_labels = refcell.atom_site_labels
    cov_factor, metal_factor = refcell.refmoleclist[0].get_adjacency_parameters()
    cov_factor = config.COV_FACTOR if cov_factor is None else cov_factor
    metal_factor = config.METAL_FACTOR if metal_factor is None else metal_factor

    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO
    # Get moleclist for the unit cell
    unitcell.moleclist = []

    for mol in all_molecules:
        newmolec = Molecule.from_positional(mol.labels, mol.coord, mol.frac_coord)
        mol_atom_site_labels = [atom_site_labels[idx] for idx in mol.ref_indices]
        newmolec.set_origin("reconstruct_unitcell")
        newmolec.add_parent(unitcell, mol.cell_indices)
        newmolec.add_parent(refcell, mol.ref_indices)
        newmolec.set_adjacency_parameters(cov_factor, metal_factor)
        newmolec.set_atoms(
            create_adjacencies=True,
            atom_site_labels=mol_atom_site_labels,
            use_bond_info=use_bond_info,
        )
        for atom, idx in zip(newmolec.atoms or [], mol.cell_indices):
            atom.add_parent(unitcell, index=idx)
        for atom, idx in zip(newmolec.atoms or [], mol.ref_indices):
            atom.add_parent(refcell, index=idx)
        if newmolec.iscomplex or newmolec.has_ia_iia:
            logger.debug("Is complex: %s", newmolec.formula)
            logger.debug("Splitting complex: %s", newmolec.formula)
            newmolec.split_complex()
        elif newmolec.has_post_transition_metal:
            logger.debug("Has post-transition metal: %s", newmolec.formula)
            newmolec.split_complex(post_tms=True)
        else:
            newmolec.add_parent(newmolec, indices=[*range(0, newmolec.natoms, 1)])
        unitcell.moleclist.append(newmolec)

    for mol in unitcell.moleclist:
        mol.analyze_coordination()
        mol.detect_special_moieties()

    return unitcell


def assign_unitcell_species(unitcell, reference_species_list):
    """
    Update unitcell in place by assigning unique indices and species list.
    Args:
        unitcell:
            Reconstructed unit cell containing `moleclist`.
        reference_species_list:
            List of reference species with assigned unique indices.

    Returns:
        unitcell:
            Updated unit cell with `unique_indices` and `species_list`.
    """
    unitcell.unique_indices = []
    unitcell.species_list = []

    # Pre-group reference species by subtype
    ref_molecules = [
        ref
        for ref in reference_species_list
        if ref.subtype == "molecule" and ref.is_non_complex_molecule
    ]
    ref_ligands = [ref for ref in reference_species_list if ref.subtype == "ligand"]
    ref_metals = [ref for ref in reference_species_list if ref.subtype == "metal"]

    for mol in unitcell.moleclist:
        # Case 1: non-complex molecule
        if mol.is_non_complex_molecule:
            for ref in ref_molecules:
                if compare_reference_indices(ref, mol):
                    mol.unique_index = ref.unique_index
                    unitcell.unique_indices.append(mol.unique_index)
                    unitcell.species_list.append(mol)
                    break
            continue

        # Case 2: complex molecule with ligands and metals
        # --- ligands ---
        for lig in mol.ligands:
            for ref in ref_ligands:
                if compare_reference_indices(ref, lig):
                    lig.unique_index = ref.unique_index
                    unitcell.unique_indices.append(lig.unique_index)
                    unitcell.species_list.append(lig)
                    break

        # --- metals ---
        for met in mol.metals:
            met_parent_ref = met.get_parent_index("reference")
            for ref in ref_metals:
                if ref.get_parent_index("reference") == met_parent_ref:
                    met.unique_index = ref.unique_index
                    unitcell.unique_indices.append(met.unique_index)
                    unitcell.species_list.append(met)
                    break

    return unitcell


# ===============================================================
# Helper functions for fragment generation and reconstruction
# ===============================================================


def _get_updated_indices(symop_idx, refcell, unitcell, sym_atoms):
    """
    Get the updated indices of atoms in the new structure that match those in the unit cell.
    Args:
        symop_idx : index of the symmetry operation
        refcell : reference cell object containing the reference structure
        unitcell : cell object containing the unit cell information
        sym_atoms : ase atoms object by applying symmetry operations to the reference structure
    Returns:
        indices_lists : list of tuples, where each tuple contains the index of the atom
                        in the new structure and the index of the atom in the unit cell
    """

    logger.debug("Applying symmetry operations #%2d", symop_idx)
    indices_lists = []

    cell_vector = np.asarray(unitcell.cell_vector)
    cell_fracs = np.asarray(unitcell.frac_coord)
    new_fracs = np.asarray(sym_atoms.get_scaled_positions())

    # Candidates grouped by element, so each generated atom is only measured
    # against the atoms it could possibly be.
    cell_atoms_by_label: dict[str, list[int]] = {}
    for kdx, label in enumerate(unitcell.labels):
        cell_atoms_by_label.setdefault(label, []).append(kdx)

    for jdx, (n_l, n_f) in enumerate(zip(refcell.labels, new_fracs)):
        candidates = cell_atoms_by_label.get(n_l)
        if not candidates:
            continue

        # Minimum image: both sets are wrapped into the cell, so an atom sitting
        # on a face can come out at 0.0 on one side and 1.0 on the other.
        separation = cell_fracs[candidates] - n_f
        separation -= np.round(separation)
        distances = np.linalg.norm(separation @ cell_vector, axis=1)

        nearest = int(np.argmin(distances))
        if distances[nearest] <= _SITE_MATCH_TOLERANCE:
            # jdx is the index of the atom in the new structure
            # kdx is the index of the atom in the unit cell
            indices_lists.append((jdx, candidates[nearest]))
    return indices_lists


def _generate_blocklist_in_updated(
    refcell,
    unitcell,
    indices_in_ref,
    indices_in_updated,
    use_bond_info: bool | None = None,
):
    """
    Get fragments indices in the updated structure by mapping from reference moieties.
    If no moiety information is available, split by distance.
    """

    moiety_indices = getattr(refcell, "moiety_indices", None) if refcell else None
    atom_site_labels = getattr(refcell, "atom_site_labels", None) if refcell else None
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    # Map reference index -> updated cell index
    ref_to_updated = {
        ref_idx: indices_in_updated[idx] for idx, ref_idx in enumerate(indices_in_ref)
    }
    # If no moiety information is available, split by distance
    if moiety_indices is None:
        updated_labels = extract_from_list(
            indices_in_updated, unitcell.labels, dimension=1
        )
        updated_coord = extract_from_list(
            indices_in_updated, unitcell.coord, dimension=1
        )
        blocklist = split_species(updated_labels, np.asarray(updated_coord))
        return blocklist

    # Generate blocklist based on distance or CIF bond information
    # depending on use_bond_info (default: config.USE_BOND_INFO)
    tmp_blocklist = []
    for moiety in moiety_indices:
        cell_indices = []
        ref_indices = []
        for ref_idx in moiety:
            if ref_idx in ref_to_updated:
                cell_indices.append(ref_to_updated[ref_idx])
                ref_indices.append(ref_idx)
        if not cell_indices:
            continue

        moiety_labels = extract_from_list(cell_indices, unitcell.labels, dimension=1)
        moiety_coord = extract_from_list(cell_indices, unitcell.coord, dimension=1)
        if atom_site_labels is None:
            moiety_atom_site_labels = None
        else:
            moiety_atom_site_labels = [atom_site_labels[i] for i in ref_indices]

        blocks = split_species(
            labels=moiety_labels,
            positions=np.asarray(moiety_coord),
            indices=cell_indices,
            atom_site_labels=moiety_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
        )

        tmp_blocklist.extend(cast("list[Any]", blocks))

    value_to_index = {val: idx for idx, val in enumerate(indices_in_updated)}
    blocklist = [
        [value_to_index[val] for val in sublist if val in value_to_index]
        for sublist in tmp_blocklist
    ]
    return blocklist


def _build_fragments_from_blocklist(
    blocklist,
    refcell,
    unitcell,
    indices_in_ref,
    indices_in_updated,
    use_bond_info: bool | None = None,
):
    """
    Build Molecule fragment objects from a blocklist.

    Args:
        blocklist (list):
            List of blocks, where each block is a list of atom indices in the updated structure.
        refcell (object):
            Reference cell object containing reference molecules.
        unitcell (object):
            Unit cell object to be reconstructed.
        indices_in_ref (list):
            List of atom indices in the reference structure.
        indices_in_updated (list):
            List of atom indices in the updated structure.
        use_bond_info (bool, optional):
            Whether to use CIF bond information for adjacency determination.
            Defaults to None.
    Returns:
        list:
            List of Molecule fragment objects.
    """

    atom_site_labels = refcell.atom_site_labels
    cov_factor, metal_factor = refcell.refmoleclist[0].get_adjacency_parameters()
    cov_factor = config.COV_FACTOR if cov_factor is None else cov_factor
    metal_factor = config.METAL_FACTOR if metal_factor is None else metal_factor

    updated_labels = extract_from_list(indices_in_updated, unitcell.labels, dimension=1)
    updated_coord = extract_from_list(indices_in_updated, unitcell.coord, dimension=1)
    updated_fracs = extract_from_list(
        indices_in_updated, unitcell.frac_coord, dimension=1
    )

    fragments = []
    for block in blocklist:
        mol_labels = extract_from_list(block, updated_labels, dimension=1)
        mol_coord = extract_from_list(block, updated_coord, dimension=1)
        mol_frac_coord = extract_from_list(block, updated_fracs, dimension=1)

        cell_indices = extract_from_list(block, indices_in_updated, dimension=1)
        ref_indices = extract_from_list(block, indices_in_ref, dimension=1)
        mol_atom_site_labels = [atom_site_labels[idx] for idx in ref_indices]

        # Create Molecular fragment object
        frag = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
        frag.set_origin("_build_fragments_from_blocklist")

        # Add parents
        frag.add_parent(unitcell, indices=cell_indices)
        frag.add_parent(refcell, indices=ref_indices)

        # Geometry & bonding
        frag.set_fractional_coord(mol_frac_coord)
        frag.set_adjacency_parameters(cov_factor, metal_factor)
        frag.set_atoms(
            create_adjacencies=True,
            atom_site_labels=mol_atom_site_labels,
            use_bond_info=use_bond_info,
        )
        # Index bookkeeping
        frag.ref_indices = ref_indices
        frag.cell_indices = cell_indices

        fragments.append(frag)
    return fragments


def _candidate_translations(keep_frag, move_frag, cell_vector):
    """Pick the whole-cell shifts to try when joining ``move_frag`` to ``keep_frag``.

    A fragment cut off at the cell edge has to be moved by one whole cell, or
    two, or three, before it sits next to the rest of its molecule. This decides
    which of those shifts are worth testing.

    Take one atom from each fragment. Exactly one shift brings that pair as close
    together as the cell allows. If they would then be close enough to bond, that
    shift is a candidate. Repeat for all pairs; only a few shifts come out.

    Smallest shift first, because most fragments need no shift at all.
    """
    cell = np.asarray(cell_vector)
    inv_cell = np.linalg.inv(cell)
    # From the Cartesian coordinates, since frac_coord may still say where the
    # fragment was before an earlier merge moved it.
    keep_fracs = np.asarray(keep_frag.coord) @ inv_cell
    move_fracs = np.asarray(move_frag.coord) @ inv_cell

    reach = (
        max(get_radii(keep_frag.labels))
        + max(get_radii(move_frag.labels))
        + _CONTACT_MARGIN
    )

    separation = keep_fracs[:, None, :] - move_fracs[None, :, :]
    shift = np.round(separation)
    distance = np.linalg.norm((separation - shift) @ cell, axis=-1)
    candidates = set(map(tuple, shift[distance <= reach].astype(int).tolist()))

    return sorted(candidates, key=lambda t: (max(map(abs, t)), sum(map(abs, t))))


def _merge_fragments_iterative(
    fragments,
    target_ref,
    refcell,
):
    """
    Iteratively merge fragment pairs in-place until no further merges are possible.

    Notes
    -----
    This function modifies `fragments` in place.
    """
    while True:
        merged_any = False
        for (i, frag_i), (j, frag_j) in combinations(list(enumerate(fragments)), 2):
            # --- skip conditions ---
            if frag_i.formula == "H" and frag_j.formula == "H":
                continue
            if frag_i.natoms + frag_j.natoms > len(target_ref):
                continue
            if set(frag_i.ref_indices) & set(frag_j.ref_indices):
                continue

            logger.debug(
                "Fragments to be merged %s %s",
                [frag_i.formula, frag_j.formula],
                [frag_i.subtype, frag_j.subtype],
            )

            merged = _merge_fragment_pair(
                (frag_i, frag_j),
                refcell,
            )
            if merged is None:
                continue
            # Successful merge
            fragments[i] = merged
            fragments.pop(j)
            merged_any = True
            break

        if not merged_any:
            break

    return fragments


def _merge_fragment_pair(
    frag_pair: "tuple[Any, Any]",
    refcell: Any,
    use_bond_info: bool | None = None,
):
    """
    Attempt to merge two fragments by translating one fragment across the unit cell.

    Notes
    -----
    - This function does NOT modify the input fragments.
    - Returns a new Molecule if a valid merge is found, otherwise None.
    """

    cell_vector = refcell.cell_vector

    # Adjacency parameters
    cov_factor, metal_factor = refcell.refmoleclist[0].get_adjacency_parameters()
    cov_factor = config.COV_FACTOR if cov_factor is None else cov_factor
    metal_factor = config.METAL_FACTOR if metal_factor is None else metal_factor

    atom_site_labels = getattr(refcell, "atom_site_labels", None)
    bond_data = getattr(refcell, "geom_bond_cif", None)

    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    # Keep the larger fragment fixed, move the smaller one
    keep_frag, move_frag = sorted(frag_pair, key=lambda f: f.natoms, reverse=True)

    inv_cell = np.linalg.inv(np.asarray(cell_vector))
    translations = _candidate_translations(keep_frag, move_frag, cell_vector)
    if not translations:
        return None

    for t in translations:
        # --- merge coordinates and metadata (ALWAYS lists) ---
        merged_labels = [*keep_frag.labels, *move_frag.labels]

        if t == (0, 0, 0):
            moved_coord = move_frag.coord
        else:
            moved_coord = translate(t, move_frag.coord, cell_vector)

        merged_coord = [*keep_frag.coord, *moved_coord]
        # Fractions follow the merged Cartesian coordinates. Carrying over the
        # mover's own fractions would describe the fragment where it used to
        # sit, not where this merge put it, and the next merge reads them back.
        merged_fracs = (np.asarray(merged_coord) @ inv_cell).tolist()

        merged_ref_indices = [*keep_frag.ref_indices, *move_frag.ref_indices]
        merged_cell_indices = [*keep_frag.cell_indices, *move_frag.cell_indices]

        if atom_site_labels is None:
            merged_atom_site_labels = None
        else:
            merged_atom_site_labels = [atom_site_labels[i] for i in merged_ref_indices]

        # --- fast reject: must form exactly one species ---
        numspecs = split_species(
            labels=merged_labels,
            positions=np.asarray(merged_coord),
            atom_site_labels=merged_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            count_species_only=True,
        )
        if numspecs != 1:
            continue

        blocklist = cast(
            "list[Any]",
            split_species(
                labels=merged_labels,
                positions=np.asarray(merged_coord),
                atom_site_labels=merged_atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
            ),
        )

        if not blocklist or len(blocklist) != 1:
            continue

        # --- successful merge ---
        merged = Molecule.from_positional(merged_labels, merged_coord, merged_fracs)
        merged.set_subtype("fragment")
        merged.set_origin("_merge_fragment_pair")
        merged.add_parent(refcell, indices=merged_ref_indices)

        merged.ref_indices = merged_ref_indices
        merged.cell_indices = merged_cell_indices

        merged.set_adjacency_parameters(cov_factor, metal_factor)
        merged.set_element_count()
        merged.get_centroid()
        merged.set_atoms(
            create_adjacencies=True,
            atom_site_labels=merged_atom_site_labels,
            use_bond_info=use_bond_info,
        )
        return merged

    return None


def _group_fragments_by_reference(fragments, refmoleclist):
    """Group fragments by reference molecule"""
    fragments_by_ref = [[] for _ in range(len(refmoleclist))]

    for ref_idx, ref in enumerate(refmoleclist):
        target_indices = set(ref.get_parent_indices("reference"))

        for frag_idx, frag in enumerate(fragments):
            frag_indices = set(frag.ref_indices)

            if frag_indices.issubset(target_indices):
                fragments_by_ref[ref_idx].append(frag)
                logger.debug(
                    "Fragment %d (%s) is a subset of reference %d (%s)",
                    frag_idx,
                    frag.formula,
                    ref_idx,
                    ref.formula,
                )
    # Debug summary
    for ref_idx, (ref, frag_list) in enumerate(zip(refmoleclist, fragments_by_ref)):
        logger.debug("Reference %d: %s", ref_idx, ref.formula)
        logger.debug("Assigned fragments: %s", [frag.formula for frag in frag_list])

    return fragments_by_ref
