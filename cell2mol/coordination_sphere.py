#!/usr/bin/env python
########################################################
# Third-party data and code included in this file:
#
# 1) Reference data (MIT License)
#    Reference shape data adapted from work by
#    Pere Alemany, Efrem Bernuz, Abel Carreras, and Miquel Llunell
#    Copyright (c) 2021 Pere Alemany, Efrem Bernuz,
#    Abel Carreras and Miquel Llunell
#      https://github.com/GrupEstructuraElectronicaSimetria/cosymlib
#
# 2) Code (BSD 3-Clause License)
#    Portions of code to calculate CShM adapted from:
#      https://github.com/radi0sus/cshm-cc/blob/main/cshm-cc.py
#    Original author:
#      Sebastian Dechert
#    Copyright (c) 2025, Sebastian Dechert
#
# Licenses:
#   - MIT License
#   - BSD 3-Clause License
#
# Modifications:
#   - integrated into cell2mol
########################################################

import numpy as np
import os
from cell2mol import __file__
import yaml
from cell2mol.element_utils import labels2formula
from cell2mol.operations import compute_centroid
from cell2mol.connectivity import (
    add_atom,
    identify_haptic_mode,
)
from cell2mol.elementdata import ElementData
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm
from scipy.stats import special_ortho_group
from scipy.linalg import svd
from cell2mol.utils import config

import logging

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

# Load YAML filem
path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "ideal_structures_center.yaml"
)
with open(path, "r") as file:
    data = yaml.safe_load(file)
ideal_shapes_from_cosymlib = {key: np.array(value) for key, value in data.items()}


shape_structure_references_simplified = {
    "2 Vertices": [
        ["L-2", 1, "Dinfh", "Linear"],
        ["vT-2", 2, "C2v", "Bent (V-shape, 109.47°)"],
        ["vOC-2", 3, "C2v", "Bent (L-shape, 90°)"],
    ],
    "3 Vertices": [
        ["TP-3", 1, "D3h", "Trigonal planar"],
        ["fvOC-3", 3, "C3v", "fac-Trivacant octahedron"],
        ["mvOC-3", 4, "C2v", "T-shaped"],
    ],
    "4 Vertices": [
        ["T-4", 2, "Td", "Tetrahedral"],
        ["SP-4", 1, "D4h", "Square planar"],
        ["SS-4", 3, "C2v", "Seesaw"],
    ],
    "5 Vertices": [
        ["PP-5", 1, "D5h", "Pentagon"],
        ["TBPY-5", 3, "D3h", "Trigonal bipyramidal"],
        ["SPY-5", 4, "C4v", "Square pyramidal"],
    ],
    "6 Vertices": [
        ["HP-6", 1, "D6h", "Hexagon"],
        ["PPY-6", 2, "C5v", "Pentagonal pyramidal"],
        ["OC-6", 3, "Oh", "Octahedral"],
        ["TPR-6", 4, "D3h", "Trigonal prismatic"],
    ],
    "7 Vertices": [
        ["HP-7", 1, "D7h", "Heptagon"],
        ["HPY-7", 2, "C6v", "Hexagonal pyramidal"],
        ["PBPY-7", 3, "D5h", "Pentagonal bipyramidal"],
        ["CTPR-7", 5, "C2v", "Capped trigonal prismatic"],
    ],
    "8 Vertices": [
        ["OP-8", 1, "D8h", "Octagon"],
        ["HPY-8", 2, "C7v", "Heptagonal pyramidal"],
        ["HBPY-8", 3, "D6h", "Hexagonal bipyramidal"],
        ["CU-8", 4, "Oh", "Cube"],
        ["SAPR-8", 5, "D4d", "Square antiprismatic"],
        ["TDD-8", 6, "D2d", "Dodecahedral"],
    ],
    "9 Vertices": [
        ["EP-9", 1, "D9h", "Enneagon"],
        ["OPY-9", 2, "C8v", "Octagonal pyramid"],
        ["HBPY-9", 3, "D7h", "Heptagonal bipyramid"],
        ["JTC-9", 4, "C3v", "Johnson triangular cupola J3"],
        ["JCCU-9", 5, "C4v", "Capped cube J8"],
        ["CCU-9", 6, "C4v", "Spherical-relaxed capped cube"],
        ["JCSAPR-9", 7, "C4v", "Capped square antiprism J10"],
        ["CSAPR-9", 8, "C4v", "Spherical capped square antiprism"],
        ["JTCTPR-9", 9, "D3h", "Tricapped trigonal prism J51"],
        ["TCTPR-9", 10, "D3h", "Spherical tricapped trigonal prism"],
        ["JTDIC-9", 11, "C3v", "Tridiminished icosahedron J63"],
        ["HH-9", 12, "C2v", "Hula-hoop"],
        ["MFF-9", 13, "Cs", "Muffin"],
    ],
    "10 Vertices": [
        ["DP-10", 1, "D10h", "Decagon"],
        ["EPY-10", 2, "C9v", "Enneagonal pyramid"],
        ["OBPY-10", 3, "D8h", "Octagonal bipyramid"],
        ["PPR-10", 4, "D5h", "Pentagonal prism"],
        ["PAPR-10", 5, "D5d", "Pentagonal antiprism"],
        ["JBCCU-10", 6, "D4h", "Bicapped cube J15"],
        ["JBCSAPR-10", 7, "D4d", "Bicapped square antiprism J17"],
        ["JMBIC-10", 8, "C2v", "Metabidiminished icosahedron J62"],
        ["JATDI-10", 9, "C3v", "Augmented tridiminished icosahedron J64"],
        ["JSPC-10", 10, "C2v", "Sphenocorona J87"],
        ["SDD-10", 11, "D2", "Staggered Dodecahedron (2:6:2)"],
        ["TD-10", 12, "C2v", "Tetradecahedron (2:6:2)"],
        ["HD-10", 13, "D4h", "Hexadecahedron (2:6:2) or (1:4:4:1)"],
    ],
    "11 Vertices": [
        ["HP-11", 1, "D11h", "Hendecagon"],
        ["DPY-11", 2, "C10v", "Decagonal pyramid"],
        ["EBPY-11", 3, "D9h", "Enneagonal bipyramid"],
        ["JCPPR-11", 4, "C5v", "Capped pentagonal prism J9"],
        ["JCPAPR-11", 5, "C5v", "Capped pentagonal antiprism J11"],
        ["JAPPR-11", 6, "C2v", "Augmented pentagonal prism J52"],
        ["JASPC-11", 7, "Cs", "Augmented sphenocorona J87"],
    ],
    "12 Vertices": [
        ["DP-12", 1, "D12h", "Dodecagon"],
        ["HPY-12", 2, "C11v", "Hendecagonal pyramid"],
        ["DBPY-12", 3, "D10h", "Decagonal bipyramid"],
        ["HPR-12", 4, "D6h", "Hexagonal prism"],
        ["HAPR-12", 5, "D6d", "Hexagonal antiprism"],
        ["TT-12", 6, "Td", "Truncated tetrahedron"],
        ["COC-12", 7, "Oh", "Cuboctahedron"],
        ["ACOC-12", 8, "D3h", "Anticuboctahedron J27"],
        ["IC-12", 9, "Ih", "Icosahedron"],
        ["JSC-12", 10, "C4v", "Johnson square cupola J4"],
        ["JEPBPY-12", 11, "D6h", "Johnson elongated pentagonal bipyramid J16"],
        ["JBAPPR-12", 12, "C2v", "Biaugmented pentagonal prism J53"],
        ["JSPMC-12", 13, "Cs", "Sphenomegacorona J88"],
    ],
    "20 Vertices": [["DD-20", 1, "Ih", "Dodecahedron"]],
    "24 Vertices": [
        ["TCU-24", 1, "Oh", "Truncated cube"],
        ["TOC-24", 2, "Oh", "Truncated octahedron"],
    ],
    "48 Vertices": [["TCOC-48", 1, "Oh", "Truncated cuboctahedron"]],
    "60 Vertices": [["TRIC-60", 1, "Ih", "Truncated icosahedron (fullerene)"]],
}


def define_coordination_geometry(metal: object, coord_group: list) -> object:
    """Define the coordination geometry of a metal center based on its coordinating groups"""

    symbols = []
    positions = []
    coord_haptic_type = []
    symbols.append(metal.label)
    positions.append(metal.coord)

    logger.debug("coord_group formula %s", [group.formula for group in coord_group])
    logger.debug(
        "coord_group hapticity %s",
        [
            group.is_haptic if group.subtype != "metal" else False
            for group in coord_group
        ],
    )
    logger.debug(
        "coord_group atoms %s",
        [
            [a.label for a in group.atoms]
            if group.subtype != "metal"
            else [group.label]
            for group in coord_group
        ],
    )

    count = 0
    for group in coord_group:
        if group.subtype == "metal":
            symbols.append(group.label)
            positions.append(group.coord)
            count += 1
        elif not group.is_haptic:
            for atom in group.atoms:
                symbols.append(atom.label)
                positions.append(atom.coord)
                count += 1

        else:
            # haptic ligand
            haptic_center_coord = compute_centroid(
                np.array([atom.coord for atom in group.atoms])
            )
            symbols.append(str(group.haptic_type))
            positions.append(haptic_center_coord.tolist())
            count += 1
            coord_haptic_type.append(group.haptic_type)

    posgeom_dev = shape_measure(symbols, positions)
    coord_nr = count
    if len(posgeom_dev) > 0:
        coordination_geometry = min(posgeom_dev, key=posgeom_dev.get)
        geom_deviation = min(posgeom_dev.values())
    else:
        coordination_geometry = "Undefined"
        geom_deviation = "Undefined"

    # for haptic ligands, it's the mid point of haptic ligands
    logger.info("The number of coordinating points: %s", coord_nr)
    logger.info("Possible coordination geometries: %s", posgeom_dev)
    logger.info("The type of hapticity : %s", coord_haptic_type)
    logger.info(
        "The most likely geometry is '%s' with deviation value %s",
        coordination_geometry,
        geom_deviation,
    )

    # return coordination_geometry
    return coord_nr, coordination_geometry, geom_deviation


def shape_measure(symbols: list, positions: list) -> dict:
    """
    Calculate shape measures for a coordination environment.

    Args:
        symbols (list):
            Atomic symbols including the metal atom (used to determine CN).
        positions (list):
            Cartesian coordinates of atoms.

    Returns:
        dict:
            Mapping of ideal geometry name to continuous shape measure (CSM).
    """
    # Coordination number (excluding metal atom)
    cn = len(symbols) - 1

    if cn <= 0:
        return {}
    if cn == 1:
        return {"Linear": 0.0}
    try:
        ref_geom = np.array(
            shape_structure_references_simplified[f"{cn} Vertices"],
            dtype=object,
        )
    except KeyError:
        logger.warning(
            "%d-vertex reference geometry not found in shape_structure_references",
            cn,
        )
        return {}

    posgeom_dev = {}
    for rg, _, _, geom in ref_geom:
        ideal_shape = ideal_shapes_from_cosymlib.get(rg)
        if ideal_shape is None:
            continue

        cshm = calc_cshm_fast(positions, ideal_shape)
        posgeom_dev[geom] = round(float(cshm), 3)
    return posgeom_dev


# def normalize_structure(coordinates):
#     # center and normalize the structure for CShM calculations
#     centered_coords = coordinates - np.mean(coordinates, axis=0)
#     norm = np.sqrt(np.mean(np.sum(centered_coords**2, axis=1)))
#     return centered_coords / norm


def normalize_structure(coordinates):
    """
    Center and normalize the structure for CShM calculations
    """
    if len(coordinates) == 0:
        logger.warning("normalize_structure: empty coordinates")
        return coordinates  # return as is

    centered_coords = coordinates - np.mean(coordinates, axis=0)

    sq = np.sum(centered_coords**2, axis=1)
    if sq.size == 0:
        logger.warning("normalize_structure: no atoms after centering")
        return centered_coords

    norm = np.sqrt(np.mean(sq))

    if norm == 0:
        logger.warning("normalize_structure: zero norm (all atoms coincident)")
        return centered_coords

    return centered_coords / norm


def calc_cshm_fast(coordinates, ideal_shape, num_trials=100):
    # faster Hungarian algorithm optimization
    # check number of trials, if it is too low, it calculates the
    # local and not the global minimum
    input_structure = normalize_structure(coordinates)
    ideal_sq_norms = np.sum(ideal_shape**2)
    # try different rotations first, then optimize assignment
    min_cshm = float("inf")

    # generate some initial rotations to avoid local minima
    for trial in range(num_trials):
        if trial == 0:
            # first trial with identity rotation
            R_init = np.eye(3)
        else:
            # random rotation matrix for subsequent trials
            # generate a random rotation matrix

            R_init = special_ortho_group.rvs(3)

            # Ensure it's a proper rotation (det=1)
            if np.linalg.det(R_init) < 0:
                R_init[:, 0] *= -1

        # apply initial rotation to ideal shape
        rotated_ideal_init = np.dot(ideal_shape, R_init)

        # compute cost matrix based on squared Euclidean distances
        cost_matrix = np.linalg.norm(
            input_structure[:, None, :] - rotated_ideal_init[None, :, :], axis=2
        )

        # solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # rearrange ideal_shape based on optimal assignment
        permuted_ideal = ideal_shape[col_ind]

        # compute optimal rotation using SVD
        H = np.dot(input_structure.T, permuted_ideal)
        U, _, Vt = svd(H)
        R = np.dot(Vt.T, U.T)

        rotated_ideal = np.dot(permuted_ideal, R)
        scale = np.sum(input_structure * rotated_ideal) / ideal_sq_norms
        if input_structure.size == 0:
            logger.warning("calc_cshm_fast: empty input_structure")
            cshm = float("inf")
        else:
            cshm = np.mean(
                np.sum((input_structure - scale * rotated_ideal) ** 2, axis=1)
            )

        min_cshm = min(min_cshm, cshm)

    return min_cshm * 100


def handle_metal_coordination(metal: object) -> list:
    """
    Determines the coordination environment of a metal center.

    Process:
    1. Identify which non-metal atoms are coordinating to the metal.
    2. Map these atoms back to their parent ligands.
    3. Refine the coordination sphere (validate connectivity and resolve hapticity).
    4. Instantiate and return Group objects representing the coordination blocks.

    Returns:
        List[Group]: A list of validated coordination Group objects.
    """
    from cell2mol.classes.group import Group

    # Format metal info for consistent logging
    metal_info = (
        f"{metal.label}{f' ({metal.atom_site_label})' if metal.atom_site_label else ''}"
    )

    # Ensure the metal has its connected atoms list initialized
    if getattr(metal, "connected_nonmetal_atoms", None) is not None:
        metal.get_connected_nonmetal_atoms()

    if not metal.connected_nonmetal_atoms:
        logger.info(f"No coordinating non-metal atoms found for metal {metal_info}")
        return []

    mol = metal.get_parent("molecule")
    ligands = mol.ligands

    if not ligands:
        logger.info(f"No ligands found for molecule {mol.formula}")
        return []

    # Initialize a mapping of ligand indices to their coordinating atom indices
    conn_idx_by_ligands = {lig_idx: [] for lig_idx, _ in enumerate(ligands)}

    # Map the molecule-wide index of each atom in each ligand for fast lookup
    lig_mol_indices = {
        lig_idx: [atom.get_parent_index("molecule") for atom in lig.atoms]
        for lig_idx, lig in enumerate(ligands)
    }

    # Group the metal's connected atoms by the ligand they belong to
    for atom in metal.connected_nonmetal_atoms:
        atom_mol_idx = atom.get_parent_index("molecule")
        for lig_idx, indices in lig_mol_indices.items():
            if atom_mol_idx in indices:
                conn_idx_by_ligands[lig_idx].append(atom_mol_idx)
                # detailed logging
                atom_info = f"{atom.label}{f' ({atom.atom_site_label})' if atom.atom_site_label else ''}"
                logger.debug(
                    "Atom %s (mol_idx %d) is part of ligand %s (lig_idx %d) coordinating metal %s",
                    atom_info,
                    atom_mol_idx,
                    ligands[lig_idx].formula,
                    lig_idx,
                    metal_info,
                )
                break

    logger.debug(f"Initial indices mapped by ligands: {conn_idx_by_ligands}")

    # Refine the coordination sphere:
    # This validates geometry and re-evaluates hapticity if atoms are pruned.
    final_refined_data = correct_coordination_sphere(
        metal=metal,
        ligands=ligands,
        conn_idx_by_ligands=conn_idx_by_ligands,
        mol=mol,
    )

    coordination_groups = []

    # Iterate through refined results to create Group objects
    for lig_idx, groups_list in final_refined_data.items():
        current_lig_m_indices = lig_mol_indices[lig_idx]
        current_lig = ligands[lig_idx]
        if getattr(current_lig, "metals", None) is None:
            object.__setattr__(current_lig, "metals", [])
        current_lig.metals.append(metal)
        for group_info in groups_list:
            atoms = group_info["atoms"]

            # 1. Create the Group instance from atom list
            group_obj = Group.from_atom_list(atoms)
            # 2. Set origin for traceability
            group_obj.origin = "handle_metal_coordination"

            # 3. Analyze coordination mode
            group_obj.get_hapticity()
            group_obj.get_denticity()

            # 4. Map indices
            group_mol_indices = [a.get_parent_index("molecule") for a in atoms]
            group_ligand_indices = [
                current_lig_m_indices.index(m_idx) for m_idx in group_mol_indices
            ]
            # 5. Establish Parents
            group_obj.add_parent(mol, indices=group_mol_indices)
            group_obj.set_inherit_adjmatrix("molecule")
            group_obj.add_parent(ligands[lig_idx], indices=group_ligand_indices)

            # 6. Link Metal (ensure list exists)
            if getattr(group_obj, "metals", None) is None:
                object.__setattr__(group_obj, "metals", [])
            group_obj.metals.append(metal)

            coordination_groups.append(group_obj)

            # Detailed log of the final stabilized coordination mode
            logger.info(
                "Final Coordinated Group [%s]: Formula=%s, Haptic=%s, Type=%s",
                metal_info,
                group_info["formula"],
                group_info["is_haptic"],
                group_info["haptic_type"],
            )

    return coordination_groups


def correct_coordination_sphere(
    metal: object, ligands: list, conn_idx_by_ligands: dict, mol: object
) -> dict:
    """
    Refine the coordination sphere and return the final coordinated groups.
    Args:
        metal: Metal Atom object.
        ligands: List of ligand Molecule objects.
        conn_idx_by_ligands: Dictionary mapping each ligand index to the atom indices
            connected to the metal within that ligand.
        mol: TMC complex Molecule object.
    Returns:
        dict[int, list]: Dictionary mapping each ligand index to a list of validated
            haptic or non-haptic coordination groups.
    """
    final_coordination_results = {}

    for jdx, connected_idx in conn_idx_by_ligands.items():
        current_pool = connected_idx
        stable_groups = []
        removed_ligand_indices = []
        logger.debug(
            "Processing ligand %s (index %d) with initial connected indices: %s",
            ligands[jdx].formula,
            jdx,
            current_pool,
        )
        while True:
            results = partition_connected_indices(
                current_pool, mol, use_bond_info=config.USE_BOND_INFO
            )
            if not results:
                break

            any_change_in_iteration = False
            surviving_pool = []
            temp_stable_groups = []

            for group in results.values():
                if config.USE_BOND_INFO:
                    validated_gr_atoms = group["gr_atoms"]
                    was_changed = False
                else:
                    validated_gr_atoms, was_changed = validate_coordinated_atoms(
                        group,
                        metal,
                        ligands[jdx],
                        removed_ligand_indices=removed_ligand_indices,
                    )

                surviving_pool.extend(
                    [a.get_parent_index("molecule") for a in validated_gr_atoms]
                )

                if was_changed:
                    any_change_in_iteration = True

                # Store the data of the group as it stands in this iteration
                temp_stable_groups.append(
                    {
                        "formula": group["formula"],
                        "is_haptic": group["is_haptic"],
                        "haptic_type": group["haptic_type"],
                        "atoms": validated_gr_atoms,
                        "count": len(validated_gr_atoms),
                    }
                )

            if any_change_in_iteration:
                current_pool = surviving_pool
                continue
            else:
                # No changes means temp_stable_groups is now the final state
                stable_groups = temp_stable_groups
                break

        final_coordination_results[jdx] = stable_groups

    return final_coordination_results


def move_to_front(atoms, target_atom):
    """Move target_atom to the front while keeping the rest order unchanged."""
    return [target_atom] + [a for a in atoms if a is not target_atom]


def unique_atoms(atoms):
    seen = set()
    unique = []

    for atom in atoms:
        atom_id = atom.get_parent_index("molecule")

        if atom_id not in seen:
            unique.append(atom)
            seen.add(atom_id)

    return unique


def prioritize_coordinating_atoms_to_validate(
    gr_atoms_dict, sorted_gr_atoms, metal, ligand, is_haptic, haptic_type
):
    priority_atoms = []

    group_labels = gr_atoms_dict["labels"]
    group_atom_site_labels = gr_atoms_dict["atom_site_labels"]
    margins = gr_atoms_dict["margins"]
    group_formula = labels2formula(group_labels)
    lig_mol_indices = {
        atom.get_parent_index("molecule"): idx for idx, atom in enumerate(ligand.atoms)
    }

    def add_priority_atom(atom):
        if atom not in priority_atoms:
            priority_atoms.append(atom)

    # Si priority
    if "Si" in group_labels:
        for idx, atom in enumerate(sorted_gr_atoms):
            if atom.label != "Si":
                continue
            margin = float(np.round(margins[idx], 3))

            if margin > 0.25:
                # If Si is present and has a large margin, prioritize it for validation.
                add_priority_atom(atom)
            else:
                neighboring_atoms = []
                for adj in atom.adjacency:
                    if adj not in atom.metal_adjacency:
                        neighboring_atoms.append(atom.get_parent("molecule").atoms[adj])

                for neighbor in neighboring_atoms:
                    if (
                        neighbor.label == "H"
                        and neighbor.atom_site_label in group_atom_site_labels
                    ):
                        logger.debug(
                            "Neighboring H atom %s (%s) of Si atom %s are both in coordianting group %s. Prioritizing Si for validation.",
                            neighbor.label,
                            neighbor.atom_site_label,
                            atom.atom_site_label,
                            group_formula,
                        )
                        add_priority_atom(atom)
                        break

                is_added, _, _ = add_atom(
                    labels=ligand.labels,
                    coords=ligand.coord,
                    site=lig_mol_indices[atom.get_parent_index("molecule")],
                    ligand=ligand,
                    element="H",
                    metal=metal,
                )

                if not is_added:
                    logger.warning(
                        "Atom %s (%s) failed validation. Returning early to re-evaluate.",
                        atom.label,
                        atom.atom_site_label,
                    )
                    add_priority_atom(atom)

    # B priority
    if "B" not in group_labels:
        return priority_atoms

    if is_haptic:
        logger.debug(
            "Haptic coordination detected (%s). Stop validation for B atoms in group %s",
            haptic_type,
            group_formula,
        )
        return priority_atoms

    is_bh4_like = ligand.formula == "H4-B" or (
        group_labels.count("B") >= 1 and group_labels.count("H") >= 2
    )

    logger.debug(
        "Evaluating BH4-like priority: ligand formula %s, group formula %s, is_bh4_like=%s",
        ligand.formula,
        group_formula,
        is_bh4_like,
    )

    if is_bh4_like:
        for atom in sorted_gr_atoms:
            if atom.label == "B":
                add_priority_atom(atom)

    return priority_atoms


def validate_coordinated_atoms(group, metal, ligand, removed_ligand_indices):
    """
    Checks if atoms in gr_atoms are truly connected to the metal.
    Args:
        group: A dictionary containing the group information.
        metal: The metal Atom object.
        ligand: The ligand Molecule object to which these atoms belong.
        removed_ligand_indices: List of removed atom indices in ligands.
    Returns: (list of surviving atoms, boolean changed_flag)
    """
    gr_atoms = group["gr_atoms"]
    is_haptic = group["is_haptic"]
    haptic_type = group["haptic_type"]

    if not gr_atoms:
        return [], False

    if len(gr_atoms) == 1:
        atom = gr_atoms[0]
        logger.debug(
            "Only one coordinating atom found: %s (%s) (distance: %s). Skipping connectivity validation.",
            atom.label,
            atom.atom_site_label,
            np.round(np.linalg.norm(metal.coord - atom.coord), 3),
        )
        return gr_atoms, False

    if ligand.formula == "H2" and len(gr_atoms) == 2:
        return gr_atoms, False

    lig_mol_indices = {
        a.get_parent_index("molecule"): i for i, a in enumerate(ligand.atoms)
    }

    # 1. Sort atoms by margin: Farthest atoms first to handle
    # Primary: Hydrogen comes before Non-hydrogen atoms
    # Secondary: largest margin first
    sorted_gr_atoms = sorted(
        gr_atoms,
        key=lambda a: (
            a.label == "H",
            np.linalg.norm(metal.coord - a.coord) - metal.radii - a.radii,
        ),
        reverse=True,
    )

    gr_atoms_dict = {
        "labels": [],
        "atom_site_labels": [],
        "distances": [],
        "margins": [],
    }

    for atom in sorted_gr_atoms:
        distance = float(np.round(np.linalg.norm(metal.coord - atom.coord), 3))
        margin = float(np.round(distance - metal.radii - atom.radii, 3))

        gr_atoms_dict["labels"].append(atom.label)
        gr_atoms_dict["atom_site_labels"].append(atom.atom_site_label)
        gr_atoms_dict["distances"].append(distance)
        gr_atoms_dict["margins"].append(margin)

    if not sorted_gr_atoms or not sorted_gr_atoms[0].atom_site_label:
        gr_atoms_dict["atom_site_labels"] = None

    # 2. Detailed Debug Logging
    logger.debug(
        "Sorted coordinated atoms (farthest first): %s", gr_atoms_dict["labels"]
    )
    logger.debug(
        "Sorted coordinated atom site labels: %s", gr_atoms_dict["atom_site_labels"]
    )
    logger.debug("Sorted coordinated atom distances: %s", gr_atoms_dict["distances"])
    logger.debug("Sorted coordinated atom margins: %s", gr_atoms_dict["margins"])

    # if haptic:
    #     logger.debug(
    #         "Haptic coordination detected. Skipping validation. %s", group_formula
    #     )
    #     return gr_atoms, False

    # if "H" not in gr_atoms_labels:
    #     outlier_atoms = find_atom_outlier(sorted_gr_atoms, metal, haptic)
    #     atoms_to_validate = outlier_atoms
    # else:
    #     atoms_to_validate = sorted_gr_atoms
    #     logger.debug(
    #         "Hydrogen atoms detected in coordination sphere. Prioritizing their validation."
    #     )
    priority_atoms = prioritize_coordinating_atoms_to_validate(
        gr_atoms_dict, sorted_gr_atoms, metal, ligand, is_haptic, haptic_type
    )
    if priority_atoms:
        logger.debug(
            "Priority atoms identified for validation: %s %s",
            [a.label for a in priority_atoms],
            [a.atom_site_label for a in priority_atoms],
        )

        removed_atom_mol_indices = set()

        for atom in priority_atoms:
            atom_mol_idx = atom.get_parent_index("molecule")

            atom.reset_mconnec(metal)
            removed_ligand_indices.append(lig_mol_indices[atom_mol_idx])
            removed_atom_mol_indices.add(atom_mol_idx)

            if getattr(metal, "removed_from_coordination", None) is None:
                object.__setattr__(metal, "removed_from_coordination", [])

            metal.removed_from_coordination.append(atom)

            logger.debug(
                "Updated removed ligand indices: %s %s",
                removed_ligand_indices,
                lig_mol_indices[atom_mol_idx],
            )

        # Filter gr_atoms to exclude all priority atoms removed in this pass
        updated_gr_atoms = [
            a
            for a in gr_atoms
            if a.get_parent_index("molecule") not in removed_atom_mol_indices
        ]

        return updated_gr_atoms, True

    # atoms_to_validate = [a for a in sorted_gr_atoms if a not in priority_atoms]

    # logger.debug(
    #     "Final atoms selected for validation: %s %s",
    #     [a.label for a in atoms_to_validate],
    #     [a.atom_site_label for a in atoms_to_validate],
    # )

    # for atom in atoms_to_validate:
    #     atom_mol_idx = atom.get_parent_index("molecule")

    #     if atom_mol_idx in lig_mol_indices:
    #         is_added, _, _ = add_atom(
    #             labels=ligand.labels,
    #             coords=ligand.coord,
    #             site=lig_mol_indices[atom_mol_idx],
    #             ligand=ligand,
    #             element="H",
    #             removed_idx=removed_ligand_indices,
    #             metal=metal,
    #         )

    #         if not is_added:
    #             # --- EARLY RETURN LOGIC ---
    #             logger.warning(
    #                 f"Atom {atom.label} ({atom.atom_site_label}) failed validation. Returning early to re-evaluate."
    #             )

    #             # Reset connectivity for the failed atom
    #             atom.reset_mconnec(metal)
    #             removed_ligand_indices.append(lig_mol_indices[atom_mol_idx])
    #             if getattr(metal, "removed_from_coordination", None) is None:
    #                 object.__setattr__(metal, "removed_from_coordination", [])
    #             metal.removed_from_coordination.append(atom)
    #             logger.debug(
    #                 f"Updated removed ligand indices: {removed_ligand_indices} {lig_mol_indices[atom_mol_idx]}"
    #             )
    #             # Filter gr_atoms to exclude only this specific failed atom
    #             # All other atoms are still 'potentially' valid in the next iteration
    #             updated_gr_atoms = [
    #                 a
    #                 for a in gr_atoms
    #                 if a.get_parent_index("molecule") != atom_mol_idx
    #             ]

    #             return updated_gr_atoms, True
    #         else:
    #             logger.debug(f"Atom {atom.label} ({atom.atom_site_label}) retained")

    # If the loop finishes without hitting 'if not is_added', nothing was removed
    return gr_atoms, False


def partition_connected_indices(
    connected_idx, molecule, use_bond_info: bool | None = None
) -> list:
    from cell2mol.operations import extract_from_list
    from cell2mol.connectivity import split_species

    refcell = molecule.get_parent("reference")
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
    cov_factor = getattr(molecule, "cov_factor", config.COV_FACTOR)

    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO
    logger.debug("  Partitioning connected indices into blocks %s", connected_idx)

    if not connected_idx:
        logger.debug("  No connected indices provided.")
        return {}

    conn_labels = extract_from_list(connected_idx, molecule.labels, dimension=1)
    conn_coord = extract_from_list(connected_idx, molecule.coord, dimension=1)
    conn_radii = extract_from_list(connected_idx, molecule.radii, dimension=1)
    conn_atoms = extract_from_list(connected_idx, molecule.atoms, dimension=1)
    if molecule.atom_site_labels is not None:
        conn_atom_site_labels = extract_from_list(
            connected_idx, molecule.atom_site_labels, dimension=1
        )
    else:
        conn_atom_site_labels = None

    blocklist = split_species(
        labels=conn_labels,
        positions=conn_coord,
        radii=conn_radii,
        indices=None,
        atom_site_labels=conn_atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
        cov_factor=cov_factor,
        apply_graph=True,
    )
    logger.debug("  Identified %s blocks from connected atoms", len(blocklist))
    logger.debug("    Blocks: %s", [block for block in blocklist])
    results = {}
    for idx, block in enumerate(blocklist):
        gr_atoms = extract_from_list(block, conn_atoms, dimension=1)
        is_haptic, haptic_type, topology = identify_haptic_mode(
            gr_atoms, use_bond_info=use_bond_info
        )
        if is_haptic:
            logger.debug(
                "    Block with atoms %s is haptic with type(s) %s",
                [atom.label for atom in gr_atoms],
                haptic_type,
            )
        else:
            logger.debug(
                "    Block with atoms %s is non-haptic",
                [atom.label for atom in gr_atoms],
            )
        results[idx] = {
            "formula": labels2formula([atom.label for atom in gr_atoms]),
            "is_haptic": is_haptic,
            "haptic_type": haptic_type,
            "gr_atoms": gr_atoms,
        }
        # logger.debug("Results for block %s: %s", idx, results[idx])
    return results


def find_atom_outlier(atoms, metal, haptic, threshold_percent=50.0):
    """
    Takes the list of atoms, calculates distances to metal,
    and returns atoms identified as outliers.
    """
    n = len(atoms)
    if n < 2:
        return []

    # 1. Calculate margins (maintaining the order from your sorted list)
    sorted_margins = np.array(
        [np.linalg.norm(metal.coord - a.coord) - metal.radii - a.radii for a in atoms]
    )
    sorted_indices = np.arange(n)  # indices corresponding to the sorted_margins order
    outlier_indices = []

    if 3 <= n <= 8:
        # 90% confidence
        q_crit_table = {3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507, 8: 0.468}
        # 95% confidence level (alpha = 0.05)
        # q_crit_table = {3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568, 8: 0.526}
        n3_harsh_factor = 0.98

        largest = sorted_margins[0]
        second_largest = sorted_margins[1]
        second_smallest = sorted_margins[-2]
        smallest = sorted_margins[-1]

        margin_range = largest - smallest

        if margin_range <= 0:
            logger.debug("Result: Margins are identical. Data retained.")
            return []

        high_gap = largest - second_largest
        low_gap = second_smallest - smallest

        q_high = high_gap / margin_range
        q_low = low_gap / margin_range
        q_crit = q_crit_table[n]
        if n == 3:
            q_crit *= n3_harsh_factor

        logger.debug(
            f"n={n} | "
            f"largest={largest:.3f}, second_largest={second_largest:.3f}, "
            f"second_smallest={second_smallest:.3f}, smallest={smallest:.3f}"
        )
        logger.debug(
            f"high_gap={high_gap:.3f}, low_gap={low_gap:.3f}, range={margin_range:.3f}"
        )
        logger.debug(f"Q_high={q_high:.4f}, Q_low={q_low:.4f}, Q_crit={q_crit:.4f}")

        high_is_outlier = q_high > q_crit
        low_is_outlier = q_low > q_crit

        # ------------------------------------------------------------
        # Case 1: largest margin is isolated
        # Example: [0.642, -0.276, -0.310]
        # The largest-margin atom is likely non-bonded.
        # ------------------------------------------------------------
        if high_is_outlier and not low_is_outlier:
            outlier_indices = [int(sorted_indices[0])]
            logger.debug(
                "Result: Largest margin is an outlier. "
                "Rejecting the furthest atom as non-bonded."
            )
        # ------------------------------------------------------------
        # Case 2: smallest margin is isolated
        # Example: [0.65, 0.58, 0.52, -0.20]
        # The smallest-margin atom may be the only true bonded atom.
        # The larger-margin atoms may be distance-based non-bonded.
        # ------------------------------------------------------------
        elif low_is_outlier and not high_is_outlier:
            outlier_indices = [int(idx) for idx in sorted_indices[:-1]]
            logger.debug(
                "Result: Smallest margin is an outlier. "
                "Keeping the smallest-margin atom and rejecting the rest."
            )
            logger.debug(
                f"Kept atom index: {int(sorted_indices[-1])}; "
                f"Rejected atom indices: {outlier_indices}"
            )
        # ------------------------------------------------------------
        # Case 3: both sides appear outlying
        # This is ambiguous. Usually safer to reject only the largest margin.
        # ------------------------------------------------------------
        elif high_is_outlier and low_is_outlier:
            if q_high >= q_low:
                outlier_indices = [int(sorted_indices[0])]
                logger.debug(
                    "Result: Both high and low margins look outlying. "
                    "High-side outlier is stronger, rejecting largest-margin atom."
                )
            else:
                outlier_indices = [int(idx) for idx in sorted_indices[:-1]]
                logger.debug(
                    "Result: Both high and low margins look outlying. "
                    "Low-side outlier is stronger, keeping smallest-margin atom only."
                )

        # ------------------------------------------------------------
        # Case 4: no statistical outlier
        # Margins are regarded as uniform enough.
        # ------------------------------------------------------------
        else:
            logger.debug("Result: Margins are uniform enough. Data retained.")

    # --- Case: n=2 (Percent Difference) ---
    elif n == 2:
        high = sorted_margins[0]
        low = sorted_margins[1]

        # Robust to negative values and near-zero mean.
        denom = max(abs(high), abs(low), 1e-8)
        relative_gap = abs(high - low) / denom * 100.0

        logger.debug(
            f"n: {n} | margins: [{high:.3f}, {low:.3f}] | "
            f"relative_gap: {relative_gap:.4f} | "
            f"threshold_percent: {threshold_percent}"
        )

        if relative_gap > threshold_percent:
            outlier_indices = [0]
            logger.debug(
                "Result: n=2 signed-margin relative gap is large. "
                "Rejecting larger-margin atom."
            )
        else:
            logger.debug("Result: n=2 margins are similar enough. Data retained.")

    else:
        pass

    logger.debug(
        f"Analysis results: {{'n': {n}, 'outlier_indices': {outlier_indices}}}"
    )

    adj_matrix, atom_labels, atom_types, is_single_simple_ring = (
        build_adjacency_matrix_within_atoms(atoms, {metal.get_parent_index("molecule")})
    )
    logger.debug(f"atom_labels: {atom_labels}")
    logger.debug(f"atom_types: {atom_types}")
    logger.debug(f"is_single_simple_ring: {is_single_simple_ring}")
    if haptic:
        logger.debug("Coordination is haptic. No outlier removal applied.")
        return []

    if n == 3 and atom_types[0] == "internal" and not is_single_simple_ring:
        if 0 in outlier_indices:
            pass
            logger.debug(
                f"{atom_labels[0]} is already marked as an outlier and is internal"
            )
        else:
            logger.debug(f"Check for outlier {atom_labels[0]}")
            outlier_indices.insert(0, 0)

    # Return the actual atom object that is an outlier
    return [atoms[i] for i in outlier_indices] if outlier_indices else []
    # return [atoms[outlier_index]] if outlier_index is not None else []


def build_adjacency_matrix_within_atoms(atoms, metal_indices={}):
    import networkx as nx

    raw = [(a.label, a.get_parent_index("molecule"), a.adjacency) for a in atoms]
    non_metal_atoms = [
        (label, idx, neighbors)
        for label, idx, neighbors in raw
        if idx not in metal_indices
    ]

    non_metal_idx_set = {idx for _, idx, _ in non_metal_atoms}
    matrix_position = {idx: pos for pos, (_, idx, _) in enumerate(non_metal_atoms)}

    n = len(non_metal_atoms)
    adj_matrix = [[0] * n for _ in range(n)]
    for _, atom_idx, neighbors in non_metal_atoms:
        for neighbor_idx in neighbors:
            if neighbor_idx in non_metal_idx_set:
                i, j = matrix_position[atom_idx], matrix_position[neighbor_idx]
                adj_matrix[i][j] = adj_matrix[j][i] = 1

    atom_labels = [f"{label}(mol idx:{idx})" for label, idx, _ in non_metal_atoms]
    bond_counts = [sum(adj_matrix[i]) for i in range(n)]
    atom_types = [
        "terminal" if count == 1 else "internal" if count > 1 else "isolated"
        for count in bond_counts
    ]

    G = nx.from_numpy_array(np.array(adj_matrix))

    # --- Ring detection ---
    cycle_basis = nx.cycle_basis(G)
    ring_sets = [set(cycle) for cycle in cycle_basis]
    n_rings = len(ring_sets)
    ring_atoms = set().union(*ring_sets) if ring_sets else set()
    ring_atoms = list(ring_atoms)
    all_atoms_in_rings = len(ring_atoms) == G.number_of_nodes()
    is_single_simple_ring = (n_rings == 1) and all_atoms_in_rings

    return adj_matrix, atom_labels, atom_types, is_single_simple_ring


def discriminate_coord_atoms(atoms, metal, gap_cutoff=0.25):
    sorted_margins = np.array(
        [np.linalg.norm(metal.coord - a.coord) - metal.radii - a.radii for a in atoms]
    )

    min_margin = min(margin for margin in sorted_margins)

    coordinating = []
    rejected = []
    delta_from_best_margin = []
    for a, margin in zip(atoms, sorted_margins):
        delta = margin - min_margin

        delta_from_best_margin.append(delta)

        if delta <= gap_cutoff:
            coordinating.append(a)
        else:
            rejected.append(a)
    logger.debug(
        f"Discriminating coordinating atoms based on margin gap cutoff {gap_cutoff}: "
        f"min_margin={min_margin:.3f}, "
        f"delta_from_best_margin={delta_from_best_margin}, "
        f"coordinating={[a.label for a in coordinating]} {[a.atom_site_label for a in coordinating]}, "
        f"rejected={[a.label for a in rejected]} {[a.atom_site_label for a in rejected]}"
    )
    return coordinating, rejected
