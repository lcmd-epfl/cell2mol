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
from cell2mol.operations import compute_centroid
from cell2mol.connectivity import (
    build_adjacency,
    is_single_ring,
    add_atom,
)
from cell2mol.elementdata import ElementData
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm
from scipy.stats import special_ortho_group
from scipy.linalg import svd
from collections import defaultdict
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


# def shape_measure(symbols: list, positions: list) -> dict:
#     """Shape measure calculation adapted from a set of coordinates"""
#     # coordination number of metal center
#     cn = len(symbols) - 1
#     if cn == 0:
#         posgeom_dev = {}
#     elif cn == 1:
#         posgeom_dev = {"Linear": 0.0}
#     else:
#         try:
#             ref_geom = np.array(
#                 shape_structure_references_simplified["{} Vertices".format(cn)],
#                 dtype=object,
#             )
#             ideal_shapes = {}
#             for idx, rg in enumerate(ref_geom[:, 0]):
#                 geom = ref_geom[:, 3][idx]
#                 ideal_shapes[geom] = ideal_shapes_from_cosymlib[rg]

#             for geom, ideal_shape in ideal_shapes.items():
#                 chsm = calc_cshm_fast(positions, ideal_shape)
#                 posgeom_dev[geom] = round(float(chsm), 3)
#             return posgeom_dev
#         except:
#             logger.warning("%s Vertices not found in shape_structure_references", cn)
#             return {}


def normalize_structure(coordinates):
    # center and normalize the structure for CShM calculations
    centered_coords = coordinates - np.mean(coordinates, axis=0)
    norm = np.sqrt(np.mean(np.sum(centered_coords**2, axis=1)))
    return centered_coords / norm


def calc_cshm_fast(coordinates, ideal_shape, num_trials=100):
    # faster Hungarian algorithm optimization
    # check number of trials, if it is to low, it calculates the
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
        cshm = np.mean(np.sum((input_structure - scale * rotated_ideal) ** 2, axis=1))

        min_cshm = min(min_cshm, cshm)

    return min_cshm * 100


def handle_nonhaptic_coordination(group: object, use_bond_info: bool | None = None):
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    if group.metals is None:
        group.get_connected_metals()

    # Pair each atom with its index in the original list
    indexed_atoms = list(enumerate(group.atoms))

    # Sort the indexed list of atoms, prioritizing hydrogen atoms
    sorted_indexed_atoms = sorted(
        indexed_atoms, key=lambda x: (x[1].label != "H", x[1].label)
    )

    # Extract the sorted atoms and their original indices into separate lists
    sorted_atoms = [atom[1] for atom in sorted_indexed_atoms]
    original_indices = [atom[0] for atom in sorted_indexed_atoms]

    ## First Correction (former verify_connectivity)
    conn_idx = []
    conn_idx_by_metal = {jdx: [] for jdx, met in enumerate(group.metals)}
    final_ligand_indices = []
    good_atoms = []
    removed_idx = []
    for idx, atom in zip(original_indices, sorted_atoms):
        isremoved = False

        ## Now there is an extra loop for each metal of the group.
        for jdx, met in enumerate(group.metals):
            if isremoved:
                continue
            lig = group.get_parent("ligand")
            ligand_idx = atom.get_parent_index("ligand")

            tmplabels = [atom.label, met.label]
            tmpcoord = [atom.coord, met.coord]
            if atom.atom_site_label is not None and met.atom_site_label is not None:
                atom_site_labels = [atom.atom_site_label, met.atom_site_label]
            else:
                atom_site_labels = None

            refcell = atom.get_parent("reference")
            bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
            cov_factor = getattr(lig, "cov_factor", config.COV_FACTOR)
            metal_factor = getattr(lig, "metal_factor", config.METAL_FACTOR)

            tmp_adjmat = build_adjacency(
                labels=tmplabels,
                positions=tmpcoord,
                atom_site_labels=atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
                metal_only=True,
            )
            if tmp_adjmat is None:
                continue
            else:
                tmp_adjnum = tmp_adjmat.sum(axis=1)
                if any(tmp_adjnum) > 0:
                    logger.debug(
                        "Atom %s (ligand index %s) is connected to metal %s (metal index %s)",
                        atom.label,
                        ligand_idx,
                        met.label,
                        jdx,
                    )
                    if use_bond_info:
                        isadded = True
                        logger.debug(
                            "Connectivity verified for atom %s with ligand index %s based on CIF bonds",
                            atom.label,
                            ligand_idx,
                        )
                        conn_idx.append(idx)
                        final_ligand_indices.append(atom.get_parent_index("ligand"))
                        good_atoms.append(atom)
                        conn_idx_by_metal[jdx].append(idx)
                    else:
                        isadded, newlab, newcoord = add_atom(
                            lig.labels,
                            lig.coord,
                            ligand_idx,
                            lig,
                            "H",
                            removed_idx,
                        )
                        if isadded:
                            logger.debug(
                                "Connectivity verified for atom %s with ligand index %s",
                                atom.label,
                                ligand_idx,
                            )
                            conn_idx.append(idx)
                            final_ligand_indices.append(atom.get_parent_index("ligand"))
                            good_atoms.append(atom)
                            conn_idx_by_metal[jdx].append(idx)
                        else:
                            logger.debug(
                                "Correct mconnec of atom %s with ligand index %s",
                                atom.label,
                                ligand_idx,
                            )
                            isremoved = True
                            removed_idx.append(ligand_idx)
                            ### Reset Connectivity of the atom and the parents
                            atom.reset_mconnec(met)
                            met.get_coord_sphere()
                            met.get_coord_sphere_formula()

    conn_idx = sorted(list(set(conn_idx)))
    split_groups = []
    final_ligand_indices_by_metal = {jdx: [] for jdx, met in enumerate(group.metals)}

    for jdx, indices in conn_idx_by_metal.items():
        metal = group.metals[jdx]
        if indices:
            logger.debug(
                "metal %s%s (index %s) connected to %s",
                metal.label,
                f" ({metal.atom_site_label})" if metal.atom_site_label else "",
                jdx,
                [
                    atom.atom_site_label if atom.atom_site_label else atom.label
                    for atom in (group.atoms[i] for i in indices)
                ],
            )
            new_group = [i for i in indices]
            split_groups.append(new_group)

    for jdx, indices in enumerate(split_groups):
        for idx in indices:
            atom = group.atoms[idx]
            if atom.get_parent_index("ligand") is not None:
                final_ligand_indices_by_metal[jdx].append(
                    atom.get_parent_index("ligand")
                )

    final_group_indices = split_groups

    grouped = defaultdict(list)
    for k, v in final_ligand_indices_by_metal.items():
        grouped[tuple(v)].append(k)
    group_metals_indices = [v for v in grouped.values()]
    # logger.debug("Final group: %s", [a.label for a in group.atoms])
    # logger.debug("Final group indices: %s", final_group_indices)
    # logger.debug("Final ligand indices by metal: %s", final_ligand_indices_by_metal)
    # logger.debug("Group metals indices: %s", group_metals_indices)

    return (
        group,
        final_group_indices,
        final_ligand_indices_by_metal,
        group_metals_indices,
    )


def handle_haptic_coordination(group: object, use_bond_info: bool | None = None):
    if use_bond_info is None:
        use_bond_info = config.USE_BOND_INFO

    refcell = group.get_parent("reference")
    bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

    single_ring = is_single_ring(
        labels=group.labels,
        positions=group.coord,
        atom_site_labels=group.atom_site_labels,
        bond_data=bond_data,
        use_bond_info=use_bond_info,
    )
    logger.debug("Is single ring: %s", single_ring)

    conn_idx = []
    conn_idx_by_metal = {jdx: [] for jdx, met in enumerate(group.metals)}
    for idx, atom in enumerate(group.atoms):
        for jdx, met in enumerate(group.metals):
            lig = group.get_parent("ligand")
            ligand_idx = atom.get_parent_index("ligand")
            tmplabels = [atom.label, met.label]
            tmpcoord = [atom.coord, met.coord]
            if atom.atom_site_label is not None and met.atom_site_label is not None:
                atom_site_labels = [atom.atom_site_label, met.atom_site_label]
            else:
                atom_site_labels = None

            refcell = atom.get_parent("reference")
            bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
            cov_factor = getattr(lig, "cov_factor", config.COV_FACTOR)
            metal_factor = getattr(lig, "metal_factor", config.METAL_FACTOR)

            tmp_adjmat = build_adjacency(
                labels=tmplabels,
                positions=tmpcoord,
                atom_site_labels=atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
                metal_only=True,
            )
            if tmp_adjmat is None:
                continue
            else:
                tmp_adjnum = tmp_adjmat.sum(axis=1)
                if any(tmp_adjnum) > 0:
                    logger.debug(
                        "Atom %s (ligand index %s) is connected to metal %s (metal index %s)",
                        atom.label,
                        ligand_idx,
                        met.label,
                        jdx,
                    )
                    conn_idx.append(idx)
                    conn_idx_by_metal[jdx].append(idx)

    conn_idx = sorted(list(set(conn_idx)))
    split_groups = []
    final_ligand_indices_by_metal = {jdx: [] for jdx, met in enumerate(group.metals)}

    for jdx, indices in conn_idx_by_metal.items():
        metal = group.metals[jdx]
        if indices:
            logger.debug(
                "metal %s%s (index %s) connected to %s",
                metal.label,
                f" ({metal.atom_site_label})" if metal.atom_site_label else "",
                jdx,
                [
                    atom.atom_site_label if atom.atom_site_label else atom.label
                    for atom in (group.atoms[i] for i in indices)
                ],
            )
            new_group = [i for i in indices]
            split_groups.append(new_group)

    for jdx, indices in enumerate(split_groups):
        for idx in indices:
            atom = group.atoms[idx]
            if atom.get_parent_index("ligand") is not None:
                final_ligand_indices_by_metal[jdx].append(
                    atom.get_parent_index("ligand")
                )

    final_group_indices = split_groups
    grouped = defaultdict(list)
    for k, v in final_ligand_indices_by_metal.items():
        grouped[tuple(v)].append(k)
    group_metals_indices = [v for v in grouped.values()]

    return (
        group,
        final_group_indices,
        final_ligand_indices_by_metal,
        group_metals_indices,
    )
