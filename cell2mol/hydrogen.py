#!/usr/bin/env python
"""
This module provides utilities to detect missing hydrogen atoms in molecules
using adjacency and geometric analysis.
"""

import logging
import numpy as np
from cell2mol.operations import get_angle, get_unit_vector

logger = logging.getLogger("cell2mol.check_hydrogen")


def detect_missing_hydrogens(
    atomic_number, center_coord, neighbor_coords, neighbor_labels
):
    """Detect missing hydrogen atoms from local adjacency geometry.

    Args:
        atomic_number (int): atomic number of the central atom
        center_coord (array-like): coordinates of the central atom
        neighbor_coords (list): coordinates of bonded neighboring atoms
        neighbor_labels (list): element symbols of bonded atoms

    Returns:
        tuple:
            missing_h (bool): whether hydrogens are missing
            report (str): textual explanation
            num_missing_h (int): estimated number of missing hydrogens
    """
    missing_h = False
    report = ""

    observed_coordination = len(neighbor_coords)

    bond_vectors = [
        get_unit_vector(np.subtract(coord, center_coord)) for coord in neighbor_coords
    ]

    geometry, expected_coordination, geometry_report = infer_coordination_geometry(
        bond_vectors
    )

    if observed_coordination == 1:
        if neighbor_labels[0] not in {"O", "N"}:
            missing_h = True
    elif observed_coordination < expected_coordination:
        missing_h = True

    num_missing_h = max(expected_coordination - observed_coordination, 0)

    report += f"Summary of facts:\n - Atom has {observed_coordination} adjacent atoms\n"

    if observed_coordination == 1 and missing_h:
        report += (
            f" - Single adjacent atom {neighbor_labels[0]} "
            f"(possibly methyl group with missing H)\n"
        )
    else:
        report += (
            f" - Geometry {geometry} suggests coordination {expected_coordination}\n"
        )

    report += geometry_report

    return missing_h, report, num_missing_h


def infer_coordination_geometry(bond_vectors):
    """Infer coordination geometry from bond vectors.

    Args:
        bond_vectors (list): list of unit bond vectors

    Returns:
        tuple:
            geometry (str): assigned geometry name
            coordination_number (int): expected coordination number
            geometry_report (str): diagnostic angle information
    """
    atol = 4e-1
    geometry = "Unassigned"
    coordination_number = 0
    geometry_report = ""

    if len(bond_vectors) == 1:
        return "Point", 1, geometry_report

    angles = []
    for i, a in enumerate(bond_vectors):
        for j, b in enumerate(bond_vectors):
            if i != j:
                angle = get_angle(a, b)
                if angle not in angles:
                    angles.append(angle)

    avg_angle = np.mean(angles)

    geometry_report += f"Angles (rad): {angles} Avg: {avg_angle}\n"
    geometry_report += (
        f"Angles (deg): {np.degrees(angles).tolist()} Avg: {np.degrees(avg_angle)}\n"
    )

    angle_differences = [
        abs(avg_angle - np.pi),  # linear
        abs(avg_angle - 2.094395),  # trigonal
        abs(avg_angle - 1.570796),  # square planar
        abs(avg_angle - 1.911136),  # tetrahedral
    ]

    min_difference = min(angle_differences)
    best_geometry_idx = angle_differences.index(min_difference)

    geometry_report += f"Diffs: {angle_differences} Minval: {min_difference}\n"

    if min_difference <= atol:
        if best_geometry_idx == 0:
            geometry, coordination_number = "Linear", 2
        elif best_geometry_idx == 1:
            geometry, coordination_number = "Triangular", 3
        elif best_geometry_idx == 2:
            geometry, coordination_number = "SquarePlanar", 4
        elif best_geometry_idx == 3:
            geometry, coordination_number = "Tetrahedron", 4

    return geometry, coordination_number, geometry_report


def check_missing_hydrogens(reference_molecules):
    """Check for missing hydrogen atoms in reference molecules.

    Args:
        reference_molecules (list): list of reference molecule objects

    Returns:
        tuple:
            has_missing_h (bool): True if missing hydrogens are detected
            missing_h_detected (bool): last evaluated missing-H state
            missing_h_in_carbon (bool): missing H in carbon atoms
            missing_h_in_coordinated_water (bool): missing H in coordinated water
            missing_h_in_water (bool): missing H in isolated water
    """
    missing_h_in_carbon = False
    missing_h_in_water = False
    missing_h_in_coordinated_water = False
    missing_h_detected = False

    coord_water_exceptions = {"Re", "V", "Mo", "W", "Fe", "Tc"}
    fullerenes = {"C60", "C72", "C80"}

    logger.info("Checking hydrogen consistency")

    for mol_idx, ref in enumerate(reference_molecules):
        if (
            not ref.iscomplex
            and not ref.has_IA_IIA
            and not ref.has_post_transition_metal
        ):
            if ref.natoms == 1 and "O" in ref.labels:
                missing_h_in_water = True
                logger.warning(
                    "Isolated O atom detected (possible water with missing H)"
                )
                continue

            if ref.formula in {"C-O", "C-N"} or ref.formula in fullerenes:
                continue

            for atom_idx, atom in enumerate(ref.atoms):
                if atom.label != "C" or atom.adjacency is None:
                    continue

                neighbor_coords = [ref.coord[i] for i in atom.adjacency]
                neighbor_labels = [ref.atoms[i].label for i in atom.adjacency]

                missing_h_detected, report, _ = detect_missing_hydrogens(
                    atom.atnum,
                    atom.coord,
                    neighbor_coords,
                    neighbor_labels,
                )

                if missing_h_detected:
                    logger.warning(
                        f"Missing H in molecule {mol_idx} "
                        f"({ref.formula}), C atom index {atom_idx}"
                    )
                    logger.warning(report)
                    missing_h_in_carbon = True

        else:
            for lig in ref.ligands:
                if (
                    lig.natoms == 1
                    and "O" in lig.labels
                    and lig.denticity <= 1
                    and not any(m.label in coord_water_exceptions for m in lig.metals)
                ):
                    missing_h_in_coordinated_water = True

    has_missing_h = (
        missing_h_in_carbon or missing_h_in_coordinated_water or missing_h_in_water
    )

    if not has_missing_h:
        logger.info("No missing hydrogen atoms detected")

    return (
        has_missing_h,
        missing_h_detected,
        missing_h_in_carbon,
        missing_h_in_coordinated_water,
        missing_h_in_water,
    )
