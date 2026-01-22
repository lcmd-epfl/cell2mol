#!/usr/bin/env python
"""
This module provides utilities
to detect missing hydrogen atoms in molecules using adjacency and geometric analysis.
to place hydrogen atoms based on existing neighbors and add hydrogen for protonation state.
"""

import logging
import numpy as np
from typing import Tuple
from cell2mol.operations import get_angle, unit_vector, perp_unit, kabsch_rotation
from cell2mol.elementdata import ElementData

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


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
        unit_vector(np.subtract(coord, center_coord)) for coord in neighbor_coords
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
            missing_h_in_carbon (bool): missing H in coordinated carbon atoms
            missing_h_in_coordinated_water (bool): missing H in coordinated water
            missing_h_in_water (bool): missing H in isolated water
    """
    missing_h_in_carbon = False
    missing_h_in_water = False
    missing_h_in_coordinated_water = False
    missing_h_detected = False
    # coord_water_exceptions = {}
    # coord_water_exceptions = {"Re", "V", "Mo", "W", "Fe", "Tc", "Os", "Cr", "Nb", "U"}
    coord_water_exceptions = {"Re", "V", "Mo", "W", "Fe", "Tc"}
    fullerenes = {"C60", "C72", "C80"}

    logger.info("Checking any missing hydrogens in reference molecules...")

    for mol_idx, ref in enumerate(reference_molecules):
        if (
            not ref.iscomplex
            and not ref.has_ia_iia
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
        missing_h_in_carbon,
        missing_h_in_coordinated_water,
        missing_h_in_water,
    )


def place_hydrogens(
    C,
    N1=None,
    N2=None,
    N3=None,
    r_CH=1.09,
    hybridization="auto",
    sp2_angle_window=(95, 145),
    sp3_angle_window=(95, 125),
):
    """
    Place hydrogens on a carbon with 1–3 existing neighbors.

    Rules
    -----
    - 1 neighbor: add 2 H (sp2), add 3 H (sp3)
    - 2 neighbors: add 1 H (sp2), add 2 H (sp3)
    - 3 neighbors: add 1 H (sp3)  [sp2 invalid -> raises]

    Auto behavior
    -------------
    - 1 neighbor: defaults to 'sp3'
    - 2 neighbors: infer from angle windows (same as before)
    - 3 neighbors: 'sp3'

    Returns
    -------
    Hs : (k,3) array of hydrogen coordinates.
    """
    C = np.asarray(C, float)
    neighbors = [v for v in (N1, N2, N3) if v is not None]
    n_nb = len(neighbors)
    if n_nb == 0 or n_nb > 3:
        raise ValueError("This function supports 1–3 neighbors.")

    # Unit vectors from C toward neighbors
    us = []
    for i, N in enumerate(neighbors):
        u = np.asarray(N, float) - C
        if np.linalg.norm(u) < 1e-12:
            raise ValueError(f"Neighbor {i + 1} coincides with C.")
        us.append(unit_vector(u))

    # Tetrahedral template (four directions)
    u1 = unit_vector(np.array([1, 1, 1], float))
    u2 = unit_vector(np.array([1, -1, -1], float))
    u3 = unit_vector(np.array([-1, 1, -1], float))
    u4 = unit_vector(np.array([-1, -1, 1], float))
    # Utemp = [u1, u2, u3, u4]

    mode = hybridization.lower()
    if mode not in ("auto", "sp2", "sp3"):
        raise ValueError("hybridization must be 'auto', 'sp2', or 'sp3'.")

    # ---------- CASE: 1 neighbor ----------
    if n_nb == 1:
        a = us[0]
        if mode == "auto":
            mode = "sp3"  # default to methyl (CH3)
        if mode == "sp2":
            # Planar CH2
            p1 = unit_vector(-a)  # opposite to neighbor
            p2 = perp_unit(a)  # any unit vector ⟂ a defines the plane
            c60 = 0.5
            s60 = np.sqrt(3) / 2.0
            d1 = unit_vector(c60 * p1 + s60 * p2)
            d2 = unit_vector(c60 * p1 - s60 * p2)
            H1 = C + r_CH * d1
            H2 = C + r_CH * d2
            return np.vstack([H1, H2])  # two hydrogens
        else:  # sp3 → three hydrogens (methyl)
            # Align template so one vertex aligns with neighbor; fix rotation using a ⟂ axis
            b_perp = perp_unit(a)
            P = np.stack([u1, u2], axis=1)  # template pair to pin orientation
            Q = np.stack([a, b_perp], axis=1)  # target pair
            R = kabsch_rotation(P, Q)
            dH2 = unit_vector(R @ u2)
            dH3 = unit_vector(R @ u3)
            dH4 = unit_vector(R @ u4)
            H2 = C + r_CH * dH2
            H3 = C + r_CH * dH3
            H4 = C + r_CH * dH4
            return np.vstack([H2, H3, H4])  # three hydrogens

    # ---------- CASE: 2 neighbors ----------
    if n_nb == 2:
        a, b = us
        # Angle between neighbors
        cosang = np.clip(a @ b, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))

        def add_sp2_two():
            # In-plane bisector opposite to existing bonds
            dH = -(a + b)
            if np.linalg.norm(dH) < 1e-8:
                raise ValueError(
                    "Neighbors nearly opposite (sp-like). Cannot place sp2 hydrogen reliably."
                )
            dH = unit_vector(dH)
            return np.array([C + r_CH * dH])  # one hydrogen

        def add_sp3_two():
            # Align two template directions to a,b then use the other two for H's
            P = np.stack([u1, u2], axis=1)  # 3x2 template
            Q = np.stack([a, b], axis=1)  # 3x2 targets
            R = kabsch_rotation(P, Q)
            dH1 = unit_vector(R @ u3)
            dH2 = unit_vector(R @ u4)
            H1 = C + r_CH * dH1
            H2 = C + r_CH * dH2
            return np.vstack([H1, H2])  # two hydrogens

        if mode == "auto":
            in_sp3 = sp3_angle_window[0] <= angle <= sp3_angle_window[1]
            in_sp2 = sp2_angle_window[0] <= angle <= sp2_angle_window[1]
            if in_sp3 and not in_sp2:
                return add_sp3_two()
            if in_sp2 and not in_sp3:
                return add_sp2_two()
            # closest target
            target_sp3 = 109.47
            target_sp2 = 120.0
            if abs(angle - target_sp3) < abs(angle - target_sp2):
                return add_sp3_two()
            else:
                return add_sp2_two()
        elif mode == "sp3":
            return add_sp3_two()
        else:  # sp2
            return add_sp2_two()

    # ---------- CASE: 3 neighbors ----------
    if n_nb == 3:
        if mode == "auto":
            mode = "sp3"
        if mode != "sp3":
            raise ValueError(
                "With 3 neighbors, only 'sp3' is supported (adds one hydrogen)."
            )
        a, b, c = us
        # Align three template directions to the three neighbors; remaining vertex gives H
        P = np.stack([u1, u2, u3], axis=1)  # 3x3 template
        Q = np.stack([a, b, c], axis=1)  # 3x3 targets
        R = kabsch_rotation(P, Q)
        dH = unit_vector(R @ u4)
        H = C + r_CH * dH
        return np.array([H])  # one hydrogen

    # Should not reach here
    raise RuntimeError("Unhandled neighbor count.")


def add_hydrogens(
    labels: list,
    coords: np.ndarray,
    site: int,
    ligand: object,
    num_hydrogens: int,
    element: str = "H",
) -> Tuple[bool, list, np.ndarray]:
    """Add hydrogens to a given atom site."""

    isadded = True
    newlab = labels.copy()
    newcoord = coords.copy()

    for idx, atom in enumerate(ligand.atoms):
        if idx != site:
            continue

        apos = np.array(atom.coord, copy=True)
        bonded_atom_coord = []
        bonded_atom_labels = []

        for adj in atom.adjacency:
            n_label = ligand.get_parent("molecule").labels[adj]
            n_coord = ligand.get_parent("molecule").coord[adj]

            # Skip d- and f-block elements
            if elemdatabase.elementblock[n_label] in {"d", "f"}:
                continue

            bonded_atom_coord.append(n_coord)
            bonded_atom_labels.append(n_label)

        logger.debug(
            "atom=%s site=%s adjacency=%s bonded_labels=%s num_hydrogens=%d",
            atom.label,
            atom.atom_site_label,
            atom.adjacency,
            bonded_atom_labels,
            num_hydrogens,
        )

        if atom.label == "C":
            ismissingH, report, num_missingH = detect_missing_hydrogens(
                atom.atnum,
                atom.coord,
                bonded_atom_coord,
                bonded_atom_labels,
            )
            logger.debug(
                "detect_missing_hydrogens -> %s, %s, %d",
                ismissingH,
                report,
                num_missingH,
            )
        else:
            ismissingH, report, num_missingH = True, "", 2

        Hs = None

        if num_hydrogens == 2:
            if len(bonded_atom_labels) == 1:
                Hs = place_hydrogens(
                    apos,
                    bonded_atom_coord[0],
                    hybridization="sp2",
                )
            elif len(bonded_atom_labels) == 2:
                Hs = place_hydrogens(
                    apos,
                    bonded_atom_coord[0],
                    bonded_atom_coord[1],
                    hybridization="sp3",
                )

        elif num_hydrogens == 3:
            if len(bonded_atom_labels) == 1:
                Hs = place_hydrogens(
                    apos,
                    bonded_atom_coord[0],
                    hybridization="sp3",
                )

        if Hs is not None and Hs.shape[0] == num_hydrogens:
            for h in Hs:
                newcoord = np.vstack([newcoord, h])
                newlab.append(str(element))

            logger.debug(
                "Added %d %s atoms to site=%d (%s, %s)",
                num_hydrogens,
                element,
                site,
                atom.label,
                atom.atom_site_label,
            )

    return isadded, newlab, newcoord
