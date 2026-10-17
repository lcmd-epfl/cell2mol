"""Compare atom properties, species composition, and CIF charge reports."""

import logging

logger = logging.getLogger(__name__)


def compare_atoms(at1, at2, check_coordinates: bool = False):
    """Compare element symbols, available charges and spins, and optional coordinates."""
    if at1.label != at2.label:
        return False
    if check_coordinates:
        if at1.coord[0] != at2.coord[0]:
            return False
        if at1.coord[1] != at2.coord[1]:
            return False
        if at1.coord[2] != at2.coord[2]:
            return False
    if hasattr(at1, "charge") and hasattr(at2, "charge"):
        if at1.charge != at2.charge:
            return False
    if hasattr(at1, "spin") and hasattr(at2, "spin"):
        if at1.spin != at2.spin:
            return False
    return True


def compare_metals(at1, at2, check_coordinates: bool = False):
    """Compare metal element symbols and coordination formulas, optionally coordinates."""
    if at1.subtype != "metal" or at2.subtype != "metal":
        return False

    if at1.label != at2.label:
        return False

    if at1.coord_sphere_formula is None:
        at1.build_coordination_sphere_formula()
    if at2.coord_sphere_formula is None:
        at2.build_coordination_sphere_formula()
    if at1.coord_sphere_formula != at2.coord_sphere_formula:
        return False

    if check_coordinates:
        if at1.coord[0] != at2.coord[0]:
            return False
        if at1.coord[1] != at2.coord[1]:
            return False
        if at1.coord[2] != at2.coord[2]:
            return False

    return True


def compare_species(
    mol1,
    mol2,
):
    """Compare atom counts, neutral electron counts, and element-pair adjacency counts."""
    if mol1.natoms != mol2.natoms:
        return False

    if mol1.eleccount != mol2.eleccount:
        return False

    element_count1 = mol1.compute_element_count()
    element_count2 = mol2.compute_element_count()
    if not (element_count1 == element_count2).all():
        return False

    adj_types1 = mol1.compute_adjacency_types()
    adj_types2 = mol2.compute_adjacency_types()
    assert adj_types1 is not None
    assert adj_types2 is not None
    return bool((adj_types1 == adj_types2).all())


def compare_reference_indices(ref, mol):
    """Compare atom counts, formulas, and reference atom indices, ignoring order."""
    if (ref.natoms == mol.natoms) and (ref.formula == mol.formula):
        ref_parent_indices = sorted(ref.get_parent_indices("reference"))
        mol_parent_indices = sorted(mol.get_parent_indices("reference"))
        if ref_parent_indices == mol_parent_indices:
            issame = True
        else:
            issame = False
    else:
        issame = False
    return issame


def compare_total_charge(refcell):
    """Record assigned versus CIF charges; return False if any known pair disagrees."""
    for i, ref in enumerate(refcell.refmoleclist):
        if ref.totcharge_cif is not None and ref.totcharge is not None:
            ref.totcharge_agree = ref.totcharge_cif == ref.totcharge
            logger.debug(
                f"Ref.Molecule {i} - total charge: "
                f"CIF={ref.totcharge_cif}, cell2mol={ref.totcharge}, "
                f"Agree={ref.totcharge_agree}"
            )

    if any(ref.totcharge_agree is False for ref in refcell.refmoleclist):
        return False
    else:
        return True


def compare_metal_oxidation_states(refcell):
    """Compare assigned and reported metal oxidationstate counts.

    Return None when reports or complete name matches are unavailable.
    """
    # Include unassigned metals as label_None tokens.
    metal_os_tag_list = []
    for ref in refcell.refmoleclist:
        if ref.metals:
            for metal in ref.metals:
                if metal.charge is not None:
                    m_ox = metal.charge
                else:
                    m_ox = None
                label = metal.label
                metal_os_tag = (
                    f"{label}_{m_ox}" if m_ox is not None else f"{label}_None"
                )
                metal_os_tag_list.append(metal_os_tag)

    if not refcell.reported_metal_os:
        logger.info(
            "No reported metal oxidation states found in CIF, skipping comparison."
        )
        return None

    matched_list = getattr(refcell, "reported_metal_os_matched")
    confidence = getattr(refcell, "metal_os_match_confidence")

    if confidence is None:
        logger.warning("No confidence level could be determined, skipping comparison.")
        return None
    elif confidence < 1.0:
        logger.info(
            f"Confidence level below 1.0 ({confidence:.2f}), skipping comparison of metal oxidation states."
        )
        return None
    if not matched_list:
        logger.warning(
            "No reported metal oxidation states could be matched to the metals found in the structure."
        )
        return None

    metal_os_count = {}
    for metal_os in metal_os_tag_list:
        if metal_os in metal_os_count:
            metal_os_count[metal_os] += 1
        else:
            metal_os_count[metal_os] = 1

    # Compare occurrence counts, including repeated oxidation states.
    reported_os_count = {}
    for reported_metal_os in matched_list:
        if reported_metal_os in reported_os_count:
            reported_os_count[reported_metal_os] += 1
        else:
            reported_os_count[reported_metal_os] = 1
    logger.info(
        f"Metal oxidation states found: {metal_os_count}, Reported metal oxidation states: {reported_os_count}"
    )
    return metal_os_count == reported_os_count
