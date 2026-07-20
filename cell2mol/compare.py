import logging
from cell2mol.elementdata import ElementData
from cell2mol.standardize_metal_os import standardize_reported_metal_os

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


def compare_atoms(at1, at2, check_coordinates: bool = False):
    # logger.debug("Comparing Atoms: %s and %s", at1.label, at2.label)

    # Compares Species, Coordinates, Charge and Spin
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
    # logger.debug("Comparing Metals: %s and %s", at1.label, at2.label)

    if at1.subtype != "metal" or at2.subtype != "metal":
        # logger.debug("Different subtypes: %s and %s", at1.subtype, at2.subtype)
        return False

    if at1.label != at2.label:
        # logger.debug("Different labels: %s and %s", at1.label, at2.label)
        return False

    if at1.coord_sphere_formula is None:
        at1.get_coord_sphere_formula()
    if at2.coord_sphere_formula is None:
        at2.get_coord_sphere_formula()
    if at1.coord_sphere_formula != at2.coord_sphere_formula:
        # logger.debug("Different coordination sphere")
        # logger.debug("Coordination sphere 1: %s", at1.coord_sphere_formula)
        # logger.debug("Coordination sphere 2: %s", at2.coord_sphere_formula)
        return False

    if check_coordinates:
        if at1.coord[0] != at2.coord[0]:
            return False
        if at1.coord[1] != at2.coord[1]:
            return False
        if at1.coord[2] != at2.coord[2]:
            return False

    return True


def compare_species(mol1, mol2, check_coordinates: bool = False):
    elems = elemdatabase.elementnr.keys()

    # logger.debug("Comparing Species: %s and %s", mol1.formula, mol2.formula)

    # a pair of species is compared on the basis of:
    # 1) the total number of atoms
    if mol1.natoms != mol2.natoms:
        # logger.debug("FALSE, different natoms")
        return False

    # 2) the total number of electrons (as sum of atomic number)
    if mol1.eleccount != mol2.eleccount:
        # logger.debug("FALSE, different eleccount")
        return False

    # 3) the number of atoms of each type
    if mol1.element_count is None:
        mol1.set_element_count()
    if mol2.element_count is None:
        mol2.set_element_count()

    assert mol1.element_count is not None
    assert mol2.element_count is not None

    for kdx, elem in enumerate(mol1.element_count):
        if elem != mol2.element_count[kdx]:
            # logger.debug("FALSE, different %s count", elem)
            return False
    # 4) the number of adjacencies between each pair of element types
    if mol1.adj_types is None:
        mol1.set_adj_types()
    if mol2.adj_types is None:
        mol2.set_adj_types()
    assert mol1.adj_types is not None
    assert mol2.adj_types is not None

    count = 0
    for kdx, (elem, row1) in enumerate(zip(elems, mol1.adj_types)):
        for ldx, (elem2, val1) in enumerate(zip(elems, row1)):
            val2 = mol2.adj_types[kdx, ldx]
            if val1 != val2:
                count += 1
                # logger.debug("FALSE, different adjacency count")
                # logger.debug(
                #     "COMPARE_SPECIES. %d %d %s - %s : %d - %d",
                #     kdx,
                #     ldx,
                #     elem,
                #     elem2,
                #     val1,
                #     val2,
                # )

    if count > 0:
        return False
    else:
        return True


def compare_reference_indices(ref, mol):
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
    for i, ref in enumerate(refcell.refmoleclist):
        if ref.totcharge_cif is not None and ref.totcharge is not None:
            ref.totcharge_agree = ref.totcharge_cif == ref.totcharge
            logger.debug(
                f"Ref.Molecule {i} - total charge: CIF={ref.totcharge_cif}, cell2mol={ref.totcharge}, Agree={ref.totcharge_agree}"
            )

    if any(ref.totcharge_agree is False for ref in refcell.refmoleclist):
        return False
    else:
        return True


def compare_metal_oxidation_states(refcell):
    """
    Compare the list of metals found in the structure with the reported metal oxidation states.
    Returns True if they match, False otherwise.
    """
    # Create a dictionary to count occurrences of each metal

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

    # Create a dictionary to count occurrences of each reported oxidation state
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
