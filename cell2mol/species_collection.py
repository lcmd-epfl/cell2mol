"""Collect species, hydrogen flags, and charge candidates across molecules."""

from __future__ import annotations

import logging

from cell2mol.compare import compare_species, compare_metals

logger = logging.getLogger(__name__)


def _ligands_match(lig, other) -> bool:
    """Compare nitrosyl geometry, donor groups, and hapticity before species matching."""
    if lig.is_nitrosyl is None:
        lig.detect_nitrosyl()
    if other.is_nitrosyl is None:
        other.detect_nitrosyl()
    if lig.haptic_type is None:
        lig.update_hapticity()
    if other.haptic_type is None:
        other.update_hapticity()

    if lig.is_nitrosyl and other.is_nitrosyl:
        return lig.NO_type == other.NO_type

    lig_groups_labels = [g.labels for g in lig.groups or []]
    other_groups_labels = [g.labels for g in other.groups or []]

    if (
        len(lig_groups_labels) == len(other_groups_labels)
        and sorted(lig_groups_labels) == sorted(other_groups_labels)
        and lig.haptic_type == other.haptic_type
    ):
        return compare_species(lig, other)

    return False


def _ensure_complex_split(mol) -> None:
    """Split metal-containing molecules into ligands and metals when needed."""
    if mol.ligands is not None:
        return
    if mol.is_metal_complex or mol.has_s_block_metal:
        mol.split_complex()
    elif mol.is_post_transition_metal_complex:
        mol.split_complex(post_tms=True)


def collect_unique_species(moleclist):
    """Collect representatives and assign each occurrence a unique_index.

    Metal-containing molecules contribute metals and ligands; metal-free
    molecules contribute themselves. Return (unique_species, unique_indices,
    species_list), with the last two lists aligned.
    """
    unique_species = []
    unique_indices: list[int] = []
    species_list = []

    typelist_mols: list[list] = []
    typelist_ligs: list[list] = []
    typelist_mets: list[list] = []

    specs_found = -1

    for idx, mol in enumerate(moleclist or []):
        logger.debug("Molecule (%d) formula=%s", idx, mol.formula)

        if mol.is_metal_free_molecule:
            kdx = None
            for ldx, typ in enumerate(typelist_mols):
                if compare_species(mol, typ[0]):
                    kdx = typ[1]
                    logger.debug(
                        "molecule %s (%d) is the same with type %d in type list",
                        mol.formula,
                        idx,
                        ldx,
                    )
                    break

            if kdx is None:
                specs_found += 1
                kdx = specs_found
                typelist_mols.append([mol, kdx])
                unique_species.append(mol)
                logger.debug(
                    "New molecule found with: formula=%s and added in species type %d",
                    mol.formula,
                    kdx,
                )

            unique_indices.append(kdx)
            mol.unique_index = kdx
            species_list.append(mol)
            continue

        # Collect metals before ligands to preserve occurrence order.
        _ensure_complex_split(mol)

        for jdx, met in enumerate(mol.metals or []):
            kdx = None
            for ldx, typ in enumerate(typelist_mets):
                if compare_metals(met, typ[0]):
                    kdx = typ[1]
                    logger.debug(
                        "Metal %s (%d) is the same with type %d in type list",
                        met.formula,
                        jdx,
                        ldx,
                    )
                    break

            if kdx is None:
                specs_found += 1
                kdx = specs_found
                typelist_mets.append([met, kdx])
                unique_species.append(met)
                logger.debug(
                    "New metal found with: formula=%s and added in species type %d",
                    met.formula,
                    kdx,
                )

            unique_indices.append(kdx)
            met.unique_index = kdx
            species_list.append(met)

        for jdx, lig in enumerate(mol.ligands or []):
            kdx = None
            for ldx, typ in enumerate(typelist_ligs):
                if _ligands_match(lig, typ[0]):
                    kdx = typ[1]
                    logger.debug(
                        "ligand %s (%d) is the same with type %d in type list",
                        lig.formula,
                        jdx,
                        ldx,
                    )
                    break

            if kdx is None:
                specs_found += 1
                kdx = specs_found
                typelist_ligs.append([lig, kdx])
                unique_species.append(lig)
                logger.debug(
                    "New ligand found with: formula=%s added in species type %d",
                    lig.formula,
                    kdx,
                )

            unique_indices.append(kdx)
            lig.unique_index = kdx
            species_list.append(lig)

    logger.info("Unique species: %s", [s.formula for s in unique_species])
    logger.info("Unique indices: %s", unique_indices)
    logger.info("Species list: %s", [s.formula for s in species_list])

    return unique_species, unique_indices, species_list


def collect_hydrogen_flags(moleclist, species_list):
    """Check isolated H/D atoms and missing hydrogens across molecules and species.

    Return flags for isolated H, missing H on carbon, coordinated donors,
    and water. has_missing_H combines the three missing-H flags.
    """
    # Only isolated H/D atoms set the hydrogen error flag.
    has_isolated_h = False
    for mol in moleclist or []:
        if mol.natoms != 1:
            continue
        label = (mol.atoms or [])[0].label
        site_label = mol.atom_site_labels[0] if mol.atom_site_labels else "N/A"
        if label in {"H", "D"}:
            has_isolated_h = True
            logger.warning(
                "  Isolated hydrogen found %s (%s)", mol.labels[0], site_label
            )
        else:
            logger.warning("  Isolated atom found %s (%s)", mol.labels[0], site_label)

    logger.info("Has isolated hydrogen: %s", has_isolated_h)

    missing = []

    for species in species_list:
        # Metals have no hydrogen-check method.
        if species.subtype == "metal":
            continue
        if species.check_hydrogens():
            missing.append(species)

    logger.info(
        "Missing hydrogens found in %d/%d species: %s",
        len(missing),
        len(species_list or []),
        [s.formula for s in missing],
    )

    flags = {
        "has_isolated_H": has_isolated_h,
        "missing_H_in_Carbon": any(
            getattr(species, "missing_H_in_Carbon", False) for species in species_list
        ),
        "missing_H_on_CoordDonor": any(
            getattr(species, "missing_H_on_CoordDonor", False)
            for species in species_list
        ),
        "missing_H_in_Water": any(
            getattr(species, "missing_H_in_Water", False) for species in species_list
        ),
    }
    flags["has_missing_H"] = (
        flags["missing_H_in_Carbon"]
        or flags["missing_H_on_CoordDonor"]
        or flags["missing_H_in_Water"]
    )

    if flags["has_missing_H"]:
        logger.info(
            "Missing hydrogens | carbon=%s, coordinated_donor=%s, water=%s",
            flags["missing_H_in_Carbon"],
            flags["missing_H_on_CoordDonor"],
            flags["missing_H_in_Water"],
        )

    return flags


def collect_plausible_charges(
    unique_species, species_list, *, skip_missing_h: bool = False
):
    """Collect charge candidates for unique_species followed by species_list.

    Return (candidates, failed, inconsistencies). None marks missing candidates.
    With skip_missing_h=True, skipped entries remain None but do not count
    as charge-enumeration failures. No final charge is selected.
    """
    plausible_charges: list[list[int] | None] = []

    all_targets = [(species, "unique species") for species in unique_species or []]
    all_targets.extend([(species, "species list") for species in species_list or []])

    # Separate hydrogen-related skips from enumeration failures.
    skipped_for_missing_h: set[int] = set()

    for idx, (species, context_label) in enumerate(all_targets):
        logger.info(
            "Get plausible charge states for %s: %s", context_label, species.formula
        )

        if skip_missing_h and species.subtype != "metal":
            if species.has_missing_H is None:
                species.check_hydrogens()
            if species.has_missing_H:
                logger.warning(
                    "Species %s has missing hydrogens; skipping charge-state "
                    "enumeration and recording None",
                    species.formula,
                )
                plausible_charges.append(None)
                skipped_for_missing_h.add(idx)
                continue

        if species.subtype == "metal":
            plausible = species.get_plausible_oxidation_states()
            plausible_charges.append(plausible if plausible else None)
        else:
            plausible = species.get_plausible_charge_states()
            plausible_charges.append(
                [cs.species_total_charge for cs in plausible] if plausible else None
            )

    unexplained = [
        idx
        for idx, charges in enumerate(plausible_charges)
        if charges is None and idx not in skipped_for_missing_h
    ]
    error_plausible_charges = bool(unexplained)

    if skipped_for_missing_h:
        logger.info(
            "Plausible charges: %d species entr(ies) skipped for missing "
            "hydrogens (not counted as a charge error)",
            len(skipped_for_missing_h),
        )
    if unexplained:
        logger.error(
            "Plausible charges: %d species entr(ies) with no charges found: %s",
            len(unexplained),
            [all_targets[i][0].formula for i in unexplained],
        )

    inconsistent = find_charge_inconsistencies(
        all_targets, plausible_charges, skipped_for_missing_h
    )

    return plausible_charges, error_plausible_charges, inconsistent


def find_charge_inconsistencies(all_targets, plausible_charges, skipped_for_missing_h):
    """Find differing charge candidates among copies sharing a unique_index.

    all_targets contains (species, context_label) pairs aligned with candidates.
    Ignore skipped positions and charge-list order. Return
    {unique_index: distinct candidate lists}, logging differences without raising.
    """
    inconsistent: dict[int, list[list[int] | None]] = {}

    if not plausible_charges:
        return inconsistent

    by_unique_index: dict[int, list[list[int] | None]] = {}
    for idx, (species, _context) in enumerate(all_targets):
        if idx in skipped_for_missing_h:
            continue
        unique_index = getattr(species, "unique_index", None)
        if unique_index is None:
            continue
        by_unique_index.setdefault(unique_index, []).append(plausible_charges[idx])

    for unique_index, recorded in sorted(by_unique_index.items()):
        # Candidate order and duplicates do not affect equality.
        distinct = {
            None if charges is None else tuple(sorted(set(charges)))
            for charges in recorded
        }
        if len(distinct) <= 1:
            continue

        inconsistent[unique_index] = [
            None if entry is None else list(entry)
            for entry in sorted(distinct, key=lambda e: (e is None, e or ()))
        ]
        formula = next(
            (
                species.formula
                for species, _ in all_targets
                if getattr(species, "unique_index", None) == unique_index
            ),
            "?",
        )
        logger.warning(
            "Plausible charges disagree between copies of the same species "
            "%s (unique_index=%d): %s. The charge finally assigned depends "
            "on which copy is stored first.",
            formula,
            unique_index,
            inconsistent[unique_index],
        )

    return inconsistent


def map_charges_to_molecules(unique_species, moleclist) -> bool:
    """Map representative charges to copies sharing a unique_index.

    Return True if any charge assignment fails.
    """
    # Defer the import to avoid a cycle through cell2mol.classes.
    from cell2mol.charge.charge_assignment import set_charge_state

    error_assign_charge = False

    for species in unique_species or []:
        species_unique_index = getattr(species, "unique_index", None)
        for mol in moleclist or []:
            try:
                if mol.is_metal_free_molecule:
                    if mol.unique_index == species_unique_index:
                        set_charge_state(species, mol, mode=1)
                    continue

                for lig in mol.ligands or []:
                    if lig.unique_index != species_unique_index:
                        continue
                    try:
                        set_charge_state(species, lig, mode=1)
                    except Exception:
                        error_assign_charge = True
                    if lig.totcharge is None:
                        logger.warning(
                            "Ligand %s charge not set from Species %s",
                            lig.formula,
                            species_unique_index,
                        )
                        error_assign_charge = True

                for met in mol.metals or []:
                    if met.unique_index != species_unique_index:
                        continue
                    try:
                        species_charge = getattr(species, "charge", None)
                        if species_charge is not None:
                            met.set_charge(species_charge)
                    except Exception:
                        error_assign_charge = True

            except Exception as e:
                error_assign_charge = True
                logger.error(
                    "Error mapping charge for Species %s: %s", species_unique_index, e
                )

    return error_assign_charge


def update_plausible_charges(collection) -> None:
    """Refresh charge candidates and inconsistency flags on a molecule collection."""
    (
        collection.plausible_charges,
        collection.error_plausible_charges,
        collection.inconsistent_plausible_charges,
    ) = collect_plausible_charges(
        collection.unique_species, collection.species_list, skip_missing_h=True
    )
    collection.error_inconsistent_plausible_charges = bool(
        collection.inconsistent_plausible_charges
    )
