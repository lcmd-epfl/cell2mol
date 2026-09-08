"""Species bookkeeping shared by every collection of molecules.

A *collection* is any group of molecules charge-balanced as one unit: a CIF
reference cell, the blocks of a multi-molecule xyz, or a lone molecule.
Deduplication must see them all at once -- one shared set of type lists is what
gives the same ligand in two molecules a single ``unique_index``, hence one
charge decision, and what makes the balancer's sum the charge of the whole
collection rather than of one molecule.

Nothing here touches the lattice: ``compare_species`` and ``compare_metals``
work on composition and adjacency only.
"""

from __future__ import annotations

import logging

from cell2mol.compare import compare_species, compare_metals

logger = logging.getLogger(__name__)


def _same_ligand(lig, other) -> bool:
    """Whether two ligands are the same specie: cheap screens (nitrosyl kind,
    group labels, hapticity) before the full ``compare_species``."""
    if lig.is_nitrosyl is None:
        lig.evaluate_as_nitrosyl()
    if other.is_nitrosyl is None:
        other.evaluate_as_nitrosyl()
    if lig.haptic_type is None:
        lig.get_hapticity()
    if other.haptic_type is None:
        other.get_hapticity()

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


def _ensure_split(mol) -> None:
    """Make sure a metal-containing molecule has been split into ligands/metals."""
    if mol.ligands is not None:
        return
    if mol.iscomplex or mol.has_ia_iia:
        mol.split_complex()
    elif mol.has_post_transition_metal:
        mol.split_complex(post_tms=True)


def collect_unique_species(moleclist):
    """Deduplicate the species of ``moleclist``, between molecules as well as
    within one.

    Metal-containing molecules contribute their metals and ligands, metal-free
    ones themselves. Stamps ``unique_index`` on each, which is how a
    representative's charge later finds its copies.

    Args:
        moleclist: The molecules of the collection.

    Returns:
        unique_species: One representative per distinct specie.
        unique_indices: Index into ``unique_species`` of each occurrence.
        species_list: Every occurrence, aligned with ``unique_indices``.
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

        if mol.is_non_complex_molecule:
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
                    "New molecule found with: formula=%s and added in specie type %d",
                    mol.formula,
                    kdx,
                )

            unique_indices.append(kdx)
            mol.unique_index = kdx
            species_list.append(mol)
            continue

        # Metal-containing molecule: metals first, then ligands. Both lists
        # are produced by split_complex, so make sure it has run.
        _ensure_split(mol)

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
                    "New metal found with: formula=%s and added in specie type %d",
                    met.formula,
                    kdx,
                )

            unique_indices.append(kdx)
            met.unique_index = kdx
            species_list.append(met)

        for jdx, lig in enumerate(mol.ligands or []):
            kdx = None
            for ldx, typ in enumerate(typelist_ligs):
                if _same_ligand(lig, typ[0]):
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
                    "New ligand found with: formula=%s added in specie type %d",
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


def collect_missing_hydrogens(moleclist, species_list):
    """Screen a collection for missing hydrogens.

    Scans the molecules for lone H/D atoms, then screens every non-metal specie
    (metals carry no hydrogens) and OR-folds the per-specie flags into
    collection-level ones. Call after the species have been collected.

    Args:
        moleclist: The molecules of the collection.
        species_list: Every specie occurrence in the collection.

    Returns:
        A dict of the five flags: ``has_isolated_H``, ``missing_H_in_Carbon``,
        ``missing_H_on_CoordDonor``, ``missing_H_in_Water`` and the
        ``has_missing_H`` that ORs the last three.
    """
    # A lone H/D is a dangling hydrogen; any other lone atom is only reported.
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

    species = species_list or []
    missing = []
    for specie in species:
        # Metals subclass Atom (not Specie) and carry no hydrogens, so they have
        # no check_hydrogens() and are always clean -- skip them.
        if specie.subtype == "metal":
            continue
        if specie.check_hydrogens():
            missing.append(specie)

    logger.info(
        "Missing hydrogens found in %d/%d species: %s",
        len(missing),
        len(species),
        [s.formula for s in missing],
    )

    flags = {
        "has_isolated_H": has_isolated_h,
        "missing_H_in_Carbon": any(
            getattr(s, "missing_H_in_Carbon", False) for s in species
        ),
        "missing_H_on_CoordDonor": any(
            getattr(s, "missing_H_on_CoordDonor", False) for s in species
        ),
        "missing_H_in_Water": any(
            getattr(s, "missing_H_in_Water", False) for s in species
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
    """Collect the plausible charges of every specie; nothing is chosen here.

    Args:
        unique_species: One representative per distinct specie.
        species_list: Every specie occurrence in the collection.
        skip_missing_h: Record ``None`` for a specie short of hydrogens without
            flagging an error -- the hydrogen check already reports that.

    Returns:
        plausible_charges: The charges of ``unique_species`` followed by those
            of all ``species_list``, so a specie recurs at two positions.
            ``None`` marks one whose charges were not found.
        error_plausible_charges: True if any specie was left without charges.
        inconsistent: ``{unique_index: differing charge sets}``, from
            ``check_charge_consistency``.
    """
    plausible_charges: list[list[int] | None] = []

    all_targets = [(specie, "unique specie") for specie in unique_species or []]
    all_targets.extend([(specie, "species list") for specie in species_list or []])

    # Positions that are None because enumeration was never attempted, as
    # opposed to attempted and failed.
    skipped_for_missing_h: set[int] = set()

    for idx, (specie, context_label) in enumerate(all_targets):
        logger.info(
            "Get plausible charge states for %s: %s", context_label, specie.formula
        )

        if skip_missing_h and specie.subtype != "metal":
            if specie.has_missing_H is None:
                specie.check_hydrogens()
            if specie.has_missing_H:
                logger.warning(
                    "Specie %s has missing hydrogens; skipping charge-state "
                    "enumeration and recording None",
                    specie.formula,
                )
                plausible_charges.append(None)
                skipped_for_missing_h.add(idx)
                continue

        if specie.subtype == "metal":
            plausible = specie.get_plausible_os()
            plausible_charges.append(plausible if plausible else None)
        else:
            plausible = specie.get_plausible_charge_states()
            plausible_charges.append(
                [cs.specie_total_charge for cs in plausible] if plausible else None
            )

    unexplained = [
        idx
        for idx, charges in enumerate(plausible_charges)
        if charges is None and idx not in skipped_for_missing_h
    ]
    error_plausible_charges = bool(unexplained)

    if skipped_for_missing_h:
        logger.info(
            "Plausible charges: %d specie entr(ies) skipped for missing "
            "hydrogens (not counted as a charge error)",
            len(skipped_for_missing_h),
        )
    if unexplained:
        logger.error(
            "Plausible charges: %d specie entr(ies) with no charges found: %s",
            len(unexplained),
            [all_targets[i][0].formula for i in unexplained],
        )

    inconsistent = check_charge_consistency(
        all_targets, plausible_charges, skipped_for_missing_h
    )

    return plausible_charges, error_plausible_charges, inconsistent


def check_charge_consistency(all_targets, plausible_charges, skipped_for_missing_h):
    """Find unique species whose copies did not enumerate to the same charges.

    Copies sharing a ``unique_index`` are the same specie, so they must yield
    the same options; when they do not, the charge chosen depends on which copy
    was stored first (the balancer reads ``unique_species`` only) rather than on
    the chemistry. The usual cause is that bond perception is not atom-order
    invariant. Recorded, not raised: order-dependent is not necessarily wrong.

    Args:
        all_targets: ``(specie, context_label)`` pairs, aligned with
            ``plausible_charges``.
        plausible_charges: The charges found for each of those entries.
        skipped_for_missing_h: Positions never attempted, which are ignored.

    Returns:
        ``{unique_index: the distinct charge sets its copies produced}``, empty
        when every copy agreed.
    """
    inconsistent: dict[int, list[list[int] | None]] = {}

    if not plausible_charges:
        return inconsistent

    by_unique_index: dict[int, list[list[int] | None]] = {}
    for idx, (specie, _context) in enumerate(all_targets):
        if idx in skipped_for_missing_h:
            continue
        unique_index = getattr(specie, "unique_index", None)
        if unique_index is None:
            continue
        by_unique_index.setdefault(unique_index, []).append(plausible_charges[idx])

    for unique_index, recorded in sorted(by_unique_index.items()):
        # Order within a charge list carries no meaning, so compare as sets.
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
                spec.formula
                for spec, _ in all_targets
                if getattr(spec, "unique_index", None) == unique_index
            ),
            "?",
        )
        logger.warning(
            "Plausible charges disagree between copies of the same specie "
            "%s (unique_index=%d): %s. The charge finally assigned depends "
            "on which copy is stored first.",
            formula,
            unique_index,
            inconsistent[unique_index],
        )

    return inconsistent


def map_charges_to_molecules(unique_species, moleclist) -> bool:
    """Propagate the charges of the unique species onto every copy.

    Returns True if any assignment failed, for ``error_assign_charge``.
    """
    # Imported here: specie_assigner pulls in cell2mol.classes, which imports
    # this module, so a module-level import would close the cycle.
    from cell2mol.charge.specie_assigner import set_charge_state

    error_assign_charge = False

    for specie in unique_species or []:
        specie_unique_index = getattr(specie, "unique_index", None)
        for mol in moleclist or []:
            try:
                if mol.is_non_complex_molecule:
                    if mol.unique_index == specie_unique_index:
                        set_charge_state(specie, mol, mode=1)
                    continue

                for lig in mol.ligands or []:
                    if lig.unique_index != specie_unique_index:
                        continue
                    try:
                        set_charge_state(specie, lig, mode=1)
                    except Exception:
                        error_assign_charge = True
                    if lig.totcharge is None:
                        logger.warning(
                            "Ligand %s charge not set from Specie %s",
                            lig.formula,
                            specie_unique_index,
                        )
                        error_assign_charge = True

                for met in mol.metals or []:
                    if met.unique_index != specie_unique_index:
                        continue
                    try:
                        specie_charge = getattr(specie, "charge", None)
                        if specie_charge is not None:
                            met.set_charge(specie_charge)
                    except Exception:
                        error_assign_charge = True

            except Exception as e:
                # Catch-all for unexpected logic errors in the outer loop
                error_assign_charge = True
                logger.error(
                    "Error mapping charge for Specie %s: %s", specie_unique_index, e
                )

    return error_assign_charge
