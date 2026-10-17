from __future__ import annotations

import numpy as np
import itertools
import networkx as nx
from cell2mol.connectivity import try_add_ligand_atom
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING, cast
from cell2mol.classes.protonation import Protonation
from cell2mol.charge.special_cases.manual import MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.element_utils import HALOGENS
from cell2mol.charge.special_cases import (
    check_fullerene_sphericity,
    find_all_porphyrin_macrocycles,
    _find_charged_moiety,
)
from cell2mol.charge.special_cases.porphyrin import (
    _generate_porphyrin_protonation_states as _generate_porphyrin_protonation_states,
    _metal_pocket_macrocycles as _metal_pocket_macrocycles,
    _order_pocket_trans as _order_pocket_trans,
)
from cell2mol.charge.xyz2mol import atomic_valence
from cell2mol.hydrogen import detect_missing_carbon_hydrogens, add_hydrogens
import logging
from cell2mol.elementdata import ElementData

elemdatabase = ElementData()

if TYPE_CHECKING:
    from cell2mol.classes.species import Species
    from cell2mol.classes.ligand import Ligand

logger = logging.getLogger(__name__)


@dataclass
class ProtonationGroupResult:
    site_proton_counts: Dict[int, int]
    ligand_donor_electrons: Dict[int, int]
    needs_combinatorial: bool
    combinatorial_indices: List[int]


# Classes at or above this size allow only none/all protonation.
ALL_OR_NOTHING_CLASS_SIZE = 3


############################################################
# Protonation-state enumeration
############################################################


def enumerate_protonation_states(species: Species) -> list[Protonation] | None:
    """Protonation states for a ligand or metal-free molecule; None for anything
    else.
    """
    if species.type != "species":
        return None

    if species.subtype == "group":
        return None

    if species.subtype == "molecule":
        if species.is_metal_free_molecule:
            return get_asis_protonation_state(species)
        else:
            logger.info(
                "Species %s (%s) is a complex molecule. Do not protonate.",
                species.formula,
                species.subtype,
            )
            return None

    if species.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        return get_asis_protonation_state(species)

    has_fullerene = (
        species.has_fullerene
        if species.has_fullerene is not None
        else species.detect_fullerene()
    )

    if has_fullerene:
        if species.coord is not None and not check_fullerene_sphericity(species.coord):
            logger.warning(
                "%s passed fullerene topology check but failed sphericity "
                "check - possible disorder/AC artifact; proceeding anyway",
                species.formula,
            )
        logger.debug("Fullerene cage detected for %s", species.formula)
        return get_asis_protonation_state(species)

    has_porphyrin = (
        species.has_porphyrin
        if species.has_porphyrin is not None
        else species.detect_porphyrin()
    )
    # Build clean N4 cores automatically; flag expanded rings or outside donors for review.
    if has_porphyrin:
        macrocycles = find_all_porphyrin_macrocycles(
            species.get_atomic_numbers(), species.adjacency_matrix
        )
        is_expanded = any(len(mac_n) >= 5 for mac_n, _c, _core in macrocycles)
        logger.debug(
            "%d porphyrin-family macrocycle(s) detected for %s "
            "(expanded: %s, contracted: %s)",
            len(macrocycles),
            species.formula,
            is_expanded,
            [is_contracted for _n, is_contracted, _c in macrocycles],
        )
        # The builder filters ineligible core N; outside donors do not change free-base sites.
        if not is_expanded:
            return _generate_porphyrin_protonation_states(species, macrocycles)
        # Expanded k>=5, but its metals may cut it into ordinary N4 pockets.
        pockets = _metal_pocket_macrocycles(species)
        if pockets:
            logger.info(
                "%s: expanded macrocycle resolves into %d metal-bound N4 "
                "pocket(s) %s; protonating a trans pair in each.",
                species.formula,
                len(pockets),
                [pocket for _n, _c, pocket in pockets],
            )
            return _generate_porphyrin_protonation_states(species, pockets)

    if species.subtype == "ligand":
        parent = cast("Species", species.get_parent("molecule"))
        if (
            parent is not None
            and parent.has_s_block_metal
            and not parent.is_metal_complex
        ):
            return get_asis_protonation_state(species)

    ligand = cast("Ligand", species)
    protonation_states: list[Protonation] = []

    newlab = ligand.labels.copy()
    newcoord = ligand.coord.copy()
    n_protons_added = 0
    site_proton_counts = np.zeros(ligand.natoms, dtype=int)
    ligand_donor_electrons = np.zeros(ligand.natoms, dtype=int)
    combinatorial_site_indices: list[int] = []
    process_both_modes: list[int] = []
    protonated_indices_to_reset: list[int] = []  # old : reset_H_indices

    max_combinations = 16
    logger.info("Processing %s (%s):", species.formula, species.subtype)

    logger.info("Ligand formula: %s, natoms: %d", ligand.formula, ligand.natoms)
    groups = ligand.groups or []
    logger.debug(
        "Ligand groups info: %s",
        [
            (
                g.formula,
                g.natoms,
                [a.atom_site_label for a in (g.atoms or [])],
                [m.atom_site_label for m in (g.metals or [])],
            )
            for g in groups
        ],
    )
    # Exclude moiety oxygens from combinatorial protonation,
    # since they are already handled by the charged moiety rules.
    moiety_atoms = _charged_moiety_atoms(ligand)

    for g in groups:
        parent_indices = g.get_parent_indices("ligand")

        if g.is_haptic:  # HAPTIC GROUPS
            result = _handle_haptic_group(ligand, g, parent_indices)
        else:  # NON-HAPTIC GROUPS
            result = _handle_non_haptic_group(ligand, g, parent_indices, moiety_atoms)

        for idx, val in result.site_proton_counts.items():
            site_proton_counts[idx] += val

        for idx, val in result.ligand_donor_electrons.items():
            ligand_donor_electrons[idx] += val

        if result.needs_combinatorial:
            combinatorial_site_indices.extend(result.combinatorial_indices)
    logger.debug("    combinatorial_site_indices: %s", combinatorial_site_indices)
    # Group equivalent donor sites before enumeration.
    site_classes = _environment_classes(ligand, combinatorial_site_indices)
    if combinatorial_site_indices:
        logger.debug(
            "    grouped %d combinatorial site(s) into %d environment class(es): %s",
            len(combinatorial_site_indices),
            len(site_classes),
            site_classes,
        )
    # Multiply per-class proton-count choices to count combinations.
    count_choices = [_class_count_choices(cls) for cls in site_classes]
    n_combinations = 1
    for choices in count_choices:
        n_combinations *= len(choices)

    # If count combinations exceed the limit, use none/all per class.
    limit_exceeded = n_combinations > max_combinations
    if limit_exceeded:
        count_choices = [(0, len(cls)) for cls in site_classes]
        n_all_or_nothing = 2 ** len(site_classes)
        logger.info(
            "  %d environment class(es) give %d combinatorial protonation "
            "state(s) (more than the limit of %d); falling back to "
            "all-or-nothing per class (%d state(s)).",
            len(site_classes),
            n_combinations,
            max_combinations,
            n_all_or_nothing,
        )
        if n_all_or_nothing > max_combinations:
            merged = _merge_by_donor_signature(ligand, site_classes)
            count_choices = [range(len(cls) + 1) for cls in merged]
            n_count_only = 1
            for choices in count_choices:
                n_count_only *= len(choices)
            logger.info(
                "  falling back to count-only protonation over %d donor "
                "signature(s) %s (%d state(s)).",
                len(merged),
                merged,
                n_count_only,
            )
            if n_count_only > max_combinations:
                reason = (
                    f"too many combinatorial protonation states even count-only "
                    f"per donor signature ({n_count_only} > {max_combinations})"
                )
                return _warn_and_return(species, reason, return_asis=True)
            site_classes = merged

    for idx, a in enumerate(ligand.atoms or []):
        atom_label = (
            f"{a.label} ({a.atom_site_label})" if a.atom_site_label else a.label
        )
        n_protons_added += site_proton_counts[idx]
        if site_proton_counts[idx] == 1:
            logger.debug(
                "    Single proton addition for atom %s (ligand idx %d)",
                atom_label,
                idx,
            )
            _, newlab, newcoord = try_add_ligand_atom(
                newlab, list(newcoord), idx, ligand, element="H", unconditional=True
            )
        elif site_proton_counts[idx] >= 2:
            logger.debug(
                "    Multiple proton addition for atom %s (ligand idx %d): %d protons, ligand_donor_electrons %d",
                atom_label,
                idx,
                site_proton_counts[idx],
                ligand_donor_electrons[idx],
            )

            start_idx = len(newlab)
            end_idx = start_idx + site_proton_counts[idx]
            _, newlab, newcoord = add_hydrogens(
                newlab,
                np.asarray(newcoord),
                idx,
                ligand,
                num_hydrogens=site_proton_counts[idx],
            )

            if ligand_donor_electrons[idx] >= 2 and idx in combinatorial_site_indices:
                process_both_modes.append(idx)
                logger.debug(
                    "    Atom %s (ligand idx %d) is also proccessed for combinatorial protonation.",
                    atom_label,
                    idx,
                )
                protonated_indices_to_reset.extend(list(range(start_idx, end_idx)))

    is_fully_deterministic = len(combinatorial_site_indices) == 0
    force_deterministic_mode = process_both_modes

    if is_fully_deterministic or force_deterministic_mode:
        protonation_states.append(
            Protonation(
                symbols=newlab,
                coord=np.asarray(newcoord),
                cov_factor=ligand.cov_factor,
                site_proton_counts=site_proton_counts.tolist(),
                ligand_donor_electrons=ligand_donor_electrons.tolist(),
                mode="deterministic",
                parent=ligand,
            )
        )
        if not force_deterministic_mode:
            return protonation_states

    if protonated_indices_to_reset:
        logger.debug(
            "Ligand natoms: %d, Protonation state: natoms %d, number of added protons: %d",
            ligand.natoms,
            len(newlab),
            n_protons_added,
        )
        logger.debug(
            "Sites process_both_modes: %s %s",
            process_both_modes,
            [(ligand.atoms or [])[idx].atom_site_label for idx in process_both_modes],
        )
        logger.debug(
            "Remove previously added protons at indices: %s for combinatorial protonation.",
            protonated_indices_to_reset,
        )
        newlab = [
            label
            for idx, label in enumerate(newlab)
            if idx not in protonated_indices_to_reset
        ]
        newcoord = [
            coord
            for idx, coord in enumerate(newcoord)
            if idx not in protonated_indices_to_reset
        ]
        for idx in process_both_modes:
            n_protons_added -= site_proton_counts[idx]
            site_proton_counts[idx] = 0
            ligand_donor_electrons[idx] = 0

    local_labels = newlab.copy()
    local_coords = newcoord.copy()
    local_site_proton_counts = site_proton_counts.copy()
    local_ligand_donor_electrons = ligand_donor_electrons.copy()
    local_n_protons_added = n_protons_added

    # Use the first c sites as each count’s representative; merged classes are approximate.
    combinations = list(itertools.product(*count_choices))
    combinations.sort(key=sum)  # order by total protons added

    for counts in combinations:
        newlab = local_labels.copy()
        newcoord = local_coords.copy()
        n_protons_added = local_n_protons_added
        site_proton_counts = local_site_proton_counts.copy()
        ligand_donor_electrons = local_ligand_donor_electrons.copy()

        for count, site_class in zip(counts, site_classes):
            for idx in site_class[:count]:  # representative subset of size `count`
                site_proton_counts[idx] = 1
                n_protons_added += 1
                _, newlab, newcoord = try_add_ligand_atom(
                    newlab,
                    list(newcoord),
                    idx,
                    ligand,
                    element="H",
                    unconditional=True,
                )
        prot = Protonation(
            symbols=newlab,
            coord=np.asarray(newcoord),
            cov_factor=ligand.cov_factor,
            site_proton_counts=site_proton_counts.tolist(),
            ligand_donor_electrons=ligand_donor_electrons.tolist(),
            mode="combinatorial",
            parent=ligand,
        )

        if prot.is_valid:
            protonation_states.append(prot)

    return protonation_states


def get_asis_protonation_state(species: Species) -> list[Protonation]:
    """Wrap the unchanged species in a Protonation state for charge enumeration."""
    logger.debug(
        "Creating as-is (no-proton) protonation for %s (%s)",
        species.formula,
        species.subtype,
    )

    asis_protonation = Protonation(
        symbols=species.labels,
        coord=species.coord,
        cov_factor=species.cov_factor,
        site_proton_counts=[0] * len(species.labels),
        ligand_donor_electrons=[0] * len(species.labels),
        mode="none",
        parent=species,
    )

    return [asis_protonation]


############################################################
# Fallback and warning handling
############################################################


def _warn_and_return(
    species: Species, reason: str, return_asis: bool = False
) -> list[Protonation] | None:
    """Record a protonation warning and return None, or an as-is state if requested."""
    logger.warning("%s: %s -- not auto-handled", species.formula, reason)
    species.protonation_warning = reason
    if return_asis:
        logger.info(
            "Returning as-is protonation state and setting protonation_warning."
        )
        return get_asis_protonation_state(species)
    logger.info("Returning None protonation state and setting protonation_warning.")
    return None


############################################################
# Combinatorial site grouping
############################################################


def _environment_classes(ligand: "Ligand", indices: list[int]) -> list[list[int]]:
    """Group indices by Weisfeiler-Lehman topological equivalence; sort by smallest index."""
    unique = sorted(set(indices))
    if not unique:
        return []

    adj = np.asarray(ligand.adjacency_matrix)
    n = adj.shape[0]
    neighbours = [np.nonzero(adj[i])[0].tolist() for i in range(n)]

    # Refine element-labelled colours until the partition stops splitting.
    label_rank = {lab: r for r, lab in enumerate(sorted(set(ligand.labels[:n])))}
    colours: dict[int, int] = {i: label_rank[ligand.labels[i]] for i in range(n)}
    n_colours = len(set(colours.values()))
    for _ in range(n):
        signatures: dict[int, tuple[int, tuple[int, ...]]] = {
            i: (colours[i], tuple(sorted(colours[j] for j in neighbours[i])))
            for i in range(n)
        }
        remap = {sig: rank for rank, sig in enumerate(sorted(set(signatures.values())))}
        colours = {i: remap[signatures[i]] for i in range(n)}
        if len(remap) == n_colours:  # partition stable
            break
        n_colours = len(remap)

    grouped: dict[int, list[int]] = {}
    for idx in unique:
        grouped.setdefault(colours[idx], []).append(idx)
    return sorted((sorted(cls) for cls in grouped.values()), key=lambda cls: cls[0])


def _class_count_choices(cls: list[int]):
    """Allow every proton count for one or two equivalent sites; use none/all for larger classes."""
    if len(cls) >= ALL_OR_NOTHING_CLASS_SIZE:
        return (0, len(cls))
    return range(len(cls) + 1)


def _merge_by_donor_signature(
    ligand: "Ligand", site_classes: list[list[int]]
) -> list[list[int]]:
    """Collapse classes sharing a donor signature, keeping only their endpoints exact."""
    merged: dict[tuple[str, int | None, tuple[str, ...]], list[int]] = {}
    for cls in site_classes:
        merged.setdefault(_donor_signature(ligand, cls[0]), []).extend(cls)
    return sorted((sorted(cls) for cls in merged.values()), key=lambda cls: cls[0])


def _donor_signature(
    ligand: "Ligand", idx: int
) -> tuple[str, int | None, tuple[str, ...]]:
    """Return the donor element, metal-neighbor count, and sorted nonmetal-neighbor elements."""
    atoms = ligand.atoms
    molecule = ligand.get_parent("molecule")
    assert atoms is not None, "Donor signatures require ligand atoms"
    assert molecule is not None, "Donor signatures require a parent molecule"
    a = atoms[idx]
    mol_labels = molecule.labels
    metal_adj = set(a.metal_contact_indices)
    non_metal = sorted(mol_labels[j] for j in a.neighbor_indices if j not in metal_adj)
    return (a.label, a.metal_contact_count, tuple(non_metal))


############################################################
# Group protonation rules
############################################################


def _charged_moiety_atoms(species) -> set[int]:
    """Find moiety oxygens whose charges are already fixed; exclude them from protonation."""
    try:
        return {
            idx
            for _centre, atom_locals, _kind, _nc in _find_charged_moiety(species)
            for idx in atom_locals
        }
    except Exception as exc:  # detection is an optimisation, never a hard gate
        logger.debug("Charged-moiety detection failed for %s: %s", species.formula, exc)
        return set()


def _handle_haptic_group(ligand, g, parent_indices) -> ProtonationGroupResult:
    """Return protonation changes for a haptic group without mutating global state."""

    site_proton_counts: Dict[int, int] = {}
    ligand_donor_electrons: Dict[int, int] = {}
    needs_combinatorial = False
    combinatorial_indices: List[int] = []

    logger.debug("        HANDLE_HAPTIC_GROUP: %s %s", g.formula, g.haptic_type)
    logger.debug("        parent_indices (ligand): %s", parent_indices)

    for idx in parent_indices:
        a = ligand.atoms[idx]

        logger.debug(
            "        HANDLE_HAPTIC: idx=%d, label=%s %s, neighbor_count=%d, metal_contact_count=%d",
            idx,
            a.label,
            f", atom_site_label={a.atom_site_label}" if a.atom_site_label else "",
            a.neighbor_count,
            a.metal_contact_count,
        )

    # Helper: add up to N hydrogens on parent_indices

    def _assign_protonation_sites(max_protons: int):
        molecule = ligand.get_parent("molecule")

        adjacency_dict = {}
        metal_adjacency_dict = {}
        for idx in parent_indices:
            atom = ligand.atoms[idx]
            adjacency_dict[idx] = [
                molecule.labels[adj]
                for adj in atom.neighbor_indices
                if adj not in atom.metal_contact_indices
            ]
            metal_adjacency_dict[idx] = [
                molecule.labels[adj] for adj in atom.metal_contact_indices
            ]

        def _select_sites(strict_mode: bool):
            """Select protonation sites with chemical priority."""
            protonation_sites = []
            added = 0
            skip_cnt = 0

            if strict_mode:
                candidates = []

                for idx in parent_indices:
                    adj_labels = adjacency_dict[idx]
                    nC = adj_labels.count("C")
                    nH = adj_labels.count("H")
                    nTot = len(adj_labels)

                    # priority (lower number = higher priority)
                    if nTot == 2 and nC == 1 and nH == 1:  # C1H1
                        priority = 0
                    if nTot == 3 and nC == 1 and nH == 2:  # C1H2
                        priority = 1
                    elif nTot == 2 and nC == 2 and nH == 0:  # C2H0
                        priority = 2
                    elif nTot == 3 and nC == 2 and nH == 1:  # C2H1
                        priority = 3
                    else:
                        priority = 4

                    candidates.append((priority, idx))
                logger.debug("  Candidates: %s", candidates)
                candidates.sort(key=lambda x: x[0])

                for _, idx in candidates:
                    atom = ligand.atoms[idx]

                    if max_protons == 2 and added == 1 and skip_cnt < 2:
                        skip_cnt += 1
                        logger.debug(
                            f"  Skipping {idx} (count: {skip_cnt}) %s %s",
                            atom.label,
                            atom.atom_site_label,
                        )
                        continue

                    if added >= max_protons:
                        break

                    protonation_sites.append(idx)
                    added += 1
                    logger.debug(
                        "  Added proton to %d %s %s %s %s",
                        idx,
                        atom.label,
                        atom.atom_site_label,
                        adjacency_dict[idx],
                        metal_adjacency_dict[idx],
                    )

                return protonation_sites

            for idx in parent_indices:
                atom = ligand.atoms[idx]
                adj_labels = adjacency_dict[idx]
                metal_adj_labels = metal_adjacency_dict[idx]

                if len(metal_adj_labels) >= 2:
                    continue

                nC = adj_labels.count("C")
                nTot = len(adj_labels)

                if nTot == 3 and nC == 3:
                    if max_protons == 2 and added == 1 and skip_cnt < 2:
                        skip_cnt += 1
                        logger.debug(
                            f"  Skipping {idx} (count: {skip_cnt}) %s %s",
                            atom.label,
                            atom.atom_site_label,
                        )
                        continue

                    protonation_sites.append(idx)
                    added += 1
                    logger.debug(
                        "  Added proton to %d %s %s %s %s",
                        idx,
                        atom.label,
                        atom.atom_site_label,
                        adjacency_dict[idx],
                        metal_adjacency_dict[idx],
                    )

                    if added >= max_protons:
                        break

            return protonation_sites

        protonation_sites = _select_sites(strict_mode=True)

        if len(protonation_sites) < max_protons:
            logger.debug("  Falling back to relaxed protonation rules")
            protonation_sites = _select_sites(strict_mode=False)

        if len(protonation_sites) == max_protons:
            site_set = set(protonation_sites)
            for idx in parent_indices:
                if idx in site_set:
                    site_proton_counts[idx] = 1

    # Cp-like rings
    if "eta5(Cp)" in g.haptic_type:
        _assign_protonation_sites(1)

    elif "eta6(benzene)" in g.haptic_type:
        pass

    elif "CHT" in g.haptic_type:  # can be anion or cation
        pass

    elif "COT" in g.haptic_type:
        _assign_protonation_sites(2)

    elif "pentalene" in g.haptic_type:
        _assign_protonation_sites(2)

    # As5 / Pentaphosphole (substitution dependent)
    # e.g. GOCSID, VENNEH
    elif "eta5(As5)" in g.haptic_type:
        issubstituted = False
        for idx in parent_indices:
            a = ligand.atoms[idx]
            if a.metal_contact_count == 1:
                for adj in a.neighbor_indices:
                    if ligand.get_parent("molecule").labels[adj] != "As":
                        issubstituted = True
        _assign_protonation_sites(0 if issubstituted else 1)

    # e.g. IMUCAX
    elif "eta5(P5)" in g.haptic_type:
        issubstituted = False
        for idx in parent_indices:
            a = ligand.atoms[idx]
            if a.metal_contact_count == 1:
                for adj in a.neighbor_indices:
                    if ligand.get_parent("molecule").labels[adj] != "P":
                        issubstituted = True
        _assign_protonation_sites(0 if issubstituted else 1)

    elif "eta3(C3)" in g.haptic_type:
        _assign_protonation_sites(1)
    # Other hapticities
    else:
        if ligand.formula == "H8-C8" or (
            g.topology["is_single_simple_ring"] and g.topology["ring_sizes"][0] == 8
        ):
            logger.debug("  Special case: cyclooctatetraene detected")
        else:
            logger.debug(
                "  Unrecognized haptic type: %s Ligand: %s",
                g.haptic_type,
                ligand.formula,
            )
            logger.debug("  Topology analysis: %s", g.topology)

            molecule = ligand.get_parent("molecule")

            adjacency_dict = {}
            metal_adjacency_dict = {}

            for idx in parent_indices:
                atom = ligand.atoms[idx]

                adjacency_dict[idx] = [
                    molecule.atoms[adj]
                    for adj in atom.neighbor_indices
                    if adj not in atom.metal_contact_indices
                ]

                metal_adjacency_dict[idx] = [
                    molecule.atoms[adj] for adj in atom.metal_contact_indices
                ]

            for idx in parent_indices:
                atom = ligand.atoms[idx]
                atom_label = (
                    f"{atom.label} ({atom.atom_site_label})"
                    if atom.atom_site_label
                    else atom.label
                )
                if atom.label != "C":
                    logger.debug(
                        "  Non-carbon atom (%d) %s is not protonated in haptic group.",
                        idx,
                        atom_label,
                    )
                else:
                    neighbor_coords = [atom.coord for atom in adjacency_dict[idx]]
                    neighbor_labels = [atom.label for atom in adjacency_dict[idx]]

                    _, report, num_missing_h = detect_missing_carbon_hydrogens(
                        atom.coord,
                        neighbor_coords,
                        neighbor_labels,
                    )
                    if num_missing_h > 0:
                        needs_combinatorial = True
                        combinatorial_indices.append(idx)
                        logger.debug(
                            "  Needing combinatorial protonation for atom %d (%s): %s",
                            idx,
                            atom_label,
                            report,
                        )
                    else:
                        logger.debug(
                            "  No missing hydrogens detected for atom %d (%s). %s",
                            idx,
                            atom_label,
                            report,
                        )
    return ProtonationGroupResult(
        site_proton_counts=site_proton_counts,
        ligand_donor_electrons=ligand_donor_electrons,
        needs_combinatorial=needs_combinatorial,
        combinatorial_indices=combinatorial_indices,
    )


def _handle_non_haptic_group(
    ligand,
    g,
    parent_indices,
    moiety_atoms: set[int] | None = None,
) -> ProtonationGroupResult:
    """Return non-haptic protonation changes, excluding moiety atoms; no global mutation."""
    moiety_atoms = moiety_atoms or set()

    site_proton_counts: Dict[int, int] = {}
    ligand_donor_electrons: Dict[int, int] = {}
    needs_combinatorial = False
    combinatorial_indices: List[int] = []

    logger.debug("        HANDLE_NON_HAPTIC_GROUP: %s", g.formula)
    logger.debug("        parent_indices (ligand): %s", parent_indices)

    for idx in parent_indices:
        a = ligand.atoms[idx]
        atom_label = a.label + (f" ({a.atom_site_label})" if a.atom_site_label else "")
        logger.debug(
            "        HANDLE_NON_HAPTIC: idx=%d, label=%s, neighbor_count=%d, metal_contact_count=%d",
            idx,
            atom_label,
            a.neighbor_count,
            a.metal_contact_count,
        )

        # Skip moiety oxygens whose charges are already fixed.
        if idx in moiety_atoms:
            logger.debug(
                "        Atom %s (idx %d) belongs to a charged moiety; "
                "its charge is fixed there, so it is not protonated.",
                atom_label,
                idx,
            )
            continue

        adj_labels = []
        for adj in a.neighbor_indices:
            if adj not in a.metal_contact_indices:
                adj_labels.append(ligand.get_parent("molecule").labels[adj])

        # Simple ionic cases
        if a.label in HALOGENS:
            if a.neighbor_count == 0:
                site_proton_counts[idx] = 1

        # Oxygen
        elif a.label == "O":
            if len(adj_labels) == 1:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # Sulfur / Selenium
        elif a.label in {"S", "Se"}:
            if len(adj_labels) == 1:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # Hydrogen (handle manually)
        elif a.label == "H":
            if len(adj_labels) >= 1:
                pass

        # Nitrogen
        elif a.label == "N":
            if len(adj_labels) >= 3:
                pass
            elif len(adj_labels) == 2:
                if adj_labels.count("N") == 2:
                    site_proton_counts[idx] = 1

                else:
                    # Pyridine N is neutral; other ring compositions remain combinatorial sites.
                    graph = nx.from_numpy_array(ligand.adjacency_matrix.astype(float))
                    rings = nx.minimum_cycle_basis(graph)

                    def _is_pyridine_ring(ring) -> bool:
                        if len(ring) != 6:
                            return False
                        ring_labels = [ligand.labels[j] for j in ring]
                        return (
                            ring_labels.count("N") == 1 and ring_labels.count("C") == 5
                        )

                    in_six_ring = any(
                        idx in ring and _is_pyridine_ring(ring) for ring in rings
                    )
                    in_five_ring = any(idx in ring and len(ring) == 5 for ring in rings)
                    if in_six_ring:
                        pass  # pyridine-like
                        logger.debug(
                            "Ligand formula: %s, Atom site label: %s, Adjacency labels: %s, Type: %s",
                            ligand.formula,
                            a.atom_site_label,
                            adj_labels,
                            "pyridine-like",
                        )
                    elif in_five_ring:
                        logger.debug(
                            "Ligand formula: %s, Atom site label: %s, Adjacency labels: %s, Type: %s",
                            ligand.formula,
                            a.atom_site_label,
                            adj_labels,
                            "pyrrolide-type",
                        )
                        needs_combinatorial = True
                        combinatorial_indices.append(idx)
                    else:
                        needs_combinatorial = True
                        combinatorial_indices.append(idx)
            elif len(adj_labels) == 1:
                # Nitrosyl ligand (handle manually)
                needs_combinatorial = True
                combinatorial_indices.append(idx)

            else:  # only N atom in the ligand
                pass
        # Phosphorus
        elif a.label == "P":
            if len(adj_labels) >= 3:
                pass
            else:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # Carbon
        elif a.label == "C":
            if ligand.formula in {"C-N", "C-P", "C-As", "C-Sb"}:
                site_proton_counts[idx] = 1

            elif ligand.formula in {"C-O", "C-S", "C-Se", "C-Te"}:
                pass
            else:
                numN = adj_labels.count("N")
                numO = adj_labels.count("O")
                numH = adj_labels.count("H")
                numC = adj_labels.count("C")

                if len(adj_labels) == 1:
                    if numN == 1:
                        pass
                    else:
                        needs_combinatorial = True
                        combinatorial_indices.append(idx)
                elif len(adj_labels) == 2:
                    if numH == 2 and ligand.formula == "H2-C":
                        site_proton_counts[idx] = 2
                    else:
                        graph = nx.from_numpy_array(
                            ligand.adjacency_matrix.astype(float)
                        )
                        cycles = nx.cycle_basis(graph)
                        in_cycles = [c for c in cycles if idx in c]

                        if len(in_cycles) == 1:
                            if numN == 2 and len(in_cycles[0]) == 5:
                                logger.debug(
                                    "Ligand formula: %s, Atom site label: %s, Adjacency labels: %s, Type: %s",
                                    ligand.formula,
                                    a.atom_site_label,
                                    adj_labels,
                                    "possible NHC",
                                )
                                site_proton_counts[idx] = 2
                                ligand_donor_electrons[idx] = 2
                            elif numC == 2:
                                needs_combinatorial = True
                                combinatorial_indices.append(idx)
                            elif (numO == 1 and numC == 1) or (numN == 1 and numC == 1):
                                site_proton_counts[idx] = 2
                                ligand_donor_electrons[idx] = 2
                                needs_combinatorial = True
                                combinatorial_indices.append(idx)
                            else:
                                needs_combinatorial = True
                                combinatorial_indices.append(idx)
                        else:
                            site_proton_counts[idx] = 2
                            ligand_donor_electrons[idx] = 2
                            needs_combinatorial = True
                            combinatorial_indices.append(idx)
                else:
                    needs_combinatorial = True
                    combinatorial_indices.append(idx)

        # Silicon
        elif a.label == "Si":
            if len(adj_labels) == 1:
                site_proton_counts[idx] = 3
                ligand_donor_electrons[idx] = 2
            elif len(adj_labels) == 2:
                graph = nx.from_numpy_array(ligand.adjacency_matrix.astype(float))
                cycles = nx.cycle_basis(graph)
                in_cycles = [c for c in cycles if idx in c]

                if len(in_cycles) == 1:
                    site_proton_counts[idx] = 2
                    ligand_donor_electrons[idx] = 2
                else:
                    needs_combinatorial = True
                    combinatorial_indices.append(idx)
            else:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # Boron
        elif a.label == "B":
            if len(adj_labels) < 3:
                site_proton_counts[idx] = 1
            else:
                pass

        # Fallback
        else:
            atomic_num = elemdatabase.elementnr[a.label]
            min_valence = min(atomic_valence[atomic_num], default=0)

            if len(adj_labels) < min_valence:
                needs_combinatorial = True
                combinatorial_indices.append(idx)
                logger.debug(
                    "Atom %s (atomic number %d) has %d non-metal neighbors, "
                    "below the minimum valence of %d. "
                    "Combinatorial protonation is required.",
                    a.label,
                    atomic_num,
                    len(adj_labels),
                    min_valence,
                )
            else:
                logger.debug(
                    "Atom %s (atomic number %d) has %d non-metal neighbors, "
                    "which satisfies the minimum valence of %d. "
                    "No combinatorial protonation is required.",
                    a.label,
                    atomic_num,
                    len(adj_labels),
                    min_valence,
                )

    return ProtonationGroupResult(
        site_proton_counts=site_proton_counts,
        ligand_donor_electrons=ligand_donor_electrons,
        needs_combinatorial=needs_combinatorial,
        combinatorial_indices=combinatorial_indices,
    )
