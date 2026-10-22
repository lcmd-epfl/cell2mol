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

    # Aromatic ring charge belongs to the ring system, not to each metal's
    # group: a pentalene bound eta5/eta5, eta5/eta3 or eta5/eta2/eta2 is -2
    # either way. Charge each haptically bound all-carbon ring system once.
    ring_system_atoms: set[int] = set()
    for atoms, charge, label in _haptic_ring_systems(ligand, groups):
        counts = _protonate_ring(ligand, label, atoms, charge)
        for idx, val in counts.items():
            site_proton_counts[idx] += val
        ring_system_atoms.update(atoms)

    # An open conjugated chain of sp2 atoms bound to one metal (allyl, azaallyl,
    # azadienyl, pentadienyl) is a pi ligand even when its composition is not in
    # the haptic table. Its atoms keep their hydrogens; only a bare carbon is
    # a proton site. Groups that are subsets of such a chain are skipped.
    pi_chain_atoms: set[int] = set()
    for g in groups:
        parent_indices = g.get_parent_indices("ligand")
        if _is_conjugated_pi_chain(ligand, parent_indices):
            pi_chain_atoms.update(parent_indices)
            logger.debug(
                "  Group %s is an open conjugated pi chain %s; treated as haptic",
                g.formula,
                [ligand.atoms[i].atom_site_label for i in parent_indices],
            )

    for g in groups:
        parent_indices = g.get_parent_indices("ligand")

        if parent_indices and set(parent_indices) <= ring_system_atoms:
            logger.debug(
                "        Group %s (%s) lies in a charged ring system; "
                "handled at the ligand level",
                g.formula,
                g.haptic_type,
            )
            continue

        is_pi_chain = _is_conjugated_pi_chain(ligand, parent_indices)
        if (
            not is_pi_chain
            and not g.is_haptic
            and parent_indices
            and set(parent_indices) <= pi_chain_atoms
        ):
            logger.debug(
                "        Group %s lies in a conjugated pi chain; handled there",
                g.formula,
            )
            continue

        if g.is_haptic or is_pi_chain:  # HAPTIC GROUPS
            result = _handle_haptic_group(ligand, g, parent_indices)
        else:  # NON-HAPTIC GROUPS
            result = _handle_non_haptic_group(ligand, g, parent_indices, moiety_atoms)

        # An atom shared by several groups (a bridging donor) needs its protons
        # once, not once per metal.
        for idx, val in result.site_proton_counts.items():
            site_proton_counts[idx] = max(site_proton_counts[idx], val)

        for idx, val in result.ligand_donor_electrons.items():
            ligand_donor_electrons[idx] = max(ligand_donor_electrons[idx], val)

        if result.needs_combinatorial:
            combinatorial_site_indices.extend(result.combinatorial_indices)

    # A site one group protonated outright (a bare ring carbon) must not be
    # protonated a second time by another group's combinatorial sweep.
    # Sites with donated electrons keep both modes (process_both_modes).
    already_fixed = [
        idx
        for idx in combinatorial_site_indices
        if site_proton_counts[idx] > 0 and ligand_donor_electrons[idx] == 0
    ]
    if already_fixed:
        logger.debug(
            "    Sites %s are protonated deterministically by a ring rule; "
            "removing them from the combinatorial sweep",
            sorted(set(already_fixed)),
        )
        combinatorial_site_indices = [
            idx for idx in combinatorial_site_indices if idx not in already_fixed
        ]
    combinatorial_site_indices = list(dict.fromkeys(combinatorial_site_indices))
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

    ring_charge = _haptic_ring_charge(ligand, g, parent_indices)
    if ring_charge is not None:
        site_proton_counts.update(
            _protonate_ring(
                ligand, f"{g.formula} {g.haptic_type}", parent_indices, ring_charge
            )
        )

    # Other hapticities
    if ring_charge is None:
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


############################################################
# Ring rules for haptic groups
############################################################

# Protons that make a neutral, closed-shell frame of each aromatic ring: one
# per unit of ring charge. Rings listed at 0 are neutral but still get their
# bare carbons protonated.
_RING_CHARGE_BY_HAPTIC_TYPE = (
    ("eta5(Cp)", 1),
    ("eta6(benzene)", 0),
    ("CHT", 0),  # tropylium (+1) or cycloheptatrienyl anion (-1); left to the sweep
    ("eta8(COT)", 2),
    ("pentalene", 2),
    ("eta3(C3)", 1),
)


# Charge of an all-carbon ring system bound haptically, keyed by its fused ring
# sizes. Zero-charge systems are listed so their bare carbons still get protons.
_RING_SYSTEM_CHARGE = {
    (5,): 1,  # cyclopentadienyl
    (6,): 0,  # arene
    (7,): 0,  # cycloheptatrienyl: cation or anion, left to the sweep
    (8,): 2,  # cyclooctatetraene dianion; requires eta8(COT) on this ring
    (5, 5): 2,  # pentalene dianion
    (5, 6): 1,  # indenyl
    (5, 5, 6): 1,  # fluorenyl
    (6, 6): 0,  # naphthalene
}


def _haptic_ring_systems(ligand, groups) -> list[tuple[list[int], int, str]]:
    """All-carbon fused ring systems that carry a haptic group, with their charge.

    Return (ring atoms, charge magnitude, label) per system whose fused ring
    sizes are listed in _RING_SYSTEM_CHARGE. Other systems keep the per-group rules.
    """
    haptic_atoms = {
        idx for g in groups if g.is_haptic for idx in g.get_parent_indices("ligand")
    }
    if not haptic_atoms:
        return []

    adj = np.asarray(ligand.adjacency_matrix)
    heavy = [i for i, label in enumerate(ligand.labels) if label != "H"]
    graph = nx.Graph()
    graph.add_nodes_from(heavy)
    graph.add_edges_from((i, j) for i in heavy for j in heavy if i < j and adj[i][j])
    # Fusing rings: cycles that share an atom belong to one system.
    cycles = [set(c) for c in nx.minimum_cycle_basis(graph) if len(c) <= 8]
    fused = nx.Graph()
    fused.add_nodes_from(range(len(cycles)))
    fused.add_edges_from(
        (a, b)
        for a in range(len(cycles))
        for b in range(a + 1, len(cycles))
        if cycles[a] & cycles[b]
    )

    systems = []
    for component in nx.connected_components(fused):
        atoms = sorted(set().union(*(cycles[c] for c in component)))
        if not haptic_atoms & set(atoms):
            continue
        if any(ligand.labels[i] != "C" for i in atoms):
            continue
        sizes = tuple(sorted(len(cycles[c]) for c in component))
        # An eight-carbon ring can also be neutral cyclooctadiene. Only
        # explicit eta8(COT) coordination identifies the dianionic COT rule.
        if sizes == (8,) and not any(
            g.is_haptic
            and "eta8(COT)" in (g.haptic_type or "")
            and set(atoms) <= set(g.get_parent_indices("ligand"))
            for g in groups
        ):
            continue
        charge = _RING_SYSTEM_CHARGE.get(sizes)
        if charge is None:
            logger.debug(
                "  Ring system %s with ring sizes %s has no ring-charge rule; "
                "its groups are handled one by one",
                [ligand.atoms[i].atom_site_label for i in atoms],
                sizes,
            )
            continue
        label = f"ring system {sizes}"
        logger.debug(
            "  %s on %s: ring charge %d",
            label,
            [ligand.atoms[i].atom_site_label for i in atoms],
            charge,
        )
        systems.append((atoms, charge, label))
    return systems


def _is_conjugated_pi_chain(ligand, parent_indices) -> bool:
    """True for three or more connected sp2 C/N atoms with no ring: an open
    pi system (allyl, azaallyl, dienyl, azadienyl) bound side-on to one metal.
    """
    if len(parent_indices) < 3:
        return False
    if any(ligand.labels[i] not in {"C", "N"} for i in parent_indices):
        return False
    adj = np.asarray(ligand.adjacency_matrix)
    members = set(parent_indices)
    graph = nx.Graph()
    graph.add_nodes_from(parent_indices)
    graph.add_edges_from(
        (i, j) for i in parent_indices for j in parent_indices if i < j and adj[i][j]
    )
    if not nx.is_connected(graph) or not nx.is_tree(graph):
        return False
    # sp2: at most three non-metal neighbours (hydrogens included).
    return all(int(adj[i].sum()) <= 3 for i in parent_indices)


def _haptic_ring_charge(ligand, g, parent_indices) -> int | None:
    """Magnitude of the ring charge a haptic group carries, or None if no rule applies."""
    haptic_type = g.haptic_type or ""
    for key, charge in _RING_CHARGE_BY_HAPTIC_TYPE:
        if key in haptic_type:
            return charge

    # As5 / pentaphosphole-like rings: anionic only when unsubstituted (GOCSID, VENNEH).
    if "eta5(As5)" in haptic_type:
        molecule = ligand.get_parent("molecule")
        for idx in parent_indices:
            a = ligand.atoms[idx]
            if a.metal_contact_count == 1 and any(
                molecule.labels[adj] != "As" for adj in a.neighbor_indices
            ):
                return 0
        return 1
    return None


def _bare_ring_carbons(ligand, parent_indices) -> list[int]:
    """Group carbons with two ring neighbours and nothing else: no hydrogen, no
    substituent. A bridgehead of a fused system has three ring neighbours and
    is not bare.
    """
    adj = ligand.adjacency_matrix
    members = set(parent_indices)
    bare = []
    for idx in parent_indices:
        if ligand.labels[idx] != "C":
            continue
        neighbours = np.nonzero(adj[idx])[0].tolist()
        if len(neighbours) == 2 and all(n in members for n in neighbours):
            bare.append(idx)
    return bare


def _adjacent_bare_pairs(ligand, bare: list[int]) -> list[tuple[int, int]]:
    """Bonded pairs of bare carbons: an in-plane alkyne unit, neutral as a pair."""
    adj = ligand.adjacency_matrix
    graph = nx.Graph()
    graph.add_nodes_from(bare)
    graph.add_edges_from((i, j) for i in bare for j in bare if i < j and adj[i][j])
    return sorted(
        tuple(sorted(e)) for e in nx.max_weight_matching(graph, maxcardinality=True)
    )


def _protonate_ring(ligand, label, parent_indices, ring_charge: int) -> Dict[int, int]:
    """One proton per bare carbon, then `ring_charge` more on the ring, minus two
    per adjacent bare pair.

    The frame must stay drawable as a neutral closed shell: the sites are chosen
    so the remaining sp2 atoms of the group admit a perfect matching.
    """
    bare = _bare_ring_carbons(ligand, parent_indices)
    pairs = _adjacent_bare_pairs(ligand, bare)
    quota = max(ring_charge - 2 * len(pairs), 0)

    counts: Dict[int, int] = {idx: 1 for idx in bare}
    if bare:
        logger.debug(
            "  %s: bare ring carbon(s) %s get one proton each; %d adjacent "
            "pair(s) %s counted as a neutral alkyne; %d of %d ring proton(s) left",
            label,
            [ligand.atoms[i].atom_site_label for i in bare],
            len(pairs),
            [[ligand.atoms[i].atom_site_label for i in pair] for pair in pairs],
            quota,
            ring_charge,
        )

    sites = _select_ring_sites(ligand, parent_indices, quota, exclude=bare)
    if quota and len(sites) < quota:
        logger.debug(
            "  %s: could only place %d of %d ring proton(s)",
            label,
            len(sites),
            quota,
        )
    for idx in sites:
        counts[idx] = counts.get(idx, 0) + 1
    if sites:
        logger.debug(
            "  %s: ring proton(s) on %s",
            label,
            [ligand.atoms[i].atom_site_label for i in sites],
        )
    return counts


def _select_ring_sites(
    ligand, parent_indices, quota: int, exclude: list[int]
) -> list[int]:
    """Choose `quota` group atoms to protonate so the rest still pair into double bonds.

    Preference: a sigma-bound (kappa1) ring carbon first, since the carbanion
    belongs there even when it is substituted; then unsubstituted C-H carbons;
    then carbons bound to a single metal; then the most spread-out placement
    (1,4 on an eight-ring, not 1,2). Return fewer sites when nothing satisfies
    the matching requirement.
    """
    if quota <= 0:
        return []

    adj = ligand.adjacency_matrix
    sigma = {
        idx
        for g in (ligand.groups or [])
        if not g.is_haptic
        for idx in g.get_parent_indices("ligand")
    }
    members = [idx for idx in parent_indices if idx not in set(exclude)]
    graph = nx.Graph()
    graph.add_nodes_from(parent_indices)
    graph.add_edges_from(
        (i, j) for i in parent_indices for j in parent_indices if i < j and adj[i][j]
    )
    distance = dict(nx.all_pairs_shortest_path_length(graph))

    def substituted(idx: int) -> bool:
        neighbours = np.nonzero(adj[idx])[0].tolist()
        return not any(ligand.labels[n] == "H" for n in neighbours)

    def multi_metal(idx: int) -> bool:
        return (ligand.atoms[idx].metal_contact_count or 0) >= 2

    def unbound(idx: int) -> bool:
        return (ligand.atoms[idx].metal_contact_count or 0) == 0

    def leaves_kekule(chosen: tuple[int, ...]) -> bool:
        rest = [idx for idx in parent_indices if idx not in chosen]
        if len(rest) % 2:
            return False
        if not rest:
            return True
        sub = graph.subgraph(rest)
        matching = nx.max_weight_matching(sub, maxcardinality=True)
        return 2 * len(matching) == len(rest)

    def spread(chosen: tuple[int, ...]) -> int:
        return sum(distance[i][j] for i, j in itertools.combinations(chosen, 2))

    def key(chosen: tuple[int, ...]):
        return (
            sum(substituted(i) and i not in sigma for i in chosen),
            -sum(i in sigma for i in chosen),
            sum(unbound(i) for i in chosen),
            sum(multi_metal(i) for i in chosen),
            -spread(chosen),
            chosen,
        )

    max_candidates = 12  # C(12, 2) = 66 combinations, C(12, 3) = 220
    members = sorted(
        members,
        key=lambda i: (
            substituted(i) and i not in sigma,
            i not in sigma,
            unbound(i),
            multi_metal(i),
            i,
        ),
    )[:max_candidates]
    combos = sorted(itertools.combinations(members, quota), key=key)
    for chosen in combos:
        if leaves_kekule(chosen):
            return list(chosen)
    logger.debug(
        "  No %d-site protonation of %s leaves a Kekule structure; using the "
        "preferred placement anyway",
        quota,
        [ligand.atoms[i].atom_site_label for i in parent_indices],
    )
    return list(combos[0]) if combos else []


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
