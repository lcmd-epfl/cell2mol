from __future__ import annotations

import numpy as np
import itertools
import networkx as nx
from cell2mol.connectivity import add_atom
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING, cast
from cell2mol.classes.protonation import Protonation
from cell2mol.charge.utils import MANUAL_CHARGE_ASSIGN_SPECIES, HALOGENS
from cell2mol.charge.special_cases import (
    check_fullerene_sphericity,
    find_all_porphyrin_macrocycles,
    porphyrin_reference_protonation_sites,
    _find_charged_moiety,
)
from cell2mol.charge.xyz2mol import atomic_valence
from cell2mol.hydrogen import detect_missing_hydrogens, add_hydrogens
import logging
from cell2mol.elementdata import ElementData

elemdatabase = ElementData()

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.ligand import Ligand

logger = logging.getLogger(__name__)


@dataclass
class ProtonationGroupResult:
    site_proton_counts: Dict[int, int]
    ligand_donor_electrons: Dict[int, int]
    needs_combinatorial: bool
    combinatorial_indices: List[int]


def enumerate_protonation_states(specie: Specie) -> list[Protonation] | None:
    """Protonation states for a ligand or non-complex molecule; None for anything
    else.
    """
    # ============================================================
    # Applicability guards
    # ============================================================
    if specie.type != "specie":
        return None

    if specie.subtype == "group":
        return None

    if specie.subtype == "molecule":
        if specie.is_non_complex_molecule:
            return get_asis_protonation_state(specie)
        else:
            logger.info(
                "Specie %s (%s) is a complex molecule. Do not protonate.",
                specie.formula,
                specie.subtype,
            )
            return None

    if specie.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        return get_asis_protonation_state(specie)

    has_fullerene = (
        specie.has_fullerene
        if specie.has_fullerene is not None
        else specie.evaluate_has_fullerene()
    )

    if has_fullerene:
        if specie.coord is not None and not check_fullerene_sphericity(specie.coord):
            logger.warning(
                "%s passed fullerene topology check but failed sphericity "
                "check - possible disorder/AC artifact; proceeding anyway",
                specie.formula,
            )
        logger.debug("Fullerene cage detected for %s", specie.formula)
        return get_asis_protonation_state(specie)

    # Porphyrin/phthalocyanine/corrole/corrin N4 macrocycle: enumerate the
    # protonation states relevant to a metal-coordinated tetrapyrrolic
    # ligand (see _generate_porphyrin_protonation_states).
    has_porphyrin = (
        specie.has_porphyrin
        if specie.has_porphyrin is not None
        else specie.evaluate_has_porphyrin()
    )
    # Only a clean k = 4 macrocycle with every metal donor inside a core is built
    # automatically. Anything harder -- expanded (k >= 5), or outside donors --
    # emits the as-is state and sets protonation_warning for manual review.
    if has_porphyrin:
        macrocycles = find_all_porphyrin_macrocycles(
            specie.get_atomic_numbers(), specie.adjmat
        )
        is_expanded = any(len(mac_n) >= 5 for mac_n, _c, _core in macrocycles)
        logger.debug(
            "%d porphyrin-family macrocycle(s) detected for %s "
            "(expanded: %s, contracted: %s)",
            len(macrocycles),
            specie.formula,
            is_expanded,
            [is_contracted for _n, is_contracted, _c in macrocycles],
        )
        # Closed-form k=4: build it. The generator's site filter drops core nitrogens
        # that cannot take a proton, and a donor that is not a ring nitrogen at all
        # (an axial phosphine, say) is never a candidate site -- so it says nothing
        # about the free base and must not divert the ligand to the general engine.
        if not is_expanded:
            return _generate_porphyrin_protonation_states(specie, macrocycles)
        # Expanded k>=5, but its metals may cut it into ordinary N4 pockets.
        pockets = _metal_pocket_macrocycles(specie)
        if pockets:
            logger.info(
                "%s: expanded macrocycle resolves into %d metal-bound N4 "
                "pocket(s) %s; protonating a trans pair in each.",
                specie.formula,
                len(pockets),
                [pocket for _n, _c, pocket in pockets],
            )
            return _generate_porphyrin_protonation_states(specie, pockets)
        # Expanded k>=5 with no pockets: as-is state + warning flag.
        reason = "expanded porphyrin (k>=5, oxidation-level-dependent free base)"
        specie.protonation_warning = reason
        # return _warn_and_return(specie, reason)

    if specie.subtype == "ligand":
        parent = cast("Specie", specie.get_parent("molecule"))
        if parent is not None and parent.has_ia_iia and not parent.iscomplex:
            return get_asis_protonation_state(specie)

    # ============================================================
    # From here on: ligand protonation engine
    # ============================================================
    ligand = cast("Ligand", specie)
    protonation_states: list[Protonation] = []

    newlab = ligand.labels.copy()
    newcoord = ligand.coord.copy()
    # ============================================================
    # Initialization
    # ============================================================
    n_protons_added = 0
    site_proton_counts = np.zeros(ligand.natoms, dtype=int)
    ligand_donor_electrons = np.zeros(ligand.natoms, dtype=int)
    combinatorial_site_indices: list[int] = []
    # nonlocal_site_metal: dict[int, "Metal"] = {}
    process_both_modes: list[int] = []
    protonated_indices_to_reset: list[int] = []  # old : reset_H_indices

    max_combinations = 16
    logger.info("Processing %s (%s):", specie.formula, specie.subtype)

    # ============================================================
    # GROUP-LEVEL ANALYSIS
    # ============================================================
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
    # Oxygens belonging to a carboxylate / sulfonate / ... already have their
    # charge fixed by _find_charged_moiety, so they are not protonation sites
    # (see _charged_moiety_atoms).
    moiety_atoms = _charged_moiety_atoms(ligand)

    for g in groups:
        parent_indices = g.get_parent_indices("ligand")

        # --------------------------------------------------
        # Dispatch to helper
        # --------------------------------------------------
        if g.is_haptic:  # HAPTIC GROUPS
            result = _handle_haptic_group(ligand, g, parent_indices)
        else:  # NON-HAPTIC GROUPS
            result = _handle_non_haptic_group(ligand, g, parent_indices, moiety_atoms)

        # --------------------------------------------------
        # Merge results (THIS IS THE IMPORTANT PART)
        # --------------------------------------------------
        for idx, val in result.site_proton_counts.items():
            site_proton_counts[idx] += val

        for idx, val in result.ligand_donor_electrons.items():
            ligand_donor_electrons[idx] += val

        if result.needs_combinatorial:
            combinatorial_site_indices.extend(result.combinatorial_indices)
    # ============================================================
    # Check combinatorial_site_indices for decision
    # ============================================================
    logger.debug("    combinatorial_site_indices: %s", combinatorial_site_indices)
    # Collapse symmetry-equivalent coordinating atoms into one all-or-nothing
    # site each, so a symmetric ligand
    site_classes = _environment_classes(ligand, combinatorial_site_indices)
    if combinatorial_site_indices:
        logger.debug(
            "    grouped %d combinatorial site(s) into %d environment class(es): %s",
            len(combinatorial_site_indices),
            len(site_classes),
            site_classes,
        )
    # A small class of k equivalent atoms contributes k+1 protonation options
    # (0..k of them protonated); a large one is all-or-nothing (see
    # _class_count_choices). The total number of combinatorial states is the
    # product of the per-class option counts.
    count_choices = [_class_count_choices(cls) for cls in site_classes]
    n_combinations = 1
    for choices in count_choices:
        n_combinations *= len(choices)

    # When the full 0..k-per-class product is over the limit, fall back to
    # all-or-nothing per class: each class is protonated either fully (all its
    # equivalent atoms) or not at all, giving 2**(#classes) states instead of
    # the product of (|class|+1).
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
                return _warn_and_return(specie, reason, return_asis=True)
            site_classes = merged

    # ============================================================
    # DETERMINISTIC ATOM ADDITION
    # ============================================================
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
            _, newlab, newcoord = add_atom(
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

    # ============================================================
    # Deterministic protonation
    # ============================================================
    is_fully_deterministic = len(combinatorial_site_indices) == 0
    force_deterministic_mode = process_both_modes

    if is_fully_deterministic or force_deterministic_mode:
        protonation_states.append(
            Protonation.from_positional(
                labels=newlab,
                coord=np.asarray(newcoord),
                cov_factor=ligand.cov_factor,
                n_protons_added=n_protons_added,
                site_proton_counts=site_proton_counts.tolist(),
                ligand_donor_electrons=ligand_donor_electrons.tolist(),
                mode="deterministic",
                parent=ligand,
            )
        )
        if not force_deterministic_mode:
            return protonation_states

    # ============================================================
    # Combinatorial protonation
    # ============================================================
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

    # Atoms in a class are symmetry-equivalent, so protonating any c of them gives
    # the same structure: one representative subset (the first c) per count. After
    # a count-only merge they are only alike, so the subset is one of several.
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
                _, newlab, newcoord = add_atom(
                    newlab,
                    list(newcoord),
                    idx,
                    ligand,
                    element="H",
                    unconditional=True,
                )
        prot = Protonation.from_positional(
            labels=newlab,
            coord=np.asarray(newcoord),
            cov_factor=ligand.cov_factor,
            n_protons_added=n_protons_added,
            site_proton_counts=site_proton_counts.tolist(),
            ligand_donor_electrons=ligand_donor_electrons.tolist(),
            mode="combinatorial",
            parent=ligand,
        )

        if prot.status:
            protonation_states.append(prot)

    return protonation_states


def get_asis_protonation_state(specie: Specie) -> list[Protonation]:
    """The specie's own structure wrapped in a single Protonation, nothing added. Not
    a chemical protonation -- it exists so enumeration always has a Protonation to
    read. Used for species that must not be protonated but still need a charge.
    """
    logger.debug(
        "Creating as-is (no-proton) protonation for %s (%s)",
        specie.formula,
        specie.subtype,
    )

    asis_protonation = Protonation.from_positional(
        labels=specie.labels,
        coord=specie.coord,
        cov_factor=specie.cov_factor,
        n_protons_added=0,
        site_proton_counts=[0] * len(specie.labels),
        ligand_donor_electrons=[0] * len(specie.labels),
        mode="none",
        parent=specie,
    )

    return [asis_protonation]


def _warn_and_return(
    specie: Specie, reason: str, return_asis: bool = False
) -> list[Protonation] | None:
    """Decline to enumerate protonation for a hard cases: log a
    warning, record ``reason`` on ``specie.protonation_warning`` so the charge
    result is flagged for review, and return None. If ``return_asis`` is True,
    return a single as-is protonation state instead of None.
    """
    logger.warning("%s: %s -- not auto-handled", specie.formula, reason)
    specie.protonation_warning = reason
    if return_asis:
        logger.info(
            "Returning as-is protonation state and setting protonation_warning."
        )
        return get_asis_protonation_state(specie)
    logger.info("Returning None protonation state and setting protonation_warning.")
    return None


# From this many symmetry-equivalent sites up, a class is protonated
# all-or-nothing rather than 0..k. See _class_count_choices.
ALL_OR_NOTHING_CLASS_SIZE = 3


def _class_count_choices(cls: list[int]):
    """How many of a class's equivalent sites may be protonated. One or two sites
    enumerate every count (acetylacetonate's two oxygens share ONE proton, the
    enol); three or more are all-or-nothing, since protonating some-but-not-all
    would invent an asymmetry the structure does not show.
    """
    if len(cls) >= ALL_OR_NOTHING_CLASS_SIZE:
        return (0, len(cls))
    return range(len(cls) + 1)


def _donor_signature(ligand: "Ligand", idx: int) -> tuple:
    """Element, metals bridged and substituents -- what makes two donors alike."""
    a = ligand.atoms[idx]
    mol_labels = ligand.get_parent("molecule").labels
    metal_adj = set(a.metal_adjacency)
    non_metal = sorted(mol_labels[j] for j in a.adjacency if j not in metal_adj)
    return (a.label, a.mconnec, tuple(non_metal))


def _merge_by_donor_signature(
    ligand: "Ligand", site_classes: list[list[int]]
) -> list[list[int]]:
    """Collapse classes sharing a donor signature, keeping only their endpoints exact."""
    merged: dict[tuple, list[int]] = {}
    for cls in site_classes:
        merged.setdefault(_donor_signature(ligand, cls[0]), []).extend(cls)
    return sorted((sorted(cls) for cls in merged.values()), key=lambda cls: cls[0])


def _environment_classes(ligand: "Ligand", indices: list[int]) -> list[list[int]]:
    """Partition ``indices`` into topological-equivalence classes by Weisfeiler-Lehman
    colour refinement, so protonation treats a class as one all-or-nothing site
    (2**#classes states instead of 2**#atoms). Sorted by smallest atom index.
    """
    unique = sorted(set(indices))
    if not unique:
        return []

    adj = np.asarray(ligand.adjmat)
    n = adj.shape[0]
    neighbours = [np.nonzero(adj[i])[0].tolist() for i in range(n)]

    # Seed each atom's colour with an integer rank of its element label (kept
    # int throughout so signatures stay comparable), then refine until the
    # partition stops splitting (at most n rounds).
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


def _order_pocket_trans(metal_coord, nitrogens: list[int], coords) -> list[int]:
    """Order four donors so 0/2 and 1/3 are the trans pairs: of the three pairings,
    the one minimising the intra-pair dot products of the metal->N vectors.
    """
    units = {}
    for idx in nitrogens:
        vec = np.asarray(coords[idx], float) - np.asarray(metal_coord, float)
        norm = np.linalg.norm(vec)
        units[idx] = vec / norm if norm > 0 else vec

    a, b, c, d = nitrogens
    pairings = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    (p0, p1), (p2, p3) = min(
        pairings,
        key=lambda pairing: sum(float(np.dot(units[p], units[q])) for p, q in pairing),
    )
    return [p0, p2, p1, p3]


def _metal_pocket_macrocycles(
    specie: Specie,
) -> list[tuple[list[int], bool, list[int]]]:
    """One N4 pocket per metal, as pseudo-macrocycles for the k=4 builder: an expanded
    ring has one free base per pocket, not one overall. Declines unless every metal
    takes exactly four unsubstituted nitrogens.
    """
    molecule = specie.get_parent("molecule")
    atoms = specie.atoms or []
    if molecule is None or not atoms:
        return []

    adjmat = np.asarray(specie.adjmat)
    pockets: dict[int, list[int]] = {}
    for idx, atom in enumerate(atoms):
        for metal_idx in atom.metal_adjacency or []:
            heavy = sum(
                1 for j in np.nonzero(adjmat[idx])[0] if specie.labels[j] != "H"
            )
            if atom.label != "N" or heavy > 2:
                return []
            pockets.setdefault(metal_idx, []).append(idx)

    if not pockets or any(len(nitrogens) != 4 for nitrogens in pockets.values()):
        return []

    return [
        (
            _order_pocket_trans(
                molecule.atoms[metal_idx].coord, nitrogens, specie.coord
            ),
            False,
            sorted(nitrogens),
        )
        for metal_idx, nitrogens in pockets.items()
    ]


def _generate_porphyrin_protonation_states(
    specie: Specie,
    macrocycles: list[tuple[list[int], bool, list[int]]],
) -> list[Protonation]:
    """Free-base protonation states for pyrrolic macrocycles: 2 N-H for a classic N4
    (trans pair), both corrole (3) and corrin (1) for a ring-contracted k=4 since
    connectivity cannot tell them apart, and m0-1/m0/m0+1 for an expanded ring
    whose count is oxidation-level dependent. Falls back to the as-is state.
    """
    asis_state = get_asis_protonation_state(specie)[0]

    # Baseline free-base sites (m0): alternating N-H per ring.
    all_nitrogens: list[int] = []
    base_sites: list[int] = []
    is_expanded = False
    is_contracted = False
    for macrocycle_nitrogens, is_contracted_ring, _core_atoms in macrocycles:
        if macrocycle_nitrogens is None or len(macrocycle_nitrogens) < 4:
            continue
        all_nitrogens.extend(macrocycle_nitrogens)
        base_sites.extend(
            porphyrin_reference_protonation_sites(
                macrocycle_nitrogens, is_contracted_ring
            )
        )
        if len(macrocycle_nitrogens) >= 5:
            is_expanded = True
        elif is_contracted_ring:
            is_contracted = True

    # Preserve insertion order while de-duplicating.
    base_sites = list(dict.fromkeys(base_sites))

    if not base_sites:
        return [asis_state]

    # A free-base site must be a ring nitrogen that binds the metal and still has its
    # lone pair: one already carrying an H (N-confused pyrrole) or a substituent
    # (N-alkylated porphyrin) would give a spurious [NH2+] or a four-bonded N that
    # will not kekulize. This also sets the count -- an N-alkyl occupies one of the
    # two trans positions, so that macrocycle takes a single proton, not two.
    adjmat = np.asarray(specie.adjmat)
    atoms = specie.atoms or []

    def _can_be_protonated(idx: int) -> bool:
        neighbours = np.nonzero(adjmat[idx])[0]
        if (atoms[idx].mconnec or 0) == 0:
            return False
        if any(specie.labels[j] == "H" for j in neighbours):
            return False
        # A pyrrole-type N has exactly two ring carbons; a third heavy
        # neighbour is an exocyclic substituent.
        return sum(1 for j in neighbours if specie.labels[j] != "H") <= 2

    base_add = [n for n in base_sites if _can_be_protonated(n)]
    extra_bare = [
        n for n in all_nitrogens if n not in base_sites and _can_be_protonated(n)
    ]

    # Classic N4: m0 only. Expanded: bracket m0 +/- 1. Ring-contracted k=4: emit
    # both corrole (3 N-H) and corrin (1 N-H); connectivity cannot tell them apart.
    site_sets: list[list[int]] = [base_add]
    if is_expanded:
        if extra_bare:
            site_sets.append(base_add + [extra_bare[0]])  # m0 + 1
        if len(base_add) > 1:
            site_sets.append(base_add[:-1])  # m0 - 1
    elif is_contracted and len(base_add) >= 1:
        site_sets.append(base_add[:1])  # corrin: 1 N-H

    states: list[Protonation] = []
    for sites in site_sets:
        newlab = list(specie.labels)
        newcoord = list(specie.coord)
        for site in sites:
            _, newlab, newcoord = add_atom(
                newlab, newcoord, site, specie, element="H", unconditional=True
            )

        site_proton_counts = [0] * len(specie.labels)
        for site in sites:
            site_proton_counts[site] = 1

        states.append(
            Protonation.from_positional(
                labels=newlab,
                coord=np.asarray(newcoord),
                cov_factor=specie.cov_factor,
                n_protons_added=len(sites),
                site_proton_counts=site_proton_counts,
                mode="porphyrin",
                parent=specie,
                ligand_donor_electrons=[0] * len(specie.labels),
            )
        )

    return states


def _handle_haptic_group(ligand, g, parent_indices) -> ProtonationGroupResult:
    """
    Handle protonation rules for haptic ligand groups.

    This function does NOT mutate global state.
    It returns a ProtonationGroupResult describing intended changes.
    """

    site_proton_counts: Dict[int, int] = {}
    ligand_donor_electrons: Dict[int, int] = {}
    needs_combinatorial = False
    combinatorial_indices: List[int] = []

    selected = False

    logger.debug("        HANDLE_HAPTIC_GROUP: %s %s", g.formula, g.haptic_type)
    logger.debug("        parent_indices (ligand): %s", parent_indices)

    for idx in parent_indices:
        a = ligand.atoms[idx]

        logger.debug(
            "        HANDLE_HAPTIC: idx=%d, label=%s %s, connec=%d, mconnec=%d",
            idx,
            a.label,
            f", atom_site_label={a.atom_site_label}" if a.atom_site_label else "",
            a.connec,
            a.mconnec,
        )

    # --------------------------------------------------
    # Helper: add up to N hydrogens on parent_indices
    # --------------------------------------------------

    def _assign_protonation_sites(max_protons: int):
        molecule = ligand.get_parent("molecule")

        # ---------- Build adjacency maps ----------
        adjacency_dict = {}
        metal_adjacency_dict = {}
        for idx in parent_indices:
            atom = ligand.atoms[idx]
            adjacency_dict[idx] = [
                molecule.labels[adj]
                for adj in atom.adjacency
                if adj not in atom.metal_adjacency
            ]
            metal_adjacency_dict[idx] = [
                molecule.labels[adj] for adj in atom.metal_adjacency
            ]

        def _select_sites(strict_mode: bool):
            """Select protonation sites with chemical priority."""
            protonation_sites = []
            added = 0
            skip_cnt = 0

            # ================= STRICT MODE =================
            if strict_mode:
                candidates = []

                # --- gather and score candidates ---
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
                # --- sort by chemical priority ---
                candidates.sort(key=lambda x: x[0])

                # --- select respecting spacing rule ---
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

            # ================= FALLBACK MODE =================
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

        # ---------- First pass (strict chemistry) ----------
        protonation_sites = _select_sites(strict_mode=True)

        # ---------- Fallback pass ----------
        if len(protonation_sites) < max_protons:
            logger.debug("  Falling back to relaxed protonation rules")
            protonation_sites = _select_sites(strict_mode=False)

        # ---------- Apply results ----------
        if len(protonation_sites) == max_protons:
            site_set = set(protonation_sites)
            for idx in parent_indices:
                if idx in site_set:
                    site_proton_counts[idx] = 1

    # --------------------------------------------------
    # Cp-like rings
    # --------------------------------------------------
    if "eta5(Cp)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)

    elif "eta6(benzene)" in g.haptic_type and not selected:
        selected = True

    elif "CHT" in g.haptic_type and not selected:  # can be anion or cation
        selected = True

    elif "COT" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(2)

    elif "pentalene" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(2)

    # --------------------------------------------------
    # As5 / Pentaphosphole (substitution dependent)
    # --------------------------------------------------
    # e.g. GOCSID, VENNEH
    elif "eta5(As5)" in g.haptic_type and not selected:
        selected = True
        issubstituted = False
        for idx in parent_indices:
            a = ligand.atoms[idx]
            if a.mconnec == 1:
                for adj in a.adjacency:
                    if ligand.get_parent("molecule").labels[adj] != "As":
                        issubstituted = True
        _assign_protonation_sites(0 if issubstituted else 1)

    # e.g. IMUCAX
    elif "eta5(P5)" in g.haptic_type and not selected:
        selected = True
        issubstituted = False
        for idx in parent_indices:
            a = ligand.atoms[idx]
            if a.mconnec == 1:
                for adj in a.adjacency:
                    if ligand.get_parent("molecule").labels[adj] != "P":
                        issubstituted = True
        _assign_protonation_sites(0 if issubstituted else 1)

    elif "eta3(C3)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)
    # --------------------------------------------------
    # Other hapticities
    # --------------------------------------------------
    else:
        if ligand.formula == "H8-C8" or (
            g.topology["is_single_simple_ring"] and g.topology["ring_sizes"][0] == 8
        ):
            logger.debug("  Special case: cyclooctatetraene detected")
            # _assign_protonation_sites(2)
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

            # ---------- Build adjacency maps ----------
            for idx in parent_indices:
                atom = ligand.atoms[idx]

                adjacency_dict[idx] = [
                    molecule.atoms[adj]
                    for adj in atom.adjacency
                    if adj not in atom.metal_adjacency
                ]

                metal_adjacency_dict[idx] = [
                    molecule.atoms[adj] for adj in atom.metal_adjacency
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

                    missing_h_detected, report, num_missing_h = (
                        detect_missing_hydrogens(
                            atom.atnum,
                            atom.coord,
                            neighbor_coords,
                            neighbor_labels,
                        )
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


def _charged_moiety_atoms(specie) -> set[int]:
    """Atoms whose charge ``_find_charged_moiety`` already fixes -- a carboxylate or
    sulfonate's terminal oxygens. Not protonation sites: the group is already a
    valid closed-shell anion, and excluding them keeps a polycarboxylate tractable
    and stops the two mechanisms double-counting.
    """
    try:
        return {
            idx
            for _centre, atom_locals, _kind, _nc in _find_charged_moiety(specie)
            for idx in atom_locals
        }
    except Exception as exc:  # detection is an optimisation, never a hard gate
        logger.debug("Charged-moiety detection failed for %s: %s", specie.formula, exc)
        return set()


def _handle_non_haptic_group(
    ligand,
    g,
    parent_indices,
    moiety_atoms: set[int] | None = None,
) -> ProtonationGroupResult:
    """Protonation rules for a non-haptic group. ``moiety_atoms`` are skipped as
    sites. Mutates no global state; all changes come back in the result.
    """
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
            "        HANDLE_NON_HAPTIC: idx=%d, label=%s, connec=%d, mconnec=%d",
            idx,
            atom_label,
            a.connec,
            a.mconnec,
        )

        # A carboxylate/sulfonate oxygen is already a closed-shell anion whose
        # charge _find_charged_moiety supplies, so it is not a protonation site.
        if idx in moiety_atoms:
            logger.debug(
                "        Atom %s (idx %d) belongs to a charged moiety; "
                "its charge is fixed there, so it is not protonated.",
                atom_label,
                idx,
            )
            continue

        # -----------------------------------------
        # Collect non-metal adjacent atom labels
        # -----------------------------------------
        adj_labels = []
        metal_adj_labels = []
        for adj in a.adjacency:
            if adj not in a.metal_adjacency:
                adj_labels.append(ligand.get_parent("molecule").labels[adj])
            else:
                metal_adj_labels.append(ligand.get_parent("molecule").labels[adj])

        # -----------------------------------------
        # Simple ionic cases
        # -----------------------------------------
        if a.label in HALOGENS:
            if a.connec == 0:
                site_proton_counts[idx] = 1

        # -----------------------------------------
        # Oxygen
        # -----------------------------------------
        elif a.label == "O":
            if len(adj_labels) == 1:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # -----------------------------------------
        # Sulfur / Selenium
        # -----------------------------------------
        elif a.label in {"S", "Se"}:
            if len(adj_labels) == 1:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # -----------------------------------------
        # Hydrides (handle manually)
        # -----------------------------------------
        elif a.label == "H":
            if len(adj_labels) >= 1:
                pass

        # -----------------------------------------
        # Nitrogen
        # -----------------------------------------
        elif a.label == "N":
            if len(adj_labels) >= 3:
                pass
            elif len(adj_labels) == 2:
                if adj_labels.count("N") == 2:
                    site_proton_counts[idx] = 1

                else:
                    # A coordinating N on a pyridine ring is a neutral donor -- no
                    # proton, no combinatorial site. Ring size 6 alone is not enough:
                    # pyridine is one N and five carbons, so a diazine or an O/S-
                    # containing 6-ring is a different donor and has to fall through
                    # to the combinatorial engine rather than be silently skipped.
                    # Size 6 also excludes 5-ring (pyrrolide) N, which is anionic.
                    graph = nx.from_numpy_array(ligand.adjmat.astype(float))
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
        # -----------------------------------------
        # Phosphorus
        # -----------------------------------------
        elif a.label == "P":
            if len(adj_labels) >= 3:
                pass
            else:
                needs_combinatorial = True
                combinatorial_indices.append(idx)

        # -----------------------------------------
        # Carbon
        # -----------------------------------------
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
                        # site_proton_counts[idx] = 1
                        needs_combinatorial = True
                        combinatorial_indices.append(idx)
                elif len(adj_labels) == 2:
                    # if numN == 1 and numO == 1:  # amide  # exception: FIQHIA
                    #     site_proton_counts[idx] = 1
                    if numH == 2 and ligand.formula == "H2-C":
                        site_proton_counts[idx] = 2
                    else:
                        graph = nx.from_numpy_array(ligand.adjmat.astype(float))
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

        # -----------------------------------------
        # Silicon
        # -----------------------------------------
        elif a.label == "Si":
            if len(adj_labels) == 1:
                site_proton_counts[idx] = 3
                ligand_donor_electrons[idx] = 2
            elif len(adj_labels) == 2:
                graph = nx.from_numpy_array(ligand.adjmat.astype(float))
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

        # -----------------------------------------
        # Boron
        # -----------------------------------------
        elif a.label == "B":
            if len(adj_labels) < 3:
                site_proton_counts[idx] = 1
            else:
                pass

        # -----------------------------------------
        # Fallback
        # -----------------------------------------
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
