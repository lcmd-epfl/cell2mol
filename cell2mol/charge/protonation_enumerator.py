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
    """Enumerate possible protonation states for the specie.

    Protonation states are only generated for:
    - ligands
    - non-complex molecules
    For all other species, returns None.
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
    # A detected porphyrin-family macrocycle is dispatched by difficulty:
    #
    #   * Clean closed-form porphyrin (k = 4 with every metal donor inside a
    #     core) -- a classic porphyrin / corrole / phthalocyanine, or a fully
    #     covered fused bis-porphyrin -- is the ONLY family handled
    #     automatically, via the closed-form free-base builder.
    #
    #   * Everything harder is deliberately declined here: an expanded
    #     macrocycle (k >= 5: penta-/hexa-/octaphyrin, whose free-base N-H count
    #     is oxidation-level dependent) or a k = 4 core that also binds a metal
    #     through donors outside it (e.g. EFISEV, furan-fused). These emit only the
    #     as-is protonation state and set specie.protonation_warning, flagging
    #     the charge result for manual review. The automatic generators for
    #     these cases are parked in _experimental_macrocycle_protonation.
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
        uncovered = _metal_donors_outside_tetrapyrrole_core(specie, macrocycles)
        # Clean closed-form k=4 (incl. fully-covered bis-porphyrin): auto-handle.
        if not is_expanded and not uncovered:
            return _generate_porphyrin_protonation_states(specie, macrocycles)
        # Expanded k>=5, or fused/ring-modified k=4: as-is state + warning flag.
        if is_expanded:
            reason = "expanded porphyrin (k>=5, oxidation-level-dependent free base)"
            specie.protonation_warning = reason
        else:
            uncovered_labels = [
                (specie.atoms or [])[i].atom_site_label or (specie.atoms or [])[i].label
                for i in uncovered
            ]
            reason = (
                f"fused/ring-modified k=4 core with {len(uncovered)} metal "
                f"donor(s) beyond the tetrapyrrole core {uncovered_labels}"
            )
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
            # Remember which metal each combinatorial site coordinates, so a
            # tetrapyrrolic ligand whose macrocycle the strict detector missed
            # (a fused / ring-modified bis-porphyrin such as EHOMUL) can still
            # be grouped into per-metal N4 pockets below.
            # group_metals = list(g.metals or [])
            # for idx in result.combinatorial_indices:
            #     if len(group_metals) == 1:
            #         nonlocal_site_metal[idx] = group_metals[0]

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
            reason = (
                f"too many combinatorial protonation states even all-or-nothing "
                f"per class ({n_all_or_nothing} > {max_combinations})"
            )
            return _warn_and_return(specie, reason, return_asis=True)

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

    # Per environment class, ``count_choices`` (built above) says HOW MANY of
    # its equivalent atoms may get a proton. Because the atoms in a class are
    # symmetry-equivalent, protonating any c of them yields the same structure,
    # so a single representative subset (the first c) is emitted per count --
    # e.g. two equivalent sites [0, 1] give exactly three states: none, one (of
    # the pair), both.
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
    """Return the specie's structure as-is, wrapped in a single Protonation.

    No hydrogens are added (``mode="none"``, ``n_protons_added=0``, zero
    ``site_proton_counts`` / ``ligand_donor_electrons``): the labels, coords and
    adjacency are exactly the specie's own. It is not a chemical protonation --
    it exists only so charge-state enumeration always has a Protonation to read
    (adjacency, per-site counts, donor electrons) even when nothing is added.

    Used for species that must not be protonated but still need a charge state:
    non-complex molecules, manual-charge formulas, fullerenes, ligands
    coordinating to alkali/alkaline-earth metals, and macrocycles declined by
    the porphyrin engine.
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
    """How many of an environment class's equivalent sites may be protonated.

    A class of one or two sites enumerates every count (0..k). Mono-protonation
    of a two-site class is real chemistry: acetylacetonate's two equivalent
    oxygens share ONE proton, so dropping its nH=1 state would lose the enol.

    From three equivalent sites up the partial counts are refused, and only
    "none" or "all" are offered. The class exists precisely because the atoms
    are indistinguishable, so protonating some-but-not-all of them invents an
    asymmetry the structure does not show.
    """
    if len(cls) >= ALL_OR_NOTHING_CLASS_SIZE:
        return (0, len(cls))
    return range(len(cls) + 1)


def _environment_classes(ligand: "Ligand", indices: list[int]) -> list[list[int]]:
    """Partition ``indices`` (ligand atom indices) into topological-equivalence
    classes by Weisfeiler-Lehman colour refinement on the element-labelled
    ligand graph. Two atoms share a class if they stay indistinguishable under
    iterated hashing of their neighbour labels, letting the combinatorial
    protonation treat a class as one all-or-nothing site rather than enumerating
    every per-atom combination (2**#classes states instead of 2**#atoms).

    Classes are returned sorted by their smallest atom index; each lists its
    atom indices sorted. ``indices`` need not be unique.
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


def _metal_donors_outside_tetrapyrrole_core(
    specie: Specie,
    macrocycles: list[tuple[list[int], bool, list[int]]],
) -> list[int]:
    """Metal-bound donors of ``specie`` (any element) that the ring-nitrogen
    free-base model can't represent -- i.e. every metal donor that is not a
    ring nitrogen of a detected tetrapyrrole core (indices into ``specie.atoms``).

    This covers both a donor lying entirely outside the cores (a pendant or
    second-pocket donor) and an in-core donor that isn't a ring nitrogen (a
    metal-bound meso or N-confused carbon). A non-empty result means the
    porphyrin fast-path misses a binding mode, forcing fall-through to the
    general engine. Ring nitrogens are pooled over ALL macrocycles, so a
    bis-corrole's two sets of metal-bound nitrogens are both covered.
    """
    core: set[int] = set()
    core_nitrogens: set[int] = set()
    for nitrogens, _is_contracted, core_atoms in macrocycles:
        core.update(core_atoms)
        core_nitrogens.update(nitrogens)

    # A metal donor is "uncovered" if it lies outside every core, OR is an
    # in-core atom that is not a ring nitrogen (a metal-bound non-N core atom
    # the ring-nitrogen free-base model can't represent). core_nitrogens is the
    # union over ALL macrocycles -- so, e.g., a bis-corrole's two sets of ring
    # nitrogens are both covered.
    outside_donors = [
        idx
        for idx, atom in enumerate(specie.atoms or [])
        if (atom.mconnec or 0) > 0 and (idx not in core or idx not in core_nitrogens)
    ]
    return outside_donors


def _generate_porphyrin_protonation_states(
    specie: Specie,
    macrocycles: list[tuple[list[int], bool, list[int]]],
) -> list[Protonation]:
    """Candidate protonation states for a specie with one or more pyrrolic
    macrocycles, protonating each ring to its neutral free-base tautomer
    (alternating ring N-H). Protons added = sum over rings:

    - Classic N4 porphyrin/phthalocyanine (meso-bridged): 2 N-H (trans pair;
      the 4-H dication is disabled).
    - Ring-contracted k=4 (one direct link): two states -- a corrole
      (aromatic, 3 N-H, trianionic free base) and a corrin (saturated,
      1 N-H, monoanionic free base). The two are ambiguous from connectivity.
    - Expanded porphyrin (k>=5): m0 = alternating N-H (3 for a hexaphyrin).

    Classic N4 families emit exactly this one free-base count (bis-porphyrin ->
    4). A ring-contracted k=4 emits both the corrole and corrin counts, and an
    expanded porphyrin has an oxidation-level-dependent count ([26]hexaphyrin=3,
    [28]=4, ...) so three states m0-1/m0/m0+1 are emitted -- in each case the
    charge/metal-balance step picks, with invalid parities dropped downstream.
    Replaces the general combinatorial search (too slow on large macrocycles).
    Falls back to the as-is state if no macrocycle nitrogens are found.
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

    # Skip ring nitrogens that already carry an H (e.g. an N-confused pyrrole
    # N-H); adding another would build a spurious [NH2+].
    adjmat = np.asarray(specie.adjmat)

    def _already_has_h(idx: int) -> bool:
        return any(specie.labels[j] == "H" for j in np.nonzero(adjmat[idx])[0])

    base_add = [n for n in base_sites if not _already_has_h(n)]
    extra_bare = [
        n for n in all_nitrogens if n not in base_sites and not _already_has_h(n)
    ]

    # Classic N4: just m0. Expanded: also bracket m0 +/- 1. Ring-contracted
    # k=4: base_add already holds the corrole 3 N-H (trianionic free base);
    # also emit the corrin 1 N-H (monoanionic free base). The two are hard to
    # tell apart from connectivity alone, so both counts are offered and the
    # charge/metal-balance step keeps whichever is valid.
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
    """Atom indices whose formal charge ``_find_charged_moiety`` already fixes.

    These are the terminal oxygens of a carboxylate / carbonate / sulfinate /
    sulfonate / sulfate. Such an oxygen is NOT a protonation site: the group is
    already a valid closed-shell anion, so bond perception needs no hydrogen on
    it, and its charge is supplied directly by the moiety scan.

    Excluding them is what keeps a polycarboxylate tractable -- a ligand with n
    carboxylates would otherwise open 2n combinatorial sites and blow up the
    protonation enumeration, which is the very cost the moiety shortcut in
    ``get_candidate_charges`` exists to avoid. It also keeps the two mechanisms
    from double-counting: no proton ever lands on a moiety atom, so every
    protonation site is an anionic centre the moiety sum did NOT account for.

    A genuine ``-C(=O)OH`` is unaffected -- its hydroxyl oxygen has two
    non-metal neighbours (C and H), so it is not terminal and the scan never
    reports it.
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
    """
    Handle protonation rules for non-haptic ligand groups.

    ``moiety_atoms`` are atoms whose charge is already fixed by
    ``_find_charged_moiety`` (carboxylate/sulfonate oxygens); they are skipped
    as protonation sites.

    No global state is mutated.
    All intended changes are returned via ProtonationGroupResult.
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
                    # A coordinating N that lies on a 6-membered ring is a
                    # neutral pyridine-type donor (no proton, no combinatorial
                    # site). Accept membership in ANY minimal 6-ring, so the N
                    # of a fused / bridged / substituted pyridine system
                    # (quinoline, phenanthroline, bipyridine, naphthyridine,
                    # ...) qualifies too -- not just a bare pyridine ligand.
                    # The size-6 test is kept to exclude an N on a 5-membered
                    # ring (pyrrolide-type), which is an anionic donor handled
                    # combinatorially instead.
                    graph = nx.from_numpy_array(ligand.adjmat.astype(float))
                    rings = nx.minimum_cycle_basis(graph)
                    in_six_ring = any(idx in ring and len(ring) == 6 for ring in rings)
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
