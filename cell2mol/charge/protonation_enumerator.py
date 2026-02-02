import numpy as np
import itertools
import networkx as nx
from cell2mol.connectivity import add_atom
from cell2mol.element_utils import get_post_transition_metal_idxs, get_metalloid_idxs
from dataclasses import dataclass
from typing import Dict, List
from cell2mol.classes.protonation import Protonation
from cell2mol.charge.utils import FULLERENES, MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.hydrogen import detect_missing_hydrogens
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProtonationGroupResult:
    addedlist: Dict[int, int]
    block: List[int]
    elemlist: Dict[int, str]
    metal_electrons: Dict[int, int]
    pos_carbenes: Dict[int, int]
    needs_nonlocal: bool
    non_local_indices: List[int]


def enumerate_protonation_states(specie: object) -> list[Protonation]:
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
            return get_empty_protonation_state(specie)
        else:
            return None

    if specie.formula in MANUAL_CHARGE_ASSIGN_SPECIES or specie.formula in FULLERENES:
        return get_empty_protonation_state(specie)

    num_post_tm = len(get_post_transition_metal_idxs(specie.labels))
    num_metalloids = len(get_metalloid_idxs(specie.labels))
    if (num_post_tm + num_metalloids) == specie.natoms:
        logger.info(
            "Specie %s (%s) consists only of metalloids/post-transition metals. ",
            specie.formula,
            specie.subtype,
        )
        logger.info("Skipping protonation enumeration.")
        return None
    # return get_empty_protonation_state(specie)

    if specie.subtype == "ligand":
        parent = specie.get_parent("molecule")
        if parent.has_ia_iia and not parent.iscomplex:
            return get_empty_protonation_state(specie)

    # ============================================================
    # From here on: ligand protonation engine
    # ============================================================
    ligand = specie
    protonation_states: list = []

    natoms = ligand.natoms
    newlab = ligand.labels.copy()
    newcoord = ligand.coord.copy()
    # ============================================================
    # Initialization
    # ============================================================
    added_atoms = 0
    addedlist = np.zeros(natoms, dtype=int)
    block = np.zeros(natoms, dtype=int)
    elemlist = np.empty(natoms, dtype=str)
    metal_electrons = np.zeros(natoms, dtype=int)
    pos_carbenes = np.zeros(natoms, dtype=int)

    non_local_groups_indices: list[int] = []
    reset_H_indices: list[int] = []

    limit_of_nonlocal_sites = 4  # Arbitrary limit to avoid combinatorial explosion

    logger.info("Processing %s (%s):", specie.formula, specie.subtype)

    # ============================================================
    # GROUP-LEVEL ANALYSIS
    # ============================================================
    for g in ligand.groups:
        parent_indices = g.get_parent_indices("ligand")

        # --------------------------------------------------
        # Dispatch to helper
        # --------------------------------------------------
        if g.is_haptic:  # HAPTIC GROUPS
            result = _handle_haptic_group(ligand, g, parent_indices)
        else:  # NON-HAPTIC GROUPS
            result = _handle_non_haptic_group(ligand, g, parent_indices)
        # --------------------------------------------------
        # Merge results (THIS IS THE IMPORTANT PART)
        # --------------------------------------------------
        for idx, val in result.addedlist.items():
            addedlist[idx] += val

        for idx in result.block:
            block[idx] = 1

        for idx, elem in result.elemlist.items():
            elemlist[idx] = elem

        for idx, val in result.metal_electrons.items():
            metal_electrons[idx] += val

        for idx, val in result.pos_carbenes.items():
            pos_carbenes[idx] = val

        if result.needs_nonlocal:
            non_local_groups_indices.extend(result.non_local_indices)

    # ============================================================
    # Check non_local_groups_indices for decision
    # ============================================================
    logger.debug("    non_local_groups_indices: %s", non_local_groups_indices)
    if len(non_local_groups_indices) > limit_of_nonlocal_sites:
        logger.info(
            "  %d non-local protonation sites detected (more than the limit of %d). ",
            len(non_local_groups_indices),
            limit_of_nonlocal_sites,
        )
        combinations = list(
            itertools.product([0, 1], repeat=len(non_local_groups_indices))
        )
        logger.info("  Total combinations to evaluate: %d. ", len(combinations))
        # logger.info("  Skipping protonation enumeration for %s.", specie.formula)
        logger.info("  Generating empty protonation state only for %s.", specie.formula)
        return get_empty_protonation_state(specie)
    # ============================================================
    # LOCAL ATOM ADDITION
    # ============================================================
    for idx, a in enumerate(ligand.atoms):
        if (addedlist[idx] - block[idx]) <= 0:
            continue

        if addedlist[idx] == 1:
            logger.debug(
                "    Single addition for atom %s%s (group index: %d)",
                a.label,
                f" ({a.atom_site_label})" if a.atom_site_label else "",
                idx,
            )
            isadded, newlab, newcoord = add_atom(
                newlab, newcoord, idx, ligand, elemlist[idx], unconditional=True
            )
        elif addedlist[idx] > 1:
            if pos_carbenes[idx]:
                logger.debug(
                    "    Position %s (%s) %d is a carbene site, added_list %d block %d",
                    a.label,
                    f" ({a.atom_site_label})" if a.atom_site_label else "",
                    idx,
                    addedlist[idx],
                    block[idx],
                )
            isadded, newlab, newcoord = add_atom(
                newlab, newcoord, idx, ligand, elemlist[idx], unconditional=True
            )
        # else:
        #     logger.debug(
        #         "    Multiple additions (%d) for atom index %d",
        #         (addedlist[idx] - block[idx]),
        #         idx,
        #     )
        #     if pos_carbenes[idx]:
        #         reset_H_indices.extend(
        #             list(
        #                 range(len(newlab), len(newlab) + (addedlist[idx] - block[idx]))
        #             )
        #         )
        #     isadded, newlab, newcoord = add_hydrogens(
        #         newlab, newcoord, idx, ligand, (addedlist[idx] - block[idx])
        #     )

        if isadded:
            added_atoms += 1
            block[idx] = 1
        else:
            addedlist[idx] = 0
            block[idx] = 1

    # ============================================================
    # LOCAL PROTONATION
    # ============================================================
    if not non_local_groups_indices:
        protonation_states.append(
            Protonation.from_positional(
                newlab,
                newcoord,
                ligand.cov_factor,
                added_atoms,
                addedlist,
                block,
                metal_electrons,
                elemlist,
                parent=specie,
            )
        )
        return protonation_states

    # ============================================================
    # NON-LOCAL PROTONATION
    # ============================================================
    local_labels = newlab.copy()
    local_coords = newcoord.copy()
    local_addedlist = addedlist.copy()
    local_block = block.copy()
    local_added_atoms = added_atoms

    combinations = list(itertools.product([0, 1], repeat=len(non_local_groups_indices)))
    combinations.sort(key=sum)

    for com in combinations:
        newlab = local_labels.copy()
        newcoord = local_coords.copy()
        addedlist = local_addedlist.copy()
        block = local_block.copy()
        added_atoms = local_added_atoms
        elemlist = np.empty(len(newlab), dtype=str)
        metal_electrons = np.zeros(len(newlab), dtype=int)

        for flag, idx in zip(com, non_local_groups_indices):
            if flag == 1:
                elemlist[idx] = "H"
                addedlist[idx] = 1
                isadded, newlab, newcoord = add_atom(
                    newlab, newcoord, idx, ligand, "H", unconditional=True
                )
                if isadded:
                    added_atoms += 1

        prot = Protonation.from_positional(
            newlab,
            newcoord,
            ligand.cov_factor,
            added_atoms,
            addedlist,
            block,
            metal_electrons,
            elemlist,
            o_s=sum(com),
            typ="Non-local",
            parent=specie,
        )

        if prot.status:
            protonation_states.append(prot)

    return protonation_states


def get_empty_protonation_state(specie: object) -> list[Protonation]:
    """
    Create a placeholder protonation state with no added hydrogens.

    This "empty" protonation state does NOT represent a chemical
    protonation. It is created solely as a preprocessing step for
    charge-state enumeration, where a Protonation object is required
    even when no atoms are added.
    """
    logger.debug(
        "Creating empty protonation placeholder for %s (%s)",
        specie.formula,
        specie.subtype,
    )

    natoms = len(specie.labels)

    addedlist = [0] * natoms
    block = [0] * natoms
    metal_electrons = [0] * natoms
    elemlist = np.empty(natoms, dtype=str)

    empty_protonation = Protonation.from_positional(
        specie.labels,
        specie.coord,
        specie.cov_factor,
        0,
        addedlist,
        block,
        metal_electrons,
        elemlist,
        typ="Empty",
        parent=specie,
    )

    return [empty_protonation]


def _handle_haptic_group(ligand, g, parent_indices) -> ProtonationGroupResult:
    """
    Handle protonation rules for haptic ligand groups.

    This function does NOT mutate global state.
    It returns a ProtonationGroupResult describing intended changes.
    """

    addedlist: Dict[int, int] = {}
    block: List[int] = []
    elemlist: Dict[int, str] = {}
    metal_electrons: Dict[int, int] = {}
    pos_carbenes: Dict[int, int] = {}
    needs_nonlocal = False
    non_local_indices: List[int] = []

    selected = False

    logger.debug("        HANDLE_HAPTIC_GROUP: %s %s", g.formula, g.haptic_type)
    logger.debug("        parent_indices: %s", parent_indices)

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

        adjacency_dict = {}
        metal_adjacency_dict = {}

        # ---------- Build adjacency maps ----------
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
                    elif nTot == 2 and nC == 2 and nH == 0:  # C2H0
                        priority = 1
                    elif nTot == 3 and nC == 2 and nH == 1:  # C2H1
                        priority = 2
                    else:
                        continue

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
                    addedlist[idx] = addedlist.get(idx, 0) + 1
                    elemlist[idx] = "H"
                else:
                    block.append(idx)

    # --------------------------------------------------
    # Cp-like rings
    # --------------------------------------------------
    if "eta5(Cp)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)

    elif "eta6(benzene)" in g.haptic_type and not selected:
        selected = True
        for idx in parent_indices:
            block.append(idx)

    elif "CHT" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(0)

    elif "COT" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(2)

    elif "pentalene" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(2)

    # --------------------------------------------------
    # As5 / Pentaphosphole (substitution dependent)
    # --------------------------------------------------
    # e.g. GOCSID
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

    # --------------------------------------------------
    # Other hapticities
    # --------------------------------------------------
    else:
        if (
            g.topology["is_single_simple_ring"] and g.ring_sizes[0] == 8
        ) or ligand.formula == "H8-C8":
            logger.debug("  Special case: cyclooctatetraene detected")
            _assign_protonation_sites(2)
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
                neighbor_coords = [atom.coord for atom in adjacency_dict[idx]]
                neighbor_labels = [atom.label for atom in adjacency_dict[idx]]

                missing_h_detected, report, num_missing_h = detect_missing_hydrogens(
                    atom.atnum,
                    atom.coord,
                    neighbor_coords,
                    neighbor_labels,
                )
                if num_missing_h > 0:
                    needs_nonlocal = True
                    non_local_indices.append(idx)
                    logger.debug(
                        "  Needing non-local protonation for atom %d (%s): %s",
                        idx,
                        atom.atom_site_label,
                        report,
                    )
                    # logger.debug(
                    #     "  Missing H detected on atom %d (%s): %s",
                    #     idx,
                    #     atom.atom_site_label,
                    #     report,
                    # )
                    # addedlist[idx] = 1
                    # elemlist[idx] = "H"
                else:
                    block.append(idx)

    # elif "eta2(C2)" in g.haptic_type and not selected:
    #     selected = True
    #     for idx in parent_indices:
    #         if ligand.atoms[idx].mconnec == 1:
    #             block.append(idx)

    # elif "eta3(C3)" in g.haptic_type and not selected:
    #     selected = True
    # _assign_protonation_sites_middle(1)
    # _assign_protonation_sites_eta3_carbons(1)
    # _assign_protonation_sites(0)
    # non_local
    # elif "eta4(C4)" in g.haptic_type and not selected:
    #     selected = True
    #     for idx in parent_indices:
    #         if ligand.atoms[idx].mconnec == 1:
    #             block.append(idx)

    # elif "eta5(C5)" in g.haptic_type and not selected:
    #     selected = True
    #     _assign_protonation_sites(1)

    # elif "eta6(C6)" in g.haptic_type and not selected:
    #     selected = True
    #     for idx in parent_indices:
    #         if ligand.atoms[idx].mconnec == 1:
    #             block.append(idx)

    # elif "eta7(C7)" in g.haptic_type and not selected:
    #     selected = True
    #     _assign_protonation_sites(1)

    # elif "eta8(C8)" in g.haptic_type and not selected:
    #     selected = True
    #     _assign_protonation_sites(1)
    # --------------------------------------------------
    # Fallback: unrecognized hapticity
    # --------------------------------------------------
    # elif not selected:
    #     if "eta5(C4-N)" in g.haptic_type:
    #         _assign_protonation_sites(1)

    # logger.info("Haptic group not recognized. %s", g.haptic_type)

    return ProtonationGroupResult(
        addedlist=addedlist,
        block=block,
        elemlist=elemlist,
        metal_electrons=metal_electrons,
        pos_carbenes=pos_carbenes,
        needs_nonlocal=needs_nonlocal,
        non_local_indices=non_local_indices,
    )


def _handle_non_haptic_group(
    ligand,
    g,
    parent_indices,
) -> ProtonationGroupResult:
    """
    Handle protonation rules for non-haptic ligand groups.

    No global state is mutated.
    All intended changes are returned via ProtonationGroupResult.
    """

    addedlist: Dict[int, int] = {}
    block: List[int] = []
    elemlist: Dict[int, str] = {}
    metal_electrons: Dict[int, int] = {}
    pos_carbenes: Dict[int, int] = {}
    needs_nonlocal = False
    non_local_indices: List[int] = []

    ions = {"F", "Cl", "Br", "I", "As"}

    logger.debug("        HANDLE_NON_HAPTIC_GROUP: %s", g.formula)
    logger.debug("        parent_indices: %s", parent_indices)

    for idx in parent_indices:
        a = ligand.atoms[idx]

        logger.debug(
            "        HANDLE_NON_HAPTIC: idx=%d, label=%s%s, connec=%d, mconnec=%d",
            idx,
            a.label,
            f", atom_site_label={a.atom_site_label}" if a.atom_site_label else "",
            a.connec,
            a.mconnec,
        )
        # -----------------------------------------
        # Collect non-metal adjacent atom labels
        # -----------------------------------------
        adj_labels = []
        for adj in a.adjacency:
            if adj not in a.metal_adjacency:
                adj_labels.append(ligand.get_parent("molecule").labels[adj])

        # -----------------------------------------
        # Simple ionic cases
        # -----------------------------------------
        if a.label in ions:
            if a.connec == 0:
                addedlist[idx] = 1
                elemlist[idx] = "H"
            else:
                block.append(idx)

        # -----------------------------------------
        # Oxygen
        # -----------------------------------------
        elif a.label == "O":
            if a.connec == 2 and len(adj_labels) == 1:
                needs_nonlocal = True
                non_local_indices.append(idx)
            else:
                block.append(idx)

        # -----------------------------------------
        # Sulfur / Selenium
        # -----------------------------------------
        elif a.label in {"S", "Se"}:
            if a.connec == 1:
                addedlist[idx] = 1
                elemlist[idx] = "H"
            elif a.connec == 2 and len(adj_labels) == 1:
                needs_nonlocal = True
                non_local_indices.append(idx)
            else:
                block.append(idx)

        # -----------------------------------------
        # Hydrides (handle manually)
        # -----------------------------------------
        # elif a.label == "H":
        #     if len(adj_labels) <= 1:
        #         addedlist[idx] = 1
        #         elemlist[idx] = "Cl"
        #     else:
        #         block.append(idx)

        # -----------------------------------------
        # Nitrogen
        # -----------------------------------------
        elif a.label == "N":
            if ligand.natoms == 2 and ligand.is_nitrosyl:
                if ligand.NO_type == "Linear":
                    addedlist[idx] = 1
                    elemlist[idx] = "O"
                    metal_electrons[idx] = 1
                else:  # Bent
                    addedlist[idx] = 1
                    elemlist[idx] = "H"
            else:
                if len(adj_labels) >= 3:
                    block.append(idx)
                elif adj_labels.count("N") == 2:
                    addedlist[idx] = 1
                    elemlist[idx] = "H"
                else:
                    G = nx.from_numpy_array(ligand.adjmat.astype(float))
                    cycles = nx.cycle_basis(G)
                    in_cycles = [c for c in cycles if idx in c]

                    if len(in_cycles) == 1 and len(in_cycles[0]) == 6:
                        block.append(idx)
                    else:
                        needs_nonlocal = True
                        non_local_indices.append(idx)

        # -----------------------------------------
        # Phosphorus
        # -----------------------------------------
        elif a.label == "P":
            if len(adj_labels) >= 3:
                block.append(idx)
            elif len(adj_labels) == 1:
                if adj_labels[0] in {"N", "C"}:
                    block.append(idx)
                elif adj_labels[0] == "P":
                    addedlist[idx] = 1
                    elemlist[idx] = "H"
                else:
                    needs_nonlocal = True
                    non_local_indices.append(idx)
            else:
                needs_nonlocal = True
                non_local_indices.append(idx)

        # -----------------------------------------
        # Carbon
        # -----------------------------------------
        elif a.label == "C":
            if ligand.formula in {"C-N", "C-P", "C-As", "C-Sb"}:
                addedlist[idx] = 1
                elemlist[idx] = "H"

            elif ligand.formula in {"C-O", "C-S", "C-Se", "C-Te"}:
                block.append(idx)

            else:
                numN = adj_labels.count("N")
                numO = adj_labels.count("O")
                numH = adj_labels.count("H")
                numC = adj_labels.count("C")

                if len(adj_labels) == 1:
                    addedlist[idx] = 1
                    elemlist[idx] = "H"

                elif len(adj_labels) == 2:
                    if numN == 1 and numO == 1:  # amide
                        addedlist[idx] = 1
                        elemlist[idx] = "H"
                    elif numH == 2 and ligand.formula == "H2-C":
                        addedlist[idx] = 2
                        elemlist[idx] = "H"
                    else:
                        G = nx.from_numpy_array(ligand.adjmat.astype(float))
                        cycles = nx.cycle_basis(G)
                        in_cycles = [c for c in cycles if idx in c]

                        if len(in_cycles) == 1:
                            if numN == 2:
                                addedlist[idx] = 2
                                elemlist[idx] = "H"
                                metal_electrons[idx] = 2
                            elif numC == 2:
                                needs_nonlocal = True
                                non_local_indices.append(idx)
                            elif (numO == 1 and numC == 1) or (numN == 1 and numC == 1):
                                pos_carbenes[idx] = 1
                                addedlist[idx] = 2
                                elemlist[idx] = "H"
                                metal_electrons[idx] = 2
                                needs_nonlocal = True
                                non_local_indices.append(idx)
                            else:
                                needs_nonlocal = True
                                non_local_indices.append(idx)
                        else:
                            pos_carbenes[idx] = 1
                            addedlist[idx] = 2
                            elemlist[idx] = "H"
                            metal_electrons[idx] = 2
                            needs_nonlocal = True
                            non_local_indices.append(idx)

                else:
                    needs_nonlocal = True
                    non_local_indices.append(idx)

        # -----------------------------------------
        # Silicon
        # -----------------------------------------
        elif a.label == "Si":
            if len(adj_labels) == 1:
                addedlist[idx] = 3
                elemlist[idx] = "H"
                metal_electrons[idx] = 2
            elif len(adj_labels) == 2:
                G = nx.from_numpy_array(ligand.adjmat.astype(float))
                cycles = nx.cycle_basis(G)
                in_cycles = [c for c in cycles if idx in c]

                if len(in_cycles) == 1:
                    addedlist[idx] = 2
                    elemlist[idx] = "H"
                    metal_electrons[idx] = 2
                else:
                    needs_nonlocal = True
                    non_local_indices.append(idx)
            else:
                needs_nonlocal = True
                non_local_indices.append(idx)

        # -----------------------------------------
        # Boron
        # -----------------------------------------
        elif a.label == "B":
            if len(adj_labels) < 4:
                addedlist[idx] = 1
                elemlist[idx] = "H"
            else:
                block.append(idx)

        # -----------------------------------------
        # Fallback
        # -----------------------------------------
        else:
            needs_nonlocal = True
            non_local_indices.append(idx)

    return ProtonationGroupResult(
        addedlist=addedlist,
        block=block,
        elemlist=elemlist,
        metal_electrons=metal_electrons,
        pos_carbenes=pos_carbenes,
        needs_nonlocal=needs_nonlocal,
        non_local_indices=non_local_indices,
    )
