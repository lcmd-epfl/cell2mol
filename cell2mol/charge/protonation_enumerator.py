import numpy as np
import itertools
import networkx as nx
from cell2mol.connectivity import add_atom
from cell2mol.element_utils import (
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    get_metalloid_idxs,
)
from dataclasses import dataclass
from typing import Dict, List

from cell2mol.hydrogen import add_hydrogens
from cell2mol.classes.protonation import Protonation
import logging

logger = logging.getLogger(__name__)

fullerene = ["C60", "C72", "C80"]
manual_assign = ["O4-Cl", "N3", "N2", "N-O", "I3", "I4", "I5", "I6"]


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

    if specie.formula in manual_assign or specie.formula in fullerene:
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

        ia_iia = get_alkali_alkaline_earth_metal_idxs(
            [metal.label for metal in g.metals]
        )

        # --------------------------------------------------------
        # Alkali / alkaline-earth only coordination
        # --------------------------------------------------------
        if len(ia_iia) == len(g.metals):
            for idx in parent_indices:
                block[idx] = 1
            continue

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
        if addedlist[idx] == 0 or block[idx] == 1:
            continue

        if addedlist[idx] == 1:
            isadded, newlab, newcoord = add_atom(
                newlab, newcoord, idx, ligand, elemlist[idx], unconditional=True
            )
        else:
            if pos_carbenes[idx]:
                reset_H_indices.extend(
                    list(range(len(newlab), len(newlab) + addedlist[idx]))
                )
            isadded, newlab, newcoord = add_hydrogens(
                newlab, newcoord, idx, ligand, addedlist[idx]
            )

        if isadded:
            added_atoms += addedlist[idx]
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

    # --------------------------------------------------
    # Helper: add up to N hydrogens on parent_indices
    # --------------------------------------------------
    def _assign_protonation_sites(max_protons: int):
        tmp = 0
        for idx in parent_indices:
            a = ligand.atoms[idx]
            if a.mconnec == 1:
                if tmp < max_protons:
                    addedlist[idx] = addedlist.get(idx, 0) + 1
                    elemlist[idx] = "H"
                    tmp += 1
                else:
                    block.append(idx)

    # --------------------------------------------------
    # Cp-like rings
    # --------------------------------------------------
    if "eta5(Cp)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)

    elif "eta7(C7)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)

    elif "eta8(C8)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)

    # --------------------------------------------------
    # AsCp / Pentaphosphole (substitution dependent)
    # --------------------------------------------------
    elif "eta5(AsCp)" in g.haptic_type and not selected:
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

    elif "eta3(Cp)" in g.haptic_type and not selected:
        selected = True
        _assign_protonation_sites(1)

    elif "eta4(C,C,C,C)" in g.haptic_type and not selected:
        selected = True
        for idx in parent_indices:
            if ligand.atoms[idx].mconnec == 1:
                block.append(idx)

    elif "eta2(C,C)" in g.haptic_type and not selected:
        selected = True
        for idx in parent_indices:
            if ligand.atoms[idx].mconnec == 1:
                block.append(idx)

    elif "eta4(C,C,C,O)" in g.haptic_type and not selected:
        selected = True
        for idx in parent_indices:
            if ligand.atoms[idx].mconnec == 1:
                block.append(idx)

    elif "eta6(C6)" in g.haptic_type and not selected:
        selected = True
        for idx in parent_indices:
            if ligand.atoms[idx].mconnec == 1:
                block.append(idx)

    # --------------------------------------------------
    # Fallback: unrecognized hapticity
    # --------------------------------------------------
    elif not selected:
        if len(g.haptic_type) == 1 and g.haptic_type[0] == "eta5(C4-N)":
            _assign_protonation_sites(1)

        logger.info("Haptic group not recognized, using fallback. %s", g.haptic_type)

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
            "        HANDLE_NON_HAPTIC: idx=%d, label=%s, connec=%d, mconnec=%d",
            idx,
            a.label,
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
