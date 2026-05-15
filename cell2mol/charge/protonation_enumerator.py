import numpy as np
import itertools
import networkx as nx
from cell2mol.connectivity import add_atom
from cell2mol.element_utils import get_post_transition_metal_idxs, get_metalloid_idxs
from dataclasses import dataclass
from typing import Dict, List
from cell2mol.classes.protonation import Protonation
from cell2mol.charge.utils import FULLERENES, MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.hydrogen import detect_missing_hydrogens, add_hydrogens
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProtonationGroupResult:
    site_proton_counts: Dict[int, int]
    ligand_donor_electrons: Dict[int, int]
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

    if specie.subtype == "ligand":
        parent = specie.get_parent("molecule")
        if parent.has_ia_iia and not parent.iscomplex:
            return get_empty_protonation_state(specie)

    # ============================================================
    # From here on: ligand protonation engine
    # ============================================================
    ligand = specie
    protonation_states: list = []

    newlab = ligand.labels.copy()
    newcoord = ligand.coord.copy()
    # ============================================================
    # Initialization
    # ============================================================
    n_protons_added = 0
    site_proton_counts = np.zeros(ligand.natoms, dtype=int)
    ligand_donor_electrons = np.zeros(ligand.natoms, dtype=int)
    non_local_groups_indices: list[int] = []
    process_both_modes: list[int] = []
    protonated_indices_to_reset: list[int] = []  # old : reset_H_indices

    limit_of_nonlocal_sites = 4  # Arbitrary limit to avoid combinatorial explosion

    logger.info("Processing %s (%s):", specie.formula, specie.subtype)

    # ============================================================
    # GROUP-LEVEL ANALYSIS
    # ============================================================
    logger.info("Ligand formula: %s, natoms: %d", ligand.formula, ligand.natoms)
    logger.debug(
        "Ligand groups info: %s",
        [
            (
                g.formula,
                g.natoms,
                [a.atom_site_label for a in g.atoms],
                [m.atom_site_label for m in g.metals],
            )
            for g in ligand.groups
        ],
    )
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
        for idx, val in result.site_proton_counts.items():
            site_proton_counts[idx] += val

        for idx, val in result.ligand_donor_electrons.items():
            ligand_donor_electrons[idx] += val

        if result.needs_nonlocal:
            non_local_groups_indices.extend(result.non_local_indices)

    # ============================================================
    # Check non_local_groups_indices for decision
    # ============================================================
    logger.debug("    non_local_groups_indices: %s", non_local_groups_indices)
    if len(non_local_groups_indices) > limit_of_nonlocal_sites:
        logger.info(
            "  %d combinatorial protonation sites detected (more than the limit of %d). ",
            len(non_local_groups_indices),
            limit_of_nonlocal_sites,
        )
        combinations = list(
            itertools.product([0, 1], repeat=len(non_local_groups_indices))
        )
        logger.info("  Total combinations to evaluate: %d. ", len(combinations))
        logger.info("  Generating empty protonation state only for %s.", specie.formula)

        return get_empty_protonation_state(specie)

    # ============================================================
    # LOCAL ATOM ADDITION
    # ============================================================
    for idx, a in enumerate(ligand.atoms):
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
                newlab, newcoord, idx, ligand, element="H", unconditional=True
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
                newlab, newcoord, idx, ligand, num_hydrogens=site_proton_counts[idx]
            )

            if ligand_donor_electrons[idx] >= 2 and idx in non_local_groups_indices:
                process_both_modes.append(idx)
                logger.debug(
                    "    Atom %s (ligand idx %d) is also proccessed for combinatorial protonation.",
                    atom_label,
                    idx,
                )
                protonated_indices_to_reset.extend(list(range(start_idx, end_idx)))

    # ============================================================
    # Heuristic protonation
    # ============================================================
    no_nonlocal_sites = len(non_local_groups_indices) == 0
    force_local_mode = process_both_modes

    if no_nonlocal_sites or force_local_mode:
        protonation_states.append(
            Protonation.from_positional(
                labels=newlab,
                coord=newcoord,
                cov_factor=ligand.cov_factor,
                n_protons_added=n_protons_added,
                site_proton_counts=site_proton_counts,
                ligand_donor_electrons=ligand_donor_electrons,
                mode="heuristic",
                parent=ligand,
            )
        )
        if not force_local_mode:
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
            [ligand.atoms[idx].atom_site_label for idx in process_both_modes],
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

    combinations = list(itertools.product([0, 1], repeat=len(non_local_groups_indices)))
    combinations.sort(key=sum)

    for com in combinations:
        newlab = local_labels.copy()
        newcoord = local_coords.copy()
        n_protons_added = local_n_protons_added
        site_proton_counts = local_site_proton_counts.copy()
        ligand_donor_electrons = local_ligand_donor_electrons.copy()

        for flag, idx in zip(com, non_local_groups_indices):
            if flag == 1:
                site_proton_counts[idx] = 1
                n_protons_added += site_proton_counts[idx]
                _, newlab, newcoord = add_atom(
                    newlab, newcoord, idx, ligand, element="H", unconditional=True
                )
        prot = Protonation.from_positional(
            labels=newlab,
            coord=newcoord,
            cov_factor=ligand.cov_factor,
            n_protons_added=n_protons_added,
            site_proton_counts=site_proton_counts,
            ligand_donor_electrons=ligand_donor_electrons,
            mode="combinatorial",
            parent=ligand,
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
    even when no protons are added.
    """
    logger.debug(
        "Creating empty protonation placeholder for %s (%s)",
        specie.formula,
        specie.subtype,
    )

    empty_protonation = Protonation.from_positional(
        labels=specie.labels,
        coord=specie.coord,
        cov_factor=specie.cov_factor,
        n_protons_added=0,
        site_proton_counts=[0] * len(specie.labels),
        ligand_donor_electrons=[0] * len(specie.labels),
        mode="none",
        parent=specie,
    )

    return [empty_protonation]


def _handle_haptic_group(ligand, g, parent_indices) -> ProtonationGroupResult:
    """
    Handle protonation rules for haptic ligand groups.

    This function does NOT mutate global state.
    It returns a ProtonationGroupResult describing intended changes.
    """

    site_proton_counts: Dict[int, int] = {}
    ligand_donor_electrons: Dict[int, int] = {}
    needs_nonlocal = False
    non_local_indices: List[int] = []

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
                        needs_nonlocal = True
                        non_local_indices.append(idx)
                        logger.debug(
                            "  Needing combinatorial protonation for atom %d (%s): %s",
                            idx,
                            atom_label,
                            report,
                        )

    return ProtonationGroupResult(
        site_proton_counts=site_proton_counts,
        ligand_donor_electrons=ligand_donor_electrons,
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

    site_proton_counts: Dict[int, int] = {}
    ligand_donor_electrons: Dict[int, int] = {}
    needs_nonlocal = False
    non_local_indices: List[int] = []

    ions = {"F", "Cl", "Br", "I", "As"}

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
        if a.label in ions:
            if a.connec == 0:
                site_proton_counts[idx] = 1

        # -----------------------------------------
        # Oxygen
        # -----------------------------------------
        elif a.label == "O":
            if len(adj_labels) == 1:
                needs_nonlocal = True
                non_local_indices.append(idx)

        # -----------------------------------------
        # Sulfur / Selenium
        # -----------------------------------------
        elif a.label in {"S", "Se"}:
            if len(adj_labels) == 1:
                needs_nonlocal = True
                non_local_indices.append(idx)

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
                    G = nx.from_numpy_array(ligand.adjmat.astype(float))
                    cycles = nx.cycle_basis(G)
                    in_cycles = [c for c in cycles if idx in c]

                    if len(in_cycles) == 1 and len(in_cycles[0]) == 6:
                        pass  # pyridine-like
                    else:
                        needs_nonlocal = True
                        non_local_indices.append(idx)
            elif len(adj_labels) == 1:
                # Nitrosyl ligand (handle manually)
                needs_nonlocal = True
                non_local_indices.append(idx)

            else:  # only N atom in the ligand
                pass
        # -----------------------------------------
        # Phosphorus
        # -----------------------------------------
        elif a.label == "P":
            if len(adj_labels) >= 3:
                pass
            elif len(adj_labels) == 1:
                if adj_labels[0] in {"N", "C"}:
                    pass
                elif adj_labels[0] == "P":
                    site_proton_counts[idx] = 1

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
                site_proton_counts[idx] = 1

            elif ligand.formula in {"C-O", "C-S", "C-Se", "C-Te"}:
                pass
            else:
                numN = adj_labels.count("N")
                numO = adj_labels.count("O")
                numH = adj_labels.count("H")
                numC = adj_labels.count("C")

                if len(adj_labels) == 1:
                    site_proton_counts[idx] = 1

                elif len(adj_labels) == 2:
                    if numN == 1 and numO == 1:  # amide
                        site_proton_counts[idx] = 1
                    elif numH == 2 and ligand.formula == "H2-C":
                        site_proton_counts[idx] = 2
                    else:
                        G = nx.from_numpy_array(ligand.adjmat.astype(float))
                        cycles = nx.cycle_basis(G)
                        in_cycles = [c for c in cycles if idx in c]

                        if len(in_cycles) == 1:
                            if numN == 2:
                                print(
                                    ligand.formula,
                                    a.atom_site_label,
                                    adj_labels,
                                    "possible NHC",
                                )
                                site_proton_counts[idx] = 2
                                ligand_donor_electrons[idx] = 2
                            elif numC == 2:
                                needs_nonlocal = True
                                non_local_indices.append(idx)
                            elif (numO == 1 and numC == 1) or (numN == 1 and numC == 1):
                                site_proton_counts[idx] = 2
                                ligand_donor_electrons[idx] = 2
                                needs_nonlocal = True
                                non_local_indices.append(idx)
                            else:
                                needs_nonlocal = True
                                non_local_indices.append(idx)
                        else:
                            site_proton_counts[idx] = 2
                            ligand_donor_electrons[idx] = 2
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
                site_proton_counts[idx] = 3
                ligand_donor_electrons[idx] = 2
            elif len(adj_labels) == 2:
                G = nx.from_numpy_array(ligand.adjmat.astype(float))
                cycles = nx.cycle_basis(G)
                in_cycles = [c for c in cycles if idx in c]

                if len(in_cycles) == 1:
                    site_proton_counts[idx] = 2
                    ligand_donor_electrons[idx] = 2
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
                site_proton_counts[idx] = 1
            else:
                pass

        # -----------------------------------------
        # Fallback
        # -----------------------------------------
        else:
            needs_nonlocal = True
            non_local_indices.append(idx)

    return ProtonationGroupResult(
        site_proton_counts=site_proton_counts,
        ligand_donor_electrons=ligand_donor_electrons,
        needs_nonlocal=needs_nonlocal,
        non_local_indices=non_local_indices,
    )
