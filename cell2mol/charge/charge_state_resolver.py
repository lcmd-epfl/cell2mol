from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from collections import defaultdict

from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation
import logging
from cell2mol.operations import reorder_element
from cell2mol.charge.utils import (
    METAL_OXIDATION_STATES,
    NOBLE_GASES,
    HALOGENS,
    aromatic_info,
    rdkit_atomic_valence,
    MANUAL_CHARGE_ASSIGN_SPECIES,
    check_rdkit_obj_connectivity,
    generate_rdkit_mol_from_AC2mol,
    generate_rdkit_mol_from_rdDetermineBonds,
)
from cell2mol.charge.special_cases import (
    generate_porphyrin_charge_state,
    _find_charged_moiety,
    generate_special_charge_states,
)

from cell2mol.elementdata import ElementData
from rdkit import Chem
from rdkit.Chem import rdchem

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.metal import Metal

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def enumerate_possible_charge_states(spec: Specie) -> list[ChargeState] | None:
    """
    Generates valid charge states for a given specie.
    Charge states are only generated for:
    - ligands
    - non-complex molecules
    Args:
        spec: The Specie object.
    Returns:
        A list of selected ChargeState objects, or None if no valid states found.
    """

    # 1. Check for protonation states
    if spec.protonation_states is None:
        spec.get_protonation_states()

    if not spec.protonation_states:
        logger.warning("No protonation states available for %s", spec.formula)
        return None

    # 2. Manual Assignments and special cases
    if spec.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        charge_state = generate_manual_charge_state(spec)
        return [charge_state] if charge_state is not None else None

    # Closed-form builders for fullerenes / Sb-halides. Keep the `is not None`
    # test (not a truthiness test): an empty list means "special case, builder
    # failed" and must still short-circuit, so a 60+-atom cage never degrades
    # into the general bond-order sweep. Only None means "not special".
    special_charge_states = generate_special_charge_states(spec)
    if special_charge_states is not None:
        return special_charge_states

    # 3. Enumeration Loop
    valid_charge_states = []
    # Get list of integer charges to attempt
    candidate_charges = get_candidate_charges(spec.protonation_states[0])

    if spec.has_porphyrin and not spec.protonation_warning:
        for prot in spec.protonation_states:
            charge_state = generate_porphyrin_charge_state(prot)
            if charge_state is not None and charge_state.status:
                valid_charge_states.append(charge_state)

    else:
        logger.debug("Considering general charge state enumeration for %s", spec.formula)
        for prot in spec.protonation_states:
            valid_charge_states_dict = generate_valid_charge_states(
                prot, candidate_charges
            )
            for _charge, charge_states in valid_charge_states_dict.items():
                for charge_state in charge_states:
                    if charge_state is not None and charge_state.status:
                        valid_charge_states.append(charge_state)

    
    for charge_state in valid_charge_states:
        prot = charge_state.protonation
        logger.debug(
            "    [Success] %s | Protonation: %s | Charge: %d | Corrected Charge: %d | Added atoms: %d | SMILES: %s",
            spec.formula,
            prot.formula,
            charge_state.uncorr_total_charge,
            charge_state.corr_total_charge,
            prot.n_protons_added,
            charge_state.smiles,
        )

    # 4. Final Selection / Filtering
    best_candidates = identify_best_charge_states(valid_charge_states)

    return best_candidates if best_candidates else None


def get_candidate_charges(prot: Protonation) -> list[int]:
    """Formal charges to try for a protonation state.

    In decreasing order of specificity:
      * Fixed-charge small formulas (diatomic donors -> 0, a lone halide -> -1,
        a noble gas -> 0).
      * With charged substituents (free carboxylate -1, ammonium +1, ...): 
        the charge is just their summed net charge (0 if none), so try
        only that value. Lets a big conjugated macrocycle skip the full sweep.
      * Large species (>=100 atoms) with no charged substituent: trim the sweep
        to {0} for a non-complex molecule, or {0, +/-1, +/-2} for a ligand.
      * Everything else: the general -4..+4 sweep.
    """
    spec = cast("Specie", prot.parent)
    formula = spec.formula

    # Quick returns for simple cases
    if formula in {"C-O", "H2-O", "C-N", "C-S", "C-Se", "C-Te", "C-P", "C-As", "C-Sb"}:
        return [0]
    elif formula in HALOGENS:
        return [-1]
    elif formula in NOBLE_GASES:
        return [0]
    else:
        # If the specie has charged substituents (e.g. a free carboxylate), 
        # its charge is just the sum of their net charges -- try only that value 
        # and skip the sweep
        charged_moieties = _find_charged_moiety(spec)
        if charged_moieties:
            candidate_charge = sum(net_charge for *_, net_charge in charged_moieties)
            logger.debug(
                "Charged moieties in %s: %s -> candidate charge %d",
                formula,
                [(kind, nc) for _c, _a, kind, nc in charged_moieties],
                candidate_charge,
            )
            return [candidate_charge] if candidate_charge != 0 else [0]
        # Large species: trim the sweep to keep bond perception tractable.
        elif spec.natoms >= 100:
            logger.debug("Trimming charge sweep for %s (%d atoms)", formula, spec.natoms)
            return [0] if spec.is_non_complex_molecule else [0, -1, 1, -2, 2]
        # General case: full -4..+4 sweep.
        return [0, -1, 1, -2, 2, -3, 3, -4, 4]


def generate_manual_charge_state(spec):
    """Generates a ChargeState for special species using formula-based lookups."""
    # 1. Configuration Registry
    REGISTRY = {
        "O4-Cl": ("Cl", "[O-]Cl(=O)(=O)=O", -1),
        "N3": ("central", "[N-]=[N+]=[N-]", -1),
        "I3": ("central", "I[I-]I", -1),
        "N2": (None, "N#N", 0),
        "I4": (None, "I[I-][I-]I", -2),
        "I5": (None, "II[I-]II", -1),
        "I6": (None, "I[I-]II[I-]I", -2),
        "H": (None, "[H-]", -1),
        "H2": (None, "[H][H]", 0),
        "N-O3": ("N", "[N+](=O)([O-])[O-]", -1),
        "Te2": (None, "[Te-][Te-]", -2),
    }

    formula = spec.formula
    target_atom, smiles, charge = REGISTRY.get(formula, (None, None, 0))

    # 2. Handle specific NO Logic
    if formula == "N-O":
        target_atom = "O"
        is_bent = getattr(spec, "NO_type", "") == "Bent"
        smiles, charge = ("[N-]=O", -1) if is_bent else ("[N]=O", 0)
    assert smiles is not None, f"No manual SMILES registered for formula {formula}"

    # 3. Determine Atom Ordering
    order = list(range(spec.natoms))
    new_order = order

    if target_atom:
        for idx, atom in enumerate(spec.atoms):
            # 'central' means the atom connected to two others of the same type
            if target_atom == "central":
                adj_labels = [
                    spec.get_parent("molecule").labels[a] for a in atom.adjacency
                ]
                if adj_labels.count(atom.label) == 2:
                    new_order = reorder_element(order, 1, idx)
                    break
            # Specific label match (e.g., "Cl" in O4-Cl)
            elif atom.label == target_atom:
                new_order = reorder_element(order, 1, idx)
                break

    # 4. RDKit Processing
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Chem.RenumberAtoms(mol, new_order)
    mol = Chem.RemoveHs(mol)

    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    total_charge = sum(atom_charges)

    logger.debug(
        "Manual Charge: %s | SMILES: %s | Order: %s", formula, smiles, new_order
    )

    return ChargeState.from_positional(
        True,
        total_charge,
        atom_charges,
        mol,
        smiles,
        charge,
        True,
        spec.protonation_states[0],
    )


def _get_possible_valences(
    atom_num: int, atomic_valence: dict[int, list[int]]
) -> list[int]:
    if atom_num in atomic_valence:
        return list(atomic_valence[atom_num])

    pt = Chem.GetPeriodicTable()
    return [int(v) for v in pt.GetValenceList(atom_num) if v >= 0]


def _specie_supported_by_rddeterminebonds(
    atoms: list[int],
    ac,  # adjacency matrix
    atomic_valence: dict[int, list[int]] = rdkit_atomic_valence,
) -> bool:
    """
    Charge-independent pre-check for a specie.
    Returns False if ANY atom either:
        (a) has no possible valences defined, or
        (b) has a bonded degree exceeding its maximum possible valence
            — meaning no bond-order assignment can satisfy it, at any charge.
        (c) has a bonded degree not in its possible valences and
        is one of P, As, Sb, Te
    """
    ac = np.asarray(ac)
    n_atoms = len(atoms)

    for i in range(n_atoms):
        atom_num = atoms[i]
        valences = _get_possible_valences(atom_num, atomic_valence)

        if not valences:
            logger.debug(f"Atom {atom_num} has no possible valences defined")
            return False

        degree = int(np.count_nonzero(ac[i]))
        max_valence = max(valences)

        if degree > max_valence:
            logger.debug(
                f"Atom {atom_num} has degree {degree} exceeding max "
                f"possible valence {max_valence} (valences: {valences}) "
                "— no bond ordering possible at any charge"
            )
            return False
        elif degree not in valences and atom_num in [15, 33, 51, 52]:  # P, As, Sb, Te
            logger.debug(
                f"Atom {atom_num} has degree {degree} not in possible valences "
                f"{valences} — may require special handling"
            )
            return False

    return True


def generate_valid_charge_states(prot, candidate_charges, allow_charged_fragments=True):
    valid_charge_states = {charge: [] for charge in candidate_charges}

    # --- Tier 0: element-level pre-check (charge-independent) ---
    if not _specie_supported_by_rddeterminebonds(prot.atnums, prot.adjmat):
        # Guaranteed to hit unordered_map for every charge — skip straight
        # to bond assignment using modified AC2mol
        logger.debug(
            "Element-level pre-check failed for %s, skipping rdDetermineBonds and trying manual bond assignment",
            prot.formula,
        )
        return determine_bond_using_modified_AC2mol(
            prot,
            candidate_charges,
            valid_charge_states,
            allow_charged_fragments=allow_charged_fragments,
        )

    # --- Tier 1: try rdDetermineBonds across ALL candidate charges ---
    valid_charge_states = determine_bond_using_rdDetermineBonds(
        prot, candidate_charges, valid_charge_states, allow_charged_fragments
    )

    if any(valid_charge_states.values()):
        logger.debug(
            "rdDetermineBonds found valid charge states for %s: %s",
            prot.formula,
            valid_charge_states,
        )
        return valid_charge_states

    # --- Tier 2: every candidate charge failed rdDetermineBonds ---
    return determine_bond_using_modified_AC2mol(
        prot,
        candidate_charges,
        valid_charge_states,
        allow_charged_fragments=allow_charged_fragments,
    )


def determine_bond_using_modified_AC2mol(
    prot, candidate_charges, valid_charge_states, allow_charged_fragments=True
):
    for charge in candidate_charges:
        rdkit_obj = generate_rdkit_mol_from_AC2mol(
            atoms=prot.atnums,
            AC=prot.adjmat,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
        )
        if rdkit_obj is not None:
            charge_state = prepare_ChargeState_from_rdkit_obj(
                rdkit_obj, prot, charge, allow_charged_fragments=allow_charged_fragments
            )
            if charge_state.status:
                valid_charge_states[charge].append(charge_state)

    return valid_charge_states


def determine_bond_using_rdDetermineBonds(
    prot, candidate_charges, valid_charge_states, allow_charged_fragments=True
):
    for charge in candidate_charges:
        try:
            rdkit_obj = generate_rdkit_mol_from_rdDetermineBonds(
                atoms=prot.atnums,
                coords=prot.coord,
                AC=prot.adjmat,
                charge=charge,
                sanitize=True,
                allow_charged_fragments=allow_charged_fragments,
            )
            if rdkit_obj is not None:
                charge_state = prepare_ChargeState_from_rdkit_obj(
                    rdkit_obj,
                    prot,
                    charge,
                    allow_charged_fragments=allow_charged_fragments,
                )
                if charge_state.status:
                    valid_charge_states[charge].append(charge_state)
        except ValueError as e:
            logger.error(
                f"  ValueError occurred using rdDetermineBonds: {e} with {charge} charge for {prot.formula}"
            )
            continue
        except IndexError as e:
            if "unordered_map::at" in str(e):
                # Shouldn't happen given the pre-check, but if it does.
                logger.error(
                    f"  IndexError occurred using rdDetermineBonds: {e} with {charge} charge for {prot.formula}"
                )
            continue

    return valid_charge_states


def prepare_ChargeState_from_rdkit_obj(
    rdkit_obj,
    prot,
    charge,
    allow_charged_fragments=True,
    ref_uncorr_atom_charges: list[int] | None = None,
):
    atom_charges = []
    total_charge = 0
    for i, atom in enumerate(rdkit_obj.GetAtoms()):
        if ref_uncorr_atom_charges is not None:
            ref_q = ref_uncorr_atom_charges[i]
            if atom.GetFormalCharge() != ref_q:
                logger.debug(
                    "Correcting atom %d (%s) %d -> %d",
                    i,
                    atom.GetSymbol(),
                    atom.GetFormalCharge(),
                    ref_q,
                )
                atom.SetFormalCharge(ref_q)
        q = atom.GetFormalCharge()
        atom_charges.append(q)
        total_charge += q

    # Final Validation and Resonance Search
    smiles = Chem.MolToSmiles(rdkit_obj)
    is_correct = check_rdkit_obj_connectivity(rdkit_obj, prot.natoms, charge)

    # logger.debug(
    #     "Generated ChargeState | SMILES: %s | Total Charge: %d | Correct: %s",
    #     smiles,
    #     total_charge,
    #     is_correct,
    # )
    charge_state = ChargeState.from_positional(
        status=is_correct,
        uncorr_total_charge=total_charge,
        uncorr_atom_charges=atom_charges,
        rdkit_obj=rdkit_obj,
        smiles=smiles,
        charge_tried=charge,
        allow=allow_charged_fragments,
        protonation=prot,
    )

    return charge_state


def get_best_resonance_state(charge_state: ChargeState) -> ChargeState:
    """
    Checks for resonance alternatives and returns the best state found.
    """
    prot = charge_state.protonation
    rdkit_obj = charge_state.rdkit_obj
    charge_tried = charge_state.uncorr_total_charge
    assert prot.natoms is not None
    natoms = prot.natoms
    parent = cast("Specie", prot.parent)
    try:
        # Generate resonance structures
        ## We use UNCONSTRAINED_ANIONS/CATIONS if the system is highly charged
        ## suppl = rdchem.ResonanceMolSupplier(rdkit_obj, rdchem.ResonanceFlags.ALLOW_INCOMPLETE_OCTETS)
        suppl = rdchem.ResonanceMolSupplier(rdkit_obj)
        num_res = len(suppl)
    except Exception as e:
        logger.error("   ResonanceMolSupplier failed for %s: %s", parent.formula, e)
        return charge_state

    if num_res <= 1:
        return charge_state

    original_smiles = Chem.MolToSmiles(rdkit_obj, canonical=True)
    logger.debug("   Resonance check for %s: found %d forms", parent.formula, num_res)
    logger.debug("   Original: %s", original_smiles)

    # The ResonanceMolSupplier ranks structures with index 0 as the most 'stable',
    # but it can still emit over-delocalized structures that are not actually
    # kekulizable/valid (e.g. an exocyclic double bond combined with a ring
    # charge that leaves no valid alternating bond pattern). Walk the ranked
    # candidates and accept the first one that round-trips through SMILES
    # parsing; per-atom valence bookkeeping (check_rdkit_obj_connectivity)
    # doesn't catch this since the inconsistency is a whole-ring kekulization
    # issue, not a per-atom one.
    best_res_mol = None
    best_smiles = None
    for candidate in suppl:
        if candidate is None:
            continue
        candidate_smiles = Chem.MolToSmiles(candidate, canonical=True)
        if candidate_smiles == original_smiles:
            logger.debug("   Found original structure in resonance forms")
            return charge_state
        if Chem.MolFromSmiles(candidate_smiles) is not None:
            best_res_mol = candidate
            best_smiles = candidate_smiles
            break
        logger.debug("   Rejected (invalid SMILES): %s", candidate_smiles)

    if best_res_mol is None or best_smiles is None:
        logger.debug(
            "   No valid resonance alternative found for %s; keeping original",
            parent.formula,
        )
        return charge_state

    logger.debug("   Best    : %s", best_smiles)
    logger.info("   Resonance form updated for %s", parent.formula)

    # Extract properties from the best resonance candidate
    atom_charges = [a.GetFormalCharge() for a in best_res_mol.GetAtoms()]
    total_charge = sum(atom_charges)

    # Perform a sanity check on the new connectivity/valence
    is_correct = check_rdkit_obj_connectivity(best_res_mol, natoms, charge_tried)

    # Return the updated ChargeState object
    return ChargeState.from_positional(
        is_correct,
        total_charge,
        atom_charges,
        best_res_mol,
        best_smiles,
        charge_tried,
        True,  # allow
        prot,
    )


def get_metal_poscharges(metal: Metal) -> list[int]:
    """
    Retrieve common oxidation states for a given metal atom.

    Oxidation state data primarily from:
    Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
    Args:
        metal (Metal): Metal atom object.
    Returns:
        poscharges (list): List of common oxidation states for the metal.
    """

    mol = cast("Molecule", metal.get_parent("molecule"))
    if mol.is_haptic is None:
        mol.get_hapticity()

    poscharges = METAL_OXIDATION_STATES[metal.label]

    # Allow 0 oxidation state for selected metals under specific conditions
    zero_os_metals = {"Fe", "Ni", "Ru"}
    if metal.label in zero_os_metals:
        has_CO = any(lig.formula == "C-O" for lig in mol.ligands or [])
        if (has_CO or mol.is_haptic) and 0 not in poscharges:
            poscharges.append(0)

    return poscharges


def identify_best_charge_states(charge_states: list[ChargeState]) -> list[ChargeState]:
    """
    Selects the best charge distributions.
    """
    # Filter out None values initially
    valid_charge_states = [ch for ch in charge_states if ch is not None and ch.status]
    if not valid_charge_states:
        return []

    # 1. Get initial best candidates indices using the core logic
    best_indices = _get_best_candidate_indices(valid_charge_states)

    # Map indices back to objects
    initial_candidates = [valid_charge_states[i] for i in best_indices]

    # 2. Group candidates by their 'corrected total charge'
    grouped_by_charge = defaultdict(list)
    for state in initial_candidates:
        grouped_by_charge[state.corr_total_charge].append(state)

    logger.debug("Found target charges: %s", list(grouped_by_charge.keys()))

    final_states = []

    # 3. Process each charge group
    for tgt_charge, candidates in grouped_by_charge.items():
        logger.debug(
            "Processing target charge %s with %d candidates", tgt_charge, len(candidates)
        )

        # CASE 1: Only one candidate for this charge
        if len(candidates) == 1:
            # best_structure = get_best_resonance_state(candidates[0])
            best_structure = candidates[0]
            final_states.append(best_structure)

        # CASE 2: Multiple candidates
        else:
            best_subset_indices = _get_best_candidate_indices(candidates)
            if not best_subset_indices:
                # Fallback: take the first if filtering somehow fails
                logger.debug("Tie-break failed, taking first.")
                final_states.append(candidates[0])
            else:
                # Take the best one from the filtered result (index 0)
                logger.debug("Tie-break successful, taking best structure.")
                best_idx = best_subset_indices[0]            # Generate resonance forms for all candidates in this group first
                final_states.append(candidates[best_idx])

            # resonance_candidates = [get_best_resonance_state(temp) for temp in candidates]

            # # We apply the same filtering criteria to the subset of resonance structures
            # best_subset_indices = _get_best_candidate_indices(resonance_candidates)

            # if not best_subset_indices:
            #     # Fallback: take the first if filtering somehow fails
            #     logger.debug("Tie-break failed, taking first.")
            #     final_states.append(candidates[0])
            # else:
            #     # Take the best one from the filtered result (index 0)
            #     logger.debug("Tie-break successful, taking best resonance structure.")
            #     best_idx = best_subset_indices[0]
            #     final_states.append(resonance_candidates[best_idx])

    return final_states


def _get_best_candidate_indices(valid_charge_states: list[ChargeState]) -> list[int]:
    """
    Helper function containing the core filtering logic.
    Calculates metrics and returns the indices of the best candidates.
    """
    nlists = len(valid_charge_states)
    if nlists == 0:
        return []

    # --- 1. Extract Metrics ---
    # Using lists to store metrics for all candidates
    uncorr_abs_total = []
    uncorr_abs_atcharge = []
    uncorr_zwitt = []
    coincide = []
    aromatic_atoms = []
    aromatic_rings = []
    added_into_aromatic = []

    # Coordinating atoms logic
    parent = cast("Specie", valid_charge_states[0].protonation.parent)
    parent_atoms = parent.atoms or []
    coordinating_atoms_indices = [
        idx for idx, atom in enumerate(parent_atoms) if (atom.mconnec or 0) > 0
    ]
    coordinating_atoms_labels = [
        atom.label for idx, atom in enumerate(parent_atoms) if (atom.mconnec or 0) > 0
    ]
    blocked_indices = [
        idx
        for idx, n_added in enumerate(
            valid_charge_states[0].protonation.site_proton_counts or []
        )
        if n_added == 0
    ]
    coord_abs_atcharge = []
    coord_raw_atcharge = []

    for chs in valid_charge_states:
        uncorr_abs_total.append(chs.uncorr_abstotal)
        uncorr_abs_atcharge.append(chs.uncorr_abs_atcharge)
        uncorr_zwitt.append(chs.uncorr_zwitt)
        coincide.append(chs.coincide)

        # Aromatic calculations
        added_indices = [
            idx
            for idx, n_added in enumerate(chs.protonation.site_proton_counts)
            if n_added > 0
        ]
        aromatic_dict = aromatic_info(chs.rdkit_obj, added_indices=added_indices)

        aromatic_atoms.append(aromatic_dict["Aromatic atoms"])
        aromatic_rings.append(aromatic_dict["Number of aromatic rings"])
        added_into_aromatic.append(aromatic_dict["Added to aromatic atoms"])

        # Coordinating atom charges
        coord_abs_atcharge.append(
            sum([abs(chs.uncorr_atom_charges[i]) for i in coordinating_atoms_indices])
        )
        coord_raw_atcharge.append(
            [chs.uncorr_atom_charges[i] for i in coordinating_atoms_indices]
        )

    # --- 2. Determine Minima/Maxima ---
    min_tot = np.min(uncorr_abs_total)
    min_abs = np.min(uncorr_abs_atcharge)
    max_aromatic = np.max(aromatic_atoms)

    indices_min_tot = {i for i, x in enumerate(uncorr_abs_total) if x == min_tot}
    indices_min_abs = {i for i, x in enumerate(uncorr_abs_atcharge) if x == min_abs}
    indices_max_aromatic = [
        i for i, x in enumerate(aromatic_atoms) if x == max_aromatic
    ]

    # logger.debug(f"   min_tot indices: {indices_min_tot}")
    # logger.debug(f"   min_abs indices: {indices_min_abs}")

    # --- 3. Build Temporary List (Filtering Rounds) ---
    tmplist = []

    # Round 1: Strict intersection
    for idx in range(nlists):
        if not valid_charge_states[idx].status:
            continue

        is_optimal = (
            (idx in indices_min_abs) and (idx in indices_min_tot) and coincide[idx]
        )

        # Special check for coordinating atoms
        is_coord_valid = False
        if (coordinating_atoms_indices == blocked_indices) and (
            "C" in coordinating_atoms_labels
        ):
            if (uncorr_abs_atcharge[idx] == coord_abs_atcharge[idx]) and coincide[idx]:
                if all(c < 0 for c in coord_raw_atcharge[idx]):
                    is_coord_valid = True

        if is_optimal or is_coord_valid:
            tmplist.append(idx)

    # Round 2: Relaxed (Allow either MinAbs OR MinTot) + Coincide + Not Zwitt
    if not tmplist:
        logger.debug("   Round 1 empty. Trying Round 2 (Min+Coincide+NotZwitt)...")
        for idx in range(nlists):
            if (
                ((idx in indices_min_abs) or (idx in indices_min_tot))
                and coincide[idx]
                and not uncorr_zwitt[idx]
            ):
                tmplist.append(idx)

    # Round 3: More Relaxed (Allow either MinAbs OR MinTot) + Coincide
    if not tmplist:
        logger.debug("   Round 2 empty. Trying Round 3 (Min+Coincide)...")
        for idx in range(nlists):
            if ((idx in indices_min_abs) or (idx in indices_min_tot)) and coincide[idx]:
                tmplist.append(idx)

    # Round 4: Most Relaxed (Allow either MinAbs OR MinTot)
    if not tmplist:
        logger.debug("   Round 3 empty. Trying Round 4 (Min only)...")
        for idx in range(nlists):
            if (idx in indices_min_abs) or (idx in indices_min_tot):
                tmplist.append(idx)

    # --- 4. Aromaticity Filtering ---
    # logger.debug(f"   Pre-aromatic tmplist: {tmplist}")

    # Logic: If max aromaticity is unique or dominates, filter tmplist
    if len(indices_max_aromatic) == nlists:
        pass  # All are equal, do nothing

    elif len(indices_max_aromatic) == 1:
        # If there is exactly one max aromatic candidate
        new_tmplist = []
        for idx in range(nlists):
            if (idx in indices_max_aromatic) and coincide[idx]:
                if idx not in tmplist:
                    tmplist.append(idx)  # Add it if not present
                else:
                    new_tmplist.append(idx)  # Keep it if present

        # Update tmplist only if we found candidates
        if new_tmplist:
            tmplist = new_tmplist

    elif len(indices_max_aromatic) > 1:
        # If multiple candidates share max aromaticity
        if len(tmplist) > 1:
            new_tmplist = tmplist.copy()
            for idx in range(nlists):
                if idx in indices_max_aromatic and coincide[idx]:
                    if idx in new_tmplist:
                        if added_into_aromatic[idx]:
                            # Exclude if H added to aromatic ring (breaking aromaticity)
                            logger.debug(f"      Removing {idx} (H added to aromatic)")
                            # Note: Original code commented out the remove, but logic implies filtering.
                            # If you want to strictly follow original commented code, do nothing.
                            # Assuming intent was to filter based on variable name logic:
                            # if added_into_aromatic[idx]: new_tmplist.remove(idx)
                            pass
                    else:
                        # Logic for adding new candidates if they are max aromatic
                        logger.debug(
                            f"      Considering adding {idx} (High aromaticity)"
                        )

            if new_tmplist:
                tmplist = new_tmplist

    return tmplist
