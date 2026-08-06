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
    Generates valid charge states (Lewis structures) for a given specie.
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

    special_charge_states = generate_special_charge_states(spec)
    if special_charge_states is not None:
        return special_charge_states

    # 3. Enumeration Loop
    valid_charge_states = []

    if spec.has_porphyrin and not spec.protonation_warning:
        for prot in spec.protonation_states:
            charge_state = generate_porphyrin_charge_state(prot)
            if charge_state is not None and charge_state.status:
                valid_charge_states.append(charge_state)

    else:
        for prot in spec.protonation_states:
            # Charges to attempt are per protonation state: an anchor derived
            # from the specie's chemistry has to be shifted into this state's
            # frame, and the closed-shell parity screen depends on how many
            # protons this state added.
            candidate_charges = get_candidate_charges(spec, prot)
            logger.debug(
                "Considering general charge state enumeration for specie %s "
                "| protonation %s (nH=%d) | %d candidate charge=%s",
                spec.formula,
                prot.formula,
                prot.n_protons_added,
                len(candidate_charges),
                candidate_charges,
            )
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
            "    [Success] %s | Protonation: %s | Protonated Charge: %d | Specie Charge: %d | Added atoms: %d | SMILES: %s",
            spec.formula,
            prot.formula,
            charge_state.protonated_total_charge,
            charge_state.specie_total_charge,
            prot.n_protons_added,
            charge_state.smiles,
        )

    # 4. Final Selection / Filtering
    best_candidates = identify_best_charge_states(valid_charge_states)

    return best_candidates if best_candidates else None


def get_candidate_charges(spec: Specie, prot: Protonation) -> list[int]:
    """Formal charges to try for ONE protonation state of a specie.

    These are charges of the PROTONATED structure, deliberately NOT re-based
    per protonation state: holding them fixed while ``n_protons_added`` grows
    is what walks the specie charge down (``ChargeState`` recovers it as about
    ``q - n_protons_added``). That sweep is the only route to strongly anionic
    answers, where deprotonated donors take the moiety sum further down.

    Sources, most specific first: fixed-charge small formulas, the summed
    charge of any charged substituents, else a sweep (see
    ``_anchored_specie_charges`` / ``_sweep_charges``). The parity screen then
    halves whatever comes back.
    """
    anchor = _anchored_specie_charges(spec)
    candidates = list(anchor) if anchor is not None else _sweep_charges(spec)

    kept = _filter_by_electron_parity(candidates, prot)
    if not kept:
        # Wrong parity everywhere proves the anchor wrong: some anionic centre
        # the moiety scan can't see (a metal-bound carbanion, say). Widen
        # DOWNWARD only -- every kind _classify_charged_moiety reports is
        # anionic, so an anchor is a lower bound. One step down also flips the
        # parity, so the widened set always survives.
        widened = [c - 1 for c in candidates]
        kept = _filter_by_electron_parity(widened, prot)
        logger.debug(
            "Anchor %s is parity-forbidden for %s; widening downward to %s",
            candidates,
            prot.formula,
            kept,
        )
    if not kept:  # defensive: never hand back an empty candidate list
        kept = candidates
    return kept


def _filter_by_electron_parity(charges: list[int], prot: Protonation) -> list[int]:
    """Drop charges that cannot give a closed-shell Lewis structure.

    Bond perception pairs every electron, so ``sum(outer electrons) - q`` must
    be even; other charges are provably unsolvable. May return an empty list --
    the caller reads that as the anchor being wrong.
    """
    atnums = prot.atnums
    if not atnums:
        return list(charges)

    pt = Chem.GetPeriodicTable()
    n_valence = sum(pt.GetNOuterElecs(int(z)) for z in atnums)
    kept = [c for c in charges if (n_valence - c) % 2 == 0]

    if kept and len(kept) < len(charges):
        logger.debug(
            "Electron-parity screen for %s (%d valence electrons): %s -> %s",
            prot.formula,
            n_valence,
            charges,
            kept,
        )
    return kept


def _anchored_specie_charges(spec: Specie) -> list[int] | None:
    """SPECIE charges when the specie's own chemistry pins them; None otherwise."""
    formula = spec.formula

    if formula in {"C-O", "H2-O", "C-N", "C-S", "C-Se", "C-Te", "C-P", "C-As", "C-Sb"}:
        return [0]
    if formula in HALOGENS:
        return [-1]
    if formula in NOBLE_GASES:
        return [0]

    # Anchor on the summed charge of any charged substituents rather than
    # sweeping -- the sweep is what makes a polycarboxylate intractable.
    charged_moieties = _find_charged_moiety(spec)
    if not charged_moieties:
        return None

    # The summed moiety charge ALONE -- don't try to guess what the scan
    # misses. Subtracting outside protonation sites over-counts, since some of
    # those sites are neutral, and sweeping down to that bound lets
    # min |charge| in identify_best_charge_states settle too positive. The
    # protonation sweep reaches the missing charge instead.
    anchor = sum(net_charge for *_, net_charge in charged_moieties)
    logger.debug(
        "Charged moieties in %s: %s -> candidate charge %d",
        formula,
        [(kind, nc) for _c, _a, kind, nc in charged_moieties],
        anchor,
    )
    return [anchor]


def _sweep_charges(spec: Specie) -> list[int]:
    """Fallback SPECIE-charge sweep for a specie with no chemical anchor."""
    # Trim the sweep for heteroatom-rich species: bond perception + resonance
    # cost scales with the charge-flexible heteroatom count.
    n_heteroatoms = sum(
        1 for lab in spec.labels if lab not in ("C", "H") and lab not in HALOGENS
    )
    n_oxygens = sum(1 for lab in spec.labels if lab == "O")
    if n_oxygens > 8:
        logger.debug(
            "Limiting candidate charges for %s (%d atoms, %d heteroatoms, %d oxygens) to [0]",
            spec.formula,
            spec.natoms,
            n_heteroatoms,
            n_oxygens,
        )
        return [0]
    if spec.is_non_complex_molecule:
        return [0, -1, 1, -2, 2, -3, 3, -4, 4]
    return [0, -1, 1, -2, 2]


def _dedupe(values: list[int]) -> list[int]:
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(values))


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
        "N": (None, "[N-3]", -3),
        "C": (None, "[C-4]", -4),
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
    if n_atoms == 1:
        logger.debug("Single atom specie - skipping rddeterminebonds")
        return False
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
    valid_charge_states_dict = {charge: [] for charge in candidate_charges}

    # --- Tier 0: element-level pre-check (charge-independent) ---
    if not _specie_supported_by_rddeterminebonds(prot.atnums, prot.adjmat):
        # Guaranteed to hit unordered_map for every charge — skip straight
        # to bond assignment using modified AC2mol
        logger.debug(
            "Element-level pre-check failed for %s, skipping rdDetermineBonds "
            "and trying to use modified AC2mol directly",
            prot.formula,
        )
        return determine_bond_using_modified_AC2mol(
            prot,
            candidate_charges,
            valid_charge_states_dict,
            allow_charged_fragments=allow_charged_fragments,
        )

    # --- Tier 1: try rdDetermineBonds across ALL candidate charges ---
    valid_charge_states_dict = determine_bond_using_rdDetermineBonds(
        prot, candidate_charges, valid_charge_states_dict, allow_charged_fragments
    )
    if any(valid_charge_states_dict.values()):
        logger.debug(
            "rdDetermineBonds found valid charge states for %s: %s",
            prot.formula,
            valid_charge_states_dict,
        )
        return valid_charge_states_dict
    logger.debug(
        "rdDetermineBonds failed to find valid charge states for %s, "
        "falling back to modified AC2mol",
        prot.formula,
    )
    # --- Tier 2: every candidate charge failed rdDetermineBonds ---
    return determine_bond_using_modified_AC2mol(
        prot,
        candidate_charges,
        valid_charge_states_dict,
        allow_charged_fragments=allow_charged_fragments,
    )


def determine_bond_using_modified_AC2mol(
    prot, candidate_charges, valid_charge_states_dict, allow_charged_fragments=True
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
                valid_charge_states_dict[charge].append(charge_state)

    return valid_charge_states_dict


def determine_bond_using_rdDetermineBonds(
    prot, candidate_charges, valid_charge_states_dict, allow_charged_fragments=True
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
                    valid_charge_states_dict[charge].append(charge_state)
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

    return valid_charge_states_dict


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

    charge_state = ChargeState.from_positional(
        status=is_correct,
        protonated_total_charge=total_charge,
        protonated_atom_charges=atom_charges,
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
    charge_tried = charge_state.protonated_total_charge
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


def get_plausible_metal_os(metal: Metal) -> list[int]:
    """
    Retrieve common oxidation states for a given metal atom.

    Oxidation state data primarily from:
    Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
    Args:
        metal (Metal): Metal atom object.
    Returns:
        metal_os (list): List of common oxidation states for the metal.
    """

    mol = cast("Molecule", metal.get_parent("molecule"))
    if mol.is_haptic is None:
        mol.get_hapticity()

    metal_os = METAL_OXIDATION_STATES[metal.label]

    # Allow 0 oxidation state for selected metals under specific conditions
    zero_os_metals = {"Fe", "Ni", "Ru"}
    if metal.label in zero_os_metals:
        has_CO = any(lig.formula == "C-O" for lig in mol.ligands or [])
        if (has_CO or mol.is_haptic) and 0 not in metal_os:
            metal_os.append(0)

    return metal_os


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
        grouped_by_charge[state.specie_total_charge].append(state)

    logger.debug("Found target charges: %s", list(grouped_by_charge.keys()))

    final_states = []

    # 3. Process each charge group
    for tgt_charge, candidates in grouped_by_charge.items():
        logger.debug(
            "Processing target charge %s with %d candidates",
            tgt_charge,
            len(candidates),
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
                best_idx = best_subset_indices[
                    0
                ]  # Generate resonance forms for all candidates in this group first
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
    specie_abs_totals = []
    specie_abs_atcharges = []
    specie_zwitt = []
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
        specie_abs_totals.append(chs.specie_abstotal)
        specie_abs_atcharges.append(chs.specie_abs_atcharge)
        specie_zwitt.append(chs.specie_zwitt)
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
            sum(
                [
                    abs(chs.protonated_atom_charges[i])
                    for i in coordinating_atoms_indices
                ]
            )
        )
        coord_raw_atcharge.append(
            [chs.protonated_atom_charges[i] for i in coordinating_atoms_indices]
        )

    # --- 2. Determine Minima/Maxima ---
    min_tot = np.min(specie_abs_totals)
    min_abs = np.min(specie_abs_atcharges)
    max_aromatic = np.max(aromatic_atoms)

    indices_min_tot = {i for i, x in enumerate(specie_abs_totals) if x == min_tot}
    indices_min_abs = {i for i, x in enumerate(specie_abs_atcharges) if x == min_abs}
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
            if (specie_abs_atcharges[idx] == coord_abs_atcharge[idx]) and coincide[idx]:
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
                and not specie_zwitt[idx]
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
        # Several candidates share max aromaticity: keep those and drop the
        # rest, which is what the `== 1` branch above already does for the
        # singleton case. This body used to copy tmplist and never modify it,
        # so aromaticity was silently ignored here and selection fell through
        # to "take the first" -- letting a non-aromatic tautomer beat an
        # equally charged aromatic one.
        if len(tmplist) > 1:
            preferred = [
                idx for idx in tmplist if idx in indices_max_aromatic and coincide[idx]
            ]
            # Among equally aromatic candidates, prefer those that did NOT put
            # an added proton on an aromatic atom -- that H saturates the ring
            # atom it is counted for.
            intact = [idx for idx in preferred if not added_into_aromatic[idx]]
            if intact:
                preferred = intact
            if preferred and len(preferred) < len(tmplist):
                logger.debug("      Aromaticity filter: %s -> %s", tmplist, preferred)
                tmplist = preferred

    return tmplist
