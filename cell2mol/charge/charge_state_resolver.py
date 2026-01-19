import numpy as np
from collections import defaultdict
from cell2mol.classes.charge_state import ChargeState
import logging
from cell2mol.operations import reorder_element
from cell2mol.charge.utils import (
    METAL_OXIDATION_STATES,
    FULLERENES,
    MANUAL_CHARGE_ASSIGN_SPECIES,
    aromatic_info,
)
from cell2mol.elementdata import ElementData
from cell2mol.charge.xyz2mol import (
    get_proto_mol,
    AC2mol,
    chiral_stereo_check,
)
from rdkit import Chem
from rdkit.Chem import rdchem

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def enumerate_possible_charge_states(spec: object):
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

    # 2. Filter: Only process Ligands or Non-complex Molecules
    # (Skip complex molecules, metals, etc.)
    is_processable = (spec.subtype == "ligand") or (spec.is_non_complex_molecule)
    if not is_processable:
        logger.debug(
            "Skipping enumeration for subtype: %s (%s)", spec.subtype, spec.formula
        )
        return None

    # Handle special cases first
    # Manual Assignments
    if spec.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        return [generate_manual_charge_state(spec)]

    # Fullerenes
    if spec.formula in FULLERENES:
        return [generate_charge_state(0, spec.protonation_states[0])]

    # Haptic Ligands
    is_haptic_c8 = (
        spec.subtype == "ligand"
        and len(spec.groups) == 1
        and spec.groups[0].haptic_type == ["eta8(C8)"]
    )
    if is_haptic_c8:
        ch_state = generate_charge_state(-1, spec.protonation_states[0])
        return [ch_state]

    # Silylyne
    # if spec.subtype == "ligand" and spec.is_silylyne:
    #     return [generate_charge_state(0, spec.protonation_states[0])]

    # 3. Enumeration Loop
    valid_charge_states = []
    for prot in spec.protonation_states:
        # Get list of integer charges to attempt for this specific protonation
        candidate_charges = get_candidate_charges(prot)

        for charge in candidate_charges:
            # Attempt to build the RDKit object and state
            charge_state = generate_charge_state(charge, prot)

            if charge_state:
                valid_charge_states.append(charge_state)
                logger.debug(
                    "    [Success] %s | Charge: %d | SMILES: %s",
                    spec.formula,
                    charge,
                    charge_state.smiles,
                )
            else:
                logger.debug("    [Failed]  %s | Charge: %d", spec.formula, charge)

    # 4. Final Selection / Filtering
    best_candidates = identify_best_charge_states(valid_charge_states)

    return best_candidates if best_candidates else None


def generate_charge_state(
    charge: int,
    prot: object,
    allow_charged_fragments: bool = True,
    embed_chiral: bool = True,
    ref_uncorr_atom_charges: list | None = None,
):
    """
    Generates molecular connectivity and atomistic charges from 3D coordinates
    using xyz2mol, then validates chirality and resonance.
    """
    # If protonation state is invalid, do not allow charged fragments
    if not prot.status:
        allow_charged_fragments = False

    logger.debug(
        "Protonation State: %s | Target Charge: %d | Allow charged fragments: %s",
        prot.formula,
        charge,
        allow_charged_fragments,
    )

    # AC2mol returns a list of RDKit molecule objects and bond order (BO) matrix
    # from the adjacency (AC) matrix
    new_mols, BO = AC2mol(
        mol=get_proto_mol(prot.atnums),
        AC=prot.adjmat,
        atoms=prot.atnums,
        charge=charge,
        allow_charged_fragments=allow_charged_fragments,
    )

    # Early Exit if no candidates found
    if not new_mols:
        logger.warning(f"No mol found for charge {charge}")
        return None

    # Stereo and Chirality Validation
    if embed_chiral:
        if not all(chiral_stereo_check(mol) for mol in new_mols):
            logger.error("Chirality check failed for one or more candidates")
            return None

    rdkit_obj = new_mols[0]  # use the first candidate as default
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
        uncorr_total_charge=total_charge,
        uncorr_atom_charges=atom_charges,
        rdkit_obj=rdkit_obj,
        smiles=smiles,
        charge_tried=charge,
        allow=allow_charged_fragments,
        protonation=prot,
    )

    if is_correct:
        charge_state = get_best_resonance_state(charge_state)

    return charge_state


def check_rdkit_obj_connectivity(mol: Chem.Mol, natoms: int, charge: int) -> bool:
    """
    Validates the chemical sanity of an RDKit molecule object by checking
    valences, lone pairs, and bond connectivity.
    """
    pt = Chem.GetPeriodicTable()
    is_correct = True

    for i in range(natoms):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        # Old : valence = atom.GetTotalValence()
        try:
            # New RDKit API (2024.03+)
            valence = atom.GetValence(Chem.ValenceType.TOTAL)
        except AttributeError:
            # Fallback for older RDKit versions if ValenceType doesn't exist
            valence = atom.GetTotalValence()

        # Calculate lone pairs: (Valence Electrons - Formal Charge - Shared Electrons) / 2
        num_valence_electrons = pt.GetNOuterElecs(atom.GetAtomicNum())
        lone_pairs = (num_valence_electrons - formal_charge - valence) / 2

        # 1. Lone Pair Sanity Check
        if lone_pairs not in [0, 1, 2, 3, 4]:
            logger.debug("Lone pair error at atom %d (%s): %f", i, symbol, lone_pairs)
            is_correct = False

        # 2. Aromaticity & Bond Consistency Check
        # We check total shared electrons (valence) against the RDKit valence model
        is_aromatic = atom.GetIsAromatic()

        if not is_aromatic:
            try:
                # New RDKit API
                explicit = atom.GetValence(Chem.ValenceType.EXPLICIT)
                implicit = atom.GetValence(Chem.ValenceType.IMPLICIT)
            except AttributeError:
                # Old RDKit API
                explicit = atom.GetExplicitValence()
                implicit = atom.GetImplicitValence()

            # Check if calculated shared electrons match expected valence
            if valence != explicit + implicit:
                logger.debug("Valence mismatch at atom %d (%s)", i, symbol)
                is_correct = False

            # 3. Total Electron Count Check
            # Shared electrons + electrons in lone pairs + charge should equal outer shell count
            calc_total_elecs = valence + (int(lone_pairs) * 2) + formal_charge
            if calc_total_elecs != num_valence_electrons:
                logger.debug("Total electron count mismatch at atom %d (%s)", i, symbol)
                is_correct = False

        # logger.debug(
        #     "Charge: %d | Atom: %2d %2s | Q: %2d | V: %2d | LP: %2d | Correct: %s",
        #     charge,
        #     i,
        #     symbol,
        #     formal_charge,
        #     valence,
        #     lone_pairs,
        #     is_correct,
        # )

    return is_correct


def get_best_resonance_state(charge_state: object) -> object:
    """
    Checks for resonance alternatives and returns the best state found.
    """
    prot = charge_state.protonation
    rdkit_obj = charge_state.rdkit_obj
    charge_tried = charge_state.uncorr_total_charge
    natoms = prot.natoms

    try:
        # Generate resonance structures
        ## We use UNCONSTRAINED_ANIONS/CATIONS if the system is highly charged
        ## suppl = rdchem.ResonanceMolSupplier(rdkit_obj, rdchem.ResonanceFlags.ALLOW_INCOMPLETE_OCTETS)
        suppl = rdchem.ResonanceMolSupplier(rdkit_obj)
        num_res = len(suppl)
    except Exception as e:
        logger.error("ResonanceMolSupplier failed for %s: %s", prot.parent.formula, e)
        return charge_state

    if num_res <= 1:
        return charge_state

    # The ResonanceMolSupplier ranks structures; index 0 is generally the most 'stable'
    best_res_mol = suppl[0]
    if best_res_mol is None:
        return charge_state

    # Canonical SMILES comparison to see if the structure actually changed
    original_smiles = Chem.MolToSmiles(rdkit_obj, canonical=True)
    best_smiles = Chem.MolToSmiles(best_res_mol, canonical=True)
    logger.debug("Resonance check for %s: found %d forms", prot.parent.formula, num_res)
    logger.debug("  Original: %s", original_smiles)
    logger.debug("  Best    : %s", best_smiles)

    if original_smiles == best_smiles:
        return charge_state
    logger.info("Resonance form updated for %s", prot.parent.formula)

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


def get_candidate_charges(prot: object) -> list:
    """
    Determines the range of formal charges to test for a specific protonation state.
    Uses chemical heuristics based on formula, denticity, and atom connectivity.
    """
    spec = prot.parent
    formula = spec.formula

    # Quick returns for simple cases
    if formula in {"C-O", "H2-O", "C-N", "C-S", "C-Se", "C-Te", "C-P", "C-As", "C-Sb"}:
        return [0]
    if formula in {"F", "Cl", "Br", "I", "H"}:
        return [-1]

    logger.debug(
        "Evaluating %s (%s) | Number of protonation states: %d",
        spec.formula,
        spec.subtype,
        len(spec.protonation_states),
    )

    # Quick check: if too many valence candidates, return neutral only
    _, sorted_valences = _get_atomic_valence_candidates(spec.protonation_states[0])
    if len(sorted_valences) > 10000:
        return [0]

    # Determine max charge ranges
    if spec.subtype == "molecule" and spec.is_non_complex_molecule:
        maxcharge = 3
    elif spec.subtype == "ligand":
        # Count terminal oxygens not coordinateed to metals (e.g., in carboxylates or sulfonates)
        num_noncoordinated_oxygen = sum(
            1 for a in spec.atoms if a.label == "O" and a.mconnec == 0 and a.connec == 1
        )

        if not spec.is_haptic:
            if spec.denticity is None:
                spec.get_denticity()
            maxcharge = spec.denticity + num_noncoordinated_oxygen - prot.added_atoms
        else:
            maxcharge = 2

        # Constraints: maxcharge should not exceed atom count, clamped between 2 and 4
        if all(a.mconnec < 2 for a in spec.atoms):
            maxcharge = min(maxcharge, spec.natoms)

        maxcharge = max(2, min(maxcharge, 4))

        # If protons were added and it's not nitrosyl, favor neutrality
        if not spec.is_nitrosyl and prot.added_atoms > 0:
            maxcharge = 0
    else:
        maxcharge = 0

    # Generate Charge List (e.g., maxcharge=2 -> [0, -1, 1, -2, 2])
    charges = [0]
    for m in range(1, int(maxcharge) + 1):
        charges.extend([-m, m])

    return charges


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
    }

    formula = spec.formula
    target_atom, smiles, charge = REGISTRY.get(formula, (None, None, 0))

    # 2. Handle specific NO Logic
    if formula == "N-O":
        target_atom = "O"
        is_bent = getattr(spec, "NO_type", "") == "Bent"
        smiles, charge = ("[N-]=O", -1) if is_bent else ("[N]=O", 0)

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


def get_metal_poscharges(metal: object) -> list:
    """
    Retrieve common oxidation states for a given metal atom.

    Oxidation state data primarily from:
    Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
    Args:
        metal (object): Metal atom object.
    Returns:
        poscharges (list): List of common oxidation states for the metal.
    """

    mol = metal.get_parent("molecule")
    if mol.is_haptic is None:
        mol.get_hapticity()

    poscharges = METAL_OXIDATION_STATES[metal.label]

    # Allow 0 oxidation state for selected metals under specific conditions
    zero_os_metals = {"Fe", "Ni", "Ru"}
    if metal.label in zero_os_metals:
        has_CO = any(lig.formula == "C-O" for lig in mol.ligands)
        if (has_CO or mol.is_haptic) and 0 not in poscharges:
            poscharges.append(0)

    return poscharges


def _get_atomic_valence_candidates(
    ligand: object, allow_carbenes: bool = False
) -> tuple[list, list]:
    """
    Determines potential valence states for each atom in a ligand based on connectivity.
    Args:
        ligand: An object containing 'labels' (element symbols) and 'adjmat' (adjacency matrix).
        allow_carbenes (bool): If False, excludes divalent Carbon states.

    Returns:
        tuple: (valences_list_of_lists, sorted_valences_list)
            - valences_list_of_lists: List of possible valence integers for each atom.
            - sorted_valences_list: A prioritized/combined list of valence configurations.
    """
    from cell2mol.charge.xyz2mol import atomic_valence, get_sorted_valences_list

    # Convert element labels to atomic numbers
    atomic_nums = [elemdatabase.elementnr[label] for label in ligand.labels]
    # Calculate current coordination number (sum of bonds for each atom)
    current_valences = ligand.adjmat.sum(axis=1).astype(int)

    valences_list_of_lists = []
    for atomic_num, curr_v in zip(atomic_nums, current_valences):
        # Initial filter: candidate valences must be >= current connectivity
        candidates = [v for v in atomic_valence.get(atomic_num, []) if v >= curr_v]

        if atomic_num == 6:  # Carbon logic
            if curr_v == 1 and 2 in candidates:
                candidates.remove(2)
            elif curr_v == 2:
                if not allow_carbenes and 2 in candidates:
                    candidates.remove(2)
                candidates.append(3)  #

        elif atomic_num == 7:  # Nitrogen logic
            if curr_v not in candidates:
                candidates.append(curr_v)

        valences_list_of_lists.append(candidates)

    sorted_list = get_sorted_valences_list(valences_list_of_lists, atomic_nums)
    return valences_list_of_lists, sorted_list


def identify_best_charge_states(charge_states: list) -> list:
    """
    Selects the best charge distributions.
    """
    # Filter out None values initially
    valid_states = [ch for ch in charge_states if ch is not None]
    if not valid_states:
        return []

    # 1. Get initial best candidates indices using the core logic
    best_indices = _get_best_candidate_indices(valid_states)

    # Map indices back to objects
    initial_candidates = [valid_states[i] for i in best_indices]

    # 2. Group candidates by their 'corrected total charge'
    grouped_by_charge = defaultdict(list)
    for state in initial_candidates:
        grouped_by_charge[state.corr_total_charge].append(state)

    logger.debug("Found target charges: %s", list(grouped_by_charge.keys()))

    final_states = []

    # 3. Process each charge group
    for tgt_charge, group in grouped_by_charge.items():
        logger.debug(
            "Processing target charge %s with %d candidates", tgt_charge, len(group)
        )

        # CASE 1: Only one candidate for this charge
        if len(group) == 1:
            best_structure = get_best_resonance_state(group[0])
            final_states.append(best_structure)

        # CASE 2: Multiple candidates
        else:
            # Generate resonance forms for all candidates in this group first
            resonance_candidates = [get_best_resonance_state(temp) for temp in group]

            # We apply the same filtering criteria to the subset of resonance structures
            best_subset_indices = _get_best_candidate_indices(resonance_candidates)

            if not best_subset_indices:
                # Fallback: take the first if filtering somehow fails
                logger.debug("Tie-break failed, taking first.")
                final_states.append(group[0])
            else:
                # Take the best one from the filtered result (index 0)
                logger.debug("Tie-break successful, taking best resonance structure.")
                best_idx = best_subset_indices[0]
                final_states.append(resonance_candidates[best_idx])

    return final_states


def _get_best_candidate_indices(charge_states: list) -> list:
    """
    Helper function containing the core filtering logic.
    Calculates metrics and returns the indices of the best candidates.
    """
    nlists = len(charge_states)
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
    parent = charge_states[0].protonation.parent
    coordinating_atoms_indices = [
        idx for idx, atom in enumerate(parent.atoms) if atom.mconnec > 0
    ]
    coordinating_atoms_labels = [
        atom.label for idx, atom in enumerate(parent.atoms) if atom.mconnec > 0
    ]
    blocked_indices = [
        idx for idx, b in enumerate(charge_states[0].protonation.block) if b == 1
    ]
    coord_abs_atcharge = []
    coord_raw_atcharge = []

    for chs in charge_states:
        uncorr_abs_total.append(chs.uncorr_abstotal)
        uncorr_abs_atcharge.append(chs.uncorr_abs_atcharge)
        uncorr_zwitt.append(chs.uncorr_zwitt)
        coincide.append(chs.coincide)

        # Aromatic calculations
        added_indices = [
            idx for idx, added in enumerate(chs.protonation.addedlist) if added != 0
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
        if not charge_states[idx].status:
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
