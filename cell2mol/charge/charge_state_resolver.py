from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import networkx as nx
from collections import defaultdict

from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation
import logging
from cell2mol.operations import reorder_element
from cell2mol.charge.utils import (
    METAL_OXIDATION_STATES,
    NOBLE_GASES,
    TEFLATE,
    HALOGENS,
    MANUAL_CHARGE_ASSIGN_SPECIES,
    aromatic_info,
    is_sb_halide_only,
    is_fullerene_cage,
    check_fullerene_sphericity,
)
from cell2mol.elementdata import ElementData
from cell2mol.charge.xyz2mol import (
    get_proto_mol,
    AC2mol,
    chiral_stereo_check,
)
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDetermineBonds

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.ligand import Ligand
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.metal import Metal

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

# AC2BO's combinatorial bond-order search can converge on a technically
# valence-consistent but chemically absurd resonance structure: e.g. an
# entire ring system drawn with alternating +1/-1 formal charges and no
# double bonds at all, just to represent a small net charge; anything
# far beyond that is a search artifact, not real chemistry.
MAX_EXCESS_CHARGE_SEPARATION = 4

ALWAYS_AC2BO_ELEMENTS = {33, 51, 83, 34, 52}  # As, Sb, Bi, Se, Te


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
        charge_state = generate_manual_charge_state(spec)
        return [charge_state] if charge_state is not None else None

    # Fullerenes
    is_fullerene, fullerene_reason = is_fullerene_cage(
        spec.get_atomic_numbers(), spec.adjmat
    )

    if is_fullerene:
        if spec.coord is not None and not check_fullerene_sphericity(spec.coord):
            logger.warning(
                "%s passed fullerene topology check but failed sphericity "
                "check - possible disorder/AC artifact; proceeding anyway",
                spec.formula,
            )
        logger.debug(
            "Fullerene cage detected for %s (%s)", spec.formula, fullerene_reason
        )
        charge_state = generate_fullerene_charge_state(spec.protonation_states[0])
        return [charge_state] if charge_state is not None else None

    # Noble gases (lattice/solvate atoms) are always neutral
    if spec.formula in NOBLE_GASES:
        charge_state = generate_noble_gas_charge_state(spec.protonation_states[0])
        return [charge_state] if charge_state is not None else None

    # Teflate (-OTeF5): fixed hexacoordinate Te(VI) bonding pattern
    if spec.formula in TEFLATE:
        charge_state = generate_teflate_charge_state(spec.protonation_states[0])
        return [charge_state] if charge_state is not None else None

    # Antimony halides: SbX6- (Sb(V)) vs SbX3/X4/X5 (Sb(III)-derived)
    if is_sb_halide_only(spec.labels):
        charge_state = generate_sb_halide_charge_state(spec.protonation_states[0])
        return [charge_state] if charge_state is not None else None

    # Haptic Ligands
    is_haptic_c8 = False
    if spec.subtype == "ligand":
        ligand = cast("Ligand", spec)
        is_haptic_c8 = (
            ligand.groups is not None
            and len(ligand.groups) == 1
            and ligand.groups[0].haptic_type == ["eta8(C8)"]
        )
    if is_haptic_c8:
        charge_state = generate_charge_state(-1, spec.protonation_states[0])
        return [charge_state] if charge_state is not None else None

    for prot in spec.protonation_states:
        logger.debug(
            "Detailed info Protonation State formula: %s Number of protons added: %d Mode : %s",
            prot.formula,
            prot.n_protons_added,
            prot.mode,
        )
        logger.debug("detailed: %s\n%s", prot.formula, prot)
        if not prot.status:
            logger.warning(
                "Invalid protonation state found %s (status=%s)",
                prot.formula,
                prot.status,
            )
            logger.debug("detailed: %s\n%s", prot.formula, prot)

    # 3. Enumeration Loop
    valid_charge_states = []

    for prot in spec.protonation_states:
        # Get list of integer charges to attempt for this specific protonation
        candidate_charges = get_candidate_charges(prot)
        for charge in candidate_charges:
            if spec.is_porphyrin and charge == 0:
                allow_charged_fragments = False
            else:
                allow_charged_fragments = True
            # Attempt to build the RDKit object and state
            charge_state = generate_charge_state(
                charge, prot, allow_charged_fragments=allow_charged_fragments
            )

            if charge_state:
                valid_charge_states.append(charge_state)
                logger.debug(
                    "    [Success] %s | Protonation: %s | Charge: %d | Added atoms: %d | SMILES: %s",
                    spec.formula,
                    prot.formula,
                    charge,
                    prot.n_protons_added,
                    charge_state.smiles,
                )
            else:
                logger.debug(
                    "    [Failed]  %s | Protonation: %s | Charge: %d  | Added atoms: %d",
                    spec.formula,
                    prot.formula,
                    charge,
                    prot.n_protons_added,
                )

    # 4. Final Selection / Filtering
    best_candidates = identify_best_charge_states(valid_charge_states)

    return best_candidates if best_candidates else None


def check_possible_valence_problems_from_ac(
    atoms,
    AC,
    *,
    extra_allowed_valences=None,
):
    """
    Check possible valence problems directly from an adjacency/connectivity matrix.

    Parameters
    ----------
    atoms : list[int]
        Atomic numbers.
    AC : array-like, shape (n_atoms, n_atoms)
        Connectivity matrix. Nonzero means bonded.
    extra_allowed_valences : dict[int, list[int]] | None
        Optional extra allowed valences by atomic number.
        Example: {15: [3, 5, 6], 33: [3, 5, 6], 51: [3, 5, 6]}

    Returns
    -------
    problems : list[dict]
        List of suspicious atoms.
    """

    atoms = [int(a) for a in atoms]
    ac = np.asarray(AC)

    n_atoms = len(atoms)

    if ac.shape != (n_atoms, n_atoms):
        raise ValueError(f"AC must have shape ({n_atoms}, {n_atoms}), got {ac.shape}")

    if not np.allclose(ac, ac.T):
        raise ValueError("AC must be symmetric")

    if np.any(np.diag(ac) != 0):
        raise ValueError("AC diagonal must be zero")

    pt = Chem.GetPeriodicTable()
    problems = []

    if extra_allowed_valences is None:
        extra_allowed_valences = {}

    for i, atomic_num in enumerate(atoms):
        symbol = pt.GetElementSymbol(atomic_num)

        if atomic_num == 0:
            continue

        degree = int(np.count_nonzero(ac[i]))

        # Always route through the modified AC2BO bond-order search
        # (see ALWAYS_AC2BO_ELEMENTS), regardless of whether this
        # particular degree happens to match a "normal" valence.
        if atomic_num in ALWAYS_AC2BO_ELEMENTS:
            neighbors = [
                {
                    "atom_idx": int(j),
                    "atomic_num": int(atoms[j]),
                    "symbol": pt.GetElementSymbol(int(atoms[j])),
                }
                for j in np.where(ac[i] != 0)[0]
            ]
            problems.append(
                {
                    "atom_idx": int(i),
                    "atomic_num": int(atomic_num),
                    "symbol": symbol,
                    "degree": degree,
                    "allowed_valences": list(pt.GetValenceList(atomic_num)),
                    "neighbors": neighbors,
                }
            )
            continue

        allowed_valences = list(pt.GetValenceList(atomic_num))

        if atomic_num in extra_allowed_valences.keys():
            allowed_valences = sorted(
                set(allowed_valences) | set(extra_allowed_valences[atomic_num])
            )

        # RDKit uses -1 for flexible/unspecified valence.
        if -1 in allowed_valences:
            continue

        if allowed_valences and (
            degree > max(allowed_valences)
            or (degree not in allowed_valences and symbol not in ["C", "O"])
        ):
            neighbors = [
                {
                    "atom_idx": int(j),
                    "atomic_num": int(atoms[j]),
                    "symbol": pt.GetElementSymbol(int(atoms[j])),
                }
                for j in np.where(ac[i] != 0)[0]
            ]

            problems.append(
                {
                    "atom_idx": int(i),
                    "atomic_num": int(atomic_num),
                    "symbol": symbol,
                    "degree": degree,
                    "allowed_valences": allowed_valences,
                    "neighbors": neighbors,
                }
            )

    return problems


def generate_rdkit_mol_from_rdDetermineBonds(
    atoms,
    coords,
    AC,
    charge=0,
    sanitize=True,
    allow_charged_fragments=True,
    embed_chiral=True,
):
    """
    Build an RDKit Mol object from atomic numbers, coordinates, and
    adjacency matrix from a protonation state, and
    then assign bond orders using rdDetermineBonds.DetermineBondOrders.
    Parameters
    ----------
    atoms : list[int]
        Atomic numbers, e.g. [6, 1, 1, 1, 1]
    coords : array-like, shape (n_atoms, 3)
        Cartesian coordinates.
    AC : array-like, shape (n_atoms, n_atoms)
        Connectivity matrix. Nonzero means bonded.
    charge : int
        Total molecular charge.
    sanitize : bool
        Whether to sanitize after bond-order assignment.
    allow_charged_fragments : bool
        Whether to allow charged fragments.
    embed_chiral : bool
        Whether to embed chiral information.

    Returns
    -------
    mol : rdkit.Chem.Mol
    """

    atoms = list(map(int, atoms))
    coords = np.asarray(coords, dtype=float)
    ac = np.asarray(AC)
    n_atoms = len(atoms)

    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords must have shape ({n_atoms}, 3), got {coords.shape}")

    if ac.shape != (n_atoms, n_atoms):
        raise ValueError(f"AC must have shape ({n_atoms}, {n_atoms}), got {ac.shape}")

    rwMol = Chem.RWMol()

    # Add atoms
    for atomic_num in atoms:
        rwMol.AddAtom(Chem.Atom(atomic_num))

    # Add connectivity as single bonds first
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if ac[i, j] != 0:
                rwMol.AddBond(i, j, Chem.BondType.SINGLE)

    mol = rwMol.GetMol()

    # Add coordinates
    conf = Chem.Conformer(n_atoms)
    conf.Set3D(True)

    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    mol.AddConformer(conf, assignId=True)

    # Assign bond orders using existing connectivity
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=charge,
        allowChargedFragments=allow_charged_fragments,
        embedChiral=embed_chiral,
    )

    if sanitize:
        Chem.SanitizeMol(mol)

    return mol


def generate_rdkit_mol_from_AC2mol(
    atoms,
    AC,
    charge=0,
    allow_charged_fragments=True,
    embed_chiral=True,
):
    """
    Build an RDKit Mol object from atomic numbers, coordinates, and
    adjacency matrix from a protonation state, and
    then assign bond orders
    - using rdDetermineBonds.DetermineBondOrders.
    - modified AC2mol from xyz2mol
    Parameters
    ----------
    atoms : list[int]
        Atomic numbers, e.g. [6, 1, 1, 1, 1]
    coords : array-like, shape (n_atoms, 3)
        Cartesian coordinates.
    AC : array-like, shape (n_atoms, n_atoms)
        Connectivity matrix. Nonzero means bonded.
    charge : int
        Total molecular charge.
    sanitize : bool
        Whether to sanitize after bond-order assignment.
    allow_charged_fragments : bool
        Whether to allow charged fragments.
    embed_chiral : bool
        Whether to embed chiral information.

    Returns
    -------
    mol : rdkit.Chem.Mol
    """
    new_mols, BO = AC2mol(
        mol=get_proto_mol(atoms),
        AC=AC,
        atoms=atoms,
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

    return new_mols[0]  # use the first candidate as default


def generate_charge_state(
    charge: int,
    prot: Protonation,
    allow_charged_fragments: bool = True,
    embed_chiral: bool = True,
    ref_uncorr_atom_charges: list[int] | None = None,
) -> ChargeState | None:
    """
    Generates molecular connectivity and formal atomic charges from 3D coordinates
    using modified AC2mol, then validates chirality and resonance.
    """
    # If protonation state is invalid, do not allow charged fragments
    if not prot.status:
        logger.warning(
            "Protonation state invalid for %s, skipping charge state generation.",
            prot.formula,
        )
        return None

    logger.debug(
        "Protonation Formula: %s | Target Charge: %d | Allow charged fragments: %s | N Protons Added: %d",
        prot.formula,
        charge,
        allow_charged_fragments,
        prot.n_protons_added,
    )

    extra_allowed_valences = {
        # 5: [3, 4],  # B, e.g. BF4-
        7: [2, 3, 4],  # N
        # 15: [3, 5, 6],  # P, e.g. PF6-
        # 33: [3, 5, 6],  # As, e.g. AsF6-
        # 51: [3, 5, 6],  # Sb, e.g. SbF6-
    }

    problems = check_possible_valence_problems_from_ac(
        atoms=prot.atnums, AC=prot.adjmat, extra_allowed_valences=extra_allowed_valences
    )
    if prot.atnums is not None and (
        len(prot.atnums) == 1 or prot.formula in {"C-Se", "C-Te"}
    ):
        logger.debug(
            "Single-atom or special case detected for %s, generating RDKit Mol object using modified AC2mol",
            prot.formula,
        )
        rdkit_obj = generate_rdkit_mol_from_AC2mol(
            atoms=prot.atnums,
            AC=prot.adjmat,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
            embed_chiral=embed_chiral,
        )

        if rdkit_obj is None:
            logger.warning(
                f"Failed to generate RDKit object of prot.formula {prot.formula} for charge {charge} using modified AC2mol"
            )
            return None

    elif problems:
        logger.debug(
            "Possible valence problems detected for %s, generating RDKit Mol object using modified AC2mol",
            prot.formula,
        )
        logger.warning("Possible valence problems detected from AC:")
        for problem in problems:
            logger.warning(
                f"  - Atom {problem['atom_idx']} ({problem['symbol']}): degree {problem['degree']}, allowed valences {problem['allowed_valences']}"
            )
        rdkit_obj = generate_rdkit_mol_from_AC2mol(
            atoms=prot.atnums,
            AC=prot.adjmat,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
            embed_chiral=embed_chiral,
        )
        if rdkit_obj is None:
            logger.warning(
                f"Failed to generate RDKit object of prot.formula {prot.formula} for charge {charge} using modified AC2mol"
            )
            return None
    else:
        try:
            rdkit_obj = generate_rdkit_mol_from_rdDetermineBonds(
                prot.atnums,
                prot.coord,
                prot.adjmat,
                charge,
                sanitize=False,
                allow_charged_fragments=allow_charged_fragments,
                embed_chiral=embed_chiral,
            )
        except Exception as e:
            logger.error(
                f"Error occurred while generating proto molecule: {e} with {charge} charge for {prot.formula}"
            )
            logger.warning(
                f"Failed to generate RDKit object of prot.formula {prot.formula} for charge {charge} using rdDetermineBonds"
            )
            return None

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
    assert prot.natoms is not None
    is_correct = check_rdkit_obj_connectivity(rdkit_obj, prot.natoms, charge)

    logger.debug(
        "Generated ChargeState | SMILES: %s | Total Charge: %d | Correct: %s",
        smiles,
        total_charge,
        is_correct,
    )
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

    # if is_correct:
    #     charge_state = get_best_resonance_state(charge_state)

    return charge_state


def check_rdkit_obj_connectivity(mol: Chem.Mol, natoms: int, charge: int) -> bool:
    """
    Validates the chemical sanity of an RDKit molecule object by checking
    valences, lone pairs, bond connectivity, and that the total formal
    charge actually matches the requested target charge.
    """
    pt = Chem.GetPeriodicTable()
    is_correct = True

    # 0. Total Formal Charge Check
    total_formal_charge = sum(
        mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(natoms)
    )
    if total_formal_charge != charge:
        logger.debug(
            "Total formal charge mismatch: got %d, expected %d",
            total_formal_charge,
            charge,
        )
        is_correct = False

    # 0b. Charge Separation Sanity Check
    # A valid Lewis structure can concentrate the net charge on a small
    # number of atoms (a real zwitterion/ylide), but a structure requiring
    # far more formal charge than the net charge demands is a bond-order
    # search artifact (see MAX_EXCESS_CHARGE_SEPARATION above), not a
    # legitimate resonance form.
    total_abs_atom_charge = sum(
        abs(mol.GetAtomWithIdx(i).GetFormalCharge()) for i in range(natoms)
    )
    if total_abs_atom_charge > abs(charge) + MAX_EXCESS_CHARGE_SEPARATION:
        logger.debug(
            "Excessive charge separation: total |formal charge| on atoms = %d, "
            "target net charge = %d (max allowed excess = %d)",
            total_abs_atom_charge,
            charge,
            MAX_EXCESS_CHARGE_SEPARATION,
        )
        is_correct = False

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
        logger.error("ResonanceMolSupplier failed for %s: %s", parent.formula, e)
        return charge_state

    if num_res <= 1:
        return charge_state

    original_smiles = Chem.MolToSmiles(rdkit_obj, canonical=True)
    logger.debug("Resonance check for %s: found %d forms", parent.formula, num_res)
    logger.debug("  Original: %s", original_smiles)

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
            return charge_state
        if Chem.MolFromSmiles(candidate_smiles) is not None:
            best_res_mol = candidate
            best_smiles = candidate_smiles
            break
        logger.debug("  Rejected (invalid SMILES): %s", candidate_smiles)

    if best_res_mol is None or best_smiles is None:
        logger.debug(
            "No valid resonance alternative found for %s; keeping original",
            parent.formula,
        )
        return charge_state

    logger.debug("  Best    : %s", best_smiles)
    logger.info("Resonance form updated for %s", parent.formula)

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


def get_candidate_charges(prot: Protonation) -> list[int]:
    """
    Determines the range of formal charges to test for a specific protonation state.
    Uses chemical heuristics based on formula, denticity, and atom connectivity.
    """
    spec = cast("Specie", prot.parent)
    formula = spec.formula

    # Quick returns for simple cases
    if formula in {"C-O", "H2-O", "C-N", "C-S", "C-Se", "C-Te", "C-P", "C-As", "C-Sb"}:
        return [0]
    if formula in {"F", "Cl", "Br", "I"}:
        return [-1]

    logger.debug(
        "Evaluating %s (%s) | Number of protonation states: %d",
        spec.formula,
        spec.subtype,
        len(spec.protonation_states or []),
    )

    negative_moieties = _find_non_coordinated_negative_moiety(spec)
    if negative_moieties:
        # Handle cases with negative moieties
        logger.debug(
            "Detected non-coordinated negative moieties for %s: %s",
            formula,
            negative_moieties,
        )

    # Determine max charge ranges
    if spec.subtype == "molecule" and spec.is_non_complex_molecule:
        maxcharge = 4
    elif spec.subtype == "ligand":
        ligand = cast("Ligand", spec)

        if not ligand.is_haptic:
            if ligand.denticity is None:
                ligand.get_denticity()

            is_porphyrin = (
                ligand.is_porphyrin
                if ligand.is_porphyrin is not None
                else ligand.evaluate_as_porphyrin()
            )

            if is_porphyrin:
                if len(negative_moieties) > 0:
                    charges = [-len(negative_moieties)]
                else:
                    charges = [0]
                logger.debug(
                    "Porphyrin detected for %s; limiting charge states to %s",
                    formula,
                    charges,
                )
                return charges

            # The count now accurately reflects structural functional groups
            maxcharge = (
                (ligand.denticity or 0) + len(negative_moieties) - prot.n_protons_added
            )
        else:
            maxcharge = 2

        # Constraints: maxcharge should not exceed atom count, clamped between 2 and 4
        if all((a.mconnec or 0) < 2 for a in ligand.atoms):
            maxcharge = min(maxcharge, ligand.natoms)

        maxcharge = max(2, min(maxcharge, 4))
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


def generate_fullerene_charge_state(prot: Protonation) -> ChargeState | None:
    """
    Builds a closed-shell, net-neutral Kekule structure for a fullerene cage.

    A fullerene skeleton is a bridgeless 3-regular graph, which by Petersen's
    theorem always has a perfect matching. Promoting each matched edge to a
    double bond gives every carbon its 4th bond directly, with no charge
    separation and no combinatorial bond-order search -- the kind of search
    that AC2mol/rdDetermineBonds cannot afford to run over 60+ atoms.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    graph = nx.from_numpy_array(np.asarray(prot.adjmat))
    matching = nx.max_weight_matching(graph, maxcardinality=True)

    if len(matching) * 2 != prot.natoms:
        logger.warning(
            "No perfect matching found for fullerene %s; cannot build a "
            "closed-shell Kekule structure",
            prot.formula,
        )
        return None

    double_bonds = {frozenset(pair) for pair in matching}

    rwmol = Chem.RWMol()
    for atomic_num in prot.atnums:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    for i, j in graph.edges():
        bond_type = (
            Chem.BondType.DOUBLE
            if frozenset((i, j)) in double_bonds
            else Chem.BondType.SINGLE
        )
        rwmol.AddBond(i, j, bond_type)

    mol = rwmol.GetMol()

    conf = Chem.Conformer(prot.natoms)
    conf.Set3D(True)
    for i, xyz in enumerate(np.asarray(prot.coord, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    try:
        # Skip aromaticity perception: RDKit's Hueckel-based model is not
        # meant for curved, fused 5/6-ring cages and can misfire on it. The
        # explicit single/double bonds above already fully describe the
        # structure.
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
        )
    except Exception as e:
        logger.warning(
            "Failed to sanitize fullerene Kekule structure for %s: %s",
            prot.formula,
            e,
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    is_correct = check_rdkit_obj_connectivity(mol, prot.natoms, 0)

    return ChargeState.from_positional(
        is_correct,
        total_charge,
        atom_charges,
        mol,
        smiles,
        0,
        True,
        prot,
    )


def generate_noble_gas_charge_state(prot: Protonation) -> ChargeState | None:
    """
    Noble gas atoms found in a crystal (solvate/lattice atoms, host cavities,
    etc.) are always neutral -- they don't form stable ions under these
    conditions, so skip the bond-order search and assign a single neutral
    atom directly.
    """
    if prot.natoms != 1 or not prot.atnums:
        logger.warning(
            "Unexpected structure for noble gas specie %s; expected a single atom",
            prot.formula,
        )
        return None

    rwmol = Chem.RWMol()
    rwmol.AddAtom(Chem.Atom(prot.atnums[0]))
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)

    smiles = Chem.MolToSmiles(mol)

    return ChargeState.from_positional(
        True,
        0,
        [0],
        mol,
        smiles,
        0,
        True,
        prot,
    )


def generate_teflate_charge_state(prot: Protonation) -> ChargeState | None:
    """
    Builds the -OTeF5 ("teflate", pentafluorooxotellurate(VI)) ligand
    directly from its connectivity: a hexacoordinate Te(VI) bonded to 5 F
    and 1 O, all single bonds, with the -1 charge sitting on the terminal
    O. The bonding pattern is fixed and unambiguous, so this skips the
    general bond-order search entirely.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    o_indices = [i for i, label in enumerate(prot.labels) if label == "O"]
    if len(o_indices) != 1:
        logger.warning(
            "Unexpected structure for teflate specie %s; expected exactly one O",
            prot.formula,
        )
        return None

    rwmol = Chem.RWMol()
    for atomic_num in prot.atnums:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    adjmat = np.asarray(prot.adjmat)
    for i in range(prot.natoms):
        for j in range(i + 1, prot.natoms):
            if adjmat[i, j] != 0:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)

    rwmol.GetAtomWithIdx(o_indices[0]).SetFormalCharge(-1)

    mol = rwmol.GetMol()

    conf = Chem.Conformer(prot.natoms)
    conf.Set3D(True)
    for i, xyz in enumerate(np.asarray(prot.coord, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        logger.warning(
            "Failed to sanitize teflate structure for %s: %s", prot.formula, e
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    is_correct = check_rdkit_obj_connectivity(mol, prot.natoms, -1)

    return ChargeState.from_positional(
        is_correct,
        total_charge,
        atom_charges,
        mol,
        smiles,
        -1,
        True,
        prot,
    )


def generate_sb_halide_charge_state(prot: Protonation) -> ChargeState | None:
    """
    Builds Sb/halogen-only species directly from connectivity.

    Mononuclear species (single Sb center): each Sb's formal charge is set
    from its own halogen coordination number, since degree directly tracks
    oxidation state here. Hexacoordinate Sb (SbX6-) gets formal charge -1
    (0 lone pairs, matching the existing xyz2mol special case for
    AsX6-/PX6-). Every other coordination number (3, 4, 5) keeps one lone
    pair, giving formal charge = 3 - degree: neutral SbX3, SbX4-, SbX5(2-).

    Polynuclear species (multiple Sb centers, e.g. bridged iodoantimonate
    clusters like [Sb7I25]4-): coordination number stops tracking
    oxidation state once halogens bridge between metal centers -- a
    degree-6 Sb here is typically still Sb(III), with the extra
    coordination coming from sharing ligands with neighboring Sb centers,
    not from oxidation. These clusters are essentially always built from
    Sb(III) + halide ligands, so the total charge is fixed at
    3 * n_Sb - n_halogens, but rather than putting +3 on every Sb and -1
    on every halogen (which cancel almost everywhere and just clutter the
    SMILES), Sb centers are drawn neutral and the charge is instead
    distributed across only as many halogens as needed to hit that total
    -- one per Sb, round-robin -- so most halogens are drawn as plain
    covalent X and only the "extra" ones needed to balance the charge are
    shown as X-.
    """
    assert (
        prot.natoms is not None and prot.adjmat is not None and prot.atnums is not None
    )

    sb_indices = [i for i, label in enumerate(prot.labels) if label == "Sb"]
    halogen_indices = [i for i, label in enumerate(prot.labels) if label in HALOGENS]

    if not sb_indices or (len(sb_indices) + len(halogen_indices)) != prot.natoms:
        logger.warning(
            "Unexpected structure for Sb-halide specie %s; expected only Sb and halogens",
            prot.formula,
        )
        return None

    adjmat = np.asarray(prot.adjmat)

    rwmol = Chem.RWMol()
    for atomic_num in prot.atnums:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    for i in range(prot.natoms):
        for j in range(i + 1, prot.natoms):
            if adjmat[i, j] != 0:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)

    is_polynuclear = len(sb_indices) > 1
    for idx in sb_indices:
        sb_atom = rwmol.GetAtomWithIdx(idx)
        sb_atom.SetNoImplicit(True)

        if not is_polynuclear:
            degree = int(np.count_nonzero(adjmat[idx]))
            sb_charge = -1 if degree == 6 else 3 - degree
            sb_atom.SetFormalCharge(sb_charge)
        # Polynuclear Sb centers stay formally neutral here -- the total
        # charge is distributed across the halogens below instead.

    if is_polynuclear:
        # Every halogen may end up with >1 explicit bond (bridging) or a
        # -1 charge, either of which violates RDKit's default halogen
        # valence table, so bypass implicit-valence handling uniformly.
        for j in halogen_indices:
            rwmol.GetAtomWithIdx(j).SetNoImplicit(True)

        # Distribute exactly enough -1 charges across the halogens so the
        # total comes out to 3 * n_Sb - n_halogens: one per Sb, round-robin,
        # so it's spread evenly across centers rather than piled onto a
        # few. Bridging halogens aren't required to carry it -- prefer a
        # terminal neighbor (bonded to just this one Sb) and only fall
        # back to a bridging one if this Sb has no terminal neighbor left.
        halogen_sb_degree = {
            j: sum(1 for idx in sb_indices if adjmat[idx, j] != 0)
            for j in halogen_indices
        }
        n_negative = len(halogen_indices) - 3 * len(sb_indices)
        assigned: set[int] = set()
        remaining = n_negative
        while remaining > 0:
            progressed = False
            for idx in sb_indices:
                if remaining <= 0:
                    break
                candidates = [
                    j
                    for j in halogen_indices
                    if j not in assigned and adjmat[idx, j] != 0
                ]
                if not candidates:
                    continue
                j = min(candidates, key=lambda j: halogen_sb_degree[j])
                rwmol.GetAtomWithIdx(j).SetFormalCharge(-1)
                assigned.add(j)
                remaining -= 1
                progressed = True
            if not progressed:
                logger.warning(
                    "Could not distribute full negative charge across "
                    "halogens for %s; %d unit(s) left unassigned",
                    prot.formula,
                    remaining,
                )
                break

    mol = rwmol.GetMol()

    conf = Chem.Conformer(prot.natoms)
    conf.Set3D(True)
    for i, xyz in enumerate(np.asarray(prot.coord, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    try:
        Chem.SanitizeMol(
            mol,
            # SANITIZE_PROPERTIES: skip, since a neutral bridging halogen
            # (2 bonds, e.g. mu-X in a polynuclear cluster) exceeds
            # RDKit's default halogen valence table.
            # SANITIZE_CLEANUP_ORGANOMETALLICS: skip too, since otherwise
            # RDKit "fixes" that same neutral bridging halogen by silently
            # reinterpreting its second bond as dative (X->Sb) instead of
            # leaving it as the plain single bond we intend.
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_CLEANUP_ORGANOMETALLICS,
        )
    except Exception as e:
        logger.warning(
            "Failed to sanitize Sb-halide structure for %s: %s", prot.formula, e
        )
        return None

    atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    total_charge = sum(atom_charges)
    smiles = Chem.MolToSmiles(mol)

    is_correct = True

    return ChargeState.from_positional(
        is_correct,
        total_charge,
        atom_charges,
        mol,
        smiles,
        total_charge,
        True,
        prot,
    )


def _find_non_coordinated_negative_moiety(
    spec: Specie,
) -> list[tuple[int, list[int], str]]:
    """
    Finds non-coordinated carboxylate (C + 2 terminal O),
    sulfonate (S + 3 terminal O)

    Returns (center_idx, oxygen_idxs, kind) tuples using *local* indices
    into spec.atoms -- the same ordering used to build any RDKit mol from
    this specie's protonation states (prot.atnums/prot.adjmat preserve
    spec.labels order, appending any added protons at the end).
    """
    atoms = spec.atoms or []
    parent_molecule = spec.get_parent("molecule")
    neighbor_source = (
        parent_molecule.atoms
        if parent_molecule is not None and parent_molecule.atoms is not None
        else atoms
    )
    local_idx_by_id = {id(a): idx for idx, a in enumerate(atoms)}

    groups: list[tuple[int, list[int], str]] = []
    for i, atom in enumerate(atoms):
        if atom.label not in ("C", "S"):
            continue

        oxygen_locals = []
        for j in atom.adjacency:
            neighbor = neighbor_source[j]
            if (
                neighbor.label == "O"
                and (neighbor.connec or 0) == 1
                and (neighbor.mconnec or 0) == 0
            ):
                local_j = local_idx_by_id.get(id(neighbor))
                if local_j is not None:
                    oxygen_locals.append(local_j)

        if atom.label == "C" and len(oxygen_locals) == 2:
            groups.append((i, oxygen_locals, "carboxylate"))
        elif atom.label == "S" and len(oxygen_locals) == 3:
            groups.append((i, oxygen_locals, "sulfonate"))

    return groups


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
            # best_structure = get_best_resonance_state(group[0])
            best_structure = group[0]
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


def _get_best_candidate_indices(charge_states: list[ChargeState]) -> list[int]:
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
    parent = cast("Specie", charge_states[0].protonation.parent)
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
            charge_states[0].protonation.site_proton_counts or []
        )
        if n_added == 0
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
