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
    BRIDGED_CLUSTER_CHARGES,
    check_rdkit_obj_connectivity,
    generate_rdkit_mol_from_AC2mol,
    generate_rdkit_mol_from_rdDetermineBonds,
)
from cell2mol.charge.special_cases import (
    generate_porphyrin_charge_state,
    _find_charged_moiety,
    generate_oxidised_donor_charge_states,
    generate_special_charge_states,
)
from cell2mol.charge.smiles_handler import (
    collapse_hypervalent_ylides,
    shift_13_ylides,
    fix_zwitterions,
    obligate_charge_separation_atoms,
)

from cell2mol.elementdata import ElementData
from rdkit import Chem

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.metal import Metal

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def enumerate_possible_charge_states(spec: Specie) -> list[ChargeState] | None:
    """Valid charge states (Lewis structures) for a ligand or non-complex molecule.

    Returns the best state per plausible charge, or None if none were found.
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
    # Why bond assignment gave up, when it gave up for a reportable reason.
    diagnostics: dict = {}

    if spec.has_porphyrin and not spec.protonation_warning:
        for prot in spec.protonation_states:
            charge_state = generate_porphyrin_charge_state(prot)
            if charge_state is not None and charge_state.status:
                valid_charge_states.append(charge_state)

    # The porphyrin path is a fast path, not a commitment: it can decline a ring
    # that matched the topology, so fall through on the RESULT, not the match.
    if not valid_charge_states:
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
                prot, candidate_charges, diagnostics=diagnostics
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

    # 5. Oxidised forms the search cannot reach. After selection, never through
    # it: these are rungs on a redox ladder, not competing drawings of one charge.
    best_candidates = _with_oxidised_donor_states(best_candidates)

    # Only meaningful when nothing was found: a specie that aborted on one
    # protonation state and succeeded on another is characterised, not skipped.
    found_nothing = not best_candidates
    spec.valence_search_too_large = bool(
        diagnostics.get("valence_search_too_large") and found_nothing
    )
    spec.bond_perception_capped = bool(
        diagnostics.get("bond_perception_capped") and found_nothing
    )

    return best_candidates if best_candidates else None


def _with_oxidised_donor_states(states: list[ChargeState]) -> list[ChargeState]:
    """Append the oxidised states of a tetrathiafulvalene-type donor. Only ever
    adds, and only charges the search did not already reach.
    """
    neutral = next((s for s in states if s.specie_total_charge == 0), None)
    if neutral is None:
        return states

    found = {s.specie_total_charge for s in states}
    added = [
        state
        for state in generate_oxidised_donor_charge_states(neutral)
        if state.specie_total_charge not in found
    ]
    if added:
        logger.debug(
            "Tetrathiafulvalene-type donor %s: adding oxidised states %s",
            neutral.protonation.formula,
            [s.specie_total_charge for s in added],
        )
    return states + added


def get_candidate_charges(spec: Specie, prot: Protonation) -> list[int]:
    """Formal charges to try for ONE protonation state, in the PROTONATED frame and
    deliberately not re-based per state -- holding them fixed while nH grows is
    what walks the specie charge down. Sources: fixed formulas, moiety anchor,
    else a sweep; a parity screen then halves the result.
    """
    anchor = _anchored_specie_charges(spec)
    candidates = list(anchor) if anchor is not None else _sweep_charges(spec)

    kept = _filter_by_electron_parity(candidates, prot)
    if not kept:
        # Wrong parity everywhere proves the anchor wrong: an anionic centre the
        # scan cannot see. Widen DOWNWARD only (an anchor is a lower bound); one
        # step down flips the parity, so the widened set always survives.
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
    """Drop charges that cannot close a shell: bond perception pairs every electron,
    so ``sum(outer electrons) - q`` must be even. May return empty, which the
    caller reads as the anchor being wrong.
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

    # Cations never anchor alone, only OFFSET detected anions: the scan has no
    # vocabulary for delocalised anions, so a cation-only specie is one whose
    # negative charge it cannot see. Falling through to the sweep finds it.
    if not any(net_charge < 0 for *_, net_charge in charged_moieties):
        return None

    # The summed moiety charge ALONE -- guessing at what the scan misses
    # over-counts, and the protonation sweep reaches it anyway.
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
    if spec.is_non_complex_molecule:
        return [0, -1, 1, -2, 2, -3, 3, -4, 4]
    return [0, -1, 1, -2, 2]


def _dedupe(values: list[int]) -> list[int]:
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(values))


def _manual_charge_state_from_adjacency(spec, charge: int):
    """ChargeState for a bridged cluster (arachno-[B3H8]-), whose 3c-2e B-H-B bridges
    no SMILES can express. The graph is taken from the specie and the known charge
    asserted on the lowest-indexed heavy atom, by convention.
    """
    adjmat = np.asarray(spec.adjmat, dtype=int)

    rwmol = Chem.RWMol()
    for atnum in spec.get_atomic_numbers():
        rwmol.AddAtom(Chem.Atom(int(atnum)))
    for i in range(spec.natoms):
        for j in range(i + 1, spec.natoms):
            if adjmat[i, j] != 0:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)

    for atom in rwmol.GetAtoms():
        # Bridge-bonded atoms already carry every bond they have; never let
        # RDKit pad the open valences with implicit hydrogens.
        atom.SetNoImplicit(True)

    heavy = [i for i, label in enumerate(spec.labels) if label != "H"]
    charge_site = heavy[0] if heavy else 0
    rwmol.GetAtomWithIdx(charge_site).SetFormalCharge(charge)

    mol = rwmol.GetMol()
    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    smiles = Chem.MolToSmiles(mol)

    logger.debug(
        "Manual bridged-cluster charge: %s | total %d on atom %d | SMILES: %s",
        spec.formula,
        charge,
        charge_site,
        smiles,
    )

    return ChargeState(
        status=True,
        protonated_total_charge=charge,
        protonated_atom_charges=atom_charges,
        rdkit_obj=mol,
        smiles=smiles,
        charge_tried=charge,
        allow=True,
        protonation=spec.protonation_states[0],
        specie_rdkit_obj=mol,
        specie_smiles=smiles,
    )


def _manual_anchor_index(mol, target_atom: str) -> int | None:
    """Index of ``target_atom`` in a registry SMILES. Hard-coding 1 breaks "N-O3",
    written [N+](=O)([O-])[O-] with nitrogen first, which permuted the mol against
    spec.atoms and left create_bonds_specie with no bonds.
    """
    if target_atom == "central":
        # The middle of a symmetric chain: azide's inner N, triiodide's inner I.
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 2:
                return atom.GetIdx()
        return None

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == target_atom:
            return atom.GetIdx()
    return None


def _manual_label_order(mol, spec) -> list[int] | None:
    """Permutation putting a registry SMILES into ``spec.atoms`` order by element.

    Only valid when every element in the SMILES appears once, which makes the
    mapping unique; returns None otherwise so the caller falls back to its anchor.
    """
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(set(symbols)) != len(symbols):
        return None

    atoms = getattr(spec, "atoms", None) or []
    if len(atoms) != len(symbols):
        return None

    position = {symbol: idx for idx, symbol in enumerate(symbols)}
    try:
        # RenumberAtoms reads this as "atom i of the result is old atom order[i]".
        return [position[atom.label] for atom in atoms]
    except KeyError:
        logger.warning(
            "Manual Charge: %s has labels %s that the registry SMILES %s does "
            "not cover; keeping the SMILES atom order",
            spec.formula,
            [a.label for a in atoms],
            Chem.MolToSmiles(mol),
        )
        return None


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
        "F6-Si": ("Si", "F[Si-2](F)(F)(F)(F)F", -2),
        "O2": (None, "[O-][O-]", -2),
        "Br3": ("central", "Br[Br-]Br", -1),
        "C-N-S": (None, "[N-]=C=S", -1),
    }

    formula = spec.formula

    # 1b. Bridged clusters are built from adjacency, not from a SMILES string.
    if formula in BRIDGED_CLUSTER_CHARGES:
        return _manual_charge_state_from_adjacency(
            spec, BRIDGED_CLUSTER_CHARGES[formula]
        )

    target_atom, smiles, charge = REGISTRY.get(formula, (None, None, 0))

    # 2. Handle specific NO Logic
    if formula == "N-O":
        target_atom = "O"
        is_bent = getattr(spec, "NO_type", "") == "Bent"
        smiles, charge = ("[N-]=O", -1) if is_bent else ("[N]=O", 0)
    assert smiles is not None, f"No manual SMILES registered for formula {formula}"

    # 3. Determine Atom Ordering
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    order = list(range(spec.natoms))
    new_order = order

    if not target_atom:
        label_order = _manual_label_order(mol, spec)
        if label_order is not None:
            new_order = label_order

    if target_atom:
        # Where the anchor sits in the registry SMILES. Hard-coding 1 breaks
        # "N-O3", written nitrogen-first, which permutes the mol against
        # spec.atoms and leaves create_bonds_specie with no bonds.
        smiles_idx = _manual_anchor_index(mol, target_atom)
        if smiles_idx is None:
            logger.warning(
                "Manual Charge: anchor %r not found in SMILES %s for %s; "
                "keeping the SMILES atom order",
                target_atom,
                smiles,
                formula,
            )
        else:
            for idx, atom in enumerate(spec.atoms):
                # 'central' means the atom connected to two others of the same type
                if target_atom == "central":
                    adj_labels = [
                        spec.get_parent("molecule").labels[a] for a in atom.adjacency
                    ]
                    if adj_labels.count(atom.label) == 2:
                        new_order = reorder_element(order, smiles_idx, idx)
                        break
                # Specific label match (e.g., "Cl" in O4-Cl)
                elif atom.label == target_atom:
                    new_order = reorder_element(order, smiles_idx, idx)
                    break

    # 4. RDKit Processing
    mol = Chem.RenumberAtoms(mol, new_order)

    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    total_charge = sum(atom_charges)

    logger.debug(
        "Manual Charge: %s | SMILES: %s | Order: %s", formula, smiles, new_order
    )

    # Supply specie fields directly: nH is always 0 here, so the specie IS the
    # protonated structure, and _build_specie_mol would sanitize away the
    # registry Lewis structure the comment above protects.
    return ChargeState(
        status=True,
        protonated_total_charge=total_charge,
        protonated_atom_charges=atom_charges,
        rdkit_obj=mol,
        smiles=smiles,
        charge_tried=charge,
        allow=True,
        protonation=spec.protonation_states[0],
        specie_rdkit_obj=mol,
        specie_smiles=smiles,
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
    atom_site_labels: list[str] | None = None,
) -> bool:
    """Charge-independent pre-check. False if any atom has no possible valences, has
    a bonded degree past its maximum valence (unsatisfiable at any charge), or is
    P/As/Sb/Te with a degree outside its possible valences.
    """
    ac = np.asarray(ac)
    n_atoms = len(atoms)
    if n_atoms == 1:
        logger.debug("Single atom specie - skipping rddeterminebonds")
        return False
    for i in range(n_atoms):
        atom_num = atoms[i]
        atom_symbol = elemdatabase.elementsym[atom_num]
        # atom_site_labels covers only the specie's original atoms; a
        # protonation state's added protons sit past the end and have no CIF
        # site label of their own.
        atom_site_label = (
            atom_site_labels[i]
            if atom_site_labels and i < len(atom_site_labels)
            else None
        )
        atom_info = (
            f"{atom_symbol} ({atom_site_label})" if atom_site_label else atom_symbol
        )
        valences = _get_possible_valences(atom_num, atomic_valence)

        if not valences:
            logger.debug(f"Atom {atom_info} has no possible valences defined")
            return False

        degree = int(np.count_nonzero(ac[i]))
        max_valence = max(valences)

        if degree > max_valence:
            logger.debug(
                f"Atom {atom_info} has degree {degree} exceeding max "
                f"possible valence {max_valence} (valences: {valences}) "
                "— no bond ordering possible at any charge"
            )
            return False
        elif degree not in valences and atom_num in [15, 33, 51, 52]:  # P, As, Sb, Te
            logger.debug(
                f"Atom {atom_info} has degree {degree} not in possible valences "
                f"{valences} — may require special handling"
            )
            return False

    return True


def generate_valid_charge_states(
    prot,
    candidate_charges,
    allow_charged_fragments=True,
    diagnostics=None,
    always_use_modified_AC2mol=False,
):
    valid_charge_states_dict = {charge: [] for charge in candidate_charges}

    # --- Tier 0: element-level pre-check (charge-independent) ---
    if not _specie_supported_by_rddeterminebonds(
        prot.atnums, prot.adjmat, atom_site_labels=prot.atom_site_labels
    ):
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
            diagnostics=diagnostics,
        )

    # --- Tier 1: try rdDetermineBonds across ALL candidate charges ---
    valid_charge_states_dict = determine_bond_using_rdDetermineBonds(
        prot,
        candidate_charges,
        valid_charge_states_dict,
        allow_charged_fragments,
        diagnostics=diagnostics,
    )
    # Every charge, not just the ones rdDetermineBonds missed. Still additive:
    # the two drawings compete within their own charge.
    if always_use_modified_AC2mol:
        logger.debug(
            "always_use_modified_AC2mol set for %s; running modified AC2mol on all "
            "candidate charges %s",
            prot.formula,
            candidate_charges,
        )
        return determine_bond_using_modified_AC2mol(
            prot,
            candidate_charges,
            valid_charge_states_dict,
            allow_charged_fragments=allow_charged_fragments,
            diagnostics=diagnostics,
        )

    unsolved = [q for q in candidate_charges if not valid_charge_states_dict[q]]

    # CONTESTED: a drawing exists at this charge but every one looks forced, so
    # downstream has nothing better to prefer. Ask AC2mol for a second opinion.
    contested = [
        q
        for q in candidate_charges
        if valid_charge_states_dict[q]
        and all(_drawing_looks_forced(state) for state in valid_charge_states_dict[q])
    ]
    if not unsolved and not contested:
        logger.debug(
            "rdDetermineBonds found valid charge states for %s: %s",
            prot.formula,
            valid_charge_states_dict,
        )
        return valid_charge_states_dict

    # Tier 2, per charge rather than all-or-nothing: the two perceivers disagree
    # about what is reachable. Safe only because identify_best_charge_states
    # reports one state per charge, so a recovery is additive.
    logger.debug(
        "rdDetermineBonds left charges %s unsolved and %s poorly drawn "
        "for %s; trying modified AC2mol",
        unsolved,
        contested,
        prot.formula,
    )
    return determine_bond_using_modified_AC2mol(
        prot,
        unsolved + contested,
        valid_charge_states_dict,
        allow_charged_fragments=allow_charged_fragments,
        diagnostics=diagnostics,
    )


def determine_bond_using_modified_AC2mol(
    prot,
    candidate_charges,
    valid_charge_states_dict,
    allow_charged_fragments=True,
    diagnostics=None,
):
    for charge in candidate_charges:
        rdkit_obj = generate_rdkit_mol_from_AC2mol(
            atoms=prot.atnums,
            AC=prot.adjmat,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
            diagnostics=diagnostics,
        )
        if rdkit_obj is not None:
            obj, fixed = fix_zwitterions(rdkit_obj)
            obj, collapsed = collapse_hypervalent_ylides(obj)
            obj, shifted = shift_13_ylides(obj)
            if fixed or collapsed or shifted:
                rdkit_obj = obj
            charge_state = prepare_ChargeState_from_rdkit_obj(
                rdkit_obj, prot, charge, allow_charged_fragments=allow_charged_fragments
            )
            if charge_state.status:
                valid_charge_states_dict[charge].append(charge_state)

    return valid_charge_states_dict


def determine_bond_using_rdDetermineBonds(
    prot,
    candidate_charges,
    valid_charge_states_dict,
    allow_charged_fragments=True,
    diagnostics=None,
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
                obj, fixed = fix_zwitterions(rdkit_obj)
                obj, collapsed = collapse_hypervalent_ylides(obj)
                obj, shifted = shift_13_ylides(obj)
                if fixed or collapsed or shifted:
                    rdkit_obj = obj
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
        except RuntimeError as e:
            # The iteration cap. RDKit raises a bare RuntimeError for it, so
            # match on the message rather than swallow every RuntimeError --
            # anything else here is a real fault and must keep propagating.
            if "Max Iterations Exceeded" not in str(e):
                raise
            logger.error(
                "  rdDetermineBonds hit its iteration cap with %s charge for %s; "
                "falling back to modified AC2mol",
                charge,
                prot.formula,
            )
            if diagnostics is not None:
                diagnostics["bond_perception_capped"] = True
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
):
    smiles = Chem.MolToSmiles(rdkit_obj)
    logger.debug(
        "Generated RDKit object for %s | Protonation: %s | Charge: %d | SMILES: %s | n_protons_added: %d",
        prot.parent.formula,
        prot.formula,
        charge,
        smiles,
        prot.n_protons_added,
    )

    atom_charges = [atom.GetFormalCharge() for atom in rdkit_obj.GetAtoms()]
    total_charge = int(sum(atom_charges))

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


def get_plausible_metal_os(metal: Metal) -> list[int]:
    """Common oxidation states for a metal atom.

    Source: Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
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
    """Best Lewis structure for each charge the specie can validly carry -- one state
    per charge, not one winner. Choosing between charges needs cell neutrality and
    metal oxidation states, so it belongs to the balancer.
    """
    # Filter out None values initially
    valid_charge_states = [ch for ch in charge_states if ch is not None and ch.status]
    if not valid_charge_states:
        return []

    # Group by specie charge FIRST, so ranking only ever compares like with
    # like. Sorted so the reported order does not depend on enumeration order.
    grouped_by_charge = defaultdict(list)
    for state in valid_charge_states:
        grouped_by_charge[state.specie_total_charge].append(state)

    logger.debug("Found target charges: %s", sorted(grouped_by_charge))

    final_states = []

    # Pick the best structure within each charge
    for tgt_charge in sorted(grouped_by_charge):
        candidates = grouped_by_charge[tgt_charge]
        logger.debug(
            "Processing target charge %s with %d candidates",
            tgt_charge,
            len(candidates),
        )
        candidates = _preferring_uncharged_carbons(candidates, tgt_charge)

        # CASE 1: Only one candidate for this charge
        if len(candidates) == 1:
            best_structure = candidates[0]

        # CASE 2: Multiple candidates
        else:
            best_subset_indices = _get_best_candidate_indices(candidates)
            if not best_subset_indices:
                logger.debug("Tie-break failed, taking first.")
                best_structure = candidates[0]
            else:
                logger.debug("Tie-break successful, taking best structure.")
                best_structure = candidates[best_subset_indices[0]]

        final_states.append(_better_drawn_sibling(best_structure, candidates))

    kept = _drop_lone_pair_stripped_charges(final_states)
    kept = _drop_cationic_donors(kept)
    kept = _drop_charged_uncoordinated_carbons(kept)
    return _drop_dominated_charges(kept)


def _better_drawn_sibling(winner: ChargeState, pool: list[ChargeState]) -> ChargeState:
    """Trade a charge-separated winner for a sibling at the same charge that is
    strictly better drawn: no worse on either ranking axis, better on one.
    Selection, not enumeration -- the cleaner drawing is already in hand.
    """
    if len(pool) < 2 or not _is_artifact_zwitterion(winner):
        return winner

    best = winner
    for candidate in pool:
        if candidate is best:
            continue
        q_new, q_best = _structural_quality(candidate), _structural_quality(best)
        if q_new[0] >= q_best[0] and q_new[1] <= q_best[1] and q_new != q_best:
            best = candidate

    if best is not winner:
        logger.debug(
            "   Better-drawn sibling at charge %+d: %s -> %s (%s)",
            winner.specie_total_charge,
            _structural_quality(winner),
            _structural_quality(best),
            best.specie_smiles,
        )
    return best


def _protonation_pattern(state: ChargeState) -> tuple:
    """Protons added per site -- which states are drawings of the same structure."""
    return tuple(state.site_proton_counts or ())


def _bond_order_signature(state: ChargeState):
    """The state's bonding without its charge: proton pattern, atom count, bonds.
    ``None`` when there is no mol, so the state compares equal to nothing.
    """
    mol = state.rdkit_obj
    if mol is None:
        return None
    bonds = tuple(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                b.GetBondTypeAsDouble(),
            )
            for b in mol.GetBonds()
        )
    )
    return tuple(state.site_proton_counts or ()), mol.GetNumAtoms(), bonds


def _drop_lone_pair_stripped_charges(states: list[ChargeState]) -> list[ChargeState]:
    """Drop a charge whose drawing has the same bonds as a less charged one.

    Same bonds and a different charge means nothing but deleted lone pairs: DMSO
    comes back as both `CS(=O)C` and `C[S+2](=O)C`, and so does every solvent with
    a lone pair. Ranking cannot catch it -- both score zero on either axis, and
    `_drop_dominated_charges` refuses |net charge| as an axis to protect ligands
    genuinely drawable at two charges (catecholate/quinone), which are safe here
    because they change bonds to do it. Equal |charge| never drops: +2 and -2 on
    one skeleton is a real ambiguity for the balancer.
    """
    if len(states) < 2:
        return states

    signatures = [_bond_order_signature(s) for s in states]
    kept = []
    for idx, signature in enumerate(signatures):
        beaten_by = (
            None
            if signature is None
            else next(
                (
                    other.specie_total_charge
                    for jdx, other in enumerate(states)
                    if jdx != idx
                    and signatures[jdx] == signature
                    and abs(other.specie_total_charge)
                    < abs(states[idx].specie_total_charge)
                ),
                None,
            )
        )
        if beaten_by is None:
            kept.append(states[idx])
        else:
            logger.debug(
                "   Dropping charge %+d: same bonds as charge %+d, the extra "
                "charge is a stripped lone pair (%s)",
                states[idx].specie_total_charge,
                beaten_by,
                states[idx].specie_smiles,
            )

    return kept or states


def _has_cationic_donor(state: ChargeState) -> bool:
    """True when an atom coordinating a metal carries a positive formal charge."""
    parent = getattr(state.protonation, "parent", None)
    atoms = getattr(parent, "atoms", None) or []
    charges = state.specie_atom_charges or []
    if not atoms or len(atoms) != len(charges):
        return False
    return any((a.mconnec or 0) > 0 and q > 0 for a, q in zip(atoms, charges))


def _drop_cationic_donors(states: list[ChargeState]) -> list[ChargeState]:
    """Drop charges that put a positive formal charge on a metal donor, as long as
    something is left. A donor gives electron density to the metal, so a cationic
    one says bond perception drew the ligand wrong rather than that the specie is a
    cation -- CF3 read as F[C+](F)F instead of the carbanion. Kept as a filter of
    last resort: where every state is like this (a linear nitrosyl, say) the drop is
    declined, so a genuinely cationic donor still gets a charge.
    """
    if len(states) < 2:
        return states

    clean = [s for s in states if not _has_cationic_donor(s)]
    if not clean or len(clean) == len(states):
        return states

    for state in states:
        if _has_cationic_donor(state):
            logger.debug(
                "   Dropping charge %+d: positive formal charge on a metal donor (%s)",
                state.specie_total_charge,
                state.specie_smiles,
            )
    return clean


def _has_charged_uncoordinated_carbon(state: ChargeState) -> bool:
    """True when a carbon that coordinates nothing carries a formal charge."""
    parent = getattr(state.protonation, "parent", None)
    atoms = getattr(parent, "atoms", None) or []
    charges = state.specie_atom_charges or []
    if not atoms or len(atoms) != len(charges):
        return False
    return any(
        a.label == "C" and (a.mconnec or 0) == 0 and q != 0
        for a, q in zip(atoms, charges)
    )


def _drawing_looks_forced(state: ChargeState) -> bool:
    """True when a charge sits somewhere only for want of anywhere better: on a
    carbon bonded to no metal, or separated with nothing obliging it. Grounds for
    a second opinion, not a verdict.
    """
    return _has_charged_uncoordinated_carbon(state) or _is_artifact_zwitterion(state)


def _preferring_uncharged_carbons(
    candidates: list[ChargeState], charge: int
) -> list[ChargeState]:
    """:func:`_drop_charged_uncoordinated_carbons` applied within one charge instead
    of between charges: same claim, so the cleaner drawing wins and no hypothesis
    is lost. Abstains when none is clean, sparing a genuine free carbanion.
    """
    clean = [c for c in candidates if not _has_charged_uncoordinated_carbon(c)]
    if not clean or len(clean) == len(candidates):
        return candidates

    logger.debug(
        "   Charge %+d: preferring %d of %d drawing(s) that leave uncoordinated "
        "carbons neutral",
        charge,
        len(clean),
        len(candidates),
    )
    return clean


def _drop_charged_uncoordinated_carbons(
    states: list[ChargeState],
) -> list[ChargeState]:
    """Drop charges that park formal charge on a carbon bonded to no metal, as long
    as something is left.
    """
    if len(states) < 2:
        return states

    clean = [s for s in states if not _has_charged_uncoordinated_carbon(s)]
    if not clean or len(clean) == len(states):
        return states

    for state in states:
        if _has_charged_uncoordinated_carbon(state):
            logger.debug(
                "   Dropping charge %+d: formal charge on a carbon that "
                "coordinates no metal (%s)",
                state.specie_total_charge,
                state.specie_smiles,
            )
    return clean


def _artifact_separation(state: ChargeState) -> int:
    """How much formal charge merely CANCELS, discounting obligate pairs:
    ``min(positive total, negative total)``. A bipyridinium dication scores 0
    (nothing cancels); a stray [C+]..[c-] scores 1, the actual defect.

    Read in the SPECIE frame: in the protonated one an added proton neutralises
    the anionic half of a pair and the min() reads 0 while the cation remains.
    Net charge cannot leak in -- min(pos, neg) is blind to it.
    """
    charges = state.specie_atom_charges or state.protonated_atom_charges or []
    mol = state.specie_rdkit_obj if state.specie_atom_charges else state.rdkit_obj
    obligate: set[int] = set()
    if mol is not None and mol.GetNumAtoms() == len(charges):
        obligate = obligate_charge_separation_atoms(mol)

    free = [q for idx, q in enumerate(charges) if idx not in obligate]
    return min(sum(q for q in free if q > 0), -sum(q for q in free if q < 0))


# Exocyclic partners whose charge-separated form keeps a ring aromatic: a
# pyridone qualifies, an aza-fulvene does not.
_YLIDIC_EXOCYCLIC_PARTNERS = frozenset({7, 8, 16, 34})  # N, O, S, Se


def _unearned_aromatic_atoms(mol) -> set[int]:
    """Atoms RDKit calls aromatic in a six-ring that has not earned it: one holding
    a divalent chalcogen, or a neutral three-connected nitrogen with no exocyclic
    double bond to O/N/S/Se.
    """
    if mol is None:
        return set()
    try:
        kekulized = Chem.Mol(mol)
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
    except Exception:  # unkekulizable: no bond orders to judge, so no discount
        return set()

    unearned: set[int] = set()
    earned: set[int] = set()
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        ring_atoms = set(ring)
        neutral_trivalent_n = any(
            mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
            and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0
            and mol.GetAtomWithIdx(idx).GetTotalDegree() == 3
            for idx in ring
        )
        ylidic = any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtomIdx(idx) not in ring_atoms
            and kekulized.GetAtomWithIdx(bond.GetOtherAtomIdx(idx)).GetAtomicNum()
            in _YLIDIC_EXOCYCLIC_PARTNERS
            for idx in ring
            for bond in kekulized.GetAtomWithIdx(idx).GetBonds()
        )
        divalent_chalcogen = any(
            mol.GetAtomWithIdx(idx).GetAtomicNum() in (16, 34)
            and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0
            and mol.GetAtomWithIdx(idx).GetTotalDegree() == 2
            for idx in ring
        )
        fails = len(ring) == 6 and (
            divalent_chalcogen or (neutral_trivalent_n and not ylidic)
        )
        (unearned if fails else earned).update(ring_atoms)

    return unearned - earned


def _structural_quality(state: ChargeState) -> tuple[int, int]:
    """``(aromatic_atoms, artifact_separation)`` -- maximise the first, minimise the
    second. Aromaticity is read in the protonated frame: it judges the drawing, and
    the specie's charge is the hypothesis under test, not evidence against it.
    Separation is read on the specie; see :func:`_artifact_separation`.
    """
    added_indices = [
        idx for idx, n in enumerate(state.protonation.site_proton_counts or []) if n > 0
    ]
    aromatic = aromatic_info(state.rdkit_obj, added_indices=added_indices)
    unearned = _unearned_aromatic_atoms(state.rdkit_obj)
    if unearned:
        logger.debug(
            "   Discounting %d unearned aromatic atom(s) %s for charge %+d (%s)",
            len(unearned),
            sorted(unearned),
            state.specie_total_charge,
            state.specie_smiles,
        )
    return int(aromatic["Aromatic atoms"]) - len(unearned), _artifact_separation(state)


def _drop_dominated_charges(states: list[ChargeState]) -> list[ChargeState]:
    """Drop a charge beaten on one structural axis without paying for it on the
    other: less cancelling charge and no less aromatic, or more aromatic at no
    more |charge|. An exact tie within one protonation falls to the lower
    |charge|.
    """
    if len(states) < 2:
        return states

    quality = [_structural_quality(s) for s in states]
    kept = []
    for i, (arom_i, sep_i) in enumerate(quality):
        dominated_by = next(
            (
                (states[j].specie_total_charge, arom_j, sep_j)
                for j, (arom_j, sep_j) in enumerate(quality)
                if j != i
                and (
                    (arom_j >= arom_i and sep_j < sep_i)
                    or (
                        arom_j > arom_i
                        and sep_j <= sep_i
                        and abs(states[j].specie_total_charge)
                        <= abs(states[i].specie_total_charge)
                    )
                    or (
                        arom_j == arom_i
                        and sep_j == sep_i
                        and _protonation_pattern(states[j])
                        == _protonation_pattern(states[i])
                        and abs(states[j].specie_total_charge)
                        < abs(states[i].specie_total_charge)
                    )
                )
            ),
            None,
        )
        if dominated_by is None:
            kept.append(states[i])
        else:
            winner_charge, winner_arom, winner_sep = dominated_by
            logger.debug(
                "   Dropping charge %+d (aromatic %d, separation %d): dominated "
                "by charge %+d (aromatic %d, separation %d)",
                states[i].specie_total_charge,
                arom_i,
                sep_i,
                winner_charge,
                winner_arom,
                winner_sep,
            )

    # Mutual domination is impossible (it needs one axis strictly better in both
    # directions), so `kept` is never empty -- but never hand back nothing.
    return kept or states


def _is_artifact_zwitterion(charge_state: ChargeState) -> bool:
    """``specie_zwitt`` with obligate separation discounted -- nitro and N-oxide
    charges are forced, so only the remainder says bond perception went astray.
    """
    mol = charge_state.specie_rdkit_obj
    charges = charge_state.specie_atom_charges or []
    if mol is None or mol.GetNumAtoms() != len(charges):
        return bool(charge_state.specie_zwitt)

    obligate = obligate_charge_separation_atoms(mol)
    if not obligate:
        return bool(charge_state.specie_zwitt)

    remaining = [q for idx, q in enumerate(charges) if idx not in obligate]
    return any(q > 0 for q in remaining) and any(q < 0 for q in remaining)


def _get_best_candidate_indices(valid_charge_states: list[ChargeState]) -> list[int]:
    """
    Helper function containing the core filtering logic.
    Calculates metrics and returns the indices of the best candidates.
    """
    nlists = len(valid_charge_states)
    if nlists == 0:
        return []

    # --- 1. Extract Metrics ---
    # The two keys are read in DIFFERENT frames on purpose: |net charge| on the
    # specie (a claim about chemistry), separation on the protonated structure
    # (a claim about the drawing).
    specie_abs_totals = []
    prot_abs_atcharges = []
    specie_zwitt = []
    artifact_zwitt = []
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
        prot_abs_atcharges.append(chs.protonated_abs_atcharge)
        specie_zwitt.append(chs.specie_zwitt)
        artifact_zwitt.append(_is_artifact_zwitterion(chs))
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
    min_abs = int(np.min(prot_abs_atcharges))

    # A zwitterion can post a smaller |net charge| purely by pairing a spurious
    # +1 against a spurious -1, so let an artifact-free candidate set the target
    # instead. Aromaticity vetoes the swap: never buy a lower |net charge| at the
    # cost of aromatic rings.
    naive_min_tot = int(np.min(specie_abs_totals))
    best_aromatic_at_naive_min = max(
        aromatic_atoms[idx]
        for idx in range(nlists)
        if specie_abs_totals[idx] == naive_min_tot
    )
    clean_at_min_abs = [
        idx
        for idx in range(nlists)
        if prot_abs_atcharges[idx] == min_abs
        and not artifact_zwitt[idx]
        and aromatic_atoms[idx] >= best_aromatic_at_naive_min
    ]
    if clean_at_min_abs:
        min_tot = int(min(specie_abs_totals[idx] for idx in clean_at_min_abs))
        if min_tot != naive_min_tot:
            logger.debug(
                "   Artifact-zwitterion guard: |net charge| target %d -> %d "
                "(a charge-separated candidate claimed the lower value without "
                "gaining aromaticity)",
                naive_min_tot,
                min_tot,
            )
    else:
        min_tot = naive_min_tot

    max_aromatic = np.max(aromatic_atoms)

    indices_min_tot = {i for i, x in enumerate(specie_abs_totals) if x == min_tot}
    indices_min_abs = {i for i, x in enumerate(prot_abs_atcharges) if x == min_abs}
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
            # coord_abs_atcharge is summed from protonated_atom_charges, so the
            # comparison is like-for-like only now that the left side is also
            # the protonated total: "all the charge separation sits on the
            # coordinating atoms".
            if (prot_abs_atcharges[idx] == coord_abs_atcharge[idx]) and coincide[idx]:
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
        # Several share max aromaticity: keep those, drop the rest. Copying
        # tmplist unmodified here silently ignored aromaticity, letting a
        # non-aromatic tautomer beat an equally charged aromatic one.
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
