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
    generate_special_charge_states,
)
from cell2mol.charge.smiles_handler import obligate_charge_separation_atoms

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
    # Why bond assignment gave up, when it gave up for a reportable reason.
    diagnostics: dict = {}

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


def _manual_charge_state_from_adjacency(spec, charge: int):
    """ChargeState for a bridged cluster, built from the specie's own adjacency.

    Species like arachno-[B3H8]- are held together by 3-centre-2-electron B-H-B
    bridges, which no SMILES string and no bond-perception search can express --
    a bridging H is bonded to two borons at once, and the borons exceed the
    valence RDKit will accept. Their total charge is nevertheless known from the
    formula, so the graph is taken straight from the specie (every contact a
    single bond) and the known charge asserted on it.

    The charge is delocalized over the cluster framework; it is recorded on the
    lowest-indexed heavy atom purely as a convention, so that the per-atom
    charges sum to the right total.
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
    """Index of a registry SMILES' anchor atom -- the one aligned against the
    matching atom of the specie so the rest of the order follows.

    Read off the SMILES rather than assumed, so adding a registry entry cannot
    silently mis-order its atoms.
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

    if target_atom:
        # Where the anchor sits in the registry SMILES. This used to be
        # hard-coded as 1, which is right for every entry except "N-O3":
        # nitrate is written [N+](=O)([O-])[O-] with its nitrogen first, so the
        # wrong atom was moved and the mol came back permuted against
        # spec.atoms. create_bonds_specie then skipped every bond as an
        # element mismatch and failed with "NO BONDS for N" (ABAMUO).
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

    # Supply the specie fields directly instead of letting ChargeState rebuild
    # them. A manual specie has no protons to strip (nH is always 0 here -- see
    # get_asis_protonation_state), so the specie IS the protonated structure.
    # _build_specie_mol would sanitize on the way through and undo the registry
    # Lewis structure the comment above protects.
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
    prot, candidate_charges, allow_charged_fragments=True, diagnostics=None
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
    #
    # Deliberately all-or-nothing, not per charge. Retrying only the charges
    # rdDetermineBonds abandoned looks strictly additive but is not:
    # enumerate_possible_charge_states returns only the BEST-ranked states, so a
    # recovered candidate that wins the ranking *displaces* the charge that was
    # being reported before, narrowing the ballot the balancer sees. Measured
    # over the manuscript corpus a per-charge retry fixed nothing and cost
    # GEKDIJ (no valid charge distribution). Revisit only alongside a change
    # that lets a specie report every valid charge, not just the best one.
    return determine_bond_using_modified_AC2mol(
        prot,
        candidate_charges,
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


def _cleaned_of_resonance_artifacts(
    charge_states: list[ChargeState],
) -> list[ChargeState]:
    """Let charge-separated candidates resonate into a cleaner form before they
    are judged.

    A candidate can carry the right total charge on a poor Lewis structure --
    AC2mol in particular tends to strand a spurious ``[N-]``/``[N+]`` pair
    across a conjugated chain. Ranked as-is such a candidate loses on
    ``specie_abs_atcharge`` to a structure that is cleaner but has the wrong
    charge, so the right answer is discarded for the wrong reason.

    Resonance is exactly the right test for "is this separation real or just a
    bad way of drawing it". Only artifact-zwitterionic candidates are tried
    (obligate separation has nothing to gain), and a resonance form is adopted
    only when it strictly reduces charge separation at the same total charge --
    ``get_best_resonance_state`` returns the first *valid* form, not
    necessarily the least charged, so its output has to be earned.
    """
    cleaned: list[ChargeState] = []
    for state in charge_states:
        if not _is_artifact_zwitterion(state):
            cleaned.append(state)
            continue

        try:
            candidate = get_best_resonance_state(state)
        except Exception as exc:  # resonance is best-effort, never fatal
            logger.debug("   Resonance cleanup failed for a candidate: %s", exc)
            cleaned.append(state)
            continue

        if (
            candidate is not state
            and candidate.status
            and candidate.specie_total_charge == state.specie_total_charge
            and candidate.specie_abs_atcharge < state.specie_abs_atcharge
        ):
            logger.debug(
                "   Resonance cleanup: q=%+d charge separation %d -> %d",
                state.specie_total_charge,
                state.specie_abs_atcharge,
                candidate.specie_abs_atcharge,
            )
            cleaned.append(candidate)
        else:
            cleaned.append(state)

    return cleaned


def identify_best_charge_states(charge_states: list[ChargeState]) -> list[ChargeState]:
    """
    Selects the best charge distributions.
    """
    # Filter out None values initially
    valid_charge_states = [ch for ch in charge_states if ch is not None and ch.status]
    if not valid_charge_states:
        return []

    valid_charge_states = _cleaned_of_resonance_artifacts(valid_charge_states)

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


def _is_artifact_zwitterion(charge_state: ChargeState) -> bool:
    """``specie_zwitt`` with obligate charge separation discounted.

    A nitro group or an N-oxide makes a specie "zwitterionic" by the plain
    any-positive-and-any-negative test, but those charges are forced -- no
    neutral Lewis structure exists for them. Only the *remaining* separation
    says anything about whether bond perception went astray, and that is what
    the ranking below should react to.
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
    # Using lists to store metrics for all candidates
    specie_abs_totals = []
    specie_abs_atcharges = []
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
        specie_abs_atcharges.append(chs.specie_abs_atcharge)
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
    min_abs = int(np.min(specie_abs_atcharges))

    # A zwitterionic candidate can post a smaller |net charge| than a
    # non-zwitterionic one purely by pairing a spurious +1 against a spurious
    # -1, and rdDetermineBonds' search is not atom-order invariant, so one
    # symmetry copy of a ligand can find such a solution while its twin does
    # not (ABOFAY: copy A -> -2, copy B -> 0, same graph, permuted indices).
    # Whenever a candidate free of *artifact* charge separation already reaches
    # the best per-atom charge separation, it describes the same amount of
    # charge with none of it cancelled, so let it set the |net charge| target
    # instead. Obligate separation (nitro, N-oxide) does not count as artifact,
    # so those candidates still compete on their own terms.
    #
    # Aromaticity vetoes the swap. Remote charges are not by themselves a sign
    # of a bad structure -- BEJFON's ligand is a methylpyridinium tethered to a
    # cyclopentadienide, permanently zwitterionic with the charges rings apart
    # -- and there the charge-free alternative is the one that is wrong,
    # dearomatising both rings to avoid the charges. Aromaticity separates the
    # two cases where the charge bookkeeping cannot: never buy a lower |net
    # charge| at the cost of aromatic rings.
    naive_min_tot = int(np.min(specie_abs_totals))
    best_aromatic_at_naive_min = max(
        aromatic_atoms[idx]
        for idx in range(nlists)
        if specie_abs_totals[idx] == naive_min_tot
    )
    clean_at_min_abs = [
        idx
        for idx in range(nlists)
        if specie_abs_atcharges[idx] == min_abs
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
