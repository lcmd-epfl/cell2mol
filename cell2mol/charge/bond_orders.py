"""Generate, repair, and validate bond-order candidates."""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from cell2mol.charge.state_ranking import (
    _has_charged_uncoordinated_carbon,
    _is_artifact_zwitterion,
)
from cell2mol.charge.molecule_builder import (
    collapse_hypervalent_ylides,
    fix_zwitterions,
    shift_13_ylides,
)
from cell2mol.charge.validation import check_rdkit_obj_connectivity
from cell2mol.charge.xyz2mol import AC2mol, chiral_stereo_check, get_proto_mol
from cell2mol.classes.charge_state import ChargeState
from cell2mol.elementdata import ElementData

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

rddeterminebonds_max_iterations = 1000


rdkit_atomic_valence: dict[int, list[int]] = {
    1: [1],  # H
    5: [3, 4],  # B
    6: [4],  # C
    7: [3, 4],  # N
    8: [2, 1, 3],  # O
    9: [1],  # F
    14: [4],  # Si
    15: [5, 3],  # P
    16: [6, 3, 2, 1],  # S
    17: [1],  # Cl
    32: [4],  # Ge
    35: [1],  # Br
    53: [1],  # I
}


############################################################
# Charge-state generation
############################################################


def generate_valid_charge_states(
    prot,
    candidate_charges,
    allow_charged_fragments=True,
    diagnostics=None,
    always_use_modified_AC2mol=False,
):
    valid_charge_states_dict = {charge: [] for charge in candidate_charges}

    if not _species_supported_by_rddeterminebonds(
        prot.atomic_numbers,
        prot.adjacency_matrix,
        atom_site_labels=prot.atom_site_labels,
    ):
        # Unsupported elements go directly to modified AC2mol.
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

    valid_charge_states_dict = determine_bond_using_rdDetermineBonds(
        prot,
        candidate_charges,
        valid_charge_states_dict,
        allow_charged_fragments,
        diagnostics=diagnostics,
    )
    # Compare both perceivers’ drawings within each charge.
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

    # Request an AC2mol alternative when every drawing at this charge looks forced.
    contested = [
        q
        for q in candidate_charges
        if valid_charge_states_dict[q]
        and all(
            _has_charged_uncoordinated_carbon(state) or _is_artifact_zwitterion(state)
            for state in valid_charge_states_dict[q]
        )
    ]
    if not unsolved and not contested:
        logger.debug(
            "rdDetermineBonds found valid charge states for %s: %s",
            prot.formula,
            valid_charge_states_dict,
        )
        return valid_charge_states_dict

    # Retry unresolved charges independently to retain all reachable states.
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
                atoms=prot.atomic_numbers,
                coords=prot.coord,
                AC=prot.adjacency_matrix,
                charge=charge,
                sanitize=True,
                allow_charged_fragments=allow_charged_fragments,
            )
            if rdkit_obj is not None:
                charge_state = _repair_and_build_charge_state(
                    rdkit_obj, prot, charge, allow_charged_fragments
                )
                if charge_state.is_valid:
                    valid_charge_states_dict[charge].append(charge_state)
        except ValueError as e:
            logger.error(
                f"  ValueError occurred using rdDetermineBonds: {e} with {charge} charge for {prot.formula}"
            )
            continue
        except RuntimeError as e:
            # Handle only the iteration-cap RuntimeError; propagate other faults.
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
                logger.error(
                    f"  IndexError occurred using rdDetermineBonds: {e} with {charge} charge for {prot.formula}"
                )
            continue

    return valid_charge_states_dict


def determine_bond_using_modified_AC2mol(
    prot,
    candidate_charges,
    valid_charge_states_dict,
    allow_charged_fragments=True,
    diagnostics=None,
):
    for charge in candidate_charges:
        rdkit_obj = generate_rdkit_mol_from_AC2mol(
            atoms=prot.atomic_numbers,
            AC=prot.adjacency_matrix,
            charge=charge,
            allow_charged_fragments=allow_charged_fragments,
            diagnostics=diagnostics,
        )
        if rdkit_obj is not None:
            charge_state = _repair_and_build_charge_state(
                rdkit_obj, prot, charge, allow_charged_fragments
            )
            if charge_state.is_valid:
                valid_charge_states_dict[charge].append(charge_state)

    return valid_charge_states_dict


############################################################
# Shared repair, validation, and state construction
############################################################


def _repair_and_build_charge_state(rdkit_obj, prot, charge, allow_charged_fragments):
    """Apply the shared repair sequence, then validate and build a charge state."""
    obj, fixed = fix_zwitterions(rdkit_obj)
    obj, collapsed = collapse_hypervalent_ylides(obj)
    obj, shifted = shift_13_ylides(obj)
    if fixed or collapsed or shifted:
        rdkit_obj = obj
    return build_charge_state_from_rdkit_mol(
        rdkit_obj, prot, charge, allow_charged_fragments=allow_charged_fragments
    )


def build_charge_state_from_rdkit_mol(
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

    is_correct = check_rdkit_obj_connectivity(rdkit_obj, prot.natoms, charge)

    charge_state = ChargeState.from_positional(
        is_valid=is_correct,
        protonated_total_charge=total_charge,
        protonated_atom_charges=atom_charges,
        protonated_rdkit_obj=rdkit_obj,
        protonated_smiles=smiles,
        protonated_target_charge=charge,
        allow_charged_fragments=allow_charged_fragments,
        protonation=prot,
    )

    return charge_state


############################################################
# RDKit molecule generation
############################################################


def generate_rdkit_mol_from_rdDetermineBonds(
    atoms,
    coords,
    AC,
    charge=0,
    sanitize=True,
    allow_charged_fragments=True,
    embed_chiral=True,
):
    """Build a molecule from atoms, coordinates and adjacency; assign bond orders with RDKit."""

    atoms = list(map(int, atoms))
    coords = np.asarray(coords, dtype=float)
    ac = np.asarray(AC)
    n_atoms = len(atoms)

    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords must have shape ({n_atoms}, 3), got {coords.shape}")

    if ac.shape != (n_atoms, n_atoms):
        raise ValueError(f"AC must have shape ({n_atoms}, {n_atoms}), got {ac.shape}")

    rwMol = Chem.RWMol()

    for atomic_num in atoms:
        rwMol.AddAtom(Chem.Atom(atomic_num))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if ac[i, j] != 0:
                rwMol.AddBond(i, j, Chem.BondType.SINGLE)

    mol = rwMol.GetMol()

    conf = Chem.Conformer(n_atoms)
    conf.Set3D(True)

    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    mol.AddConformer(conf, assignId=True)

    # Cap backtracking to prevent hangs; RDKit raises RuntimeError at the limit.
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=charge,
        allowChargedFragments=allow_charged_fragments,
        embedChiral=embed_chiral,
        maxIterations=rddeterminebonds_max_iterations,
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
    diagnostics=None,
):
    """Build a molecule with modified AC2mol; record search-limit failures in diagnostics."""
    new_mols, BO = AC2mol(
        mol=get_proto_mol(atoms),
        AC=AC,
        atoms=atoms,
        charge=charge,
        allow_charged_fragments=allow_charged_fragments,
        diagnostics=diagnostics,
    )

    if not new_mols:
        logger.warning(f"No mol found for charge {charge}")
        return None

    if embed_chiral:
        if not all(chiral_stereo_check(mol) for mol in new_mols):
            logger.error("Chirality check failed for one or more candidates")
            return None

    return new_mols[0]  # use the first candidate as default


############################################################
# Valence support helpers
############################################################


def _species_supported_by_rddeterminebonds(
    atoms: list[int],
    ac,  # adjacency matrix
    atomic_valence: dict[int, list[int]] = rdkit_atomic_valence,
    atom_site_labels: list[str] | None = None,
) -> bool:
    """Check whether atom degrees and allowed valences support rdDetermineBonds."""
    ac = np.asarray(ac)
    n_atoms = len(atoms)
    if n_atoms == 1:
        logger.debug("Single atom species - skipping rddeterminebonds")
        return False
    for i in range(n_atoms):
        atom_num = atoms[i]
        atom_symbol = elemdatabase.elementsym[atom_num]
        # Added protons have no CIF atom site labels.
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


def _get_possible_valences(
    atom_num: int, atomic_valence: dict[int, list[int]]
) -> list[int]:
    if atom_num in atomic_valence:
        return list(atomic_valence[atom_num])

    pt = Chem.GetPeriodicTable()
    return [int(v) for v in pt.GetValenceList(atom_num) if v >= 0]
