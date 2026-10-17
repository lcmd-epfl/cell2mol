"""Formula-based manual charge assignments."""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem

from cell2mol.classes.charge_state import ChargeState
from cell2mol.operations import reorder_element

logger = logging.getLogger(__name__)

MANUAL_CHARGE_ASSIGN_SPECIES = {
    "O4-Cl",
    "N3",
    "N2",
    "N-O",
    "I3",
    "I4",
    "I5",
    "I6",
    "H",
    "H2",
    "N-O3",
    "Te2",
    "N",
    "C",
    "H8-B3",
    "F6-Si",
    "O2",
    "Br3",
    "C-N-S",
}


BRIDGED_CLUSTER_CHARGES = {
    # arachno-[B3H8]- is non-deltahedral, so the cage builder does not handle it.
    "H8-B3": -1,
}


############################################################
# Manual charge-state generation
############################################################


def generate_manual_charge_state(species):
    """Generates a ChargeState for special species using formula-based lookups."""
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

    formula = species.formula

    if formula in BRIDGED_CLUSTER_CHARGES:
        return _manual_charge_state_from_adjacency(
            species, BRIDGED_CLUSTER_CHARGES[formula]
        )

    target_atom, smiles, charge = REGISTRY.get(formula, (None, None, 0))

    if formula == "N-O":
        target_atom = "O"
        is_bent = getattr(species, "NO_type", "") == "Bent"
        smiles, charge = ("[N-]=O", -1) if is_bent else ("[N]=O", 0)
    assert smiles is not None, f"No manual SMILES registered for formula {formula}"

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    order = list(range(species.natoms))
    new_order = order

    if not target_atom:
        label_order = _manual_label_order(mol, species)
        if label_order is not None:
            new_order = label_order

    if target_atom:
        # Locate the anchor in the registry SMILES before matching atom order.
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
            for idx, atom in enumerate(species.atoms):
                # 'central' means the atom connected to two others of the same type
                if target_atom == "central":
                    adj_labels = [
                        species.get_parent("molecule").labels[a]
                        for a in atom.neighbor_indices
                    ]
                    if adj_labels.count(atom.label) == 2:
                        new_order = reorder_element(order, smiles_idx, idx)
                        break
                elif atom.label == target_atom:
                    new_order = reorder_element(order, smiles_idx, idx)
                    break

    mol = Chem.RenumberAtoms(mol, new_order)

    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    total_charge = sum(atom_charges)

    logger.debug(
        "Manual Charge: %s | SMILES: %s | Order: %s", formula, smiles, new_order
    )

    # With no added H, reuse registry species fields to avoid re-sanitizing its Lewis structure.
    return ChargeState(
        is_valid=True,
        protonated_total_charge=total_charge,
        protonated_atom_charges=atom_charges,
        protonated_rdkit_obj=mol,
        protonated_smiles=smiles,
        protonated_target_charge=charge,
        allow_charged_fragments=True,
        protonation=species.protonation_states[0],
        species_rdkit_obj=mol,
        species_smiles=smiles,
    )


############################################################
# Adjacency and atom-order helpers
############################################################


def _manual_charge_state_from_adjacency(species, charge: int):
    """Build a bridged cluster from adjacency; place its known charge on the first heavy atom."""
    adjacency_matrix = np.asarray(species.adjacency_matrix, dtype=int)

    rwmol = Chem.RWMol()
    for atnum in species.get_atomic_numbers():
        rwmol.AddAtom(Chem.Atom(int(atnum)))
    for i in range(species.natoms):
        for j in range(i + 1, species.natoms):
            if adjacency_matrix[i, j] != 0:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)

    for atom in rwmol.GetAtoms():
        # Prevent implicit H on bridge-bonded atoms.
        atom.SetNoImplicit(True)

    heavy = [i for i, label in enumerate(species.labels) if label != "H"]
    charge_site = heavy[0] if heavy else 0
    rwmol.GetAtomWithIdx(charge_site).SetFormalCharge(charge)

    mol = rwmol.GetMol()
    atom_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    smiles = Chem.MolToSmiles(mol)

    logger.debug(
        "Manual bridged-cluster charge: %s | total %d on atom %d | SMILES: %s",
        species.formula,
        charge,
        charge_site,
        smiles,
    )

    return ChargeState(
        is_valid=True,
        protonated_total_charge=charge,
        protonated_atom_charges=atom_charges,
        protonated_rdkit_obj=mol,
        protonated_smiles=smiles,
        protonated_target_charge=charge,
        allow_charged_fragments=True,
        protonation=species.protonation_states[0],
        species_rdkit_obj=mol,
        species_smiles=smiles,
    )


def _manual_anchor_index(mol, target_atom: str) -> int | None:
    """Locate the anchor element or central atom in a registry SMILES."""
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


def _manual_label_order(mol, species) -> list[int] | None:
    """Map registry atoms to species order by element; return None if elements repeat."""
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(set(symbols)) != len(symbols):
        return None

    atoms = getattr(species, "atoms", None) or []
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
            species.formula,
            [a.label for a in atoms],
            Chem.MolToSmiles(mol),
        )
        return None
