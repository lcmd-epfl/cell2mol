from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from rdkit import Chem

from cell2mol.classes.charge_state import ChargeState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_TTF_CHALCOGENS = {16, 34}


############################################################
# Donor detection and charge-state generation
############################################################


def find_dithiolylidene_units(mol: Chem.Mol) -> list[int]:
    """Find five-ring 1,3-dithiol-2-ylidene carbons with exocyclic double bonds."""
    units = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        chalcogens = {
            idx
            for idx in ring
            if mol.GetAtomWithIdx(idx).GetAtomicNum() in _TTF_CHALCOGENS
        }
        if len(chalcogens) != 2:
            continue

        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                continue
            if {n.GetIdx() for n in atom.GetNeighbors()} & set(ring) != chalcogens:
                continue
            exocyclic = [n for n in atom.GetNeighbors() if n.GetIdx() not in ring]
            if len(exocyclic) != 1 or exocyclic[0].GetAtomicNum() != 6:
                continue
            bond = mol.GetBondBetweenAtoms(idx, exocyclic[0].GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE and idx not in units:
                units.append(idx)
    return units


def generate_oxidised_donor_charge_states(
    neutral: ChargeState,
) -> list[ChargeState]:
    """Build the radical-cation and dication states of a neutral TTF-type donor."""
    if neutral.species_total_charge != 0 or neutral.protonated_rdkit_obj is None:
        return []

    states = []
    for removed in (1, 2):
        mol = _oxidised_donor_mol(neutral.protonated_rdkit_obj, removed)
        if mol is None:
            continue

        atom_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        total_charge = neutral.protonated_total_charge + removed
        if sum(atom_charges) != total_charge:
            continue

        smiles = Chem.MolToSmiles(mol)
        logger.debug(
            "Oxidised donor state for %s: +%d | SMILES: %s",
            neutral.protonation.formula,
            removed,
            smiles,
        )
        states.append(
            ChargeState.from_positional(
                True,
                total_charge,
                atom_charges,
                mol,
                smiles,
                total_charge,
                True,
                neutral.protonation,
            )
        )
    return states


############################################################
# Helpers
############################################################


def _oxidised_donor_mol(mol: Chem.Mol, n_electrons_removed: int) -> Chem.Mol | None:
    """Oxidize donor units by flipping their conjugated bridge.

    For one-electron oxidation, retain one radical instead of a second cation.
    """
    units = find_dithiolylidene_units(mol)
    if len(units) != 2:
        return None

    first, second = units
    path = _alternating_bond_path(mol, first, second)
    if path is None:
        return None

    oxidised = Chem.RWMol(mol)
    for begin, end in zip(path, path[1:]):
        bond = oxidised.GetBondBetweenAtoms(begin, end)
        bond.SetBondType(
            Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.DOUBLE
            else Chem.BondType.DOUBLE
        )

    oxidised.GetAtomWithIdx(first).SetFormalCharge(1)
    far_atom = oxidised.GetAtomWithIdx(second)
    if n_electrons_removed == 2:
        far_atom.SetFormalCharge(1)
    else:
        far_atom.SetNumRadicalElectrons(1)
        far_atom.SetNoImplicit(True)

    result = oxidised.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as exc:
        logger.debug("Oxidised donor failed to sanitize: %s", exc)
        return None
    return result


def _alternating_bond_path(mol: Chem.Mol, start: int, end: int) -> list[int] | None:
    """Return a DOUBLE/SINGLE/.../DOUBLE path between donor units, or None."""
    double, single = Chem.BondType.DOUBLE, Chem.BondType.SINGLE
    queue = deque([(start, double, [start])])
    seen: set[tuple[int, Chem.BondType]] = set()

    while queue:
        atom_idx, expected, path = queue.popleft()
        if (atom_idx, expected) in seen:
            continue
        seen.add((atom_idx, expected))

        for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            next_idx = neighbour.GetIdx()
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond.GetBondType() != expected or next_idx in path:
                continue
            if next_idx == end:
                # Arrive on a double bond: the far unit's exocyclic one.
                if expected is double:
                    return path + [next_idx]
                continue
            queue.append(
                (next_idx, single if expected is double else double, path + [next_idx])
            )
    return None
