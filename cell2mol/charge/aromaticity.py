"""Inspect aromaticity and identify artifacts in perceived aromatic rings."""

from __future__ import annotations

from rdkit import Chem

from cell2mol.my_types import RDKitObject

# Exocyclic partners whose charge-separated form keeps a ring aromatic: a
# pyridone qualifies, an aza-fulvene does not.
_YLIDIC_EXOCYCLIC_PARTNERS = frozenset({7, 8, 16, 34})  # N, O, S, Se


def aromatic_info(mol: RDKitObject, added_indices=None):
    if added_indices is None:
        added_indices = []

    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    aromatic_ring_count = 0
    for bond_indices in bond_rings:
        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_indices):
            aromatic_ring_count += 1

    check = any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in added_indices)

    return {
        "Aromatic atoms": aromatic_atoms,
        "Number of aromatic rings": aromatic_ring_count,
        "Added to aromatic atoms": check,
    }


def _unearned_aromatic_atoms(mol) -> set[int]:
    """Find six-ring aromatic atoms with divalent chalcogens or neutral
    three-connected N lacking an exocyclic double bond to O/N/S/Se.
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
