"""Validate molecular structures and charge consistency."""

from __future__ import annotations

import logging

from rdkit import Chem

logger = logging.getLogger(__name__)

MAX_EXCESS_CHARGE_SEPARATION = 4


############################################################
# Structure validation
############################################################


def check_rdkit_obj_connectivity(mol: Chem.Mol, natoms: int, charge: int) -> bool:
    """Validate total charge, valences, lone pairs, electron counts and charge separation."""
    pt = Chem.GetPeriodicTable()
    is_correct = True

    total_formal_charge = sum(
        mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(natoms)
    )
    if total_formal_charge != charge:
        logger.debug(
            "   Total formal charge mismatch: got %d, expected %d",
            total_formal_charge,
            charge,
        )
        is_correct = False

    # Exclude obligate nitro charges from separation; reject malformed nitro bonds.
    bad_nitro = _malformed_nitro_atom(mol, natoms)
    if bad_nitro is not None:
        logger.debug(
            "   Malformed nitro on atom %d: two terminal O(-1) and no N=O; "
            "the correct [N+](=O)[O-] carries a different total charge",
            bad_nitro,
        )
        is_correct = False

    nitro_charge_atoms = _nitro_charge_atom_indices(mol, natoms)
    total_abs_atom_charge = sum(
        abs(mol.GetAtomWithIdx(i).GetFormalCharge())
        for i in range(natoms)
        if i not in nitro_charge_atoms
    )
    if total_abs_atom_charge > abs(charge) + MAX_EXCESS_CHARGE_SEPARATION:
        logger.debug(
            "   Excessive charge separation: total |formal charge| on atoms = %d, "
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
        try:
            valence = atom.GetValence(Chem.ValenceType.TOTAL)
        except AttributeError:
            valence = atom.GetTotalValence()

        num_valence_electrons = pt.GetNOuterElecs(atom.GetAtomicNum())
        lone_pairs = (num_valence_electrons - formal_charge - valence) / 2

        if lone_pairs not in [0, 1, 2, 3, 4]:
            logger.debug(
                "   Lone pair error at atom %d (%s): %f", i, symbol, lone_pairs
            )
            is_correct = False

        is_aromatic = atom.GetIsAromatic()

        if not is_aromatic:
            try:
                explicit = atom.GetValence(Chem.ValenceType.EXPLICIT)
                implicit = atom.GetValence(Chem.ValenceType.IMPLICIT)
            except AttributeError:
                explicit = atom.GetExplicitValence()
                implicit = atom.GetImplicitValence()

            if valence != explicit + implicit:
                logger.debug("   Valence mismatch at atom %d (%s)", i, symbol)
                is_correct = False

            calc_total_elecs = valence + (int(lone_pairs) * 2) + formal_charge
            if calc_total_elecs != num_valence_electrons:
                logger.debug(
                    "   Total electron count mismatch at atom %d (%s)", i, symbol
                )
                is_correct = False

    # Reject reducible charged C/C pairs that represent missing multiple bonds.
    for bond in mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin_atom.GetAtomicNum() != 6 or end_atom.GetAtomicNum() != 6:
            continue
        q_begin, q_end = begin_atom.GetFormalCharge(), end_atom.GetFormalCharge()
        if q_begin == 0 or q_end == 0:
            continue

        opposite_sign = (q_begin > 0) != (q_end > 0)
        both_carbanion = q_begin < 0 and q_end < 0
        if both_carbanion:
            # Only localized single/double bonds permit lone-pair pairing.
            if begin_atom.GetIsAromatic() or end_atom.GetIsAromatic():
                continue
            if bond.GetBondTypeAsDouble() >= 3:
                continue
        elif not opposite_sign:
            continue  # two carbocations cannot pair into a double bond

        logger.debug(
            "   Reducible C%+d/C%+d pair at atoms %d-%d (missing C=C double bond)",
            q_begin,
            q_end,
            begin_atom.GetIdx(),
            end_atom.GetIdx(),
        )
        is_correct = False
        break

    return is_correct


############################################################
# Nitro-group validation helpers
############################################################


def _malformed_nitro_atom(mol: Chem.Mol, natoms: int) -> int | None:
    """Find N([O-])[O-] lacking a required N=O bond.

    Reject rather than repair: adding that bond would change the requested charge.
    """
    for i in range(natoms):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 7:
            continue

        terminal_o_minus = 0
        has_double_bonded_o = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(i, neighbor.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                has_double_bonded_o = True
            elif (
                bond.GetBondType() == Chem.BondType.SINGLE
                and neighbor.GetDegree() == 1
                and neighbor.GetFormalCharge() == -1
            ):
                terminal_o_minus += 1

        if terminal_o_minus >= 2 and not has_double_bonded_o:
            return i

    return None


def _nitro_charge_atom_indices(mol: Chem.Mol, natoms: int) -> set[int]:
    """Find nitro N(+1)/O(-1) atoms exempt from the excess-charge check; no sanitization needed."""
    nitro_atoms: set[int] = set()
    for i in range(natoms):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        o_minus = None
        has_o_double = False
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(i, nb.GetIdx())
            if (
                nb.GetFormalCharge() == -1
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                o_minus = nb.GetIdx()
            elif (
                nb.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                has_o_double = True
        if o_minus is not None and has_o_double:
            nitro_atoms.add(i)
            nitro_atoms.add(o_minus)
    return nitro_atoms
