from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from cell2mol.element_utils import HALOGENS
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def is_sb_halide_only(labels) -> bool:
    """
    True for species made up of only Sb and halogens (SbX3/X4/X5/X6-type),
    regardless of halogen identity or count.
    """
    return "Sb" in labels and all(
        label in HALOGENS or label == "Sb" for label in labels
    )


def generate_sb_halide_charge_state(prot: Protonation) -> ChargeState | None:
    """Assign Sb-halide charges from connectivity: mononuclear 3-degree
    (except SbX6: -1), or 3*n_Sb-n_halogens for clusters.
    """
    assert (
        prot.natoms is not None
        and prot.adjacency_matrix is not None
        and prot.atomic_numbers is not None
    )

    sb_indices = [i for i, label in enumerate(prot.symbols) if label == "Sb"]
    halogen_indices = [i for i, label in enumerate(prot.symbols) if label in HALOGENS]

    if not sb_indices or (len(sb_indices) + len(halogen_indices)) != prot.natoms:
        logger.warning(
            "Unexpected structure for Sb-halide species %s; expected only Sb and halogens",
            prot.formula,
        )
        return None

    adjacency_matrix = np.asarray(prot.adjacency_matrix)

    rwmol = Chem.RWMol()
    for atomic_num in prot.atomic_numbers:
        rwmol.AddAtom(Chem.Atom(atomic_num))

    for i in range(prot.natoms):
        for j in range(i + 1, prot.natoms):
            if adjacency_matrix[i, j] != 0:
                rwmol.AddBond(i, j, Chem.BondType.SINGLE)

    is_polynuclear = len(sb_indices) > 1
    for idx in sb_indices:
        sb_atom = rwmol.GetAtomWithIdx(idx)
        sb_atom.SetNoImplicit(True)

        if not is_polynuclear:
            degree = int(np.count_nonzero(adjacency_matrix[idx]))
            sb_charge = -1 if degree == 6 else 3 - degree
            sb_atom.SetFormalCharge(sb_charge)

    if is_polynuclear:
        # Bypass implicit valence for bridging or negatively charged halogens.
        for j in halogen_indices:
            rwmol.GetAtomWithIdx(j).SetNoImplicit(True)

        # Distribute 3*n_Sb-n_halogens round-robin, preferring terminal halogens.
        halogen_sb_degree = {
            j: sum(1 for idx in sb_indices if adjacency_matrix[idx, j] != 0)
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
                    if j not in assigned and adjacency_matrix[idx, j] != 0
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
