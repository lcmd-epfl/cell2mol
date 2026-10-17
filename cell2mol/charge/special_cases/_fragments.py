from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

from cell2mol.charge.validation import check_rdkit_obj_connectivity
from cell2mol.charge.bond_orders import generate_rdkit_mol_from_AC2mol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _collect_substituent_fragment(
    root: int, cage_indices: list[int], adjacency_matrix: np.ndarray
) -> list[int]:
    """Return atoms reachable from root without crossing into the cage."""
    cage_set = set(cage_indices)
    visited = {root}
    frontier = [root]
    while frontier:
        next_frontier = []
        for i in frontier:
            for j in np.nonzero(adjacency_matrix[i])[0]:
                j = int(j)
                if j in cage_set or j in visited:
                    continue
                visited.add(j)
                next_frontier.append(j)
        frontier = next_frontier
    return sorted(visited)


def _charge_capped_fragment(
    fragment: list[int],
    attach_atoms: list[int],
    adjacency_matrix: np.ndarray,
    atnums,
    formula: str | None,
) -> tuple[int | None, dict[int, int], dict[frozenset[int], "Chem.BondType"]]:
    """Cap cage bonds with H and search increasing absolute charge.

    Return internal bond orders as well as charges to preserve aromatic fragments.
    """
    local_index = {g: local for local, g in enumerate(fragment)}
    k = len(fragment)
    n_caps = len(attach_atoms)
    local_ac = np.zeros((k + n_caps, k + n_caps), dtype=int)
    for a in fragment:
        for b in fragment:
            if a != b and adjacency_matrix[a, b] != 0:
                local_ac[local_index[a], local_index[b]] = 1
    for cap_offset, attach_atom in enumerate(attach_atoms):
        cap_idx = k + cap_offset
        local_ac[local_index[attach_atom], cap_idx] = 1
        local_ac[cap_idx, local_index[attach_atom]] = 1

    local_atoms = [int(atnums[g]) for g in fragment] + [1] * n_caps
    for charge in (0, -1, 1, -2, 2, -3, 3):
        mol = generate_rdkit_mol_from_AC2mol(
            atoms=local_atoms,
            AC=local_ac,
            charge=charge,
            allow_charged_fragments=True,
            embed_chiral=True,
        )
        if mol is not None:
            is_correct = check_rdkit_obj_connectivity(mol, len(local_atoms), charge)
            if not is_correct:
                continue
            frag_atom_charges = {
                g: mol.GetAtomWithIdx(local_index[g]).GetFormalCharge()
                for g in fragment
            }

            frag_bond_orders = {}
            for a in fragment:
                for b in fragment:
                    if a < b and adjacency_matrix[a, b] != 0:
                        bond = mol.GetBondBetweenAtoms(local_index[a], local_index[b])
                        frag_bond_orders[frozenset((a, b))] = bond.GetBondType()
            return charge, frag_atom_charges, frag_bond_orders

    logger.warning(
        "Could not determine a charge for a capped substituent fragment on species %s",
        formula,
    )
    return None, {}, {}
