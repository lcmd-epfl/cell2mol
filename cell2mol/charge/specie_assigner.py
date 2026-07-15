from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from rdkit import Chem
from cell2mol.classes.charge_state import ChargeState
from cell2mol.charge.utils import MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.charge.charge_state_resolver import generate_manual_charge_state
from cell2mol.charge.smiles_handler import generate_tmc_rdkit_obj_smiles

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.metal import Metal

logger = logging.getLogger(__name__)


def prepare_mol(mol):
    tmp_atcharge = np.zeros((mol.natoms), dtype=int)

    for lig in mol.ligands:
        parent_indices = lig.get_parent_indices("molecule")
        for kdx, a in enumerate(parent_indices):
            tmp_atcharge[a] = lig.atomic_charges[kdx]

    for met in mol.metals:
        parent_index = met.get_parent_index("molecule")
        tmp_atcharge[parent_index] = met.charge
    tmp_atcharge = tmp_atcharge.tolist()

    tmc_rdkit_obj, tmc_smiles = generate_tmc_rdkit_obj_smiles(mol)
    mol.set_charges(
        int(sum(tmp_atcharge)),
        atomic_charges=tmp_atcharge,
        smiles=tmc_smiles,
        rdkit_obj=tmc_rdkit_obj,
    )


def assign_charge_to_specie(specie: "Specie | Metal", final_charge: int):
    """
    Assigns the final charge to a specie object.

    Args:
        specie: The specie object to update.
        final_charge: The integer charge to assign.
    """
    specie_unique_index = getattr(specie, "unique_index", None)
    logger.debug(
        "Target: %s (unique index: %s) | Final Charge: %d",
        specie.formula,
        specie_unique_index,
        final_charge,
    )

    if specie.subtype == "metal":
        metal = cast("Metal", specie)
        metal.set_charge(final_charge)

        logger.debug(
            "Updated Metal: %s | Formula: %s | Charge: %d",
            specie_unique_index,
            metal.formula,
            metal.charge,
        )
    elif (
        getattr(specie, "is_non_complex_molecule", False) or specie.subtype == "ligand"
    ):
        target = cast("Specie", specie)
        # Extract list of available charges
        target_possible_cs = cast("list[ChargeState]", target.possible_cs or [])
        available_charges = [cs.corr_total_charge for cs in target_possible_cs]

        try:
            idx = available_charges.index(final_charge)
            cs = target_possible_cs[idx]

            # Update the species state
            target.charge_state = cs
            target.set_charges(
                cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj
            )

            logger.debug(
                "Updated State: %s | Formula: %s | Total Charge: %d | SMILES: %s",
                specie_unique_index,
                target.formula,
                target.totcharge,
                target.smiles,
            )

        except ValueError:
            logger.error(
                "Failed to assign charge %d to %s. Available options were: %s",
                final_charge,
                specie.formula,
                available_charges,
            )
            # Optional: Raise error if this state is critical
            # raise ValueError(f"Charge mismatch for {specie.formula}")


def set_charge_state(reference, target, mode: int):
    """
    Dispatcher function to apply charge states to a target species.

    Args:
        reference: The source object (holds the total charge or reference state).
        target: The object to receive the charge/state.
        mode:
            1 = Select pre-calculated state (Reference Cell).
            2 = Transfer and reorder state (Unit Cell - Direct Mapping).
    """
    final_charge = reference.totcharge

    # logger.debug(
    #     "SET_CHARGE_STATE: Mode=%d | Target=%s | RefCharge=%s",
    #     mode,
    #     target.formula,
    #     final_charge,
    # )

    if mode == 1:
        _apply_precalculated_state(target, final_charge)
    elif mode == 2:
        _transfer_state_to_unit_cell(reference, target, final_charge)
    else:
        logger.error("Invalid Mode %d passed to set_charge_state", mode)


# --- Mode 1: Reference Cell (Selection) ---
def _apply_precalculated_state(target: "Specie", final_charge):
    """Mode 1: Selects an existing charge state from target.possible_cs."""

    # 1. Determine the Charge State (cs) object
    cs = None

    if target.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        cs = generate_manual_charge_state(target)
    else:
        if target.possible_cs is None:
            target.get_possible_cs()

        # Extract charges to find the index matching final_charge
        target_possible_cs = cast("list[ChargeState]", target.possible_cs or [])
        charge_list = [c.corr_total_charge for c in target_possible_cs]

        try:
            idx = charge_list.index(final_charge)
            cs = target_possible_cs[idx]
        except ValueError:
            logger.error(
                "Charge Mismatch: Target %s needs charge %d, but options are %s",
                target.formula,
                final_charge,
                charge_list,
            )
            # Fallback or critical error handling here if necessary
            return

    # 2. Apply the State
    target.charge_state = cs

    # Validation
    if final_charge != cs.corr_total_charge:
        logger.warning(
            "Target %s: Requested charge %d != Selected state charge %d",
            target.formula,
            final_charge,
            cs.corr_total_charge,
        )

    target.set_charges(
        cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj
    )
    logger.debug(
        "Mode 1 Applied: %s (Q=%d) %s", target.formula, target.totcharge, target.smiles
    )


# --- Mode 2: Unit Cell (Direct Transfer) ---
def _transfer_state_to_unit_cell(reference, target, final_charge):
    """Mode 2: Transfers RDKit object from reference to target, reordering atoms."""

    if not target.subtype == "ligand" and not target.is_non_complex_molecule:
        return

    # We must reorder the reference RDKit object to match the Unit Cell atom labels
    rdkit_obj = _reorder_rdkit_atoms(
        reference.rdkit_obj, reference.atom_site_labels, target.atom_site_labels
    )

    # Determine SMILES
    if target.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        smiles = reference.smiles
    else:
        smiles = Chem.MolToSmiles(rdkit_obj)

    if smiles != reference.smiles:
        logger.warning(
            "Mode 2 SMILES Mismatch: %s Reordered SMILES (%s) != Reference SMILES (%s)",
            target.formula,
            smiles,
            reference.smiles,
        )

    # Recalculate properties from the reordered object
    atom_charges = []
    total_charge_calc = 0

    for i in range(target.natoms):
        atom = rdkit_obj.GetAtomWithIdx(i)
        chg = atom.GetFormalCharge()
        atom_charges.append(chg)
        total_charge_calc += chg

    # Validation
    if total_charge_calc != final_charge:
        logger.warning(
            "Mode 2 Mismatch: %s Reordered Calc Charge (%d) != Final Charge (%d)",
            target.formula,
            total_charge_calc,
            final_charge,
        )

    target.set_charges(total_charge_calc, atom_charges, smiles, rdkit_obj)
    logger.debug("Mode 2 Applied: %s (Q=%d)", target.formula, target.totcharge)


def _reorder_rdkit_atoms(ref_mol, ref_labels, target_labels):
    # 1. Create label → atom index map from ref_data
    label_to_index = {label: idx for idx, label in enumerate(ref_labels)}

    # 2. Create old index → new index map
    old_to_new = {label_to_index[label]: i for i, label in enumerate(target_labels)}
    new_to_old = {v: k for k, v in old_to_new.items()}

    # 3. Create editable mol
    new_mol = Chem.RWMol()

    # 4. Add atoms in new order
    for new_idx in range(len(target_labels)):
        old_idx = new_to_old[new_idx]
        atom = ref_mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_mol.AddAtom(new_atom)

    # 5. Add bonds based on original molecule
    for bond in ref_mol.GetBonds():
        begin_old = bond.GetBeginAtomIdx()
        end_old = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # remap to new indices
        begin_new = old_to_new[begin_old]
        end_new = old_to_new[end_old]
        new_mol.AddBond(begin_new, end_new, bond_type)

    return new_mol.GetMol()
