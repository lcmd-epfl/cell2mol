from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from rdkit import Chem
from cell2mol.charge.utils import MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.charge.charge_state_resolver import generate_manual_charge_state
from cell2mol.charge.smiles_handler import generate_tmc_rdkit_obj_smiles

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.metal import Metal

logger = logging.getLogger(__name__)


def assemble_complex_charge_state(mol):
    """Roll the per-specie charges of a metal complex up to the molecule level.

    Gathers the already-assigned atomic charges from the complex's ligands and
    metals, builds the whole metal complex rdkit_obj/SMILES via
    ``generate_tmc_rdkit_obj_smiles``, and stores the molecule's total charge,
    atomic charges, SMILES and rdkit_obj. Metal-complex only.
    """
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
        target_states = target.plausible_charge_states or []
        available_charges = [cs.specie_total_charge for cs in target_states]

        try:
            idx = available_charges.index(final_charge)
            cs = target_states[idx]

            # Update the species state (specie_* = the deprotonated specie, as
            # opposed to the protonated smiles/rdkit_obj)
            target.charge_state = cs
            target.set_charges(
                cs.specie_total_charge,
                cs.specie_atom_charges,
                cs.specie_smiles,
                cs.specie_rdkit_obj,
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
    """Mode 1: Selects an existing charge state from target.plausible_charge_states."""

    # 1. Determine the Charge State (cs) object
    cs = None

    if target.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        cs = generate_manual_charge_state(target)
    else:
        if target.plausible_charge_states is None:
            target.get_plausible_charge_states()

        # Extract charges to find the index matching final_charge
        target_states = target.plausible_charge_states or []
        charge_list = [c.specie_total_charge for c in target_states]

        try:
            idx = charge_list.index(final_charge)
            cs = target_states[idx]
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
    if final_charge != cs.specie_total_charge:
        logger.warning(
            "Target %s: Requested charge %d != Selected state charge %d",
            target.formula,
            final_charge,
            cs.specie_total_charge,
        )

    target.set_charges(
        cs.specie_total_charge,
        cs.specie_atom_charges,
        cs.specie_smiles,
        cs.specie_rdkit_obj,
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
    # Reorder atoms so position i holds the ref atom whose label == target_labels[i].
    # Use RenumberAtoms (not a manual rebuild) so ALL atom properties are kept --
    # formal charge, NoImplicit / explicit-H, aromaticity, chirality. A manual
    # rebuild that copies only atomic number + charge drops NoImplicit, letting
    # RDKit add implicit H to fill open valences (e.g. [Al-] -> [AlH2-]).
    label_to_index = {label: idx for idx, label in enumerate(ref_labels)}
    new_order = [label_to_index[label] for label in target_labels]
    return Chem.RenumberAtoms(ref_mol, new_order)
