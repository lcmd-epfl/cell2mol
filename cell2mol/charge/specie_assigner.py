import copy
import logging
import numpy as np
from rdkit import Chem
from cell2mol.classes.protonation import Protonation
from cell2mol.charge.utils import MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.charge.charge_state_resolver import (
    generate_charge_state,
    generate_manual_charge_state,
)
from cell2mol.charge.smiles_handler import generate_tmc_rdkit_obj_smiles

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


def assign_charge_to_specie(specie: object, final_charge: int):
    """
    Assigns the final charge to a specie object.

    Args:
        specie: The specie object to update.
        final_charge: The integer charge to assign.
    """
    logger.debug(
        "Target: %s (unique index: %s) | Final Charge: %d",
        specie.formula,
        specie.unique_index,
        final_charge,
    )

    if specie.subtype == "metal":
        specie.set_charge(final_charge)

        logger.debug(
            "Updated Metal: %s | Formula: %s | Charge: %d",
            specie.unique_index,
            specie.formula,
            specie.charge,
        )
    elif specie.is_non_complex_molecule or specie.subtype == "ligand":
        # Extract list of available charges
        available_charges = [cs.corr_total_charge for cs in specie.possible_cs]

        try:
            idx = available_charges.index(final_charge)
            cs = specie.possible_cs[idx]

            # Update the species state
            specie.charge_state = cs
            specie.set_charges(
                cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj
            )

            logger.debug(
                "Updated State: %s | Formula: %s | Total Charge: %d | SMILES: %s",
                specie.unique_index,
                specie.formula,
                specie.totcharge,
                specie.smiles,
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
            3 = Reconstruct state from topology (Unit Cell - Complex/Ligand).
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
    elif mode == 3:
        _construct_state_from_topology(reference, target, final_charge)
    else:
        logger.error("Invalid Mode %d passed to set_charge_state", mode)


# --- Mode 1: Reference Cell (Selection) ---
def _apply_precalculated_state(target, final_charge):
    """Mode 1: Selects an existing charge state from target.possible_cs."""

    # 1. Determine the Charge State (cs) object
    cs = None

    if target.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        cs = generate_manual_charge_state(target)
    else:
        if target.possible_cs is None:
            target.get_possible_cs()

        # Extract charges to find the index matching final_charge
        charge_list = [c.corr_total_charge for c in target.possible_cs]

        try:
            idx = charge_list.index(final_charge)
            cs = target.possible_cs[idx]
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


# --- Mode 3: Unit Cell (Topological Reconstruction) ---
def _construct_state_from_topology(reference, target, final_charge):
    """Mode 3: Reconstructs state (Protonation/Charges) based on topological mapping."""
    cs = None

    # Case A: Manual Override
    if target.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        cs = generate_manual_charge_state(target)

    # Case B: Standard Molecule (Create Empty/Neutral State)
    elif target.is_non_complex_molecule:
        logger.debug("Mode 3: Creating Empty PROTONATION for %s", target.formula)
        empty_list = [0] * len(target.labels)

        # Construct a neutral/empty protonation object
        empty_prot = Protonation.from_positional(
            labels=target.labels,
            coords=target.coord,
            cov_factor=target.cov_factor,
            total_charge=0,
            added_list=empty_list,
            added_type=empty_list,
            added_z=empty_list,
            added_lab=empty_list,
            typ="Empty",
            parent=target,
        )
        cs = generate_charge_state(final_charge, empty_prot)

    # Case C: Ligand (Reorder Protonation from Reference)
    elif target.subtype == "ligand":
        cs = _reorder_ligand_protonation(reference, target)
    # Apply the constructed state
    if cs:
        target.charge_state = cs
        if final_charge != cs.corr_total_charge:
            logger.warning(
                "Mode 3 Charge Mismatch: %d vs %d", final_charge, cs.corr_total_charge
            )

        target.set_charges(
            cs.corr_total_charge, cs.corr_atom_charges, cs.smiles, cs.rdkit_obj
        )
        logger.debug("Mode 3 Applied: %s (Q=%d)", target.formula, target.totcharge)


def _reorder_ligand_protonation(reference, target):
    """Helper for Mode 3 Case C: Complex reordering of ligand protonation states."""
    refcell = reference.get_parent("reference")

    # 1. Clone the reference protonation
    prot = reference.charge_state.protonation
    temp_prot = copy.deepcopy(prot)
    temp_prot.parent = target

    # 2. Calculate Sorting Indices based on parent Reference Cell labels
    # This maps the order of atoms in the reference ligand to the target ligand
    ref_indices = reference.get_parent_indices("reference")
    target_indices = target.get_parent_indices("reference")

    ref_labels = [refcell.atom_site_labels[idx] for idx in ref_indices]
    target_labels = [refcell.atom_site_labels[idx] for idx in target_indices]

    # Create map: Target Label -> Target Index
    label_to_index_map = {lbl: idx for idx, lbl in enumerate(target_labels)}

    # Sort reference indices based on where their labels appear in the target list
    sorted_indices = sorted(
        range(len(ref_labels)), key=lambda i: label_to_index_map[ref_labels[i]]
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Reordering Ligand %s:", target.formula)
        logger.debug("  Sorted Indices: %s", sorted_indices)

    # 3. Reorder the Protonation Object
    reordered_prot = temp_prot.reorder(sorted_indices)

    # 4. Reorder Atom Charges
    temp_uncorr_atom_charges = reference.charge_state.uncorr_atom_charges
    reordered_charges = []

    if len(temp_uncorr_atom_charges) == len(sorted_indices):
        reordered_charges = [temp_uncorr_atom_charges[idx] for idx in sorted_indices]
    else:
        # Handle case where charges length differs (e.g., added protons)
        reordered_charges = [temp_uncorr_atom_charges[idx] for idx in sorted_indices]
        # Append remaining charges (usually added atoms like H+)
        remaining = temp_uncorr_atom_charges[len(sorted_indices) :]
        reordered_charges.extend(remaining)

    # 5. Generate Charge State
    return generate_charge_state(
        reference.charge_state.uncorr_total_charge,
        reordered_prot,
        ref_uncorr_atom_charges=reordered_charges,
    )


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
