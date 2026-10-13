from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
from rdkit import Chem
from cell2mol.charge.special_cases.manual import MANUAL_CHARGE_ASSIGN_SPECIES
from cell2mol.charge.special_cases.manual import generate_manual_charge_state
from cell2mol.charge.molecule_builder import generate_tmc_rdkit_obj_smiles

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.metal import Metal

logger = logging.getLogger(__name__)


############################################################
# Charge assignment and complex assembly
############################################################


def assign_charge_to_specie(specie: "Specie | Metal", final_charge: int):
    """Apply an already-selected total charge to one specie or metal,
    rebuilding its per-atom charges and RDKit object.
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
    elif getattr(specie, "is_metal_free_molecule", False) or specie.subtype == "ligand":
        target = cast("Specie", specie)
        target_states = target.plausible_charge_states or []
        available_charges = [cs.specie_total_charge for cs in target_states]

        try:
            idx = available_charges.index(final_charge)
            cs = target_states[idx]

            # Use deprotonated specie fields.
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


def assemble_complex_charge_state(mol):
    """Assemble ligand and metal charges into the complex molecule, SMILES and total charge."""
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


############################################################
# Charge-state transfer between structure copies
############################################################


def set_charge_state(reference, target, mode: int):
    """Transfer a solved state: mode 1 maps connectivity; mode 2 maps atom site labels."""
    final_charge = reference.totcharge

    if mode == 1:
        _transfer_state_to_reference_copy(reference, target, final_charge)
    elif mode == 2:
        _transfer_state_to_unit_cell(reference, target, final_charge)
    else:
        logger.error("Invalid Mode %d passed to set_charge_state", mode)


def _transfer_state_to_reference_copy(
    reference: "Specie", target: "Specie", final_charge
):
    """Transfer by graph isomorphism; rebuild manual species or fall back to target states."""

    cs = None

    if target.formula in MANUAL_CHARGE_ASSIGN_SPECIES:
        cs = generate_manual_charge_state(target)
    elif (
        reference is not None
        and reference is not target
        and getattr(reference, "rdkit_obj", None) is not None
        and _transfer_state_by_connectivity(reference, target, final_charge)
    ):
        return
    else:
        if target.plausible_charge_states is None:
            target.get_plausible_charge_states()

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
            return

    target.charge_state = cs

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


def _transfer_state_by_connectivity(
    reference: "Specie", target: "Specie", final_charge
) -> bool:
    """Transfer the solved RDKit molecule by atom mapping; return False to request fallback."""
    new_order = _connectivity_new_order(reference, target)
    if new_order is None:
        logger.debug(
            "No graph isomorphism between unique specie %s and its copy; "
            "falling back to the copy's own charge states",
            target.formula,
        )
        return False

    rdkit_obj = Chem.RenumberAtoms(reference.rdkit_obj, new_order)
    atom_charges = [
        rdkit_obj.GetAtomWithIdx(i).GetFormalCharge() for i in range(target.natoms)
    ]
    total_charge = int(sum(atom_charges))

    if total_charge != final_charge:
        logger.warning(
            "Mode 1 transfer for %s produced charge %d, expected %d; falling back",
            target.formula,
            total_charge,
            final_charge,
        )
        return False

    target.charge_state = reference.charge_state
    target.set_charges(total_charge, atom_charges, reference.smiles, rdkit_obj)
    logger.debug(
        "Mode 1 transferred by connectivity: %s (Q=%d) %s",
        target.formula,
        target.totcharge,
        target.smiles,
    )
    return True


def _transfer_state_to_unit_cell(reference: "Specie", target: "Specie", final_charge):
    """Mode 2: Transfers RDKit object from reference to target, reordering atoms."""

    if not target.subtype == "ligand" and not target.is_metal_free_molecule:
        return

    rdkit_obj = _reorder_rdkit_atoms(
        reference.rdkit_obj, reference.atom_site_labels, target.atom_site_labels
    )

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

    atom_charges = []
    total_charge_calc = 0

    for i in range(target.natoms):
        atom = rdkit_obj.GetAtomWithIdx(i)
        chg = atom.GetFormalCharge()
        atom_charges.append(chg)
        total_charge_calc += chg

    if total_charge_calc != final_charge:
        logger.warning(
            "Mode 2 Mismatch: %s Reordered Calc Charge (%d) != Final Charge (%d)",
            target.formula,
            total_charge_calc,
            final_charge,
        )

    target.set_charges(total_charge_calc, atom_charges, smiles, rdkit_obj)
    logger.debug("Mode 2 Applied: %s (Q=%d)", target.formula, target.totcharge)


############################################################
# Connectivity and atom-order mapping helpers
############################################################


def _connectivity_graph(spec) -> nx.Graph[int]:
    """Build an element-labelled connectivity graph without bond orders."""
    graph = nx.Graph()
    atomic_numbers = [int(z) for z in spec.get_atomic_numbers()]

    mconnec = [a.mconnec or 0 for a in spec.atoms or []]
    if len(mconnec) != len(atomic_numbers):
        mconnec = [0] * len(atomic_numbers)

    for i, atomic_num in enumerate(atomic_numbers):
        graph.add_node(i, element=atomic_num, donor=(atomic_num, mconnec[i]))

    adjmat = np.asarray(spec.adjmat)
    rows, cols = np.nonzero(np.triu(adjmat, k=1))
    graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return graph


def _connectivity_new_order(ref_spec, target_spec) -> list[int] | None:
    """Return an atom mapping for RenumberAtoms, preferring aligned metal donors.

    Return None if graphs differ; a shared unique_index does not guarantee isomorphism.
    """
    # Return None for unavailable mappings so callers can use target charge states.
    if getattr(ref_spec, "natoms", None) != getattr(target_spec, "natoms", None):
        return None
    try:
        ref_graph = _connectivity_graph(ref_spec)
        target_graph = _connectivity_graph(target_spec)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug(
            "Cannot build a connectivity graph for %s: %s",
            getattr(target_spec, "formula", "?"),
            exc,
        )
        return None

    # Label donors before matching to avoid enumerating all graph automorphisms.
    mapping = nx.vf2pp_isomorphism(ref_graph, target_graph, node_label="donor")
    if mapping is None:
        mapping = nx.vf2pp_isomorphism(ref_graph, target_graph, node_label="element")
        if mapping is None:
            return None
        logger.debug(
            "No isomorphism of %s aligns metal coordination; matching elements only",
            target_spec.formula,
        )

    new_order = [0] * target_spec.natoms
    for ref_idx, target_idx in mapping.items():
        new_order[target_idx] = ref_idx
    return new_order


def _reorder_rdkit_atoms(ref_mol, ref_labels, target_labels):
    # RenumberAtoms preserves NoImplicit and prevents invented hydrogens.
    label_to_index = {label: idx for idx, label in enumerate(ref_labels)}
    new_order = [label_to_index[label] for label in target_labels]
    return Chem.RenumberAtoms(ref_mol, new_order)
