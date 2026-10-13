from __future__ import annotations

import logging

import numpy as np
from cell2mol.classes.cell import Cell
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.molecule import Molecule
from cell2mol.compare import compare_reference_indices
from cell2mol.my_types import SubType, NDArray

logger = logging.getLogger(__name__)


class UnitCell(Cell):
    """Reconstructed unit-cell molecules, mapped charges, and processing flags."""

    moleclist: list[Molecule] | None = None

    # Reconstruction failures
    error_get_fragments: bool | None = None
    error_reconstruction: bool | None = None

    # Charge-balancing failures
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None

    # None until every molecule has an assigned charge.
    is_neutral: bool | None = None

    subtype: SubType | None = Field(default="unitcell")

    def map_charges_to_unitcell(self, refmoleclist: list[Molecule]):
        """Map reference charges to matching molecules, ligands, and metals."""
        from cell2mol.charge.charge_assignment import set_charge_state

        self.error_assign_charge = False

        for mol in self.moleclist or []:
            try:
                if mol.is_metal_free_molecule:
                    logger.info(
                        "Mapping charges for Metal-free Molecule: %s", mol.formula
                    )
                    for ref in refmoleclist:
                        if ref.is_metal_free_molecule and (
                            mol.unique_index == ref.unique_index
                        ):
                            if compare_reference_indices(ref, mol):
                                try:
                                    set_charge_state(ref, mol, mode=2)
                                except Exception as e:
                                    logger.error(
                                        "Charge assignment failed for %s: %s",
                                        mol.formula,
                                        e,
                                    )
                                    self.error_assign_charge = True
                else:
                    logger.info("Mapping charges for Complex Molecule: %s", mol.formula)
                    for ref in refmoleclist:
                        # Match complex formulas before mapping their components.
                        if not ref.is_metal_free_molecule and (
                            mol.formula == ref.formula
                        ):
                            self._map_complex_charges(mol, ref)
            except Exception as e:
                logger.error(
                    "Unexpected error in unit cell mapping for %s: %s", mol.formula, e
                )
                self.error_assign_charge = True

    def _map_complex_charges(self, mol: Molecule, ref: Molecule) -> None:
        """Map ligand and metal charges using formulas and reference atom indices."""
        from cell2mol.charge.charge_assignment import set_charge_state

        for lig in mol.ligands or []:
            for ref_lig in ref.ligands or []:
                if lig.formula == ref_lig.formula:
                    try:
                        if compare_reference_indices(ref_lig, lig):
                            set_charge_state(ref_lig, lig, mode=2)
                    except Exception as e:
                        logger.error("Ligand charge mapping failed: %s", e)
                        self.error_assign_charge = True

        for met in mol.metals or []:
            for ref_met in ref.metals or []:
                if met.formula == ref_met.formula:
                    try:
                        p_idx_ref = ref_met.get_parent_index("reference")
                        p_idx_mol = met.get_parent_index("reference")
                        if p_idx_ref == p_idx_mol:
                            if ref_met.charge is None:
                                raise ValueError(
                                    f"Reference metal {ref_met.label} has no assigned charge"
                                )
                            met.set_charge(ref_met.charge)
                    except Exception as e:
                        logger.error("Metal charge mapping failed: %s", e)
                        self.error_assign_charge = True

    def check_charge_neutrality(self):
        """Set is_neutral from the total charge; use None for unassigned charges."""

        assert self.subtype == "unitcell", (
            "check_charge_neutrality should only be called on unitcell"
        )

        if self.moleclist is None:
            raise ValueError("Molecule list is None")

        total_charge = 0
        unassigned = []

        for mol in self.moleclist:
            if mol.totcharge is None:
                unassigned.append(mol.formula)
            else:
                total_charge += mol.totcharge

        if unassigned:
            logger.warning(
                "Charges not assigned for %d molecule(s): %s",
                len(unassigned),
                unassigned,
            )
            self.is_neutral = None
            return

        logger.info(
            "Total charge of the unit cell: %d",
            total_charge,
        )

        self.is_neutral = total_charge == 0

    def __repr__(self, indirect: bool = False):
        """Return cell details and reconstructed molecule formulas."""
        to_print = ""
        to_print += "------------- Cell2mol UnitCell Object --------------\n"
        to_print += Cell.__repr__(self, indirect=True)
        if self.moleclist is not None:
            to_print += f" # Molecules:          = {len(self.moleclist)}\n"
            to_print += " with Formula:                               \n"
            for idx, m in enumerate(self.moleclist):
                to_print += f"    {idx}: {m.formula} \n"
            to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use UnitCell() with keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]],
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "UnitCell":
        """Construct a unit cell using the legacy positional interface."""
        return cls(
            name=name,
            labels=labels,
            pos=np.asarray(pos),
            frac_coord=np.asarray(frac_coord),
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
