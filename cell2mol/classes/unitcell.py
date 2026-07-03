from __future__ import annotations

import logging
from typing import cast

import numpy as np
from cell2mol.classes.cell import Cell
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.specie import Specie
from cell2mol.compare import compare_reference_indices
from cell2mol.charge.specie_assigner import set_charge_state
from cell2mol.elementdata import ElementData
from cell2mol.my_types import SubType, NDArray

logger = logging.getLogger(__name__)

elemdatabase = ElementData()


Labels = list[str]


class UnitCell(Cell):
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Molecule lists
    moleclist: list[Molecule] | None = None

    # Unit cell construction related attributes
    error_get_fragments: bool | None = None
    error_reconstruction: bool | None = None

    # Balancing charges related attributes
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None

    # Charge neutrality
    is_neutral: bool | None = None

    # Frozen fields
    subtype: SubType | None = Field(default="unitcell")

    def get_species_list(self, reference_species_list: list[Specie | Metal]):
        """Get unique indices and species list."""
        self.unique_indices = []
        self.species_list = []

        # Pre-group reference species by subtype
        ref_molecules = [
            ref
            for ref in reference_species_list
            if ref.subtype == "molecule"
            and getattr(ref, "is_non_complex_molecule", False)
        ]
        ref_ligands = [ref for ref in reference_species_list if ref.subtype == "ligand"]
        ref_metals = [
            cast(Metal, ref) for ref in reference_species_list if ref.subtype == "metal"
        ]

        for mol in self.moleclist or []:
            # Case 1: non-complex molecule
            if mol.is_non_complex_molecule:
                for ref in ref_molecules:
                    if compare_reference_indices(ref, mol):
                        ref_unique_index = getattr(ref, "unique_index", None)
                        mol.unique_index = ref_unique_index
                        if ref_unique_index is not None:
                            self.unique_indices.append(ref_unique_index)
                        self.species_list.append(mol)
                        break
                continue

            # Case 2: complex molecule with ligands and metals
            # --- ligands ---
            for lig in mol.ligands or []:
                for ref in ref_ligands:
                    if compare_reference_indices(ref, lig):
                        ref_unique_index = getattr(ref, "unique_index", None)
                        lig.unique_index = ref_unique_index
                        if ref_unique_index is not None:
                            self.unique_indices.append(ref_unique_index)
                        self.species_list.append(lig)
                        break

            # --- metals ---
            for met in mol.metals or []:
                met_parent_ref = met.get_parent_index("reference")
                for ref in ref_metals:
                    if ref.get_parent_index("reference") == met_parent_ref:
                        met.unique_index = ref.unique_index
                        if met.unique_index is not None:
                            self.unique_indices.append(met.unique_index)
                        self.species_list.append(met)
                        break

        logger.info("Unique indices: %s", self.unique_indices)
        logger.info("Species list: %s", [s.formula for s in self.species_list])

    def map_charges_to_unitcell(self, refmoleclist: list[Molecule]):
        """Logic: Propagate charges from Reference Molecules to Unit Cell Molecules."""
        self.error_assign_charge = False

        for mol in self.moleclist or []:
            try:
                if mol.is_non_complex_molecule:
                    logger.info(
                        "Mapping charges for Non-Complex Molecule: %s", mol.formula
                    )
                    for ref in refmoleclist:
                        if ref.is_non_complex_molecule and (
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
                        # Complex Molecule Match by Formula
                        if not ref.is_non_complex_molecule and (
                            mol.formula == ref.formula
                        ):
                            self._map_complex_components(mol, ref)
            except Exception as e:
                logger.error(
                    "Unexpected error in unit cell mapping for %s: %s", mol.formula, e
                )
                self.error_assign_charge = True

    def _map_complex_components(self, mol, ref):
        """Helper to map ligands and metals within a complex."""
        # Map Ligands
        for lig in mol.ligands:
            for ref_lig in ref.ligands:
                if lig.formula == ref_lig.formula:
                    try:
                        if compare_reference_indices(ref_lig, lig):
                            set_charge_state(ref_lig, lig, mode=2)
                    except Exception as e:
                        logger.error("Ligand charge mapping failed: %s", e)
                        self.error_assign_charge = True

        # Map Metals
        for met in mol.metals:
            for ref_met in ref.metals:
                if met.formula == ref_met.formula:
                    try:
                        p_idx_ref = ref_met.get_parent_index("reference")
                        p_idx_mol = met.get_parent_index("reference")
                        if p_idx_ref == p_idx_mol:
                            met.set_charge(ref_met.charge)
                    except Exception as e:
                        logger.error("Metal charge mapping failed: %s", e)
                        self.error_assign_charge = True

    def check_charge_neutrality(self):
        """
        Check whether the total charge of the unit cell is neutral.

        Sets:
            self.is_neutral:
                - True  : total charge == 0
                - False : total charge != 0
                - None  : charges not fully assigned
        """

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
    @deprecated("Use unitcell() with the keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]],
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "UnitCell":
        return cls(
            name=name,
            labels=labels,
            pos=np.asarray(pos),  # Using pos which gets aliased to coord
            frac_coord=np.asarray(frac_coord),
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
