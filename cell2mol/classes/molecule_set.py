from __future__ import annotations

from cell2mol.utils.serialization import ModelFileMixin

import logging

from pydantic import Field
from typing_extensions import deprecated

from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.specie import Specie
from cell2mol.my_types import SubType, Type
from cell2mol.species_collection import (
    collect_hydrogen_flags,
    update_plausible_charges,
    collect_unique_species,
    map_charges_to_molecules,
)
from cell2mol.utils import BaseModel, config

logger = logging.getLogger(__name__)

# Rule order sets primary-error priority; retain every matching code.
MOLECULE_SET_ERROR_RULES = [
    ("has_isolated_H", 1),
    ("missing_H_in_Water", 2),
    ("missing_H_on_CoordDonor", 3),
    ("missing_H_in_Carbon", 4),
    ("error_plausible_charges", 5),
    ("error_multiple_distrib", 6),
    ("error_empty_distrib", 7),
    ("error_assign_charge", 8),
    ("error_create_bonds", 9),
    ("error_get_spin", 10),
]


class MoleculeSet(ModelFileMixin, BaseModel):
    """Molecules from one XYZ input, balanced together against an input charge."""

    name: str
    moleclist: list[Molecule] = Field(default_factory=list)

    # User-supplied total charge.
    input_charge: int | None = None

    # Unique species and the mapping of molecule species to them.
    unique_species: list[Specie | Metal] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie | Metal] | None = None

    # Hydrogen flags aggregated across molecules and species.
    has_isolated_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_on_CoordDonor: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None

    # Charge candidates for unique_species followed by species_list.
    plausible_charges: list[list[int] | None] | None = None
    error_plausible_charges: bool | None = None
    # Different charge candidates for copies sharing a unique_index.
    error_inconsistent_plausible_charges: bool | None = None
    inconsistent_plausible_charges: dict[int, list[list[int] | None]] | None = None

    # Charge, bond, and spin failures.
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    # Primary code (zero on success) and all matching codes.
    error_case: int | None = None
    error_cases_all: list[int] | None = None

    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="molecule_set")
    subtype: SubType | None = Field(default="molecule_set")

    @property
    def formula(self) -> str:
        """Join molecule formulas with a plus sign."""
        return " + ".join(mol.formula for mol in self.moleclist)

    @property
    def totcharge(self) -> int | None:
        """Return the total assigned charge, or None for missing charges or molecules."""
        if not self.moleclist:
            return None
        total = 0
        for mol in self.moleclist:
            if mol.totcharge is None:
                return None
            total += mol.totcharge
        return total

    def collect_unique_species(self):
        """Collect unique species and map all molecule species to them."""
        logger.info("Getting unique species in %s", self.subtype)

        (
            self.unique_species,
            self.unique_indices,
            self.species_list,
        ) = collect_unique_species(self.moleclist)

        return self.unique_species

    def get_plausible_charges(self) -> list[list[int] | None] | None:
        """Refresh and return charge candidates, recording inconsistencies."""
        if self.unique_species is None:
            self.collect_unique_species()

        update_plausible_charges(self)
        return self.plausible_charges

    def check_hydrogens(self):
        """Collect species if needed, then aggregate hydrogen flags."""
        if self.species_list is None:
            self.collect_unique_species()

        flags = collect_hydrogen_flags(self.moleclist, self.species_list)
        self.has_isolated_H = flags["has_isolated_H"]
        self.missing_H_in_Carbon = flags["missing_H_in_Carbon"]
        self.missing_H_on_CoordDonor = flags["missing_H_on_CoordDonor"]
        self.missing_H_in_Water = flags["missing_H_in_Water"]
        self.has_missing_H = flags["has_missing_H"]

        return self.has_missing_H

    def assign_charges(self):
        """Map balanced charges to molecules, create bonds, and assemble complexes."""
        from cell2mol.charge.charge_assignment import assemble_complex_charge_state

        logger.info("Assigning charges to the %d molecules", len(self.moleclist))

        self.error_assign_charge = map_charges_to_molecules(
            self.unique_species, self.moleclist
        )

        self.create_bonds()

        for idx, mol in enumerate(self.moleclist):
            if mol.contains_metal and not mol.error_create_bonds:
                assemble_complex_charge_state(mol)
                logger.info("Complex %d: %s %s", idx, mol.formula, mol.totcharge)
                for jdx, lig in enumerate(mol.ligands or []):
                    logger.info(
                        "    Ligand %d %s %s %s",
                        jdx,
                        lig.formula,
                        lig.totcharge,
                        lig.smiles,
                    )
                for kdx, met in enumerate(mol.metals or []):
                    logger.info("    Metal %d %s %s", kdx, met.formula, met.charge)
            else:
                logger.info(
                    "Metal-free %d: %s %s %s",
                    idx,
                    mol.formula,
                    mol.totcharge,
                    mol.smiles,
                )

        logger.info("Total charge of %s: %s", self.name, self.totcharge)

    def create_bonds(self):
        """Create bonds in every molecule and record any failures."""
        overall_error = False

        for mol in self.moleclist:
            try:
                mol.create_bonds()
            except Exception as e:
                mol.error_create_bonds = True
                logger.error("Bonding failed for %s: %s", mol.formula, e)
                logger.debug("Exception details", exc_info=True)

            if mol.error_create_bonds:
                overall_error = True

        self.error_create_bonds = overall_error

    def assign_spin(self):
        """Assign molecule and metal spins, recording any failures."""
        overall_error = False

        for mol in self.moleclist:
            mol.error_get_spin = False
            try:
                if mol.is_metal_complex:
                    for metal in mol.metals or []:
                        if metal.coord_nr is None:
                            metal.determine_coordination_geometry()
                            metal.compute_relative_metal_radius()
                            metal.build_coordination_sphere_formula()
                        metal.assign_spin()
                mol.assign_spin()
            except Exception as e:
                mol.error_get_spin = True
                overall_error = True
                logger.error("Spin assignment failed for %s: %s", mol.formula, e)
                logger.debug("Exception details", exc_info=True)

        self.error_get_spin = overall_error

    def assess_errors(self) -> int:
        """Record all matching error codes and return the first, or zero."""
        self.error_cases_all = [
            code
            for attr, code in MOLECULE_SET_ERROR_RULES
            if getattr(self, attr, False)
        ]
        self.error_case = self.error_cases_all[0] if self.error_cases_all else 0
        return self.error_case

    def __repr__(self, indirect: bool = False):
        """Return molecule formulas, assigned charges, and the total charge."""
        to_print = ""
        to_print += "----------- Cell2mol MOLECULE SET Object ------------\n"
        to_print += f" Name                         = {self.name}\n"
        to_print += f" Number of Molecules          = {len(self.moleclist)}\n"
        for idx, mol in enumerate(self.moleclist):
            to_print += f"    {idx}: {mol.formula} totcharge={mol.totcharge}\n"
        to_print += f" Total charge                 = {self.totcharge}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use MoleculeSet() with keyword arguments instead.")
    def from_positional(cls, name: str, moleclist: list[Molecule]) -> MoleculeSet:
        """Implement the base model's legacy positional constructor."""
        return cls(name=name, moleclist=moleclist)
