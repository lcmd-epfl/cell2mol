from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from pydantic import Field
from typing_extensions import deprecated

from cell2mol.charge.specie_assigner import assemble_complex_charge_state
from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule, assess_molecule_errors
from cell2mol.classes.specie import Specie
from cell2mol.my_types import Format, SubType, Type
from cell2mol.species_collection import (
    collect_missing_hydrogens,
    collect_plausible_charges,
    collect_unique_species,
    map_charges_to_molecules,
)
from cell2mol.utils import BaseModel, config

try:
    from typing import Self  # py3.11+
except ImportError:
    from typing_extensions import Self  # py3.10

logger = logging.getLogger(__name__)


class MoleculeSet(BaseModel):
    """Several discrete molecules that are charge-balanced as one unit."""

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    name: str
    moleclist: list[Molecule] = Field(default_factory=list)

    # Total charge of the whole set, as supplied by the user
    input_charge: int | None = None

    # Unique species related attributes
    unique_species: list[Specie | Metal] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie | Metal] | None = None

    # Missing hydrogens, aggregated over every molecule and specie
    has_isolated_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_on_CoordDonor: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None

    # Plausible integer charges per specie: metal oxidation states, or a
    # ligand/molecule's total charges. NOT one entry per specie -- it is
    # unique_species first, then all of species_list, so species recur at
    # two indices. None marks a specie whose charges could not be found.
    plausible_charges: list[list[int] | None] | None = None
    error_plausible_charges: bool | None = None
    # Set when two entries sharing a unique_index -- i.e. copies of the SAME
    # specie -- enumerated to different charges.
    error_inconsistent_plausible_charges: bool | None = None
    inconsistent_plausible_charges: dict[int, list[list[int] | None]] | None = None

    # Error assessment
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    # Primary (first-matching, most severe) code, and every code that fired.
    error_case: int | None = None
    error_cases_all: list[int] | None = None

    # Frozen fields
    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="molecule_set")
    subtype: SubType | None = Field(default="molecule_set")

    @classmethod
    @deprecated("Use MoleculeSet() with the keyword arguments instead.")
    def from_positional(cls, name: str, moleclist: list[Molecule]) -> Self:
        return cls(name=name, moleclist=moleclist)

    @property
    def formula(self) -> str:
        return " + ".join(mol.formula for mol in self.moleclist)

    @property
    def totcharge(self) -> int | None:
        """Sum of the molecular charges, or None until they have been assigned."""
        charges = [mol.totcharge for mol in self.moleclist]
        if not charges or any(charge is None for charge in charges):
            return None
        return sum(charges)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # Species
    # -------------------------------------------------------------------------
    def get_unique_species(self):
        """Deduplicate the species of every molecule against each other."""
        logger.info("Getting unique species in %s", self.subtype)

        (
            self.unique_species,
            self.unique_indices,
            self.species_list,
        ) = collect_unique_species(self.moleclist)

        return self.unique_species

    def get_plausible_charges(self):
        if self.unique_species is None:
            self.get_unique_species()

        (
            self.plausible_charges,
            self.error_plausible_charges,
            self.inconsistent_plausible_charges,
        ) = collect_plausible_charges(
            self.unique_species, self.species_list, skip_missing_h=True
        )
        self.error_inconsistent_plausible_charges = bool(
            self.inconsistent_plausible_charges
        )

    def check_hydrogens(self):
        """Check every molecule and specie for missing hydrogens.

        Same screen the reference cell runs, over the blocks of the xyz file.
        Call after ``get_unique_species``.
        """
        if self.species_list is None:
            self.get_unique_species()

        flags = collect_missing_hydrogens(self.moleclist, self.species_list)
        self.has_isolated_H = flags["has_isolated_H"]
        self.missing_H_in_Carbon = flags["missing_H_in_Carbon"]
        self.missing_H_on_CoordDonor = flags["missing_H_on_CoordDonor"]
        self.missing_H_in_Water = flags["missing_H_in_Water"]
        self.has_missing_H = flags["has_missing_H"]

        return self.has_missing_H

    # -------------------------------------------------------------------------
    # Charges, bonds and spin
    # -------------------------------------------------------------------------
    def assign_charges(self):
        """Propagate the balanced charges onto every molecule and bond them."""
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
                    "Non-Complex %d: %s %s %s",
                    idx,
                    mol.formula,
                    mol.totcharge,
                    mol.smiles,
                )

        logger.info("Total charge of %s: %s", self.name, self.totcharge)

    def create_bonds(self):
        """Create bonds in every molecule, folding the failures together."""
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

    def get_spin(self):
        """Assign spin multiplicity to every molecule."""
        overall_error = False

        for mol in self.moleclist:
            mol.error_get_spin = False
            try:
                if mol.iscomplex:
                    for metal in mol.metals or []:
                        if metal.coord_nr is None:
                            metal.get_coordination_geometry()
                            metal.get_coord_sphere_formula()
                        metal.get_spin()
                mol.get_spin()
            except Exception as e:
                mol.error_get_spin = True
                overall_error = True
                logger.error("Spin assignment failed for %s: %s", mol.formula, e)
                logger.debug("Exception details", exc_info=True)

        self.error_get_spin = overall_error

    def assess_errors(self):
        """Reduce the flags to error codes, on the same rules as ``Molecule``."""
        return assess_molecule_errors(self)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, path: str | Path, *, format: Format = "json"):
        if format == "json":
            self._save_as_json(path)
        elif format == "pickle":
            self._save_as_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str | Path, *, format: Format = "json") -> Self:
        if format == "json":
            return cls._load_from_json(path)
        elif format == "pickle":
            return cls._load_from_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_as_json(self, path: str | Path):
        if not str(path).endswith(".json"):
            logger.warning("Use `.json` extension instead for path: %s", path)
        with open(path, "w") as fd:
            fd.write(self.to_json(indent=4))

    @deprecated("Use json format instead")
    def _save_as_pickle(self, path: str | Path):
        logger.warning("Use json format instead")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    @classmethod
    def _load_from_json(cls, path: str | Path) -> Self:
        with open(path, "r") as fd:
            return cls.from_json(fd.read())

    @classmethod
    @deprecated("Use json format instead")
    def _load_from_pickle(cls, path: str | Path) -> Any:
        with open(path, "rb") as fil:
            return pickle.load(fil)

    def __repr__(self, indirect: bool = False):
        to_print = ""
        to_print += "----------- Cell2mol MOLECULE SET Object ------------\n"
        to_print += f" Name                         = {self.name}\n"
        to_print += f" Number of Molecules          = {len(self.moleclist)}\n"
        for idx, mol in enumerate(self.moleclist):
            to_print += f"    {idx}: {mol.formula} totcharge={mol.totcharge}\n"
        to_print += f" Total charge                 = {self.totcharge}\n"
        to_print += "---------------------------------------------------\n"
        return to_print
