from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from pydantic import Field

from cell2mol.classes.atom_set import AtomSet
from cell2mol.classes.bond import BondRecord
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.protonation import Protonation
from cell2mol.my_types import NDArray, RDKitObject

if TYPE_CHECKING:
    from cell2mol.classes.ligand import Ligand
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)


class Species(AtomSet):
    """Chemical species with charge, protonation and molecular representations."""

    # Preserve the legacy serialized category; class names use Species.
    type: str = Field(default="species", frozen=True)
    bonds: list[BondRecord] = Field(default_factory=list)
    totcharge: int | None = None
    atomic_charges: list[int] | NDArray | None = None
    rdkit_obj: RDKitObject | None = Field(default=None)
    smiles: str | None = None
    protonation_states: list[Protonation] | None = None
    charge_state: ChargeState | None = None
    plausible_charge_states: list[ChargeState] | None = Field(default=None)
    has_porphyrin: bool | None = None
    has_fullerene: bool | None = None
    has_borane: bool | None = None
    protonation_warning: str | None = None
    valence_search_too_large: bool | None = None
    bond_perception_capped: bool | None = None
    has_missing_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_on_CoordDonor: bool | None = None
    missing_H_in_Water: bool | None = None

    def __getattr__(self, name):
        if name == "bonds":
            # Old plain pickles bypass defaults and may be loaded directly
            # without ModelFileMixin. Upgrade only after the graph is restored.
            from cell2mol.utils.bond_records import upgrade_pickle_bonds

            upgrade_pickle_bonds(self)
            return self.__dict__["bonds"]
        return super().__getattr__(name)

    def make_bond_record(
        self, atom_indices: list[int], order: float = 1.0
    ) -> BondRecord:
        """Create a record in local atom order with its Cartesian distance."""
        if len(atom_indices) != 2 or any(
            not isinstance(i, (int, np.integer)) or i < 0 or i >= self.natoms
            for i in atom_indices
        ):
            raise ValueError("Bond requires two valid local atom indices")
        i, j = sorted(int(index) for index in atom_indices)
        if i == j:
            raise ValueError("A bond must connect two different atoms")
        start = self.atoms[i].coord if self.atoms is not None else self.coord[i]
        end = self.atoms[j].coord if self.atoms is not None else self.coord[j]
        distance = round(float(np.linalg.norm(np.asarray(start) - end)), 3)
        return {"atom_indices": [i, j], "order": float(order), "distance": distance}

    def add_bond(self, atom_indices: list[int], order: float = 1.0) -> BondRecord:
        """Insert or replace one local atom pair, keeping a single record."""
        record = self.make_bond_record(atom_indices, order)
        for position, existing in enumerate(self.bonds):
            if existing["atom_indices"] == record["atom_indices"]:
                self.bonds[position] = record
                return record
        self.bonds.append(record)
        return record

    def get_atom_bonds(self, index: int) -> list[BondRecord]:
        """Return this species' records incident to a local atom index."""
        if index < 0 or index >= self.natoms:
            raise ValueError("Atom index out of range")
        return [bond for bond in self.bonds if index in bond["atom_indices"]]

    def detect_porphyrin(self) -> bool:
        """Detect porphyrin-family macrocycles and log their structural types."""
        from cell2mol.charge.special_cases.porphyrin import (
            find_all_porphyrin_macrocycles,
        )

        macrocycles = find_all_porphyrin_macrocycles(
            self.get_atomic_numbers(), self.adjacency_matrix
        )
        self.has_porphyrin = bool(macrocycles)
        for nitrogens, is_contracted, _core in macrocycles:
            if len(nitrogens) > 4:
                description = "an expanded porphyrin-family macrocycle"
            elif is_contracted:
                description = "a contracted porphyrin-family macrocycle"
            else:
                description = "a porphyrin-type macrocycle"
            logger.debug("%s %s contains %s", self.subtype, self.formula, description)
        return self.has_porphyrin

    def detect_fullerene(self) -> bool:
        """Detect and store whether a fullerene cage is present."""
        from cell2mol.charge.special_cases import has_fullerene

        is_cage, _ = has_fullerene(self.get_atomic_numbers(), self.adjacency_matrix)
        self.has_fullerene = is_cage
        return self.has_fullerene

    def detect_borane(self) -> bool:
        """Detect and store whether a borane or carborane cage is present."""
        from cell2mol.charge.special_cases import is_borane_cage

        is_cage, _ = is_borane_cage(self.get_atomic_numbers(), self.adjacency_matrix)
        self.has_borane = is_cage
        return self.has_borane

    def detect_special_moieties(self):
        """Detect fullerene, porphyrin, and borane motifs and store their flags."""
        self.detect_fullerene()
        if self.has_fullerene:
            logger.debug("%s %s has a fullerene", self.subtype, self.formula)

        self.detect_porphyrin()

        self.detect_borane()
        if self.has_borane:
            logger.debug("%s %s has a borane/carborane", self.subtype, self.formula)

    def check_hydrogens(self):
        """Check for missing hydrogens and store carbon, donor, and water flags."""
        from cell2mol.hydrogen import detect_species_missing_hydrogens

        (
            has_missing_h,
            missing_h_in_carbon,
            missing_h_on_coordinated_donor,
            missing_h_in_water,
        ) = detect_species_missing_hydrogens(self)

        if has_missing_h:
            logger.info(
                "Missing hydrogens in species %s | carbon=%s, coordinated_donor=%s, water=%s",
                self.formula,
                missing_h_in_carbon,
                missing_h_on_coordinated_donor,
                missing_h_in_water,
            )

        self.has_missing_H = has_missing_h
        self.missing_H_in_Carbon = missing_h_in_carbon
        self.missing_H_on_CoordDonor = missing_h_on_coordinated_donor
        self.missing_H_in_Water = missing_h_in_water

        return self.has_missing_H

    def get_protonation_states(self):
        """Enumerate protonation states for ligands and metal-free molecules; otherwise None."""
        from cell2mol.charge.protonation_states import enumerate_protonation_states

        self.protonation_states = None
        self.protonation_warning = None

        if not (self.subtype == "ligand" or self.is_metal_free_molecule):
            return self.protonation_states

        if self.subtype == "ligand":
            ligand_self = cast("Ligand", self)
            if ligand_self.is_haptic is None:
                ligand_self.update_hapticity()
            if ligand_self.denticity_by_metal is None:
                ligand_self.update_denticity()
            if ligand_self.is_nitrosyl is None:
                ligand_self.detect_nitrosyl()

        self.protonation_states = enumerate_protonation_states(self)
        return self.protonation_states

    def get_plausible_charge_states(self):
        """Return cached charge candidates, or generate them for ligands and metal-free molecules."""
        from cell2mol.charge.state_generation import generate_charge_states

        if self.plausible_charge_states is not None:
            return self.plausible_charge_states

        self.plausible_charge_states = None

        if not (self.subtype == "ligand" or self.is_metal_free_molecule):
            return self.plausible_charge_states

        if self.protonation_states is None:
            self.get_protonation_states()
        logger.debug("Enumerating charge states for %s", self.formula)

        self.plausible_charge_states = generate_charge_states(self)
        return self.plausible_charge_states

    def set_charges(
        self,
        totcharge: int | None = None,
        atomic_charges: list[int] | NDArray | None = None,
        smiles: str | None = None,
        rdkit_obj: object = None,
    ) -> None:
        """Store total and atom charges, updating atom objects and supplied molecular forms."""
        if totcharge is not None:
            self.totcharge = totcharge
        elif totcharge is None and atomic_charges is not None:
            self.totcharge = int(np.sum(atomic_charges))
        elif totcharge is None and atomic_charges is None:
            self.totcharge = None
        if atomic_charges is not None:
            self.atomic_charges = atomic_charges
            if self.atoms is None:
                self.set_atoms()
            for idx, a in enumerate(self.atoms or []):
                a.set_charge(self.atomic_charges[idx])
        if smiles is not None:
            self.smiles = smiles
        if rdkit_obj is not None:
            self.rdkit_obj = rdkit_obj

    def reset_charge(self):
        """Clear assigned charges and molecular forms, including charges on atom objects."""
        self.totcharge = None
        self.atomic_charges = None
        self.smiles = None
        self.rdkit_obj = None
        self.plausible_charge_states = None

        for a in self.atoms or []:
            a.reset_charge()

    def __repr__(self, indirect: bool = False):
        text = (
            "" if indirect else "------------- Cell2mol SPECIES Object -------------\n"
        )
        text += AtomSet.__repr__(self, indirect=True)
        if self.totcharge is not None:
            text += f" Total Charge                 = {self.totcharge}\n"
        if self.subtype == "molecule" and cast("Molecule", self).spin is not None:
            text += f" Spin                         = {cast('Molecule', self).spin}\n"
        if self.smiles is not None:
            text += f" Smiles                       = {self.smiles}\n"
        if not indirect:
            text += "---------------------------------------------------\n"
        return text
