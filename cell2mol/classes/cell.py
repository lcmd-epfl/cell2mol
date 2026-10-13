from __future__ import annotations

from cell2mol.utils.serialization import ModelFileMixin
import logging
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.classes.molecule import Molecule
from cell2mol.utils import BaseModel, config
from cell2mol.my_types import NDArray, Type, SubType

if TYPE_CHECKING:
    from cell2mol.classes.reference import Reference
    from cell2mol.classes.unitcell import UnitCell

logger = logging.getLogger(__name__)


_CHARGE_ERRORS = [("error_assign_charge", 8), ("error_create_bonds", 9)]
_SPIN_ERRORS = [("error_get_spin", 10)]

ERROR_MAPS = {
    "reference": {
        "ref_molecules": [("ref_molecules", -1)],
        "hydrogens": [
            ("has_isolated_H", 1),
            ("missing_H_in_Water", 2),
            ("missing_H_on_CoordDonor", 3),
            ("missing_H_in_Carbon", 4),
        ],
        "plausible_charges": [("error_plausible_charges", 5)],
        "charge_assignment": _CHARGE_ERRORS,
        "spin_assignment": _SPIN_ERRORS,
        "general": [("general_error", config.ERR_GENERAL)],
        "timeout": [("timeout_error", config.ERR_TIMEOUT)],
        "memory": [("memory_error", config.ERR_MEMORY)],
    },
    "unitcell": {
        "reconstruction": [
            ("error_get_fragments", 3),
            ("error_reconstruction", 4),
        ],
        "balance_charges": [
            ("error_multiple_distrib", 6),
            ("error_empty_distrib", 7),
        ],
        "charge_assignment": _CHARGE_ERRORS,
        "spin_assignment": _SPIN_ERRORS,
        "general": [("general_error", config.ERR_GENERAL)],
        "timeout": [("timeout_error", config.ERR_TIMEOUT)],
        "memory": [("memory_error", config.ERR_MEMORY)],
    },
}


class Cell(ModelFileMixin, BaseModel):
    """Shared structure data, charge assignment, and step errors for cells."""

    model_config = {"populate_by_name": True}

    # Structure inputs
    name: str
    labels: list[str]
    coord: NDArray = Field(alias="pos")  # Legacy input alias
    frac_coord: NDArray
    cell_vector: NDArray
    cell_param: NDArray

    natoms: int | None = None
    subtype: SubType | None = None

    unique_indices: list[int] | None = None

    # Metal-free molecules, ligands, and metals used for charge balancing.
    species_list: list[Specie | Metal] | None = None

    # Step error flags, primary codes, and all matching codes.
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    error_cases: dict[str, int] | None = None
    error_cases_all: dict[str, list[int]] | None = None

    # Comparison results
    total_charge_comparison: bool | None = None
    metal_os_comparison: bool | None = None

    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="cell")

    def model_post_init(self, __context: Any) -> None:
        """Set the atom count when it is not supplied."""
        if self.natoms is None:
            self.natoms = len(self.labels)

    def set_subtype(self, subtype: SubType):
        """Store the cell subtype."""
        self.subtype = subtype

    def set_comparison_results(self, total_charge_comparison, metal_os_comparison):
        """Store total-charge and metal oxidation-state comparison results."""
        self.total_charge_comparison = total_charge_comparison
        self.metal_os_comparison = metal_os_comparison

    def assign_charges(self, refmoleclist: list[Molecule] | None = None):
        """Map charges, build bonds, assemble complexes, and log results."""
        if self.subtype == "reference":
            reference_self = cast("Reference", self)
            molecule_list = self._get_molecules()
            reference_self.map_charges_to_reference()
            log_prefix = "Reference"
        elif self.subtype == "unitcell":
            unitcell_self = cast("UnitCell", self)
            molecule_list = self._get_molecules()
            unitcell_self.map_charges_to_unitcell(refmoleclist=refmoleclist or [])
            log_prefix = "UnitCell"
        else:
            raise ValueError(f"Unknown subtype {self.subtype} of Cell")

        logger.info("=" * 40)
        logger.info("Assigning Charges to %s Molecules", log_prefix)
        logger.info("=" * 40)

        self._create_bonds_and_assemble_complexes(molecule_list)
        self._log_charge_results(molecule_list, log_prefix)

    def _create_bonds_and_assemble_complexes(self, molecule_list):
        """Build bonds and assemble complexes, recording failures per molecule."""
        from cell2mol.charge.charge_assignment import assemble_complex_charge_state

        overall_error = False

        for mol in molecule_list:
            try:
                mol.create_bonds()
            except Exception as e:
                mol.error_create_bonds = True
                logger.error("Bonding failed for %s: %s", mol, e)

            if mol.error_create_bonds:
                overall_error = True

            # Assemble complexes only after bonding succeeds.
            if not mol.is_metal_free_molecule and not mol.error_create_bonds:
                try:
                    assemble_complex_charge_state(mol)
                except Exception as e:
                    mol.error_create_bonds = True
                    overall_error = True
                    logger.error("Preparation failed for %s: %s", mol, e)

        self.error_create_bonds = overall_error

    def _log_charge_results(self, molecule_list, label: str):
        """Log assigned molecule, ligand, and metal charges."""
        for idx, mol in enumerate(molecule_list):
            logger.info("ASSIGN_CHARGES: %s Molecule %d %s", label, idx, mol.formula)

            if mol.is_metal_free_molecule:
                logger.info(
                    "Metal-free %d: %s %s %s",
                    idx,
                    mol.formula,
                    mol.totcharge,
                    mol.smiles,
                )
            else:
                logger.info("Complex %d: %s %s", idx, mol.formula, mol.totcharge)

                for jdx, lig in enumerate(mol.ligands):
                    logger.info(
                        "  Ligand %d: %s %s %s",
                        jdx,
                        lig.formula,
                        lig.totcharge,
                        lig.smiles,
                    )
                for kdx, met in enumerate(mol.metals):
                    logger.info("  Metal %d: %s %s", kdx, met.formula, met.charge)

    def _get_molecules(self) -> list[Molecule] | None:
        """Return molecules owned by the concrete cell subtype."""
        if self.subtype == "reference":
            return cast("Reference", self).refmoleclist
        if self.subtype == "unitcell":
            return cast("UnitCell", self).moleclist
        raise ValueError(f"Unknown subtype {self.subtype} of Cell")

    def assign_spin(self):
        """Assign spins to every molecule and record assignment failures."""
        logger.info("=" * 40)
        logger.info("       Assigning Spin multiplicity       ")
        logger.info("=" * 40)

        overall_error = False

        moleclist = self._get_molecules()

        if moleclist is None:
            raise ValueError(
                "Molecule list is None before spin assignment (invalid state)"
            )

        for mol in moleclist:
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
                logger.error("Spin assignment failed for %s: %s", mol, e)
                logger.debug("Exception details", exc_info=True)

        self.error_get_spin = overall_error

    def assess_errors(self, mode):
        """Record the primary and all matching error codes for one processing step."""
        if self.error_cases is None:
            self.error_cases = {}
        if self.error_cases_all is None:
            self.error_cases_all = {}

        subtype_map = ERROR_MAPS.get(self.subtype or "")
        if not subtype_map:
            raise ValueError(f"Unknown Cell subtype: {self.subtype}")

        rules = subtype_map.get(mode)
        if rules is None:
            raise ValueError(f"Invalid mode '{mode}' for subtype '{self.subtype}'")

        triggered = []
        for attr, code in rules:
            is_triggered = getattr(self, attr, False)

            logger.debug(
                "Checking %s (Code %d) in %s: Result: %s",
                attr,
                code,
                self.subtype,
                is_triggered,
            )

            if is_triggered:
                triggered.append(code)

        # Rule order sets priority; retain every match.
        self.error_cases_all[mode] = triggered
        self.error_cases[mode] = triggered[0] if triggered else 0

    def has_error(self, mode: str | None = None) -> bool:
        """Check one processing step, or all recorded steps when mode is None."""
        if self.error_cases is None:
            return False

        if mode is not None:
            return self.error_cases.get(mode, 0) != 0

        return any(code != 0 for code in self.error_cases.values())

    def __str__(self):
        """Return the formatted cell details."""
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        """Return cell metadata and lattice parameters."""
        to_print = ""
        if not indirect:
            to_print = "------------- Cell2mol CELL Object ----------------\n"
        to_print += f" Version               = {self.version}\n"
        to_print += f" Type                  = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type              = {self.subtype}\n"
        to_print += f" Name (Refcode)        = {self.name}\n"
        to_print += f" Num Atoms             = {self.natoms}\n"
        to_print += f" Cell Parameters a:c   = {self.cell_param[0:3]}\n"
        to_print += f" Cell Parameters al:ga = {self.cell_param[3:6]}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use Cell() with keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]],
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "Cell":
        """Construct a cell using the legacy positional interface."""
        return cls(
            name=name,
            labels=labels,
            pos=np.asarray(pos),
            frac_coord=np.asarray(frac_coord),
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
