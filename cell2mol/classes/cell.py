from __future__ import annotations
import pickle
import logging
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.classes.molecule import Molecule
from cell2mol.charge.specie_assigner import assemble_complex_charge_state
from cell2mol.elementdata import ElementData
from cell2mol.utils import BaseModel, config
from cell2mol.my_types import NDArray, Type, SubType, Format
from pathlib import Path

if TYPE_CHECKING:
    from cell2mol.classes.reference import Reference
    from cell2mol.classes.unitcell import UnitCell

try:
    from typing import Self  # py3.11+
except ImportError:
    from typing_extensions import Self  # py3.10
logger = logging.getLogger(__name__)

elemdatabase = ElementData()

Labels = list[str]
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


class Cell(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Required constructor parameters
    name: str
    labels: Labels
    coord: NDArray = Field(alias="pos")  # Using alias to match original parameter name
    frac_coord: NDArray
    cell_vector: NDArray
    cell_param: NDArray

    natoms: int | None = None
    subtype: SubType | None = None

    # Unique species (only defined for reference) related attributes
    unique_indices: list[int] | None = None

    # Specie for molecules, ligands, or metals
    species_list: list[Specie | Metal] | None = None

    # Error assessment
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    # error_case: int | None = None
    # Primary (first-matching, most severe) error code per processing mode.
    error_cases: dict[str, int] | None = None
    # Every code that fired for that mode. A mode can trigger several rules at
    # once -- "hydrogens" is the usual case, where a structure can be short of
    # hydrogens on a coordinated donor AND on carbon -- and reporting only the
    # first hides the rest. error_cases keeps the single primary code, so exit
    # codes and has_error() are unaffected.
    error_cases_all: dict[str, list[int]] | None = None

    # Comparison results
    total_charge_comparison: bool | None = None
    metal_os_comparison: bool | None = None

    # Frozen fields
    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="cell")

    def model_post_init(self, __context: Any) -> None:
        # Compute natoms from labels length
        if self.natoms is None:
            self.natoms = len(self.labels)

    def set_subtype(self, subtype: SubType):
        self.subtype = subtype

    def set_comparison_info(self, total_charge_comparison, metal_os_comparison):
        """Save the results of total charge and metal oxidation state comparisons."""
        self.total_charge_comparison = total_charge_comparison
        self.metal_os_comparison = metal_os_comparison

    def assign_charges(self, refmoleclist: list[Molecule] | None = None):
        """
        Master function to assign charges, create bonds, and log results.
        """
        # Determine molecule list and map charges
        if self.subtype == "reference":
            reference_self = cast("Reference", self)
            molecule_list = reference_self.refmoleclist
            reference_self.map_charges_to_reference()
            log_prefix = "Reference"
        elif self.subtype == "unitcell":
            unitcell_self = cast("UnitCell", self)
            molecule_list = unitcell_self.moleclist
            unitcell_self.map_charges_to_unitcell(refmoleclist=refmoleclist or [])
            log_prefix = "UnitCell"
        else:
            raise ValueError(f"Unknown subtype {self.subtype} of Cell")

        logger.info("=" * 40)
        logger.info("Assigning Charges to %s Molecules", log_prefix)
        logger.info("=" * 40)

        # Finalize bonds and Log results
        self._finalize_bonds_and_prepare(molecule_list)
        self._log_charge_results(molecule_list, log_prefix)

    def _finalize_bonds_and_prepare(self, molecule_list):
        """
        Shared logic to finalize bonding and prepare complex molecules.
        Only prepares molecules if bonding was successful.
        """
        overall_error = False

        for mol in molecule_list:
            # Create Bonds
            try:
                mol.create_bonds()
            except Exception as e:
                mol.error_create_bonds = True
                logger.error("Bonding failed for %s: %s", mol, e)

            # Update overall error state if any create_bonds failed
            if mol.error_create_bonds:
                overall_error = True

            #  Prepare Complex Molecules (if bonding succeeded)
            if not mol.is_non_complex_molecule and not mol.error_create_bonds:
                try:
                    assemble_complex_charge_state(mol)
                except Exception as e:
                    # We flag the error state here as well
                    mol.error_create_bonds = True
                    overall_error = True
                    logger.error("Preparation failed for %s: %s", mol, e)

        self.error_create_bonds = overall_error

    def _log_charge_results(self, molecule_list, label: str):
        """Shared logic to log the final state of molecules."""
        for idx, mol in enumerate(molecule_list):
            logger.info("ASSIGN_CHARGES: %s Molecule %d %s", label, idx, mol.formula)

            if mol.is_non_complex_molecule:
                logger.info(
                    "Non-Complex %d: %s %s %s",
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

    def assign_spin(self):
        """
        Assign spin multiplicity for all molecules in the cell.
        Attempts spin assignment for all molecules and records
        whether any failures occurred.
        """
        logger.info("=" * 40)
        logger.info("       Assigning Spin multiplicity       ")
        logger.info("=" * 40)

        overall_error = False

        if self.subtype == "reference":
            moleclist = cast("Reference", self).refmoleclist
        elif self.subtype == "unitcell":
            moleclist = cast("UnitCell", self).moleclist
        else:
            raise ValueError(f"Unknown subtype {self.subtype} of Cell")

        if moleclist is None:
            raise ValueError(
                "Molecule list is None before spin assignment (invalid state)"
            )

        for mol in moleclist:
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
                logger.error("Spin assignment failed for %s: %s", mol, e)
                logger.debug("Exception details", exc_info=True)

        self.error_get_spin = overall_error

    def predict_metal_ox(self):
        """Predict oxidation states for metals in all molecules in the cell."""
        if self.subtype == "reference":
            moleclist = cast("Reference", self).refmoleclist
        elif self.subtype == "unitcell":
            moleclist = cast("UnitCell", self).moleclist
        else:
            raise ValueError(f"Unknown subtype {self.subtype} of Cell")

        if moleclist is None:
            return

        for mol in moleclist:
            if mol.iscomplex:
                for metal in mol.metals or []:
                    if metal.coord_nr is None:
                        metal.get_coordination_geometry()
                        metal.get_coord_sphere_formula()
                    metal.predict_charge()

    def assess_errors(self, mode):
        """Assess error conditions for a specific processing mode."""
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

        # Rules are listed most-severe-first, so the first hit is the primary
        # code; keep every hit alongside it so nothing is silently dropped.
        self.error_cases_all[mode] = triggered
        self.error_cases[mode] = triggered[0] if triggered else 0

    def has_error(self, mode: str | None = None) -> bool:
        """
        Return True if an error exists.

        If mode is provided, check only that processing mode.
        If mode is None, check across all recorded modes.
        """
        if self.error_cases is None:
            return False

        if mode is not None:
            return self.error_cases.get(mode, 0) != 0

        # Check all modes
        return any(code != 0 for code in self.error_cases.values())

    def save(self, path: str | Path, *, format: Format = "pickle"):
        if format == "json":
            self._save_as_json(path)
        elif format == "pickle":
            self._save_as_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str | Path, *, format: Format = "pickle") -> Self:
        if format == "json":
            return cls._load_from_json(path)
        elif format == "pickle":
            return cls._load_from_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @deprecated("Use json format instead")
    def _save_as_pickle(
        self,
        path: str | Path,
    ):
        logger.warning("Use json format instead")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    def _save_as_json(
        self,
        path: str | Path,
    ):
        if not str(path).endswith(".json"):
            logger.warning("Use `.json` extension instead for path: %s", path)
        # Pretty print with indent=4
        with open(path, "w") as fd:
            fd.write(self.to_json(indent=4))
        # Minified version
        # with open(path, "w") as fd:
        #     fd.write(self.to_json(separators=(",", ":")))

    @classmethod
    def _load_from_json(
        cls,
        path: str | Path,
    ) -> Self:
        with open(path, "r") as fd:
            return cls.from_json(fd.read())

    @classmethod
    @deprecated("Use json format instead")
    def _load_from_pickle(
        cls,
        path: str | Path,
    ):
        with open(path, "rb") as fil:
            return pickle.load(fil)

    def __str__(self):
        """Return a string representation of the Cell object."""
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
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
        # to_print += f' Cell Vector           = {self.cell_vector}\n'
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use cell() with the keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]],
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "Cell":
        return cls(
            name=name,
            labels=labels,
            pos=np.asarray(pos),  # Using pos which gets aliased to coord
            frac_coord=np.asarray(frac_coord),
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
