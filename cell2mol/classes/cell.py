from __future__ import annotations
import pickle
import logging
from typing import Any
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.classes.molecule import Molecule
from cell2mol.charge.specie_assigner import prepare_mol
from cell2mol.elementdata import ElementData
from cell2mol.utils import BaseModel, config
from cell2mol.my_types import NDArray, Type, SubType

logger = logging.getLogger(__name__)

elemdatabase = ElementData()

Labels = list[str]
_CHARGE_ERRORS = [("error_assign_charge", 8), ("error_create_bonds", 9)]
_SPIN_ERRORS = [("error_get_spin", 10)]

ERROR_MAPS = {
    "reference": {
        "hydrogens": [
            ("has_isolated_H", 1),
            ("missing_H_in_Water", 2),
            ("missing_H_in_CoordWater", 3),
            ("missing_H_in_Carbon", 4),
        ],
        "possible_charges": [("error_get_poscharges", 5)],
        "charge_assignment": _CHARGE_ERRORS,
        "spin_assignment": _SPIN_ERRORS,
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
    error_case: int | None = None

    # Frozen fields
    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="cell")

    def model_post_init(self, __context: Any) -> None:
        # Compute natoms from labels length
        if self.natoms is None:
            self.natoms = len(self.labels)

    def set_subtype(self, subtype: SubType):
        self.subtype = subtype

    def assign_charges(self, refmoleclist: list[Molecule] | None = None):
        """
        Master function to assign charges, create bonds, and log results.
        """
        # Determine molecule list and map charges
        if self.subtype == "reference":
            molecule_list = self.refmoleclist
            self.map_charges_to_reference()
            log_prefix = "Reference"
        elif self.subtype == "unitcell":
            molecule_list = self.moleclist
            self.map_charges_to_unitcell(refmoleclist=refmoleclist)
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
                    prepare_mol(mol)
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
                    "Non-Complex %d: %s %d %s",
                    idx,
                    mol.formula,
                    mol.totcharge,
                    mol.smiles,
                )
            else:
                logger.info("Complex %d: %s %d", idx, mol.formula, mol.totcharge)

                for jdx, lig in enumerate(mol.ligands):
                    logger.info(
                        "  Ligand %d: %s %d %s",
                        jdx,
                        lig.formula,
                        lig.totcharge,
                        lig.smiles,
                    )
                for kdx, met in enumerate(mol.metals):
                    logger.info("  Metal %d: %s %d", kdx, met.formula, met.charge)

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
            moleclist = self.refmoleclist
        elif self.subtype == "unitcell":
            moleclist = self.moleclist
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
                    for metal in mol.metals:
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
            moleclist = self.refmoleclist
        elif self.subtype == "unitcell":
            moleclist = self.moleclist
        else:
            raise ValueError(f"Unknown subtype {self.subtype} of Cell")

        if moleclist is None:
            return

        for mol in moleclist:
            if mol.iscomplex:
                for metal in mol.metals:
                    if metal.coord_nr is None:
                        metal.get_coordination_geometry()
                        metal.get_coord_sphere_formula()
                    metal.predict_charge()

    def assess_errors(self, mode):
        """
        Assess error conditions based on subtype and mode.
        First truthy attribute found sets the error_case and exits.
        """
        # 1. Get the map for the current subtype
        subtype_map = ERROR_MAPS.get(self.subtype)
        if not subtype_map:
            raise ValueError(f"Unknown Cell subtype: {self.subtype}")

        # 2. Get the rules for the specific processing mode
        rules = subtype_map.get(mode)
        if rules is None:
            raise ValueError(f"Invalid mode '{mode}' for subtype '{self.subtype}'")

        # 3. Check attributes
        for attr, code in rules:
            # We use False as default so missing attributes don't trigger errors
            is_triggered = getattr(self, attr, False)

            logger.debug(
                "Checking %s (Code %d) in %s:%s. Result: %s",
                attr,
                code,
                self.subtype,
                mode,
                is_triggered,
            )

            if is_triggered:
                self.error_case = code
                return

        # 4. Fallback if no errors triggered
        self.error_case = 0

    def has_error(self) -> bool:
        """
        Return True if the cell is in an error state.
        error_case == 0 means no error.
        """
        return self.error_case != 0

    def save(self, path):
        """Save the Cell object to a file using pickle."""
        logger.info(f"SAVING cell2mol CELL ({self.subtype}) object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    def __str__(self):
        """Return a string representation of the Cell object."""
        return self.__repr__()

    def __repr__(self):
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
        pos: list[list[float]],
        frac_coord: list[list[float]],
        cell_vector: object,
        cell_param: object,
    ) -> "Cell":
        return cls(
            name=name,
            labels=labels,
            pos=pos,  # Using pos which gets aliased to coord
            frac_coord=frac_coord,
            cell_vector=cell_vector,
            cell_param=cell_param,
        )
