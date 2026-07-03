from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pydantic import Field, computed_field
from typing_extensions import deprecated
from cell2mol.element_utils import get_radii
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray, OptionalInt, RefList, SubType
from cell2mol.utils import BaseModel

if TYPE_CHECKING:
    from cell2mol.classes.bond import Bond
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.cell import Cell
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class Atom(BaseModel):
    label: str
    coord: NDArray
    frac_coord: NDArray | None = None
    radii: float | None = None

    # Cross-references to parent species (Molecule, Ligand, etc.)
    parents: list["Specie | Cell"] = Field(default_factory=list)
    parents_index: list[int] = Field(default_factory=list)

    # Seem to be set in various places depending on the context, might need to check this
    cov_factor: float | None = None
    metal_factor: float | None = None
    atom_site_label: str | None = None
    connec: int | None = None
    mconnec: int | None = None
    adjacency: list[int] = Field(default_factory=list)
    metal_adjacency: list[int] = Field(default_factory=list)
    charge: OptionalInt = None

    # Cross-references to Bond objects
    bonds: RefList["Bond"] = Field(default_factory=list)

    type: str = Field(default="atom", frozen=True)
    # Originally atom does not have a subtype, but we add it for consistency with other classes
    subtype: SubType = Field(default="atom")

    @computed_field
    @property
    def atnum(self) -> int:
        return elemdatabase.elementnr[self.label]

    @computed_field
    @property
    def block(self) -> str:
        return elemdatabase.elementblock[self.label]

    @computed_field
    @property
    def formula(self) -> str:
        return self.label

    def model_post_init(self, __context: Any) -> None:
        if self.radii is None:
            radii_array = get_radii([self.label])
            self.radii = float(radii_array[0])

    @classmethod
    @deprecated("Use atom() with the keyword arguments instead.")
    def from_positional(
        cls,
        label: str,
        coord: np.ndarray | list[float],
        frac_coord: np.ndarray | list[float] | None = None,
        radii: float | None = None,
    ) -> "Atom":
        """
        Creates an atom instance using positional arguments.

        Args:
            label: Atomic symbol
            coord: Cartesian coordinates [x,y,z]
            frac_coord: Optional fractional coordinates
            radii: Optional atomic radius

        Returns:
            atom: New atom instance
        """
        return cls(
            label=label,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=radii,
        )

    def add_parent(self, parent: "Specie | Cell", index: int, overwrite: bool = True):
        ## associates a parent specie to self. The atom indices of self in parent are given in "indices"
        ## if parent of the same subtype already in self.parent then it is overwritten
        ## this is to avoid having a substructure (e.g. a ligand) in more than one superstructure (e.g. a molecule)
        append = True
        for idx, p in enumerate(self.parents):
            if p.subtype == parent.subtype:
                if overwrite:
                    self.parents[idx] = parent
                    self.parents_index[idx] = index
                append = False
        if append:
            self.parents.append(parent)
            self.parents_index.append(index)

    def check_parent(self, subtype: str):
        ## checks if parent of a given subtype exists
        for p in self.parents:
            if p.subtype == subtype:
                return True
        return False

    def get_parent(self, subtype: str):
        ## retrieves parent of a given subtype
        for p in self.parents:
            if p.subtype == subtype:
                return p
        return None

    def get_parent_index(self, subtype: str):
        ## retrieves parent of a given subtype
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype:
                return self.parents_index[idx]
        return None

    def add_bond(self, newbond: Bond):
        at1 = newbond.atom1
        at2 = newbond.atom2
        found = False
        for b in self.bonds:
            if (b.atom1 == at1 and b.atom2 == at2) or (
                b.atom1 == at2 and b.atom2 == at1
            ):
                ### It means that the same bond has already been defined
                found = True
        if not found:
            self.bonds.append(newbond)

    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    def reset_charge(self) -> None:
        self.charge = None

    def set_charge(self, charge: int) -> None:
        self.charge = int(charge)

    def set_adjacencies(self, adjmat, madjmat, adjnum: int, madjnum: int):
        self.connec = int(adjnum)
        self.mconnec = int(madjnum)
        self.adjacency = []
        self.metal_adjacency = []

        # The atom only receives one row of adjmat, so this is not a matrix anymore.
        # Keep in mind that the idx are the indices of parent
        for idx, c in enumerate(adjmat):
            if c > 0:
                self.adjacency.append(idx)
        # The atom only receives one row of madjmat, so this is not a matrix anymore
        for idx, c in enumerate(madjmat):
            if c > 0:
                self.metal_adjacency.append(idx)

    @property
    def closest_metal(self):
        return self.get_closest_metal()

    def get_closest_metal(self):
        ## Here, the list of metal atoms must be provided
        apos = self.coord
        dist = []
        mol = cast("Molecule", self.get_parent("molecule"))
        metals = mol.metals or []
        for met in metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        closest_metal = metals[int(np.argmin(dist))]
        return closest_metal

    def set_atom_site_label(self, atom_site_label: str) -> None:
        self.atom_site_label = atom_site_label

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        to_print = ""
        if not indirect:
            to_print += "------------- Cell2mol ATOM Object ----------------\n"
        to_print += f" Type                         = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type                     = {self.subtype}\n"
        to_print += f" Label                        = {self.label}\n"
        if self.atom_site_label is not None:
            to_print += f" Atom Site Label              = {self.atom_site_label}\n"
        to_print += f" Atomic Number                = {self.atnum}\n"
        idx = self.get_parent_index("molecule")
        if idx is not None:
            to_print += f" Index in Molecule            = {idx}\n"
        idx = self.get_parent_index("ligand")
        if idx is not None:
            to_print += f" Index in Ligand              = {idx}\n"
        if self.mconnec is not None:
            to_print += f" Metal Adjacency (mconnec)    = {self.mconnec}\n"
        if self.connec is not None:
            to_print += f" Regular Adjacencies (connec) = {self.connec}\n"
        if self.charge is not None:
            to_print += f" Atom Charge                  = {self.charge}\n"
        if not indirect:
            to_print += "----------------------------------------------------\n"
        return to_print

    def reset_mconnec(self, met: "Atom", diff: int = -1):
        self_info = f"{self.label}{f' ({self.atom_site_label})' if self.atom_site_label else ''}"
        met_info = (
            f"{met.label}{f' ({met.atom_site_label})' if met.atom_site_label else ''}"
        )
        if not self.check_parent("molecule"):
            return
        self_mol_idx = self.get_parent_index("molecule")
        met_mol_idx = met.get_parent_index("molecule")
        mol = cast("Specie", self.get_parent("molecule"))
        assert self_mol_idx is not None
        assert met_mol_idx is not None
        assert mol is not None
        assert mol.atoms is not None
        assert mol.adjmat is not None
        assert mol.madjmat is not None
        assert mol.adjnum is not None
        assert mol.madjnum is not None
        logger.info(f"Reset mconnec: atom={self_info} diff={diff} to metal={met_info}")

        logger.debug(
            "Initial atom connectivity: mol_idx=%s connec=%d mconnec=%d adj=%s madj=%s",
            self_mol_idx,
            self.connec,
            self.mconnec,
            self.adjacency,
            self.metal_adjacency,
        )

        logger.debug(
            "Initial metal connectivity: mol_idx=%s connec=%d mconnec=%d adj=%s madj=%s",
            met_mol_idx,
            met.connec,
            met.mconnec,
            met.adjacency,
            met.metal_adjacency,
        )

        # ---------- Fix adjacency matrix in Molecule level ----------

        logger.info(
            "Updating molecule: atom_idx=%d metal_idx=%d", self_mol_idx, met_mol_idx
        )

        logger.debug(
            "Molecule before: atom(connec=%d,mconnec=%d) metal(connec=%d,mconnec=%d)",
            mol.atoms[self_mol_idx].connec,
            mol.atoms[self_mol_idx].mconnec,
            met.connec,
            met.mconnec,
        )

        # Update numbers
        mol.madjnum[self_mol_idx] += diff
        mol.madjnum[met_mol_idx] += diff
        mol.adjnum[self_mol_idx] += diff
        mol.adjnum[met_mol_idx] += diff

        # Update matrices
        mol.madjmat[self_mol_idx, met_mol_idx] += diff
        mol.madjmat[met_mol_idx, self_mol_idx] += diff
        mol.adjmat[self_mol_idx, met_mol_idx] += diff
        mol.adjmat[met_mol_idx, self_mol_idx] += diff

        self.set_adjacencies(
            mol.adjmat[self_mol_idx],
            mol.madjmat[self_mol_idx],
            mol.adjnum[self_mol_idx],
            mol.madjnum[self_mol_idx],
        )

        met.set_adjacencies(
            mol.adjmat[met_mol_idx],
            mol.madjmat[met_mol_idx],
            mol.adjnum[met_mol_idx],
            mol.madjnum[met_mol_idx],
        )

        logger.info(
            "Molecule after: atom(connec=%d,mconnec=%d adj=%s madj=%s)",
            mol.atoms[self_mol_idx].connec,
            mol.atoms[self_mol_idx].mconnec,
            mol.atoms[self_mol_idx].adjacency,
            mol.atoms[self_mol_idx].metal_adjacency,
        )

        logger.debug(
            "Metal after: connec=%d mconnec=%d adj=%s madj=%s",
            mol.atoms[met_mol_idx].connec,
            mol.atoms[met_mol_idx].mconnec,
            mol.atoms[met_mol_idx].adjacency,
            mol.atoms[met_mol_idx].metal_adjacency,
        )

        logger.info(
            f"Final: atom={self_info} diff={diff} to metal={met_info}",
        )

        logger.debug(
            "Final atom connectivity: mol_idx=%s connec=%d mconnec=%d adj=%s madj=%s",
            self_mol_idx,
            self.connec,
            self.mconnec,
            self.adjacency,
            self.metal_adjacency,
        )

        logger.debug(
            "Final metal connectivity: mol_idx=%s connec=%d mconnec=%d adj=%s madj=%s",
            met_mol_idx,
            met.connec,
            met.mconnec,
            met.adjacency,
            met.metal_adjacency,
        )
