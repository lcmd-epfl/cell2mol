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
    """Atom coordinates, connectivity, charge, and parent references."""

    label: str
    coord: NDArray
    frac_coord: NDArray | None = None
    radii: float | None = None

    # Parent objects and the atom index within each parent.
    parents: list["Specie | Cell"] = Field(default_factory=list)
    parents_index: list[int] = Field(default_factory=list)

    cov_factor: float | None = None
    metal_factor: float | None = None
    atom_site_label: str | None = None
    connec: int | None = None
    mconnec: int | None = None
    adjacency: list[int] = Field(default_factory=list)
    metal_adjacency: list[int] = Field(default_factory=list)
    charge: OptionalInt = None

    # References to bonds involving this atom.
    bonds: RefList["Bond"] = Field(default_factory=list)

    type: str = Field(default="atom", frozen=True)
    subtype: SubType = Field(default="atom")

    def model_post_init(self, __context: Any) -> None:
        """Fill a missing radius from the element table."""
        if self.radii is None:
            radii_array = get_radii([self.label])
            self.radii = float(radii_array[0])

    @computed_field
    @property
    def atnum(self) -> int:
        """Return the atomic number."""
        return elemdatabase.elementnr[self.label]

    @computed_field
    @property
    def block(self) -> str:
        """Return the periodic-table block."""
        return elemdatabase.elementblock[self.label]

    @computed_field
    @property
    def formula(self) -> str:
        """Return the element symbol as the atom formula."""
        return self.label

    def check_parent(self, subtype: str):
        """Return whether a parent with this subtype exists."""
        for p in self.parents:
            if p.subtype == subtype:
                return True
        return False

    def get_parent(self, subtype: str):
        """Return the parent with this subtype, or None."""
        for p in self.parents:
            if p.subtype == subtype:
                return p
        return None

    def get_parent_index(self, subtype: str):
        """Return this atom's index in the requested parent, or None."""
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype:
                return self.parents_index[idx]
        return None

    @property
    def closest_metal(self):
        """Return the nearest metal in the parent molecule."""
        return self.find_closest_metal()

    def add_parent(self, parent: "Specie | Cell", index: int, overwrite: bool = True):
        """Register a parent and this atom's index within it.

        Keep one parent per subtype; replace it when overwrite=True.
        """
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

    def add_bond(self, newbond: Bond):
        """Register a bond unless its endpoint pair is already present."""
        at1 = newbond.atom1
        at2 = newbond.atom2
        for bond in self.bonds:
            if (bond.atom1 is at1 and bond.atom2 is at2) or (
                bond.atom1 is at2 and bond.atom2 is at1
            ):
                return
        self.bonds.append(newbond)

    def reset_charge(self) -> None:
        """Clear the assigned charge."""
        self.charge = None

    def set_charge(self, charge: int) -> None:
        """Store an integer charge."""
        self.charge = int(charge)

    def set_connectivity(self, adjmat, madjmat, adjnum: int, madjnum: int):
        """Store connectivity counts and neighbors from parent adjacency rows."""
        self.connec = int(adjnum)
        self.mconnec = int(madjnum)
        self.adjacency = []
        self.metal_adjacency = []

        # Adjacency rows use indices in the parent molecule.
        for idx, c in enumerate(adjmat):
            if c > 0:
                self.adjacency.append(idx)
        for idx, c in enumerate(madjmat):
            if c > 0:
                self.metal_adjacency.append(idx)

    def find_closest_metal(self):
        """Return the nearest metal in the parent molecule."""
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
        """Store the source structure atom label."""
        self.atom_site_label = atom_site_label

    def update_metal_connectivity(self, met: "Atom", diff: int = -1):
        """Adjust the atom–metal bond and refresh both atoms' connectivity."""
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

        # Update both endpoints in the parent molecule.
        mol.madjnum[self_mol_idx] += diff
        mol.madjnum[met_mol_idx] += diff
        mol.adjnum[self_mol_idx] += diff
        mol.adjnum[met_mol_idx] += diff

        mol.madjmat[self_mol_idx, met_mol_idx] += diff
        mol.madjmat[met_mol_idx, self_mol_idx] += diff
        mol.adjmat[self_mol_idx, met_mol_idx] += diff
        mol.adjmat[met_mol_idx, self_mol_idx] += diff

        self.set_connectivity(
            mol.adjmat[self_mol_idx],
            mol.madjmat[self_mol_idx],
            mol.adjnum[self_mol_idx],
            mol.madjnum[self_mol_idx],
        )

        met.set_connectivity(
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

    def __str__(self):
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        """Return atom details; omit the border when indirect is True."""
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

    @classmethod
    @deprecated("Use atom() with the keyword arguments instead.")
    def from_positional(
        cls,
        label: str,
        coord: NDArray,
        frac_coord: NDArray | list[float] | None = None,
        radii: float | None = None,
    ) -> "Atom":
        """Construct an atom from positional arguments."""
        return cls(
            label=label,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=radii,
        )
