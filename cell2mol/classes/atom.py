from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pydantic import AliasChoices, Field, model_validator, computed_field
from typing_extensions import TypedDict, deprecated
from cell2mol.element_utils import get_radii
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray, OptionalInt, SubType
from cell2mol.utils import BaseModel

from cell2mol.utils.parent_links import migrate_parent_links, migrate_parent_pickle

if TYPE_CHECKING:
    from cell2mol.classes.bond import BondRecord
    from cell2mol.classes.atom_set import AtomSet
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.cell import Cell
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class AtomParentLink(TypedDict):
    """A parent reference paired with its atom index mapping."""

    parent: "AtomSet | Cell"
    index: int


class Atom(BaseModel):
    """Atom coordinates, connectivity, charge, and parent references."""

    symbol: str = Field(validation_alias=AliasChoices("symbol", "label"))
    coord: NDArray
    frac_coord: NDArray | None = None
    radius: float | None = Field(
        default=None, validation_alias=AliasChoices("radius", "radii")
    )

    # Parent objects and the atom index within each parent.
    parent_links: list[AtomParentLink] = Field(default_factory=list)

    atom_site_label: str | None = None
    neighbor_count: int | None = Field(
        default=None, validation_alias=AliasChoices("neighbor_count", "connec")
    )
    # Contacts involving a metal: donor-metal and metal-metal, not eta site counts.
    metal_contact_count: int | None = Field(
        default=None, validation_alias=AliasChoices("metal_contact_count", "mconnec")
    )
    neighbor_indices: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("neighbor_indices", "adjacency"),
    )
    metal_contact_indices: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("metal_contact_indices", "metal_adjacency"),
    )
    charge: OptionalInt = None

    type: str = Field(default="atom", frozen=True)
    subtype: SubType = Field(default="atom")

    @property
    def mconnec(self) -> int | None:
        """Compatibility alias for metal_contact_count."""
        return self.metal_contact_count

    @mconnec.setter
    def mconnec(self, value: int | None) -> None:
        self.metal_contact_count = value

    @property
    def metal_adjacency(self) -> list[int]:
        """Compatibility alias for metal_contact_indices."""
        return self.metal_contact_indices

    @metal_adjacency.setter
    def metal_adjacency(self, value: list[int]) -> None:
        self.metal_contact_indices = value

    @property
    def label(self) -> str:
        """Compatibility alias for symbol."""
        return self.symbol

    @label.setter
    def label(self, value: str) -> None:
        self.symbol = value

    @property
    def radii(self) -> float | None:
        """Compatibility alias for radius."""
        return self.radius

    @radii.setter
    def radii(self, value: float | None) -> None:
        self.radius = value

    @property
    def parents(self) -> list["AtomSet | Cell"]:
        """Return parent references; use add_parent() to update relationships."""
        return [link["parent"] for link in self.parent_links]

    @property
    def index_in_parents(self) -> list[int]:
        """Return parent index mappings as a compatibility view."""
        return [link["index"] for link in self.parent_links]

    @property
    def parents_index(self) -> list[int]:
        """Compatibility alias for index_in_parents."""
        return self.index_in_parents

    @property
    def connec(self) -> int | None:
        """Compatibility alias for neighbor_count."""
        return self.neighbor_count

    @connec.setter
    def connec(self, value: int | None) -> None:
        self.neighbor_count = value

    @property
    def adjacency(self) -> list[int]:
        """Compatibility alias for neighbor_indices."""
        return self.neighbor_indices

    @adjacency.setter
    def adjacency(self, value: list[int]) -> None:
        self.neighbor_indices = value

    @property
    def atnum(self) -> int:
        """Compatibility alias for atomic_number."""
        return self.atomic_number

    @model_validator(mode="before")
    @classmethod
    def _migrate_serialized_data(cls, data: Any) -> Any:
        """Merge legacy parent references and index lists on every load path."""
        return migrate_parent_links(data, "index_in_parents", "parents_index", "index")

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Migrate legacy metal-contact fields in pickle state."""
        names = {
            "mconnec": "metal_contact_count",
            "metal_adjacency": "metal_contact_indices",
        }
        names.update(
            {
                "label": "symbol",
                "radii": "radius",
                "connec": "neighbor_count",
                "adjacency": "neighbor_indices",
            }
        )
        data = state.get("__dict__", {})
        fields = state.get("__pydantic_fields_set__", set())
        for old, new in names.items():
            if old in data:
                data.setdefault(new, data.pop(old))
            if old in fields:
                fields.discard(old)
                fields.add(new)
        migrate_parent_pickle(state, "index_in_parents", "parents_index", "index")
        super().__setstate__(state)

    def model_post_init(self, __context: Any) -> None:
        """Fill a missing radius from the element table."""
        if self.radius is None:
            radii_array = get_radii([self.symbol])
            self.radius = float(radii_array[0])

    @computed_field
    @property
    def atomic_number(self) -> int:
        """Return the atomic number."""
        return elemdatabase.elementnr[self.symbol]

    @computed_field
    @property
    def block(self) -> str:
        """Return the periodic-table block."""
        return elemdatabase.elementblock[self.symbol]

    @computed_field
    @property
    def formula(self) -> str:
        """Return the element symbol as the atom formula."""
        return self.symbol

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
        for link in self.parent_links:
            if link["parent"].subtype == subtype:
                return link["index"]
        return None

    @property
    def closest_metal(self):
        """Return the nearest metal in the parent molecule."""
        return self.find_closest_metal()

    def add_parent(self, parent: "AtomSet | Cell", index: int, overwrite: bool = True):
        """Register a parent and this atom's index within it.

        Keep one parent per subtype; replace it when overwrite=True.
        """
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("Parent index must be a non-negative integer")
        parent_natoms = parent.natoms
        if parent_natoms is None:
            parent_natoms = len(parent.coord)
        if index >= parent_natoms:
            raise ValueError("Parent index out of range")
        new_link: AtomParentLink = {"parent": parent, "index": int(index)}
        for position, link in enumerate(self.parent_links):
            if link["parent"].subtype == parent.subtype:
                if overwrite:
                    self.parent_links[position] = new_link
                return
        self.parent_links.append(new_link)

    @property
    def bonds(self) -> list[BondRecord]:
        """Query the parent species; indices belong to that species.

        Prefer the molecule when available. Use Species.get_atom_bonds() for
        an explicit ligand-local view. No bond records are stored on atoms.
        """
        from cell2mol.classes.species import Species

        links = sorted(
            self.parent_links, key=lambda link: link["parent"].subtype != "molecule"
        )
        for link in links:
            parent = link["parent"]
            if isinstance(parent, Species):
                return parent.get_atom_bonds(link["index"])
        return []

    def reset_charge(self) -> None:
        """Clear the assigned charge."""
        self.charge = None

    def set_charge(self, charge: int) -> None:
        """Store an integer charge."""
        self.charge = int(charge)

    def set_connectivity(
        self,
        adjacency_matrix,
        metal_contact_matrix,
        neighbor_counts: int,
        metal_contact_counts: int,
    ):
        """Store connectivity counts and neighbors from parent adjacency rows."""
        self.neighbor_count = int(neighbor_counts)
        self.metal_contact_count = int(metal_contact_counts)
        self.neighbor_indices = []
        self.metal_contact_indices = []

        # Adjacency rows use indices in the parent molecule.
        for idx, c in enumerate(adjacency_matrix):
            if c > 0:
                self.neighbor_indices.append(idx)
        for idx, c in enumerate(metal_contact_matrix):
            if c > 0:
                self.metal_contact_indices.append(idx)

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
        self_info = f"{self.symbol}{f' ({self.atom_site_label})' if self.atom_site_label else ''}"
        met_info = (
            f"{met.label}{f' ({met.atom_site_label})' if met.atom_site_label else ''}"
        )
        if not self.check_parent("molecule"):
            return
        self_mol_idx = self.get_parent_index("molecule")
        met_mol_idx = met.get_parent_index("molecule")
        mol = cast("AtomSet", self.get_parent("molecule"))
        assert self_mol_idx is not None
        assert met_mol_idx is not None
        assert mol is not None
        assert mol.atoms is not None
        assert mol.adjacency_matrix is not None
        assert mol.metal_contact_matrix is not None
        assert mol.neighbor_counts is not None
        assert mol.metal_contact_counts is not None
        logger.info(
            f"Reset metal_contact_count: atom={self_info} diff={diff} to metal={met_info}"
        )

        logger.debug(
            "Initial atom connectivity: mol_idx=%s connec=%d metal_contact_count=%d adj=%s madj=%s",
            self_mol_idx,
            self.neighbor_count,
            self.metal_contact_count,
            self.neighbor_indices,
            self.metal_contact_indices,
        )

        logger.debug(
            "Initial metal connectivity: mol_idx=%s connec=%d metal_contact_count=%d adj=%s madj=%s",
            met_mol_idx,
            met.connec,
            met.metal_contact_count,
            met.adjacency,
            met.metal_contact_indices,
        )

        logger.info(
            "Updating molecule: atom_idx=%d metal_idx=%d", self_mol_idx, met_mol_idx
        )

        logger.debug(
            "Molecule before: atom(connec=%d,metal_contact_count=%d) metal(connec=%d,metal_contact_count=%d)",
            mol.atoms[self_mol_idx].connec,
            mol.atoms[self_mol_idx].metal_contact_count,
            met.connec,
            met.metal_contact_count,
        )

        # Update both endpoints in the parent molecule.
        mol.metal_contact_counts[self_mol_idx] += diff
        mol.metal_contact_counts[met_mol_idx] += diff
        mol.neighbor_counts[self_mol_idx] += diff
        mol.neighbor_counts[met_mol_idx] += diff

        mol.metal_contact_matrix[self_mol_idx, met_mol_idx] += diff
        mol.metal_contact_matrix[met_mol_idx, self_mol_idx] += diff
        mol.adjacency_matrix[self_mol_idx, met_mol_idx] += diff
        mol.adjacency_matrix[met_mol_idx, self_mol_idx] += diff

        self.set_connectivity(
            mol.adjacency_matrix[self_mol_idx],
            mol.metal_contact_matrix[self_mol_idx],
            mol.neighbor_counts[self_mol_idx],
            mol.metal_contact_counts[self_mol_idx],
        )

        met.set_connectivity(
            mol.adjacency_matrix[met_mol_idx],
            mol.metal_contact_matrix[met_mol_idx],
            mol.neighbor_counts[met_mol_idx],
            mol.metal_contact_counts[met_mol_idx],
        )

        logger.info(
            "Molecule after: atom(connec=%d,metal_contact_count=%d adj=%s madj=%s)",
            mol.atoms[self_mol_idx].connec,
            mol.atoms[self_mol_idx].metal_contact_count,
            mol.atoms[self_mol_idx].adjacency,
            mol.atoms[self_mol_idx].metal_contact_indices,
        )

        logger.debug(
            "Metal after: connec=%d metal_contact_count=%d adj=%s madj=%s",
            mol.atoms[met_mol_idx].connec,
            mol.atoms[met_mol_idx].metal_contact_count,
            mol.atoms[met_mol_idx].adjacency,
            mol.atoms[met_mol_idx].metal_contact_indices,
        )

        logger.info(
            f"Final: atom={self_info} diff={diff} to metal={met_info}",
        )

        logger.debug(
            "Final atom connectivity: mol_idx=%s connec=%d metal_contact_count=%d adj=%s madj=%s",
            self_mol_idx,
            self.neighbor_count,
            self.metal_contact_count,
            self.neighbor_indices,
            self.metal_contact_indices,
        )

        logger.debug(
            "Final metal connectivity: mol_idx=%s connec=%d metal_contact_count=%d adj=%s madj=%s",
            met_mol_idx,
            met.connec,
            met.metal_contact_count,
            met.adjacency,
            met.metal_contact_indices,
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
        to_print += f" Label                        = {self.symbol}\n"
        if self.atom_site_label is not None:
            to_print += f" Atom Site Label              = {self.atom_site_label}\n"
        to_print += f" Atomic Number                = {self.atomic_number}\n"
        idx = self.get_parent_index("molecule")
        if idx is not None:
            to_print += f" Index in Molecule            = {idx}\n"
        idx = self.get_parent_index("ligand")
        if idx is not None:
            to_print += f" Index in Ligand              = {idx}\n"
        if self.metal_contact_count is not None:
            to_print += f" Metal Adjacency (metal_contact_count)    = {self.metal_contact_count}\n"
        if self.neighbor_count is not None:
            to_print += f" Regular Adjacencies (connec) = {self.neighbor_count}\n"
        if self.charge is not None:
            to_print += f" Atom Charge                  = {self.charge}\n"
        if not indirect:
            to_print += "----------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use atom() with the keyword arguments instead.")
    def from_positional(
        cls,
        symbol: str,
        coord: NDArray,
        frac_coord: NDArray | list[float] | None = None,
        radius: float | None = None,
    ) -> "Atom":
        """Construct an atom from positional arguments."""
        return cls(
            symbol=symbol,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radius=radius,
        )
