from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np
from pydantic import AliasChoices, Field, model_validator
from typing_extensions import TypedDict, Self, deprecated

from cell2mol.classes.atom import Atom
from cell2mol.classes.metal import Metal
from cell2mol.element_utils import (
    get_df_block_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    get_element_count,
    get_radii,
    symbols2electrons,
    symbols2formula,
)
from cell2mol.connectivity import build_adjacency, count_bonds_by_element_pair
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray, SubType
from cell2mol.operations import compute_centroid, extract_from_list
from cell2mol.utils import BaseModel, config
import logging

from cell2mol.utils.parent_links import migrate_parent_links, migrate_parent_pickle

if TYPE_CHECKING:
    from cell2mol.classes.cell import Cell

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class AtomSetParentLink(TypedDict):
    """A parent reference paired with its atom index mapping."""

    parent: "AtomSet | Cell"
    indices: list[int]


class AtomSet(BaseModel):
    """Ordered shared atoms, geometry, connectivity and parent index maps."""

    symbols: list[str] = Field(validation_alias=AliasChoices("symbols", "labels"))
    coord: NDArray
    frac_coord: NDArray | None = None
    radii: NDArray | None = None
    atomic_numbers: list[int] | None = Field(
        default=None, validation_alias=AliasChoices("atomic_numbers", "atnums")
    )
    atom_site_labels: list[str] | None = None
    atoms: list[Atom] | None = None
    parent_links: list[AtomSetParentLink] = Field(default_factory=list)

    cov_factor: float = Field(default=config.COV_FACTOR)
    metal_factor: float = Field(default=config.METAL_FACTOR)
    adjacency_matrix: NDArray | None = Field(
        default=None, validation_alias=AliasChoices("adjacency_matrix", "adjmat")
    )
    neighbor_counts: NDArray | None = Field(
        default=None, validation_alias=AliasChoices("neighbor_counts", "adjnum")
    )
    # Atom-pair contacts with at least one metal endpoint; includes metal-metal links.
    metal_contact_matrix: NDArray | None = Field(
        default=None, validation_alias=AliasChoices("metal_contact_matrix", "madjmat")
    )
    metal_contact_counts: NDArray | None = Field(
        default=None, validation_alias=AliasChoices("metal_contact_counts", "madjnum")
    )
    subtype: SubType | None = None
    origin: str | None = None
    type: str = Field(default="atomset", frozen=True)

    @property
    def madjmat(self) -> NDArray | None:
        """Compatibility alias for metal_contact_matrix."""
        return self.metal_contact_matrix

    @madjmat.setter
    def madjmat(self, value: NDArray | None) -> None:
        self.metal_contact_matrix = value

    @property
    def madjnum(self) -> NDArray | None:
        """Compatibility alias for metal_contact_counts."""
        return self.metal_contact_counts

    @madjnum.setter
    def madjnum(self, value: NDArray | None) -> None:
        self.metal_contact_counts = value

    @property
    def labels(self) -> list[str]:
        """Compatibility alias for symbols."""
        return self.symbols

    @labels.setter
    def labels(self, value: list[str]) -> None:
        self.symbols = value

    @property
    def atnums(self) -> list[int] | None:
        """Compatibility alias for atomic_numbers."""
        return self.atomic_numbers

    @atnums.setter
    def atnums(self, value: list[int] | None) -> None:
        self.atomic_numbers = value

    @property
    def parents(self) -> list["AtomSet | Cell"]:
        """Return parent references; use add_parent() to update relationships."""
        return [link["parent"] for link in self.parent_links]

    @property
    def atom_indices_in_parents(self) -> list[list[int]]:
        """Return parent index mappings as a compatibility view."""
        return [link["indices"].copy() for link in self.parent_links]

    @property
    def parents_indices(self) -> list[list[int]]:
        """Compatibility alias for atom_indices_in_parents."""
        return self.atom_indices_in_parents

    @property
    def adjmat(self) -> NDArray | None:
        """Compatibility alias for adjacency_matrix."""
        return self.adjacency_matrix

    @adjmat.setter
    def adjmat(self, value: NDArray | None) -> None:
        self.adjacency_matrix = value

    @property
    def adjnum(self) -> NDArray | None:
        """Compatibility alias for neighbor_counts."""
        return self.neighbor_counts

    @adjnum.setter
    def adjnum(self, value: NDArray | None) -> None:
        self.neighbor_counts = value

    @model_validator(mode="before")
    @classmethod
    def _migrate_serialized_data(cls, data: Any) -> Any:
        """Merge legacy parent references and index lists on every load path."""
        return migrate_parent_links(
            data, "atom_indices_in_parents", "parents_indices", "indices"
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Migrate legacy metal-contact fields in pickle state."""
        names = {"madjmat": "metal_contact_matrix", "madjnum": "metal_contact_counts"}
        names.update(
            {
                "labels": "symbols",
                "atnums": "atomic_numbers",
                "adjmat": "adjacency_matrix",
                "adjnum": "neighbor_counts",
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
        migrate_parent_pickle(
            state, "atom_indices_in_parents", "parents_indices", "indices"
        )
        super().__setstate__(state)

    def model_post_init(self, context: Any) -> None:
        """Validate coordinate lengths and initialize missing radii."""
        super().model_post_init(context)
        assert len(self.symbols) == len(self.coord)
        if self.frac_coord is not None:
            assert len(self.coord) == len(self.frac_coord)
        if self.radii is not None:
            # Legacy data may contain list or scalar radii.
            if isinstance(self.radii, (list, tuple)):
                assert len(self.symbols) == len(self.radii)
            elif len(self.symbols) > 1:
                self.radii = np.asarray(get_radii(self.symbols))
        else:
            self.radii = np.asarray(get_radii(self.symbols))

    @property
    def formula(self) -> str:
        """Return the formula built from element labels."""
        return symbols2formula(self.symbols)

    @property
    def eleccount(self) -> int:
        """Return the electron count for a neutral structure."""
        return symbols2electrons(self.symbols)

    @property
    def natoms(self) -> int:
        """Return the number of atoms."""
        return len(self.symbols)

    @property
    def is_metal_complex(self) -> bool:
        """Return whether d- or f-block metals are present."""
        return bool(get_df_block_metal_idxs(self.symbols))

    @property
    def has_s_block_metal(self) -> bool:
        """Return whether alkali or alkaline-earth metals are present."""
        return bool(get_alkali_alkaline_earth_metal_idxs(self.symbols))

    @property
    def is_post_transition_metal_complex(self) -> bool:
        """Detect post-transition metals only when no d/f- or s-block metals are present."""
        return (
            not self.is_metal_complex
            and not self.has_s_block_metal
            and bool(get_post_transition_metal_idxs(self.symbols))
        )

    @property
    def is_metal_free_molecule(self):
        """Return whether this is a molecule without any supported metal category."""
        return (
            self.subtype == "molecule"
            and not self.is_metal_complex
            and not self.has_s_block_metal
            and not self.is_post_transition_metal_complex
        )

    @property
    def indices(self) -> list[int]:
        """Return local atom indices in structure order."""
        return list(range(self.natoms))

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

    def get_parent_indices(self, subtype: str):
        """Return atom indices in the requested parent, or None."""
        for link in self.parent_links:
            if link["parent"].subtype == subtype:
                return link["indices"].copy()
        return None

    def get_adjacency_parameters(self) -> tuple[float, float]:
        """Return covalent-radius and metal-radius scaling factors."""
        return self.cov_factor, self.metal_factor

    def set_subtype(self, subtype: SubType):
        """Store the species subtype."""
        self.subtype = subtype

    def set_origin(self, origin: str):
        """Store the structure creation source."""
        self.origin = origin

    def _register_parent(
        self, parent: "AtomSet | Cell", indices: list[int], overwrite: bool
    ) -> None:
        """Store one parent and its atom map together."""
        new_link: AtomSetParentLink = {"parent": parent, "indices": indices.copy()}
        for position, link in enumerate(self.parent_links):
            if link["parent"].subtype == parent.subtype:
                if overwrite:
                    self.parent_links[position] = new_link
                return
        self.parent_links.append(new_link)

    def add_parent(
        self, parent: "AtomSet | Cell", indices: list[int], overwrite: bool = True
    ) -> None:
        """Register a parent and compose atom maps for its ancestors."""
        if len(indices) != self.natoms:
            raise ValueError("Parent index map must contain one index per atom")
        if any(not isinstance(i, (int, np.integer)) or i < 0 for i in indices):
            raise ValueError("Parent indices must be non-negative integers")
        if isinstance(parent, AtomSet) and any(i >= parent.natoms for i in indices):
            raise ValueError("Parent index out of range")
        if not overwrite and self.check_parent(parent.subtype or ""):
            return

        ancestors = []
        if isinstance(parent, AtomSet):
            for link in parent.parent_links:
                ancestor = link["parent"]
                parent_indices = link["indices"]
                if len(parent_indices) != parent.natoms:
                    raise ValueError(
                        "Ancestor index map must contain one index per parent atom"
                    )
                ancestors.append((ancestor, [parent_indices[i] for i in indices]))
        self._register_parent(parent, indices, overwrite)
        for ancestor, own_indices in ancestors:
            self._register_parent(ancestor, own_indices, overwrite)

    def compute_centroid(self) -> NDArray:
        """Return the Cartesian centroid without storing derived coordinates."""
        return np.asarray(compute_centroid(np.array(self.coord)))

    def set_fractional_coordinates(
        self, frac_coord: NDArray | list[list[float]]
    ) -> None:
        """Store fractional coordinates with one row per atom."""
        assert len(frac_coord) == len(self.coord)
        self.frac_coord = np.asarray(frac_coord)

    def get_atomic_numbers(self):
        """Build atom objects if needed, then return their atomic numbers."""
        if self.atoms is None:
            self.set_atoms()
        self.atomic_numbers = []
        for at in self.atoms or []:
            self.atomic_numbers.append(at.atomic_number)
        return self.atomic_numbers

    def compute_element_count(self, heavy_only: bool = False) -> NDArray:
        """Return element counts without storing them, optionally excluding hydrogen."""
        return get_element_count(self.symbols, heavy_only=heavy_only)

    def build_adjacency_matrix(
        self,
        use_bond_info: bool | None = None,
        metal_only: bool = False,
    ):
        """Build and store adjacency and neighbor counts, optionally for metal links only."""
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = self.cov_factor
        metal_factor = self.metal_factor

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        adjmat = build_adjacency(
            labels=self.symbols,
            positions=self.coord,
            atom_site_labels=self.atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            metal_only=metal_only,
            warn_on_mismatch=False,
            detail=False,
        )

        if adjmat is None:
            if metal_only:
                self.metal_contact_matrix = None
                self.metal_contact_counts = None
                return self.metal_contact_matrix, self.metal_contact_counts
            else:
                self.adjacency_matrix = None
                self.neighbor_counts = None
                return self.adjacency_matrix, self.neighbor_counts

        adjnum = adjmat.sum(axis=1)

        if metal_only:
            self.metal_contact_matrix = adjmat
            self.metal_contact_counts = adjnum
            return self.metal_contact_matrix, self.metal_contact_counts
        else:
            self.adjacency_matrix = adjmat
            self.neighbor_counts = adjnum
            return self.adjacency_matrix, self.neighbor_counts

    def compute_adjacency_types(
        self, use_bond_info: bool | None = None
    ) -> NDArray | None:
        """Build adjacency if needed, then return element-pair bond counts."""
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix(use_bond_info=use_bond_info, metal_only=False)
        if self.adjacency_matrix is not None:
            return count_bonds_by_element_pair(self.symbols, self.adjacency_matrix)
        return None

    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        """Store covalent-radius and metal-radius scaling factors."""
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    def set_atoms(
        self,
        atomlist: list[Atom] | None = None,
        create_adjacencies: bool = False,
        atom_site_labels: list[str] | None = None,
        use_bond_info: bool | None = None,
    ):
        """Use supplied atoms or create them, link parents, and optionally set connectivity."""
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO
        if atomlist is not None:
            self.atoms = atomlist.copy()
            for idx, at in enumerate(self.atoms):
                at.add_parent(self, index=idx)
        else:
            self.atoms = []

            df_block_metal_idxs = get_df_block_metal_idxs(self.symbols)
            s_block_metal_idxs = get_alkali_alkaline_earth_metal_idxs(self.symbols)

            assert self.radii is not None
            for idx, label in enumerate(self.symbols):
                ismetal = (
                    elemdatabase.elementblock[label] == "d"
                    or elemdatabase.elementblock[label] == "f"
                )
                if len(get_alkali_alkaline_earth_metal_idxs([label])) > 0:
                    ismetal = True
                if len(df_block_metal_idxs) == 0 and len(s_block_metal_idxs) == 0:
                    if len(get_post_transition_metal_idxs([label])) > 0:
                        ismetal = True

                if self.frac_coord is not None:
                    if ismetal:
                        newatom = Metal.from_positional(
                            label,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radius=self.radii[idx],
                        )
                    else:
                        newatom = Atom.from_positional(
                            label,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radius=self.radii[idx],
                        )
                else:
                    if ismetal:
                        newatom = Metal.from_positional(
                            label, self.coord[idx], radius=self.radii[idx]
                        )
                    else:
                        newatom = Atom.from_positional(
                            label, self.coord[idx], radius=self.radii[idx]
                        )
                newatom.add_parent(self, index=idx)
                self.atoms.append(newatom)

        if atom_site_labels is not None:
            self.atom_site_labels = atom_site_labels.copy()
            for idx, at in enumerate(self.atoms):
                at.set_atom_site_label(atom_site_labels[idx])

        if create_adjacencies:
            if self.adjacency_matrix is None:
                self.build_adjacency_matrix(
                    use_bond_info=use_bond_info, metal_only=False
                )

            if self.metal_contact_matrix is None:
                self.build_adjacency_matrix(
                    use_bond_info=use_bond_info, metal_only=True
                )

            if (
                self.adjacency_matrix is not None
                and self.metal_contact_matrix is not None
            ):
                assert self.neighbor_counts is not None
                assert self.metal_contact_counts is not None
                for idx, at in enumerate(self.atoms):
                    at.set_connectivity(
                        self.adjacency_matrix[idx],
                        self.metal_contact_matrix[idx],
                        self.neighbor_counts[idx],
                        self.metal_contact_counts[idx],
                    )

    def inherit_adjacency(self, parent_subtype: str):
        """Copy adjacency submatrices and parent neighbor counts using the atom map."""
        exists = self.check_parent(parent_subtype)
        if not exists:
            logger.debug(f"{parent_subtype=} does not exist")
            return None
        parent = cast("AtomSet", self.get_parent(parent_subtype))
        indices = self.get_parent_indices(parent_subtype)
        assert parent is not None
        assert indices is not None
        if parent.metal_contact_counts is None:
            logger.debug(f"{parent_subtype=} does not have metal_contact_counts")
            return None
        assert parent.metal_contact_matrix is not None
        assert parent.adjacency_matrix is not None
        assert parent.neighbor_counts is not None

        # Convert generic object arrays to integer connectivity arrays.
        self.metal_contact_matrix = np.stack(
            extract_from_list(
                indices, parent.metal_contact_matrix.tolist(), dimension=2
            ),
            axis=0,
        ).astype(int)
        self.metal_contact_counts = np.stack(
            extract_from_list(
                indices, parent.metal_contact_counts.tolist(), dimension=1
            ),
            axis=0,
        ).astype(int)
        self.adjacency_matrix = np.stack(
            extract_from_list(indices, parent.adjacency_matrix.tolist(), dimension=2),
            axis=0,
        ).astype(int)
        self.neighbor_counts = np.stack(
            extract_from_list(indices, parent.neighbor_counts.tolist(), dimension=1),
            axis=0,
        ).astype(int)

    def __str__(self):
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        """Return structure, connectivity, and origin details."""
        to_print = ""
        if not indirect:
            to_print += "------------- Cell2mol ATOMSET Object --------------\n"
        to_print += f" Type                         = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type                     = {self.subtype}\n"
        to_print += f" Number of Atoms              = {self.natoms}\n"
        to_print += f" Formula                      = {self.formula}\n"
        to_print += f" Covalent Radii Factor        = {self.cov_factor}\n"
        to_print += f" Metal Radii Factor           = {self.metal_factor}\n"
        if self.adjacency_matrix is not None:
            to_print += " Has Adjacency Matrix         = YES\n"
        else:
            to_print += " Has Adjacency Matrix         = NO \n"
        if self.origin is not None:
            to_print += f" Origin                       = {self.origin}\n"
        if not indirect:
            to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use the class constructor with keyword arguments instead.")
    def from_positional(
        cls,
        symbols: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> Self:
        """Construct a species using the legacy positional interface."""
        return cls(
            symbols=symbols,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )
