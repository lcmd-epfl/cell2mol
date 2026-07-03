from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.utils import config
from cell2mol.classes.atom import Atom
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import identify_haptic_mode
from cell2mol.operations import compute_centroid
from cell2mol.elementdata import ElementData
from cell2mol.my_types import OptionalRef, OptionalRefList, SubType, NDArray
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class Group(Specie):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    haptic_type: str | None = None
    is_haptic: bool | None = None
    topology: dict[str, Any] | None = None
    checked_coordination: bool | None = None
    # Cross-reference: points to a metal in parent Molecule
    closest_metal: OptionalRef[Metal] = None
    # Cross-reference: points to metals in parent Molecule
    metals: OptionalRefList[Metal] = None
    denticity: int | None = None

    subtype: SubType | None = Field(default="group")

    @classmethod
    @deprecated("Use group(**kwargs) with keyword arguments")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> "Group":
        return cls(
            labels=labels,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )

    @classmethod
    def from_atom_list(
        cls, gr_atoms: list["Atom"], use_bond_info: bool | None = None
    ) -> "Group":
        """
        Factory method to instantiate a Group directly from a list of Atom objects.
        """
        if not gr_atoms:
            raise ValueError("Attempted to create a Group from an empty atom list.")

        # 1. Extract data from the atom list
        # We assume all atoms in the list have the same attribute availability
        first_atom = gr_atoms[0]

        # 2. Instantiate the class using the base constructor
        # Since Group inherits from Specie, we pass the basic structural data
        instance = cls(
            labels=[a.label for a in gr_atoms],
            coord=np.asarray([a.coord for a in gr_atoms]),
            frac_coord=np.asarray([a.frac_coord for a in gr_atoms])
            if first_atom.frac_coord is not None
            else None,
            radii=np.asarray([a.radii for a in gr_atoms]),
        )

        # 3. Set internal metadata
        instance.origin = "Group.from_atom_list"

        # 4. Initialize internal atom mapping and hapticity
        # This mirrors the logic you had in your standalone function
        instance.set_atoms(
            atomlist=gr_atoms,
            create_adjacencies=False,
            atom_site_labels=cast("list[str]", [a.atom_site_label for a in gr_atoms])
            if first_atom.atom_site_label is not None
            else None,
            use_bond_info=use_bond_info
            if use_bond_info is not None
            else config.USE_BOND_INFO,
        )
        # Adjacency matrices are not built here; they should be built later
        return instance

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        to_print = ""
        to_print += "------------- Cell2mol GROUP Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.metals is not None:
            to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    def remove_atom(self, index: int):
        logger.info(
            "Deleting atom (index=%d) from group with %d atoms", index, self.natoms
        )
        if index > self.natoms:
            return None
        if self.atoms is None:
            self.set_atoms()
        assert self.atoms is not None
        self.atoms.pop(index)
        self.labels.pop(index)
        self.coord = np.delete(self.coord, index, axis=0)
        self.radii = np.delete(np.asarray(self.radii), index)
        # formula, eleccount, and natoms are computed properties on Specie;
        # they update automatically from self.labels and don't need to be set.
        logger.info("Group after removing atom: %s", self)

        if self.natoms > 0:
            if self.closest_metal is not None:
                self.get_closest_metal()
            if self.is_haptic is not None:
                self.get_hapticity()
            if self.centroid is not None:
                self.get_centroid()
            if self.frac_coord is not None:
                self.frac_coord = np.delete(self.frac_coord, index, axis=0)
            if self.adjmat is not None:
                self.build_adjmatrix(metal_only=False)
            if self.madjmat is not None:
                self.build_adjmatrix(metal_only=True)

    def get_closest_metal(self):
        """
        Calculates the centroid of the group and identifies the
        nearest metal atom within the parent molecule.
        """
        mol = cast("Molecule", self.get_parent("molecule"))

        # Safety check for parent molecule and metal availability
        if mol is None or not mol.metals:
            logger.warning(
                f"Group {self.formula} has no parent molecule or metals to reference."
            )
            return

        # Compute the centroid (mean position) of all atoms in this group
        group_centroid = compute_centroid(np.array(self.coord))

        # Find the metal with the minimum Euclidean distance to the centroid
        # Using 'min' with a lambda is more efficient than building a full distance list
        self.closest_metal = min(
            mol.metals,
            key=lambda met: np.linalg.norm(group_centroid - np.array(met.coord)),
        )

        logger.debug(
            f"Closest metal for group {self.formula} identified as {self.closest_metal.label}"
        )

    def get_hapticity(self, use_bond_info: bool | None = None):
        """
        Determine haptic coordination mode(s) for the group.
        """
        if self.atoms is None:
            return None
        logger.debug(f"Determining hapticity for group {self.formula}")
        logger.debug(f"Group atoms: {[atom.label for atom in self.atoms]}")
        is_haptic, haptic_type, topology = identify_haptic_mode(
            self.atoms, use_bond_info
        )

        self.is_haptic = is_haptic
        self.haptic_type = haptic_type
        self.topology = topology

    def get_denticity(self):
        self.denticity = 0
        for a in self.atoms or []:
            self.denticity += a.mconnec or 0
        return self.denticity
