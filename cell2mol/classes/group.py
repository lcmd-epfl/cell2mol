from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import Self, deprecated
from cell2mol.utils import config
from cell2mol.classes.atom import Atom
from cell2mol.classes.metal import Metal
from cell2mol.classes.atom_set import AtomSet
from cell2mol.operations import compute_centroid
from cell2mol.my_types import OptionalRefList, SubType, NDArray
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)


class Group(AtomSet):
    """A coordinating ligand fragment containing one or more donor atoms."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    haptic_type: str | None = None
    is_haptic: bool | None = None
    topology: dict[str, Any] | None = None
    # Cross-reference: points to metals in parent Molecule
    metals: OptionalRefList[Metal] = None

    subtype: SubType | None = Field(default="group")

    @classmethod
    def from_atom_list(
        cls, gr_atoms: list["Atom"], use_bond_info: bool | None = None
    ) -> Self:
        """
        Factory method to instantiate a Group directly from a list of Atom objects.
        """
        if not gr_atoms:
            raise ValueError("Attempted to create a Group from an empty atom list.")

        # We assume all atoms in the list have the same attribute availability
        first_atom = gr_atoms[0]

        instance = cls(
            symbols=[a.symbol for a in gr_atoms],
            coord=np.asarray([a.coord for a in gr_atoms]),
            frac_coord=np.asarray([a.frac_coord for a in gr_atoms])
            if first_atom.frac_coord is not None
            else None,
            radii=np.asarray([a.radius for a in gr_atoms]),
        )

        instance.origin = "Group.from_atom_list"

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

    def find_closest_metal(self) -> Metal | None:
        """
        Return the nearest metal to the group centroid in the parent molecule.
        """
        mol = cast("Molecule | None", self.get_parent("molecule"))

        # Safety check for parent molecule and metal availability
        if mol is None or not mol.metals:
            logger.warning(
                f"Group {self.formula} has no parent molecule or metals to reference."
            )
            return

        # Compute the centroid (mean position) of all atoms in this group
        group_centroid = compute_centroid(np.array(self.coord))

        # Find the metal with the minimum Euclidean distance to the centroid
        closest_metal = min(
            mol.metals,
            key=lambda met: float(np.linalg.norm(group_centroid - np.array(met.coord))),
        )
        logger.debug(
            f"Closest metal for group {self.formula} identified as {closest_metal.symbol}"
        )
        return closest_metal

    def update_hapticity(self, use_bond_info: bool | None = None) -> None:
        """
        Determine haptic coordination mode(s) for the group.
        """
        from cell2mol.connectivity import identify_haptic_mode

        if self.atoms is None:
            return None
        logger.debug(f"Determining hapticity for group {self.formula}")
        logger.debug(f"Group atoms: {[atom.symbol for atom in self.atoms]}")
        is_haptic, haptic_type, topology = identify_haptic_mode(
            self.atoms, use_bond_info
        )

        self.is_haptic = is_haptic
        self.haptic_type = haptic_type
        self.topology = dict(topology)

    def __str__(self):
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        to_print = ""
        to_print += "------------- Cell2mol GROUP Object --------------\n"
        to_print += AtomSet.__repr__(self, indirect=True)
        if self.metals is not None:
            to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use group(**kwargs) with keyword arguments")
    def from_positional(
        cls,
        symbols: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> Self:
        return cls(
            symbols=symbols,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )
