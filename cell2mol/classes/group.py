from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.utils import config
from cell2mol.classes.atom import Atom
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.operations import compute_centroid
from cell2mol.my_types import OptionalRef, OptionalRefList, SubType, NDArray
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)


class Group(Specie):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    haptic_type: str | None = None
    is_haptic: bool | None = None
    topology: dict[str, Any] | None = None
    # Cross-reference: points to a metal in parent Molecule
    closest_metal: OptionalRef[Metal] = None
    # Cross-reference: points to metals in parent Molecule
    metals: OptionalRefList[Metal] = None
    denticity: int | None = None

    subtype: SubType | None = Field(default="group")

    @classmethod
    def from_atom_list(
        cls, gr_atoms: list["Atom"], use_bond_info: bool | None = None
    ) -> "Group":
        """
        Factory method to instantiate a Group directly from a list of Atom objects.
        """
        if not gr_atoms:
            raise ValueError("Attempted to create a Group from an empty atom list.")

        # We assume all atoms in the list have the same attribute availability
        first_atom = gr_atoms[0]

        instance = cls(
            labels=[a.label for a in gr_atoms],
            coord=np.asarray([a.coord for a in gr_atoms]),
            frac_coord=np.asarray([a.frac_coord for a in gr_atoms])
            if first_atom.frac_coord is not None
            else None,
            radii=np.asarray([a.radii for a in gr_atoms]),
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

    def remove_atom(self, index: int) -> None:
        """Remove a local atom and update group data without changing shared atoms."""
        if not 0 <= index < self.natoms:
            raise IndexError(
                f"Group atom index {index} out of range for {self.natoms} atoms"
            )
        if self.atoms is None:
            self.set_atoms()
        assert self.atoms is not None

        update_centroid = self.centroid is not None or self.frac_centroid is not None
        update_hapticity = self.is_haptic is not None
        update_denticity = self.denticity is not None
        update_closest_metal = self.closest_metal is not None
        removed_atom = self.atoms.pop(index)
        self._unlink_atom(removed_atom, index)

        self.labels.pop(index)
        self.coord = np.delete(self.coord, index, axis=0)
        if self.radii is not None:
            self.radii = np.delete(self.radii, index)
        if self.frac_coord is not None:
            self.frac_coord = np.delete(self.frac_coord, index, axis=0)
        if self.atom_site_labels is not None:
            self.atom_site_labels.pop(index)
        if self.atnums is not None:
            self.atnums.pop(index)
        # Keep existing connectivity; deleting an atom must not perceive new bonds.
        for matrix_name, count_name in (("adjmat", "adjnum"), ("madjmat", "madjnum")):
            matrix = getattr(self, matrix_name)
            if matrix is not None:
                matrix = np.delete(np.delete(matrix, index, axis=0), index, axis=1)
                setattr(self, matrix_name, matrix)
                setattr(self, count_name, matrix.sum(axis=1))
            else:
                setattr(self, count_name, None)

        self._clear_structure_results()
        if self.natoms:
            if update_centroid:
                self.compute_centroid()
            if update_closest_metal:
                self.update_closest_metal()
            if update_hapticity:
                self.update_hapticity()
            if update_denticity:
                self.update_denticity()

    def _unlink_atom(self, removed_atom: Atom, index: int) -> None:
        """Remove this group's link and update local and parent atom maps."""
        assert self.atoms is not None
        # Only unlink this group: molecule/ligand ownership and bonds stay intact.
        for position in range(len(removed_atom.parents) - 1, -1, -1):
            if removed_atom.parents[position] is self:
                removed_atom.parents.pop(position)
                removed_atom.parents_index.pop(position)
        for position, atom in enumerate(self.atoms):
            atom.add_parent(self, position)
        for indices in self.parents_indices:
            indices.pop(index)

    def _clear_structure_results(self) -> None:
        """Invalidate group results without changing shared atom charges."""
        # Invalidate group results without resetting charges on shared atoms.
        for name in (
            "totcharge",
            "atomic_charges",
            "charge_state",
            "plausible_charge_states",
            "rdkit_obj",
            "smiles",
            "protonation_states",
            "element_count",
            "has_porphyrin",
            "has_fullerene",
            "has_borane",
            "adj_types",
            "centroid",
            "frac_centroid",
            "closest_metal",
            "is_haptic",
            "haptic_type",
            "topology",
            "denticity",
        ):
            setattr(self, name, None)

    def update_closest_metal(self):
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
        self.closest_metal = min(
            mol.metals,
            key=lambda met: np.linalg.norm(group_centroid - np.array(met.coord)),
        )

        logger.debug(
            f"Closest metal for group {self.formula} identified as {self.closest_metal.label}"
        )

    def update_hapticity(self, use_bond_info: bool | None = None):
        """
        Determine haptic coordination mode(s) for the group.
        """
        from cell2mol.connectivity import identify_haptic_mode

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

    def update_denticity(self):
        self.denticity = 0
        for a in self.atoms or []:
            self.denticity += a.mconnec or 0
        return self.denticity

    def __str__(self):
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        to_print = ""
        to_print += "------------- Cell2mol GROUP Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.metals is not None:
            to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

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
