from __future__ import annotations

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.utils import config
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import identify_haptic_mode
from cell2mol.element_utils import labels2electrons, labels2formula
from cell2mol.operations import compute_centroid
from cell2mol.elementdata import ElementData
from cell2mol.my_types import HapticType, OptionalRef, OptionalRefList, SubType
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class Group(Specie):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    checked_coordination: bool | None = None
    # Cross-reference: points to a metal in parent Molecule
    closest_metal: OptionalRef[Metal] = None
    # Cross-reference: points to metals in parent Molecule
    metals: OptionalRefList[Metal] = None
    denticity: int | None = None

    subtype: SubType = Field(default="group")

    @classmethod
    @deprecated("Use group(**kwargs) with keyword arguments")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> None:
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)

    @classmethod
    def from_atom_list(
        cls, gr_atoms: list, use_bond_info: bool | None = None
    ) -> "Group":
        """
        Factory method to instantiate a Group directly from a list of Atom objects.
        """
        if not gr_atoms:
            logger.warning("Attempted to create a Group from an empty atom list.")
            return None

        # 1. Extract data from the atom list
        # We assume all atoms in the list have the same attribute availability
        first_atom = gr_atoms[0]

        # 2. Instantiate the class using the base constructor
        # Since Group inherits from Specie, we pass the basic structural data
        instance = cls(
            labels=[a.label for a in gr_atoms],
            coord=[a.coord for a in gr_atoms],
            frac_coord=[a.frac_coord for a in gr_atoms]
            if first_atom.frac_coord is not None
            else None,
            radii=[a.radii for a in gr_atoms],
        )

        # 3. Set internal metadata
        instance.origin = "Group.from_atom_list"

        # 4. Initialize internal atom mapping and hapticity
        # This mirrors the logic you had in your standalone function
        instance.set_atoms(
            atomlist=gr_atoms,
            create_adjacencies=False,
            atom_site_labels=[a.atom_site_label for a in gr_atoms]
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

    def __repr__(self):
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
        self.atoms.pop(index)
        self.labels.pop(index)
        self.coord.pop(index)
        self.radii.pop(index)
        self.formula = labels2formula(self.labels)
        ### Assuming neutral specie (so basically this is the sum of atomic numbers)
        self.eleccount = labels2electrons(self.labels)
        self.natoms = len(self.labels)
        logger.info("Group after removing atom: %s", self)

        if self.natoms > 0:
            if self.closest_metal is not None:
                self.get_closest_metal()
            if self.is_haptic is not None:
                self.get_hapticity()
            if self.centroid is not None:
                self.get_centroid()
            if self.frac_coord is not None:
                self.frac_coord.pop(index)
            if self.adjmat is not None:
                self.build_adjmatrix(metal_only=False)
            if self.madjmat is not None:
                self.build_adjmatrix(metal_only=True)

    def get_closest_metal(self):
        """
        Calculates the centroid of the group and identifies the
        nearest metal atom within the parent molecule.
        """
        mol = self.get_parent("molecule")

        # Safety check for parent molecule and metal availability
        if mol is None or not getattr(mol, "metals", None):
            logger.warning(
                f"Group {self.label} has no parent molecule or metals to reference."
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
            f"Closest metal for group {self.label} identified as {self.closest_metal.label}"
        )

    def get_hapticity(self, use_bond_info: bool | None = None):
        """
        Determine haptic coordination mode(s) for the group.
        """
        if self.atoms is None:
            return None
        logger.debug(f"Determining hapticity for group {self.formula}")
        logger.debug(f"Group atoms: {[atom.label for atom in self.atoms]}")
        is_haptic, haptic_type = identify_haptic_mode(self.atoms, use_bond_info)

        self.is_haptic = is_haptic
        self.haptic_type = haptic_type

    def get_denticity(self):
        self.denticity = 0
        for a in self.atoms:
            self.denticity += a.mconnec
        return self.denticity
