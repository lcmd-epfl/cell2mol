from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field
from typing_extensions import deprecated

from cell2mol.classes.atom import Atom
from cell2mol.my_types import Type
from cell2mol.utils import BaseModel


class Bond(BaseModel):
    """Bond between shared atom objects, with order and Cartesian distance."""

    atom1: Atom
    atom2: Atom
    order: float = Field(default=1, alias="bond_order")
    distance: float | None = None
    type: Type = Field(default="bond")

    def model_post_init(self, __context: Any) -> None:
        """Compute the bond distance, rounded to three decimal places."""
        self.distance = round(
            float(
                np.linalg.norm(np.array(self.atom1.coord) - np.array(self.atom2.coord))
            ),
            3,
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """Return atom labels, molecule indices, bond order, and distance."""
        to_print = ""
        to_print += "------------- Cell2mol BOND Object --------------\n"
        to_print += f" Type                     = {self.type}\n"
        idx1 = self.atom1.get_parent_index("molecule")
        idx2 = self.atom2.get_parent_index("molecule")
        to_print += f" Molecule Atom 1 label    = {self.atom1.label}\n"
        to_print += f" Molecule Atom 2 label    = {self.atom2.label}\n"
        to_print += f" Molecule Atom 1 index    = {idx1}\n"
        to_print += f" Molecule Atom 2 index    = {idx2}\n"
        to_print += f" Bond Order               = {self.order}\n"
        to_print += f" Distance                 = {self.distance}\n"
        to_print += "----------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use Bond() with keyword arguments instead.")
    def from_positional(cls, atom1: Atom, atom2: Atom, bond_order: float = 1) -> "Bond":
        """Construct a bond using the legacy positional interface."""
        return cls(atom1=atom1, atom2=atom2, bond_order=bond_order)
