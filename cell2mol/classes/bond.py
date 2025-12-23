from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field
from typing_extensions import deprecated

from cell2mol.classes.atom import Atom
from cell2mol.elementdata import ElementData
from cell2mol.my_types import Type
from cell2mol.utils import BaseModel
from cell2mol.utils.ref import Ref

elemdatabase = ElementData()


###############
### BOND ######
###############
class Bond(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Internal storage for atom refs (alias ensures JSON uses "atom1"/"atom2")
    atom1_ref: Ref[Atom] = Field(..., alias="atom1")
    atom2_ref: Ref[Atom] = Field(..., alias="atom2")
    order: float = Field(
        default=1, alias="bond_order"
    )  # Using alias to match original parameter name

    # Computed attribute with proper default
    distance: float | None = None

    # Frozen fields
    version: str = Field(default="2.0", frozen=True)
    type: Type = Field(default="bond")

    @property
    def atom1(self) -> Atom:
        """Get the first atom (unwrapped from Ref)."""
        return self.atom1_ref.get()

    @atom1.setter
    def atom1(self, value: Atom | Ref[Atom]) -> None:
        """Set the first atom (wraps in Ref if needed)."""
        self.atom1_ref = value if isinstance(value, Ref) else Ref(value)

    @property
    def atom2(self) -> Atom:
        """Get the second atom (unwrapped from Ref)."""
        return self.atom2_ref.get()

    @atom2.setter
    def atom2(self, value: Atom | Ref[Atom]) -> None:
        """Set the second atom (wraps in Ref if needed)."""
        self.atom2_ref = value if isinstance(value, Ref) else Ref(value)

    def __getattr__(self, name: str) -> Any:
        """Handle backward compatibility with old pickle files.

        Old pickles have 'atom1'/'atom2' as Atom objects directly.
        This converts them to Ref on first access.
        """
        if name == "atom1_ref":
            # Check if old-style atom1 exists in __dict__
            if "atom1" in self.__dict__:
                atom = self.__dict__.pop("atom1")
                ref = atom if isinstance(atom, Ref) else Ref(atom)
                object.__setattr__(self, "atom1_ref", ref)
                return ref
        elif name == "atom2_ref":
            if "atom2" in self.__dict__:
                atom = self.__dict__.pop("atom2")
                ref = atom if isinstance(atom, Ref) else Ref(atom)
                object.__setattr__(self, "atom2_ref", ref)
                return ref
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def model_post_init(self, __context: Any) -> None:
        # Compute distance between atoms (only if refs are resolved)
        if self.atom1_ref.is_resolved() and self.atom2_ref.is_resolved():
            self.distance = round(
                float(
                    np.linalg.norm(
                        np.array(self.atom1.coord) - np.array(self.atom2.coord)
                    )
                ),
                3,
            )

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol BOND Object --------------\n"
        to_print += f" Version                  = {self.version}\n"
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
    @deprecated("Use bond() with the keyword arguments instead.")
    def from_positional(
        cls, atom1: object, atom2: object, bond_order: float = 1
    ) -> "Bond":
        return cls(atom1=atom1, atom2=atom2, bond_order=bond_order)
