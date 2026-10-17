from .atom import Atom
from .metal import Metal
from .bond import Bond
from .protonation import Protonation
from .charge_state import ChargeState
from .atom_set import AtomSet
from .species import Species
from .group import Group
from .ligand import Ligand
from .molecule import Molecule
from .cell import Cell
from .reference import Reference
from .unitcell import UnitCell
from .molecule_set import MoleculeSet
from .cells import Cells
from cell2mol.utils.type_registry import TypeRegistry

# Keep the former public name available to external callers and old pickles.
Specie = Species

__all__ = [
    "Atom",
    "Metal",
    "Bond",
    "Protonation",
    "ChargeState",
    "Specie",
    "Species",
    "AtomSet",
    "Group",
    "Ligand",
    "Molecule",
    "Cell",
    "Reference",
    "UnitCell",
    "MoleculeSet",
    "Cells",
]

# Resolve model references after every class is available.
Atom.model_rebuild()
Metal.model_rebuild()
Bond.model_rebuild()
Protonation.model_rebuild()
ChargeState.model_rebuild()
AtomSet.model_rebuild()
Species.model_rebuild()
Group.model_rebuild()
Ligand.model_rebuild()
Molecule.model_rebuild()
Cell.model_rebuild()
Reference.model_rebuild()
UnitCell.model_rebuild()
MoleculeSet.model_rebuild()
Cells.model_rebuild()

# Legacy JSON graphs identify the former class as Specie.
TypeRegistry.get_instance().register(Species, name="Specie")
