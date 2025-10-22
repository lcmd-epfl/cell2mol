from typing import Annotated, Literal
from pydantic import PlainSerializer
import pydantic_numpy.typing as pnd
from rdkit import Chem
from rdkit.Chem import Mol

Spin = int
HapticType = list[str]
Type = Literal["cell", "specie", "protonation", "charge_state", "atom", "bond"]
SubType = Literal["reference", "unitcell", "molecule", "ligand", "metal", "group"]
NOType = Literal["Linear", "Bent"]

NDArray = pnd.NpNDArray


def serialize_mol(mol: Mol) -> str:
    return Chem.MolToJSON(mol)


RDKitObject = Annotated[Mol, PlainSerializer(serialize_mol)]
