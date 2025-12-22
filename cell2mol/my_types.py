from typing import Annotated, Literal

import pydantic_numpy.typing as pnd
from pydantic import BeforeValidator, PlainSerializer
from rdkit import Chem
from rdkit.Chem import Mol

Spin = int
HapticType = list[str]
Type = Literal["cell", "cells", "specie", "protonation", "charge_state", "atom", "bond"]
SubType = Literal[
    "reference", "unitcell", "molecule", "ligand", "metal", "atom", "group"
]
NOType = Literal["Linear", "Bent"]

NDArray = pnd.NpNDArray

Format = Literal["json", "pickle"]


def serialize_mol(mol: Mol) -> str:
    return Chem.MolToJSON(mol)


def deserialize_mol(value: Mol | str) -> Mol:
    if isinstance(value, Mol):
        return value
    if isinstance(value, str):
        # JSONToMols returns a list, we take the first element
        mols = Chem.JSONToMols(value)
        if mols and len(mols) > 0:
            return mols[0]
        raise ValueError(f"Failed to deserialize RDKit Mol from JSON: {value[:100]}...")
    raise ValueError(f"Cannot deserialize {type(value)} to RDKit Mol")


RDKitObject = Annotated[
    Mol, BeforeValidator(deserialize_mol), PlainSerializer(serialize_mol)
]
