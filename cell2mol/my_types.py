from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar

import numpy as np
from pydantic import (
    BeforeValidator,
    GetCoreSchemaHandler,
    PlainSerializer,
    SerializationInfo,
)
from pydantic.functional_serializers import WrapSerializer
from pydantic_core import CoreSchema, core_schema
from rdkit import Chem
from rdkit.Chem import Mol


# =============================================================================
# NumPy-safe scalar types
# =============================================================================
# These types handle numpy scalars that might be assigned to fields from
# legacy pickle data. They coerce numpy types to Python native types.
# =============================================================================


def _coerce_int(value: Any) -> int | None:
    """Coerce numpy integers to Python int."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _coerce_float(value: Any) -> float | None:
    """Coerce numpy floats to Python float."""
    if value is None:
        return None
    if isinstance(value, np.floating):
        return float(value)
    return value


def _serialize_int(value: Any) -> int | None:
    """Serialize int, handling numpy integers."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _serialize_float(value: Any) -> float | None:
    """Serialize float, handling numpy floats."""
    if value is None:
        return None
    if isinstance(value, np.floating):
        return float(value)
    return value


# Use these types for fields that might receive numpy scalars
# BeforeValidator handles new assignments, PlainSerializer handles legacy pickle data
Int = Annotated[int, BeforeValidator(_coerce_int), PlainSerializer(_serialize_int)]
OptionalInt = Annotated[
    int | None, BeforeValidator(_coerce_int), PlainSerializer(_serialize_int)
]
Float = Annotated[
    float, BeforeValidator(_coerce_float), PlainSerializer(_serialize_float)
]
OptionalFloat = Annotated[
    float | None, BeforeValidator(_coerce_float), PlainSerializer(_serialize_float)
]


# =============================================================================
# Reference Types for Cross-Object References
# =============================================================================
# These types mark fields that contain references to other BaseModel objects.
# At runtime, the field holds the actual object (e.g., Metal).
# During serialization, the object is stored in the central object store
# and replaced with its UUID string.
# =============================================================================

T = TypeVar("T")


@dataclass(frozen=True)
class RefMarker:
    """Marker to identify cross-reference fields during serialization.

    When a field is annotated with RefMarker (via Ref, RefList, etc.),
    the serialization system knows to:
    - Serialize: Replace the object with its UUID
    - Deserialize: Resolve the UUID back to the object from the registry

    This allows clean typing like `metals: RefList[Metal]` instead of
    polluted types like `metals: list[Metal | str]`.
    """

    pass


def _serialize_ref(value: Any, handler: Any, info: SerializationInfo) -> Any:
    """Serialize a reference field to UUID strings.

    For single refs: returns UUID string
    For list refs: returns list of UUID strings

    Note: The referenced objects must be serialized separately by the parent
    object's serializer. This serializer only extracts IDs.
    """
    if value is None:
        return None

    # Handle list of references
    if isinstance(value, list):
        result = []
        for item in value:
            if hasattr(item, "id"):
                result.append(item.id)
            else:
                # Keep as-is (might be UUID string already)
                result.append(item)
        return result

    # Handle single reference
    if hasattr(value, "id"):
        return value.id

    # Not a BaseModel, return as-is
    return value


# Type aliases for cross-reference fields
# Usage: metals: RefList[Metal] = Field(default_factory=list)
# The WrapSerializer ensures referenced objects are added to the store
Ref = Annotated[T, RefMarker(), WrapSerializer(_serialize_ref)]
RefList = Annotated[list[T], RefMarker(), WrapSerializer(_serialize_ref)]
OptionalRef = Annotated[T | None, RefMarker(), WrapSerializer(_serialize_ref)]
OptionalRefList = Annotated[list[T] | None, RefMarker(), WrapSerializer(_serialize_ref)]


# =============================================================================
# Simple Type Aliases
# =============================================================================

Spin = int
HapticType = list[str]
Type = Literal["cell", "cells", "specie", "protonation", "charge_state", "atom", "bond"]
SubType = Literal[
    "reference", "unitcell", "molecule", "ligand", "metal", "atom", "group"
]
NOType = Literal["Linear", "Bent"]


class _NDArrayType:
    """Custom Pydantic type for numpy arrays with proper serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            ),
        )

    @staticmethod
    def _validate(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, list):
            return np.array(value)
        raise ValueError(f"Cannot convert {type(value)} to ndarray")

    @staticmethod
    def _serialize(value: np.ndarray) -> list:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


NDArray = Annotated[np.ndarray, _NDArrayType()]

Format = Literal["json", "pickle"]


class _RDKitMolType:
    """Custom Pydantic type for RDKit Mol with proper serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            ),
        )

    @staticmethod
    def _validate(value: Any) -> Mol:
        if isinstance(value, Mol):
            return value
        if isinstance(value, str):
            mols = Chem.JSONToMols(value)
            if mols and len(mols) > 0:
                return mols[0]
            raise ValueError(f"Failed to deserialize RDKit Mol: {value[:100]}...")
        raise ValueError(f"Cannot convert {type(value)} to RDKit Mol")

    @staticmethod
    def _serialize(value: Mol) -> str:
        return Chem.MolToJSON(value)


RDKitObject = Annotated[Mol, _RDKitMolType()]
