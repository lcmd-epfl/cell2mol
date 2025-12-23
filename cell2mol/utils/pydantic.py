"""Pydantic utilities for cell2mol serialization.

This module provides a BaseModel class with support for serializing object graphs
with circular references using a central object store pattern.

JSON Format:
    {
        "_format": "cell2mol-store",
        "_version": "2.1",
        "objects": {
            "uuid-1": {"_type": "Metal", "label": "Re", ...},
            "uuid-2": {"_type": "Ligand", "metals": ["uuid-1"], ...}
        },
        "root": "uuid-root"
    }

All BaseModel instances are stored exactly once in the "objects" dict.
All references between objects use UUID strings.
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pydantic
from rdkit import Chem
from rdkit.Chem import Mol
from typing_extensions import deprecated

# Type registry for deserialization
_type_registry: dict[str, type["BaseModel"]] = {}


def register_type(cls: type["BaseModel"]) -> type["BaseModel"]:
    """Register a class in the type registry for deserialization."""
    _type_registry[cls.__name__] = cls
    return cls


def get_type(type_name: str) -> type["BaseModel"]:
    """Get a class from the type registry by name."""
    if type_name not in _type_registry:
        raise ValueError(
            f"Unknown type '{type_name}'. "
            f"Available types: {list(_type_registry.keys())}"
        )
    return _type_registry[type_name]


class BaseModel(pydantic.BaseModel, ABC):
    """Base model with central object store serialization support.

    All subclasses are automatically registered in the type registry.
    """

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the type registry."""
        super().__init_subclass__(**kwargs)
        # Register the class by its name
        _type_registry[cls.__name__] = cls

    @classmethod
    @deprecated(
        "The constructor using the keyword arguments should be used instead, this method is not safe."
    )
    @abstractmethod
    def from_positional(cls, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by the subclass")

    # =========================================================================
    # Central Object Store Serialization
    # =========================================================================

    def to_dict_store(self) -> dict[str, Any]:
        """Serialize to a dict with central object store.

        Returns a dict with:
        - "_format": "cell2mol-store"
        - "_version": "2.1"
        - "objects": dict mapping UUID -> serialized object data
        - "root": UUID of the root object

        All BaseModel references are replaced with UUID strings.
        Each object appears exactly once in the store.
        """
        store: dict[str, dict[str, Any]] = {}
        self._collect_objects(store)

        return {
            "_format": "cell2mol-store",
            "_version": "2.1",
            "objects": store,
            "root": self.id,
        }

    def _collect_objects(self, store: dict[str, dict[str, Any]]) -> None:
        """Recursively collect all BaseModel objects into the store."""
        if self.id in store:
            return  # Already collected

        # Reserve our spot BEFORE recursing (prevents infinite recursion)
        store[self.id] = {}

        # Serialize this object
        data = {"_type": type(self).__name__}

        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            data[field_name] = self._serialize_value(value, store)

        # Replace placeholder with actual data
        store[self.id] = data

    def _serialize_value(self, value: Any, store: dict[str, dict[str, Any]]) -> Any:
        """Serialize a value, replacing BaseModel instances with UUIDs."""
        if value is None:
            return None
        elif isinstance(value, BaseModel):
            # Collect the object and return its UUID
            value._collect_objects(store)
            return value.id
        elif isinstance(value, Mol):
            # RDKit Mol objects are serialized to JSON string
            return Chem.MolToJSON(value)
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to nested lists
            return self._serialize_value(value.tolist(), store)
        elif isinstance(value, (np.integer,)):
            return int(value)
        elif isinstance(value, (np.floating,)):
            return float(value)
        elif isinstance(value, (np.bool_,)):
            return bool(value)
        elif isinstance(value, list):
            return [self._serialize_value(item, store) for item in value]
        elif isinstance(value, tuple):
            return [self._serialize_value(item, store) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v, store) for k, v in value.items()}
        else:
            return value

    @classmethod
    def from_dict_store(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dict with central object store.

        Args:
            data: Dict with "_format", "objects", and "root" keys

        Returns:
            The reconstructed root object with all references resolved
        """
        if data.get("_format") != "cell2mol-store":
            raise ValueError(
                f"Invalid format: {data.get('_format')}. Expected 'cell2mol-store'"
            )

        store = data["objects"]
        root_id = data["root"]

        # First pass: create empty object shells (no initialization)
        registry: dict[str, BaseModel] = {}
        raw_data: dict[str, dict[str, Any]] = {}

        for obj_id, obj_data in store.items():
            type_name = obj_data["_type"]
            obj_cls = get_type(type_name)

            # Store raw data for later
            raw_data[obj_id] = {k: v for k, v in obj_data.items() if k != "_type"}

            # Create empty shell using __new__ (bypasses __init__ and model_post_init)
            obj = object.__new__(obj_cls)
            # Initialize the pydantic internals
            object.__setattr__(obj, "__dict__", {})
            object.__setattr__(obj, "__pydantic_fields_set__", set())
            object.__setattr__(obj, "__pydantic_extra__", None)
            object.__setattr__(obj, "__pydantic_private__", None)
            registry[obj_id] = obj

        # Second pass: populate fields with resolved references and type conversion
        for obj_id, obj in registry.items():
            obj_cls = type(obj)
            field_data = raw_data[obj_id]

            # First, set defaults for missing fields
            for field_name, field_info in obj_cls.model_fields.items():
                if field_name not in field_data:
                    if field_info.default is not pydantic.fields.PydanticUndefined:
                        object.__setattr__(obj, field_name, field_info.default)
                    elif field_info.default_factory is not None:
                        object.__setattr__(
                            obj, field_name, field_info.default_factory()
                        )

            # Then set actual values with resolution and type conversion
            for field_name, value in field_data.items():
                if field_name == "id":
                    object.__setattr__(obj, field_name, value)
                else:
                    resolved = _resolve_value_static(value, registry)
                    # Apply type conversions
                    converted = _convert_type(resolved, obj_cls, field_name)
                    object.__setattr__(obj, field_name, converted)

        return registry[root_id]

    # =========================================================================
    # JSON Serialization (convenience methods)
    # =========================================================================

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string with central object store."""
        return json.dumps(self.to_dict_store(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize from JSON string with central object store."""
        data = json.loads(json_str)
        return cls.from_dict_store(data)


def _convert_type(value: Any, obj_cls: type, field_name: str) -> Any:
    """Convert value to the expected type based on field annotation."""
    if value is None:
        return None

    # Get field info to check annotation
    field_info = obj_cls.model_fields.get(field_name)
    if field_info is None:
        return value

    annotation = field_info.annotation

    # Handle numpy arrays - check if annotation mentions NDArray or ndarray
    annotation_str = str(annotation)
    if "NDArray" in annotation_str or "ndarray" in annotation_str:
        if isinstance(value, list):
            return np.array(value)

    # Handle RDKit Mol - check if it's an RDKit JSON string
    if "RDKitObject" in annotation_str or "Mol" in annotation_str:
        if isinstance(value, str) and (
            value.startswith('{"commonchem"') or value.startswith('{"rdkitjson"')
        ):
            mols = Chem.JSONToMols(value)
            if mols and len(mols) > 0:
                return mols[0]

    return value


def _resolve_value_static(value: Any, registry: dict[str, "BaseModel"]) -> Any:
    """Resolve UUID strings to objects from the registry (static version)."""
    if isinstance(value, str) and value in registry:
        return registry[value]
    elif isinstance(value, list):
        return [_resolve_value_static(item, registry) for item in value]
    elif isinstance(value, dict):
        return {k: _resolve_value_static(v, registry) for k, v in value.items()}
    return value


# =============================================================================
# Legacy functions (for backward compatibility during transition)
# =============================================================================


def serialize_circular_references(
    value: list[BaseModel] | BaseModel | None,
) -> list[str] | str | None:
    """Serialize BaseModel references as UUID strings.

    DEPRECATED: This is only needed for the old serialization format.
    The new central object store format handles this automatically.
    """
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.id
    return [item.id for item in value]


def validate_circular_references(
    value: list[Any] | Any | None,
) -> list[Any] | Any | None:
    """Validator that accepts both objects and UUID strings.

    DEPRECATED: This is only needed for the old serialization format.
    The new central object store format handles this automatically.
    """
    return value
