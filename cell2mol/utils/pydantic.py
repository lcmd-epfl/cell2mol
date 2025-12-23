"""Pydantic utilities for cell2mol serialization.

This module provides a BaseModel class with support for serializing object graphs
with circular references using a central object store pattern.

JSON Format:
    {
        "_format": "cell2mol-store",
        "_version": "3.0",
        "objects": {
            "uuid-1": {"_type": "Metal", "label": "Re", ...},
            "uuid-2": {"_type": "Ligand", "metals": ["uuid-1"], ...}
        },
        "root": "uuid-root"
    }

All BaseModel instances are stored exactly once in the "objects" dict.
References between objects use UUID strings.

Reference Fields:
    Fields marked with RefMarker (via Ref, RefList, etc.) are automatically
    serialized as UUIDs and resolved back to objects on deserialization.
    This allows clean typing like `metals: RefList[Metal]` instead of
    polluted types like `metals: list[Metal | str]`.

Serialization Strategy:
    Uses Pydantic's native model_dump() with a context containing the object store.
    A @model_serializer handles collecting objects into the store and returning UUIDs.

Deserialization Strategy:
    Two-pass approach:
    1. Create all objects using model_construct() (refs stay as UUID strings)
    2. Resolve all Ref fields by replacing UUIDs with actual objects
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
import types
import typing
from typing import TYPE_CHECKING, Annotated, Any, Self, get_args, get_origin

import numpy as np
import pydantic
from pydantic import SerializationInfo, model_serializer
from pydantic.fields import FieldInfo
from typing_extensions import deprecated

from cell2mol.my_types import RefMarker
from cell2mol.utils.object_store import ObjectStore
from cell2mol.utils.ref import Ref
from cell2mol.utils.type_registry import TypeRegistry, get_type, register_type

if TYPE_CHECKING:
    pass


# =============================================================================
# Annotation Introspection Helpers
# =============================================================================


def _is_ref_field(annotation: Any) -> bool:
    """Check if a field annotation is marked as a reference field.

    A reference field is one annotated with RefMarker (via Ref, RefList, etc.).
    These fields contain BaseModel objects that should be serialized as UUIDs.
    """
    if annotation is None:
        return False
    if get_origin(annotation) is Annotated:
        for arg in get_args(annotation):
            if isinstance(arg, RefMarker):
                return True
    return False


def _has_ref_fields(cls: type) -> dict[str, FieldInfo]:
    """Get all Ref fields for a class."""
    return {
        name: info
        for name, info in cls.model_fields.items()
        if _is_ref_field(info.annotation)
    }


# =============================================================================
# Value Serialization Helper
# =============================================================================


def _serialize_value(value: Any, context: dict[str, Any]) -> Any:
    """Serialize a value, collecting BaseModels to the store.

    This handles all the types we need to serialize:
    - BaseModel: trigger its serialization and return UUID
    - list: recurse into each item
    - numpy types: convert to Python native types
    - RDKit Mol: use its JSON serializer
    - primitives: pass through as-is
    """
    if value is None:
        return None

    # Ref wrapper -> serialize the target and return UUID
    if isinstance(value, Ref):
        target = value.get()
        target.model_dump(mode="json", context=context)
        return value.id

    # BaseModel -> trigger serialization, return UUID
    if isinstance(value, pydantic.BaseModel) and hasattr(value, "id"):
        value.model_dump(mode="json", context=context)
        return value.id

    # NumPy types -> convert to Python native (check early to avoid issues)
    if isinstance(value, np.ndarray):
        # tolist() may produce numpy scalars, so recurse to ensure conversion
        return [_serialize_value(item, context) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)

    # List -> recurse
    if isinstance(value, list):
        return [_serialize_value(item, context) for item in value]

    # Tuple -> convert to list and recurse
    if isinstance(value, tuple):
        return [_serialize_value(item, context) for item in value]

    # Dict -> recurse into values
    if isinstance(value, dict):
        return {k: _serialize_value(v, context) for k, v in value.items()}

    # RDKit Mol -> use its JSON serializer
    if hasattr(value, "__class__") and value.__class__.__name__ == "Mol":
        from rdkit import Chem

        return Chem.MolToJSON(value)

    # Primitives and other JSON-serializable types pass through
    return value


# =============================================================================
# BaseModel with Central Object Store Serialization
# =============================================================================


class BaseModel(pydantic.BaseModel, ABC):
    """Base model with central object store serialization support.

    All subclasses are automatically registered in the type registry.

    Serialization uses Pydantic's native model_dump() with a context:
        store = {}
        data = obj.model_dump(context={"store": store})
        # store now contains all objects, data is the root UUID

    Deserialization uses a two-pass approach:
        1. model_construct() to create objects (refs are UUID strings)
        2. Resolve refs by replacing UUIDs with actual objects
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,  # Coerce numpy types on assignment
    )

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the type registry."""
        super().__init_subclass__(**kwargs)
        TypeRegistry.get_instance().register(cls)

    @classmethod
    @deprecated(
        "The constructor using keyword arguments should be used instead. "
        "This method is not safe."
    )
    @abstractmethod
    def from_positional(cls, *args: Any, **kwargs: Any) -> Self:
        raise NotImplementedError("This method should be implemented by the subclass")

    # =========================================================================
    # Serialization with Central Object Store
    # =========================================================================

    @model_serializer(mode="wrap")
    def _serialize_to_store(
        self, handler: Any, info: SerializationInfo
    ) -> dict[str, Any] | str:
        """Serialize this object, collecting it into the store if context provided.

        If context contains a 'store' ObjectStore:
            - Add this object to the store (if not already there)
            - Return just the UUID (as a reference)

        If no context:
            - Return normal serialization (for debugging/inspection)
        """
        store: ObjectStore | None = None
        if info.context:
            store = info.context.get("store")

        if store is None:
            # No store context - use normal pydantic serialization
            return handler(self)

        # Already in store - just return UUID
        if self.id in store:
            return self.id

        # Reserve spot FIRST to break cycles
        store.reserve(self.id)

        # Serialize fields ourselves (don't call handler to avoid caching issues)
        data: dict[str, Any] = {"_type": type(self).__name__}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            data[field_name] = _serialize_value(value, info.context)

        store.set(self.id, data)
        return self.id

    def to_dict_store(self) -> dict[str, Any]:
        """Serialize to a dict with central object store.

        Returns a dict with:
        - "_format": "cell2mol-store"
        - "_version": "3.0"
        - "objects": dict mapping UUID -> serialized object data
        - "root": UUID of the root object

        All BaseModel references are replaced with UUID strings.
        Each object appears exactly once in the store.
        """
        store = ObjectStore()

        # Use model_dump with store context
        # The model_serializer will populate the store
        root_id = self.model_dump(mode="json", context={"store": store})

        return {
            "_format": "cell2mol-store",
            "_version": "3.0",
            "objects": store.to_dict(),
            "root": root_id,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string with central object store."""
        return json.dumps(self.to_dict_store(), indent=indent)

    # =========================================================================
    # Deserialization with Two-Pass Reference Resolution
    # =========================================================================

    @classmethod
    def from_dict_store(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dict with central object store.

        Uses a two-pass approach:
        1. Create all objects using model_construct() (refs stay as UUID strings)
        2. Resolve all Ref fields by replacing UUIDs with actual objects

        Args:
            data: Dict with "_format", "objects", and "root" keys

        Returns:
            The reconstructed root object with all references resolved
        """
        format_version = data.get("_format")
        if format_version != "cell2mol-store":
            raise ValueError(
                f"Invalid format: {format_version}. Expected 'cell2mol-store'"
            )

        store = data["objects"]
        root_id = data["root"]

        # Handle case where root_id might be a dict (shouldn't happen but be safe)
        if isinstance(root_id, dict):
            root_id = root_id.get("id", list(store.keys())[0])

        # Pass 1: Create all objects WITHOUT calling model_post_init
        # Ref fields will temporarily hold UUID strings
        registry: dict[str, BaseModel] = {}

        for obj_id, obj_data in store.items():
            type_name = obj_data.get("_type")
            if not type_name:
                raise ValueError(f"Missing '_type' in object {obj_id}")

            obj_cls = get_type(type_name)

            # Prepare data without _type
            construct_data = {k: v for k, v in obj_data.items() if k != "_type"}

            # Create object without calling model_post_init
            # (model_post_init may access refs which are still strings)
            obj = _construct_without_post_init(obj_cls, construct_data)
            registry[obj_id] = obj

        # Pass 2: Resolve all fields that contain UUIDs
        for obj_id, obj in registry.items():
            _resolve_object_refs(obj, registry)

        # Pass 3: Call model_post_init on all objects now that refs are resolved
        for obj_id, obj in registry.items():
            if hasattr(obj, "model_post_init"):
                obj.model_post_init(None)

        root = registry.get(root_id)
        if root is None:
            raise ValueError(f"Root object '{root_id}' not found in store")

        return root  # type: ignore[return-value]

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize from JSON string with central object store."""
        data = json.loads(json_str)
        return cls.from_dict_store(data)


# =============================================================================
# Construction Helper
# =============================================================================


def _construct_without_post_init(
    cls: type["BaseModel"], data: dict[str, Any]
) -> "BaseModel":
    """Create a pydantic model instance without calling model_post_init.

    This is used during deserialization when refs are still UUID strings.
    We need to skip model_post_init until refs are resolved.

    This replicates the logic of model_construct but skips the post_init call.
    """
    obj = cls.__new__(cls)
    fields_set: set[str] = set()
    fields_values: dict[str, Any] = {}

    for name, field in cls.model_fields.items():
        if name in data:
            value = data[name]
            # Convert values to expected types (NDArray, RDKit Mol, etc.)
            value = _convert_field_type(value, field.annotation)
            fields_values[name] = value
            fields_set.add(name)
        elif not field.is_required():
            fields_values[name] = field.get_default(call_default_factory=True)

    # Set pydantic internal state
    object.__setattr__(obj, "__pydantic_fields_set__", fields_set)
    object.__setattr__(obj, "__pydantic_extra__", None)
    object.__setattr__(obj, "__pydantic_private__", None)

    # Set field values
    for k, v in fields_values.items():
        object.__setattr__(obj, k, v)

    # Deliberately skip model_post_init - caller will call it after refs are resolved
    return obj


def _is_ndarray_annotation(annotation: Any) -> bool:
    """Check if annotation represents a numpy array type.

    Handles:
    - numpy.ndarray directly
    - numpy.typing.NDArray[...]
    - pydantic_numpy types (NpNDArray, etc.)
    """
    if annotation is None:
        return False

    # Direct numpy.ndarray
    if annotation is np.ndarray:
        return True

    # Check origin for generic types like NDArray[np.float64]
    origin = get_origin(annotation)
    if origin is np.ndarray:
        return True

    # pydantic_numpy types have __origin__ = ndarray
    if hasattr(annotation, "__origin__") and annotation.__origin__ is np.ndarray:
        return True

    return False


def _is_rdkit_mol_annotation(annotation: Any) -> bool:
    """Check if annotation represents an RDKit Mol type.

    Handles:
    - rdkit.Chem.Mol directly
    - Annotated[Mol, ...] (like RDKitObject)
    - Union types like RDKitObject | None
    """
    if annotation is None:
        return False

    origin = get_origin(annotation)

    # Handle Union types (e.g., RDKitObject | None)
    # types.UnionType for `X | Y` syntax, typing.Union for Optional/Union
    if origin is types.UnionType or origin is typing.Union:
        for arg in get_args(annotation):
            if arg is not type(None) and _is_rdkit_mol_annotation(arg):
                return True
        return False

    # Check if it's an Annotated type and extract the base type
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            base_type = args[0]
            # Check if base type is Mol
            if hasattr(base_type, "__name__") and base_type.__name__ == "Mol":
                return True

    # Direct Mol type
    if hasattr(annotation, "__name__") and annotation.__name__ == "Mol":
        return True

    return False


def _is_ref_annotation(annotation: Any) -> bool:
    """Check if annotation is a Ref[T] type from utils.ref."""
    if annotation is None:
        return False

    origin = get_origin(annotation)

    # Direct Ref class
    if origin is Ref:
        return True

    # Check if it's the Ref class itself (not generic)
    if annotation is Ref:
        return True

    # Check class name for Ref (handles edge cases)
    if hasattr(annotation, "__origin__") and hasattr(annotation.__origin__, "__name__"):
        if annotation.__origin__.__name__ == "Ref":
            return True

    return False


def _convert_field_type(value: Any, annotation: Any) -> Any:
    """Convert deserialized values to their expected types.

    Handles:
    - UUID strings to Ref wrappers for Ref[T] fields
    - Lists to numpy arrays for NDArray fields
    - JSON strings to RDKit Mol for RDKitObject fields
    """
    if value is None:
        return None

    if annotation is None:
        return value

    # Wrap UUID strings in Ref for Ref[T] fields
    if _is_ref_annotation(annotation):
        if isinstance(value, str):
            return Ref(value)
        if isinstance(value, Ref):
            return value

    # Convert lists to numpy arrays for NDArray fields
    if _is_ndarray_annotation(annotation):
        if isinstance(value, list):
            return np.array(value)

    # Convert JSON strings to RDKit Mol for RDKitObject fields
    if _is_rdkit_mol_annotation(annotation):
        if isinstance(value, str) and value.startswith("{"):
            from rdkit import Chem

            mols = Chem.JSONToMols(value)
            if mols and len(mols) > 0:
                return mols[0]

    return value


# =============================================================================
# Reference Resolution Helpers
# =============================================================================


def _resolve_object_refs(obj: BaseModel, registry: dict[str, BaseModel]) -> None:
    """Resolve all fields that contain UUIDs by replacing them with objects.

    Since all BaseModel fields are serialized as UUIDs, we need to resolve
    ALL fields, not just those marked with RefMarker.
    """
    for field_name in type(obj).model_fields:
        # Skip the id field - it's a UUID that identifies the object itself,
        # not a reference to another object
        if field_name == "id":
            continue
        value = getattr(obj, field_name, None)
        resolved = _resolve_value(value, registry)
        if resolved is not value:  # Only set if changed
            # Use object.__setattr__ to bypass frozen field protection
            object.__setattr__(obj, field_name, resolved)


def _resolve_value(value: Any, registry: dict[str, BaseModel]) -> Any:
    """Resolve a value, replacing UUID strings with objects from registry."""
    if value is None:
        return None

    # Ref wrapper -> resolve the internal reference
    if isinstance(value, Ref):
        if not value.is_resolved():
            value.resolve(registry)
        return value

    # Single UUID string -> resolve to object if in registry
    if isinstance(value, str):
        return registry.get(value, value)

    # List of values -> resolve each
    if isinstance(value, list):
        return [_resolve_value(item, registry) for item in value]

    # Dict -> resolve values
    if isinstance(value, dict):
        return {k: _resolve_value(v, registry) for k, v in value.items()}

    # Already an object or other type -> return as-is
    return value
