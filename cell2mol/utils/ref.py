"""Typed reference wrapper for object graph serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

if TYPE_CHECKING:
    from cell2mol.utils.pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")


class Ref(Generic[T]):
    """A typed reference to another entity.

    During normal operation, holds the actual object.
    During serialization, converts to UUID string.
    During deserialization, starts as UUID and resolves to object.

    Usage:
        class Bond(BaseModel):
            atom1: Ref[Atom]
            atom2: Ref[Atom]

        # Create with actual object
        bond = Bond(atom1=Ref(some_atom), atom2=Ref(other_atom))

        # Access the referenced object
        atom = bond.atom1.get()
    """

    __slots__ = ("_target", "_id")

    def __init__(self, target: T | str) -> None:
        """Create a reference.

        Args:
            target: Either the actual object or a UUID string (during deserialization)
        """
        if isinstance(target, str):
            # UUID string (during deserialization)
            self._id: str = target
            self._target: T | None = None
        else:
            # Actual object
            self._target = target
            self._id = target.id  # type: ignore[union-attr]

    @property
    def id(self) -> str:
        """Get the UUID of the referenced object."""
        return self._id

    def get(self) -> T:
        """Get the referenced object.

        Raises:
            ValueError: If the reference has not been resolved yet.
        """
        if self._target is None:
            raise ValueError(f"Reference {self._id} not yet resolved")
        return self._target

    def is_resolved(self) -> bool:
        """Check if the reference has been resolved to an actual object."""
        return self._target is not None

    def resolve(self, registry: dict[str, Any]) -> None:
        """Resolve UUID to actual object from registry.

        Args:
            registry: Dict mapping UUIDs to objects.
        """
        if self._target is None and self._id in registry:
            self._target = registry[self._id]

    def __repr__(self) -> str:
        if self._target is not None:
            return f"Ref({self._target!r})"
        return f"Ref(id={self._id!r}, unresolved)"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ref):
            return self._id == other._id
        return False

    def __hash__(self) -> int:
        return hash(self._id)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Tell Pydantic how to validate and serialize Ref fields.

        Accepts:
        - Ref instances (pass through)
        - String UUIDs (wrap in Ref)
        - BaseModel objects (wrap in Ref)

        Serializes to UUID string.
        """

        def validate_ref(value: Any) -> Ref[Any]:
            if isinstance(value, Ref):
                return value
            # String UUID or BaseModel object
            return Ref(value)

        def serialize_ref(ref: Ref[Any]) -> str:
            return ref.id

        return core_schema.no_info_plain_validator_function(
            validate_ref,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_ref,
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )
