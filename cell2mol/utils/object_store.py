"""Object store for serialization/deserialization of object graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cell2mol.utils.pydantic import BaseModel


class ObjectStore:
    """Registry of serialized/deserialized objects by UUID.

    Used during serialization to collect objects and break cycles,
    and during deserialization to resolve references.
    """

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any] | BaseModel] = {}

    def __contains__(self, id: str) -> bool:
        return id in self._objects

    def reserve(self, id: str) -> None:
        """Reserve a slot to break cycles during serialization."""
        self._objects[id] = {}

    def set(self, id: str, data: dict[str, Any] | BaseModel) -> None:
        """Store an object or its serialized data."""
        self._objects[id] = data

    def get(self, id: str) -> dict[str, Any] | BaseModel | None:
        """Get an object or its serialized data by ID."""
        return self._objects.get(id)

    def items(self):
        """Iterate over (id, object) pairs."""
        return self._objects.items()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Export as a plain dict (for JSON serialization)."""
        return dict(self._objects)
