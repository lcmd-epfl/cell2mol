"""Type registry for deserialization of object graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cell2mol.utils.pydantic import BaseModel


class TypeRegistry:
    """Registry mapping type names to classes for deserialization.

    Uses singleton pattern for global access, but can be cleared for testing.
    """

    _instance: TypeRegistry | None = None

    def __init__(self) -> None:
        self._types: dict[str, type[BaseModel]] = {}

    @classmethod
    def get_instance(cls) -> TypeRegistry:
        """Get the global registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, type_cls: type[BaseModel]) -> None:
        """Register a class by its name."""
        self._types[type_cls.__name__] = type_cls

    def get(self, name: str) -> type[BaseModel]:
        """Get a class by name. Raises ValueError if not found."""
        if name not in self._types:
            raise ValueError(
                f"Unknown type '{name}'. Available: {list(self._types.keys())}"
            )
        return self._types[name]

    def clear(self) -> None:
        """Clear all registered types (useful for testing)."""
        self._types.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._types


# Convenience functions for backwards compatibility
def register_type(cls: type[BaseModel]) -> type[BaseModel]:
    """Register a class in the global type registry."""
    TypeRegistry.get_instance().register(cls)
    return cls


def get_type(type_name: str) -> type[BaseModel]:
    """Get a class from the global type registry by name."""
    return TypeRegistry.get_instance().get(type_name)
