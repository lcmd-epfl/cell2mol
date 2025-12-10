import contextvars
import uuid
from abc import ABC, abstractmethod
from typing import Any

import pydantic
from pydantic import model_validator
from typing_extensions import deprecated

# Context variable to store objects during deserialization
# Using contextvars for thread-safety
_object_registry: contextvars.ContextVar[dict[str, "BaseModel"] | None] = (
    contextvars.ContextVar("_object_registry", default=None)
)


def get_registry() -> dict[str, "BaseModel"]:
    """Get or create the object registry for the current context."""
    registry = _object_registry.get()
    if registry is None:
        registry = {}
        _object_registry.set(registry)
    return registry


def clear_registry() -> None:
    """Clear the object registry."""
    _object_registry.set(None)


class BaseModel(pydantic.BaseModel, ABC):
    """
    Wrapper class for pydantic.BaseModel with some additional methods.
    """

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)

    @classmethod
    @deprecated(
        "The constructor using the keyword arguments should be used instead, this method is not safe."
    )
    @abstractmethod
    def from_positional(cls, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by the subclass")

    @model_validator(mode="after")
    def _register_in_registry(self) -> "BaseModel":
        """Register this object in the registry after construction."""
        registry = get_registry()
        registry[self.id] = self
        return self

    def resolve_references(self) -> None:
        """
        Resolve all UUID string references to actual objects.
        Call this on the top-level object after deserialization.
        """
        registry = get_registry()
        self._resolve_references_recursive(registry, set())

    def _resolve_references_recursive(
        self, registry: dict[str, "BaseModel"], visited: set[str]
    ) -> None:
        """Recursively resolve references in this object and its children."""
        # Ensure self.id is a string
        obj_id = self.id if isinstance(self.id, str) else str(id(self))
        if obj_id in visited:
            return
        visited.add(obj_id)

        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name, None)
            if value is None:
                continue

            resolved = self._resolve_value(value, registry)
            if resolved is not value:
                # Use object.__setattr__ to bypass frozen validation
                object.__setattr__(self, field_name, resolved)

            # Recurse into resolved children
            new_value = getattr(self, field_name, None)
            if isinstance(new_value, BaseModel) and hasattr(new_value, "id"):
                new_value._resolve_references_recursive(registry, visited)
            elif isinstance(new_value, list):
                for item in new_value:
                    if isinstance(item, BaseModel) and hasattr(item, "id"):
                        item._resolve_references_recursive(registry, visited)

    def _resolve_value(self, value: Any, registry: dict[str, "BaseModel"]) -> Any:
        """Resolve a single value, replacing UUID strings with objects."""
        if isinstance(value, str) and value in registry:
            return registry[value]
        elif isinstance(value, list):
            resolved_list = []
            changed = False
            for item in value:
                if isinstance(item, str) and item in registry:
                    resolved_list.append(registry[item])
                    changed = True
                else:
                    resolved_list.append(item)
            return resolved_list if changed else value
        return value


def serialize_circular_references(
    value: list[BaseModel] | BaseModel | None,
) -> list[str] | str | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.id
    return [item.id for item in value]


def validate_circular_references(
    value: list[Any] | Any | None,
) -> list[Any] | Any | None:
    """
    Validator that accepts both objects and UUID strings for circular reference fields.
    During deserialization, UUIDs are kept as strings and resolved later.
    """
    # Just pass through - strings will be resolved later by resolve_references()
    return value
