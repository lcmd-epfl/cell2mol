"""Migrate parallel parent/index lists to inline parent-link records."""

from typing import Any


def migrate_parent_links(
    data: Any, index_field: str, legacy_index_field: str, link_index: str
) -> Any:
    """Accept old inputs without silently truncating inconsistent lists."""
    if not isinstance(data, dict):
        return data
    data = data.copy()
    parents = data.pop("parents", [])
    indices = data.pop(index_field, data.pop(legacy_index_field, []))
    if "parent_links" not in data:
        if len(parents) != len(indices):
            raise ValueError("Parent objects and index maps are inconsistent")
        data["parent_links"] = [
            {"parent": parent, link_index: index}
            for parent, index in zip(parents, indices)
        ]
    return data


def migrate_parent_pickle(
    state: dict[str, Any], index_field: str, legacy_index_field: str, link_index: str
) -> None:
    """Convert old pickle state before Pydantic restores the instance."""
    state["__dict__"] = migrate_parent_links(
        state.get("__dict__", {}), index_field, legacy_index_field, link_index
    )
    fields = state.get("__pydantic_fields_set__")
    if fields is not None:
        fields.difference_update({"parents", index_field, legacy_index_field})
        fields.add("parent_links")
