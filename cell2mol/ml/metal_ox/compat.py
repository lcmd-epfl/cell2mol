"""
Accessors that work on both cell2mol APIs the .cell files may come from:
the pre-refactor one (≤ 2026-08, e.g. commit ceadcca) and the pydantic
refactor (feat/pydantic, 2026-10+). Renames handled here:

  Reference.load(path)        -> Reference.load(path, format="pickle")
  mol.iscomplex               -> mol.is_metal_complex
  metal.rel_metal_radius_test -> metal.rel_metal_radius (same computation:
                                 haptic groups weighted as one donor)
  element_utils.labels2formula -> element_utils.symbols2formula

Membership tests use identity: pydantic models compare by field values.
"""

from __future__ import annotations

from typing import Any


def labels2formula(labels: list[str]) -> str:
    try:
        from cell2mol.element_utils import symbols2formula as to_formula
    except ImportError:
        from cell2mol.element_utils import labels2formula as to_formula
    return to_formula(labels)


def load_reference(path: str):
    from cell2mol.classes import Reference

    try:
        return Reference.load(path, format="pickle")
    except TypeError:  # pre-refactor API: no `format` kwarg, pickle by default
        return Reference.load(path)


def is_metal_complex(mol: Any) -> bool:
    if hasattr(mol, "iscomplex"):
        return bool(mol.iscomplex)
    return bool(getattr(mol, "is_metal_complex", False))


def rel_metal_radius(metal: Any) -> float | None:
    if hasattr(metal, "rel_metal_radius_test"):
        return metal.rel_metal_radius_test
    return getattr(metal, "rel_metal_radius", None)


def contains(items, obj: Any) -> bool:
    return any(x is obj for x in items or [])
