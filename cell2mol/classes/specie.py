"""Compatibility import for the former Specie class and pickle module path."""

from cell2mol.classes.species import Species

Specie = Species

__all__ = ["Specie", "Species"]
