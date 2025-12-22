"""Tests for JSON serialization/deserialization with circular references."""

from pathlib import Path

import pytest

from cell2mol.classes import Cells, Metal
from cell2mol.utils.pydantic import clear_registry


@pytest.fixture
def cells_pickle_path():
    """Path to the test cells pickle file."""
    return Path(__file__).parent.parent / "Cells_YOXKUS.cell"


@pytest.fixture(autouse=True)
def clear_registry_before_each():
    """Clear the object registry before each test."""
    clear_registry()
    yield
    clear_registry()


class TestSerialization:
    """Test serialization round-trip."""

    def test_load_pickle_save_json_load_json(self, cells_pickle_path, tmp_path):
        """Test that we can load pickle, save as JSON, and reload."""
        # Load from pickle
        cells = Cells.load(cells_pickle_path, format="pickle")

        # Save as JSON
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")

        # Reload from JSON
        cells_loaded = Cells.load(json_path, format="json")

        assert cells_loaded.name == cells.name
        assert len(cells_loaded.reference.refmoleclist or []) == len(
            cells.reference.refmoleclist or []
        )

    def test_metals_resolved_to_objects(self, cells_pickle_path, tmp_path):
        """Test that ligand.metals are resolved to Metal objects, not UUID strings."""
        # Load -> save JSON -> reload
        cells = Cells.load(cells_pickle_path, format="pickle")
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")
        cells_loaded = Cells.load(json_path, format="json")

        # Check ligand.metals are Metal objects
        for mol in cells_loaded.reference.refmoleclist or []:
            for lig in mol.ligands or []:
                assert lig.metals is not None, "Ligand should have metals"
                for metal in lig.metals:
                    assert not isinstance(metal, str), (
                        f"Metal should be object, not UUID string: {metal}"
                    )
                    assert hasattr(metal, "label"), "Metal should have label attribute"

    def test_single_instance_per_uuid(self, cells_pickle_path, tmp_path):
        """Test that objects with same UUID are the same Python instance."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")
        cells_loaded = Cells.load(json_path, format="json")

        # Get metal from molecule.metals and from ligand.metals
        assert isinstance(cells_loaded.reference.refmoleclist, list), (
            "Reference molecule list should be a list"
        )
        mol = cells_loaded.reference.refmoleclist[0]
        assert isinstance(mol.metals, list), "Molecule metals should be a list"
        mol_metal = mol.metals[0]

        # Find same metal in ligand
        for lig in mol.ligands or []:
            if lig.metals:
                for lig_metal in lig.metals:
                    assert isinstance(lig_metal, Metal), (
                        "Ligand metal should be a Metal object"
                    )
                    if lig_metal.id == mol_metal.id:
                        # Should be the SAME Python object
                        assert mol_metal is lig_metal, (
                            f"Metal with same UUID should be same instance: {id(mol_metal)} vs {id(lig_metal)}"
                        )

    def test_groups_resolved(self, cells_pickle_path, tmp_path):
        """Test that ligand.groups are properly resolved."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")
        cells_loaded = Cells.load(json_path, format="json")

        for mol in cells_loaded.reference.refmoleclist or []:
            for lig in mol.ligands or []:
                if lig.groups:
                    for group in lig.groups:
                        assert not isinstance(group, str), "Group should be object"
                        assert hasattr(group, "formula"), "Group should have formula"

    def test_metal_parents_resolved(self, cells_pickle_path, tmp_path):
        """Test that metal.parents are properly resolved to objects."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")
        cells_loaded = Cells.load(json_path, format="json")

        for mol in cells_loaded.reference.refmoleclist or []:
            for metal in mol.metals or []:
                if metal.parents:
                    for parent in metal.parents:
                        assert not isinstance(parent, str), (
                            f"Parent should be object, not UUID: {parent}"
                        )
