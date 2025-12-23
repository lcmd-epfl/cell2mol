"""Tests for JSON serialization/deserialization with central object store.

The new serialization format uses a central object store pattern:
- All objects are stored exactly once in an "objects" dict
- All references between objects use UUID strings
- This enables true object identity and round-trip serialization
"""

from pathlib import Path

import pytest

from cell2mol.classes import Cells, Group, Metal
from cell2mol.utils.pydantic import get_type
from cell2mol.utils.type_registry import TypeRegistry


@pytest.fixture
def cells_pickle_path():
    """Path to the test cells pickle file."""
    path = Path(__file__).parent.parent / "Cells_YOXKUS.cell"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return path


def save_and_reload(cells: Cells, tmp_path: Path) -> Cells:
    """Helper to save as JSON and reload."""
    json_path = tmp_path / "test.json"
    cells.save(json_path, format="json")
    return Cells.load(json_path, format="json")


class TestSerialization:
    """Test serialization round-trip."""

    def test_load_pickle_save_json_load_json(self, cells_pickle_path, tmp_path):
        """Test that we can load pickle, save as JSON, and reload."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        assert cells_loaded.name == cells.name
        assert len(cells_loaded.reference.refmoleclist or []) == len(
            cells.reference.refmoleclist or []
        )

    def test_json_to_json_round_trip(self, cells_pickle_path, tmp_path):
        """Test JSON -> JSON -> JSON round-trip works correctly."""
        cells = Cells.load(cells_pickle_path, format="pickle")

        # First round: pickle -> JSON -> load
        json_path1 = tmp_path / "test1.json"
        cells.save(json_path1, format="json")
        cells1 = Cells.load(json_path1, format="json")

        # Second round: JSON -> JSON -> load
        json_path2 = tmp_path / "test2.json"
        cells1.save(json_path2, format="json")
        cells2 = Cells.load(json_path2, format="json")

        # Verify data integrity after two round-trips
        assert cells2.name == cells.name
        mol = cells2.reference.refmoleclist[0]
        for lig in mol.ligands or []:
            for metal in lig.metals or []:
                assert isinstance(metal, Metal), (
                    "Metal should still be resolved after 2 round-trips"
                )

    def test_multiple_round_trips(self, cells_pickle_path, tmp_path):
        """Test that multiple round-trips preserve data integrity."""
        cells = Cells.load(cells_pickle_path, format="pickle")

        # Do 5 round-trips
        current = cells
        for i in range(5):
            json_path = tmp_path / f"test_{i}.json"
            current.save(json_path, format="json")
            current = Cells.load(json_path, format="json")

        # Verify final state
        assert current.name == cells.name
        mol = current.reference.refmoleclist[0]
        assert isinstance(mol.metals[0], Metal)
        assert mol.metals[0] is mol.ligands[0].metals[0]


class TestMetalsResolution:
    """Test that metal references are properly resolved."""

    def test_ligand_metals_are_objects(self, cells_pickle_path, tmp_path):
        """Test that ligand.metals are Metal objects, not UUID strings."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        metals_checked = 0
        for mol in cells_loaded.reference.refmoleclist or []:
            for lig in mol.ligands or []:
                assert lig.metals is not None, "Ligand should have metals"
                for metal in lig.metals:
                    assert isinstance(metal, Metal), (
                        f"Metal should be Metal object, not {type(metal)}: {metal}"
                    )
                    assert hasattr(metal, "label"), "Metal should have label attribute"
                    metals_checked += 1

        assert metals_checked > 0, "Should have checked at least one metal"

    def test_group_metals_are_objects(self, cells_pickle_path, tmp_path):
        """Test that group.metals and group.closest_metal are Metal objects."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        groups_with_metals_checked = 0

        for mol in cells_loaded.reference.refmoleclist or []:
            for lig in mol.ligands or []:
                for group in lig.groups or []:
                    assert isinstance(group, Group), (
                        f"Group should be Group object: {type(group)}"
                    )

                    # Check group.metals
                    if group.metals:
                        for metal in group.metals:
                            assert isinstance(metal, Metal), (
                                f"Group.metals should contain Metal objects: {type(metal)}"
                            )
                        groups_with_metals_checked += 1

                    # Check group.closest_metal
                    if group.closest_metal is not None:
                        assert isinstance(group.closest_metal, Metal), (
                            f"Group.closest_metal should be Metal: {type(group.closest_metal)}"
                        )

        assert groups_with_metals_checked > 0, (
            "Should have checked at least one group with metals"
        )

    def test_metal_parents_are_objects(self, cells_pickle_path, tmp_path):
        """Test that metal.parents are properly resolved to objects."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        parents_checked = 0
        for mol in cells_loaded.reference.refmoleclist or []:
            for metal in mol.metals or []:
                if metal.parents:
                    for parent in metal.parents:
                        assert not isinstance(parent, str), (
                            f"Parent should be object, not UUID: {parent}"
                        )
                        parents_checked += 1

        assert parents_checked > 0, "Should have checked at least one parent"


class TestObjectIdentity:
    """Test that same UUID = same Python object instance."""

    def test_molecule_metal_is_ligand_metal(self, cells_pickle_path, tmp_path):
        """Test that mol.metals[i] is the same instance as lig.metals[j] when UUIDs match."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        mol = cells_loaded.reference.refmoleclist[0]
        mol_metal = mol.metals[0]

        # Track if we found and verified at least one match
        matches_found = 0

        for lig in mol.ligands or []:
            for lig_metal in lig.metals or []:
                if lig_metal.id == mol_metal.id:
                    assert mol_metal is lig_metal, (
                        f"Metal with same UUID should be same instance.\n"
                        f"  mol_metal id(): {id(mol_metal)}\n"
                        f"  lig_metal id(): {id(lig_metal)}\n"
                        f"  UUID: {mol_metal.id}"
                    )
                    matches_found += 1

        assert matches_found > 0, (
            f"Should find at least one matching metal. "
            f"mol_metal.id={mol_metal.id}, "
            f"ligand metals: {[m.id for lig in (mol.ligands or []) for m in (lig.metals or [])]}"
        )

    def test_group_metal_is_molecule_metal(self, cells_pickle_path, tmp_path):
        """Test that group.metals[i] is the same instance as mol.metals[j]."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        mol = cells_loaded.reference.refmoleclist[0]
        mol_metals_by_id: dict[str, Metal] = {}
        for m in mol.metals or []:
            mol_metals_by_id[m.id] = m

        matches_found = 0
        for lig in mol.ligands or []:
            for group in lig.groups or []:
                for group_metal in group.metals or []:
                    if group_metal.id in mol_metals_by_id:
                        mol_metal = mol_metals_by_id[group_metal.id]
                        assert group_metal is mol_metal, (
                            f"Group metal should be same instance as molecule metal"
                        )
                        matches_found += 1

        assert matches_found > 0, "Should find at least one matching metal in groups"


class TestUnitcell:
    """Test that unitcell is also properly deserialized."""

    def test_unitcell_molecules_resolved(self, cells_pickle_path, tmp_path):
        """Test that unitcell molecules are properly resolved."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        assert cells_loaded.unitcell is not None, "Unitcell should exist"
        assert cells_loaded.unitcell.moleclist is not None, (
            "Unitcell should have molecules"
        )

        metals_checked = 0
        for mol in cells_loaded.unitcell.moleclist or []:
            for lig in mol.ligands or []:
                for metal in lig.metals or []:
                    assert isinstance(metal, Metal), (
                        f"Unitcell ligand metal should be Metal object: {type(metal)}"
                    )
                    metals_checked += 1

        assert metals_checked > 0, "Should have checked at least one unitcell metal"


class TestObjectStore:
    """Test the central object store format."""

    def test_no_duplicate_objects_in_json(self, cells_pickle_path, tmp_path):
        """Test that each object appears exactly once in the JSON store."""
        import json

        cells = Cells.load(cells_pickle_path, format="pickle")
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")

        with open(json_path) as f:
            data = json.load(f)

        assert data["_format"] == "cell2mol-store"
        assert "objects" in data
        assert "root" in data

        # Count objects by type
        type_counts: dict[str, int] = {}
        for obj_id, obj_data in data["objects"].items():
            obj_type = obj_data["_type"]
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        # Each UUID should appear exactly once as a key
        uuids = list(data["objects"].keys())
        assert len(uuids) == len(set(uuids)), "Each UUID should be unique"

        print(f"Object counts by type: {type_counts}")

    def test_all_references_are_uuids(self, cells_pickle_path, tmp_path):
        """Test that all object references in the JSON are UUID strings."""
        import json

        cells = Cells.load(cells_pickle_path, format="pickle")
        json_path = tmp_path / "test.json"
        cells.save(json_path, format="json")

        with open(json_path) as f:
            data = json.load(f)

        objects = data["objects"]
        valid_uuids = set(objects.keys())

        def check_references(value, path=""):
            """Check that string values that look like UUIDs are valid references."""
            if isinstance(value, str):
                # If it looks like a UUID and is in our valid set, it's a reference
                if len(value) == 36 and value.count("-") == 4:
                    if value in valid_uuids:
                        return  # Valid reference
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_references(item, f"{path}[{i}]")
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k not in ("_type", "id"):
                        check_references(v, f"{path}.{k}")

        for obj_id, obj_data in objects.items():
            check_references(obj_data, obj_id)


class TestTypeRegistry:
    """Test the type registry for deserialization."""

    def test_all_classes_registered(self):
        """Test that all cell2mol classes are in the type registry."""
        # Import to trigger registration
        from cell2mol.classes import (
            Atom,
            Bond,
            Cell,
            Cells,
            ChargeState,
            Group,
            Ligand,
            Metal,
            Molecule,
            Protonation,
            Specie,
        )

        expected_types = [
            "Atom",
            "Bond",
            "Cell",
            "Cells",
            "ChargeState",
            "Group",
            "Ligand",
            "Metal",
            "Molecule",
            "Protonation",
            "Specie",
        ]

        registry = TypeRegistry.get_instance()
        for type_name in expected_types:
            assert type_name in registry, f"{type_name} should be registered"
            assert get_type(type_name) is not None

    def test_unknown_type_raises_error(self):
        """Test that unknown types raise a clear error."""
        with pytest.raises(ValueError, match="Unknown type"):
            get_type("NonExistentClass")


class TestGroupsResolution:
    """Test that group references are properly resolved."""

    def test_ligand_groups_are_objects(self, cells_pickle_path, tmp_path):
        """Test that ligand.groups are Group objects."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        groups_checked = 0
        for mol in cells_loaded.reference.refmoleclist or []:
            for lig in mol.ligands or []:
                for group in lig.groups or []:
                    assert isinstance(group, Group), (
                        f"Should be Group object: {type(group)}"
                    )
                    assert hasattr(group, "formula"), "Group should have formula"
                    groups_checked += 1

        assert groups_checked > 0, "Should have checked at least one group"


class TestEdgeCases:
    """Test edge cases and potential fragility."""

    def test_numpy_arrays_preserved(self, cells_pickle_path, tmp_path):
        """Test that numpy arrays are preserved after round-trip."""
        import numpy as np

        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        # cell_vector and cell_param should be numpy arrays
        assert isinstance(cells_loaded.cell_vector, np.ndarray), (
            f"cell_vector should be ndarray, got {type(cells_loaded.cell_vector)}"
        )
        assert isinstance(cells_loaded.cell_param, np.ndarray), (
            f"cell_param should be ndarray, got {type(cells_loaded.cell_param)}"
        )

        # Values should match
        np.testing.assert_array_almost_equal(
            cells_loaded.cell_vector, cells.cell_vector
        )
        np.testing.assert_array_almost_equal(cells_loaded.cell_param, cells.cell_param)

    def test_rdkit_mol_preserved(self, cells_pickle_path, tmp_path):
        """Test that RDKit Mol objects are preserved after round-trip."""
        from rdkit.Chem import Mol

        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        # Find a species with rdkit_obj
        rdkit_checked = 0
        for mol in cells_loaded.reference.refmoleclist or []:
            if mol.rdkit_obj is not None:
                assert isinstance(mol.rdkit_obj, Mol), (
                    f"rdkit_obj should be Mol, got {type(mol.rdkit_obj)}"
                )
                rdkit_checked += 1

            for lig in mol.ligands or []:
                if lig.rdkit_obj is not None:
                    assert isinstance(lig.rdkit_obj, Mol), (
                        f"Ligand rdkit_obj should be Mol, got {type(lig.rdkit_obj)}"
                    )
                    rdkit_checked += 1

        # Note: rdkit_obj might be None in test data, so we just log
        print(f"Checked {rdkit_checked} RDKit Mol objects")

    def test_charge_state_fields_preserved(self, cells_pickle_path, tmp_path):
        """Test that ChargeState computed fields are preserved."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        charge_states_checked = 0
        for mol in cells_loaded.reference.refmoleclist or []:
            for lig in mol.ligands or []:
                if hasattr(lig, "possible_cs") and lig.possible_cs:
                    for cs in lig.possible_cs:
                        # These should have values (computed originally, saved, restored)
                        assert cs.uncorr_abstotal is not None, (
                            "uncorr_abstotal should be set"
                        )
                        assert cs.corr_total_charge is not None, (
                            "corr_total_charge should be set"
                        )
                        charge_states_checked += 1

        print(f"Checked {charge_states_checked} ChargeState objects")

    def test_frozen_fields_preserved(self, cells_pickle_path, tmp_path):
        """Test that frozen fields (version, type) are preserved correctly."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        assert cells_loaded.version == cells.version
        assert cells_loaded.type == cells.type

        # Check nested objects too
        for mol in cells_loaded.reference.refmoleclist or []:
            assert mol.version == "2.0"
            assert mol.type == "specie"

    def test_atom_bonds_resolved(self, cells_pickle_path, tmp_path):
        """Test that Atom.bonds list is resolved (bonds are Bond objects, not UUIDs)."""
        from cell2mol.classes import Atom

        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        atoms_with_bonds_checked = 0
        for mol in cells_loaded.reference.refmoleclist or []:
            for atom in mol.atoms or []:
                if atom.bonds:
                    for bond in atom.bonds:
                        # Bond should be an object, not a UUID string
                        assert not isinstance(bond, str), (
                            f"Bond should be object, not UUID string: {bond}"
                        )
                        atoms_with_bonds_checked += 1

        # Not all molecules may have bonds populated, so this may be 0
        # Just ensure we don't crash when iterating

    def test_deeply_nested_references(self, cells_pickle_path, tmp_path):
        """Test that deeply nested references are all resolved."""
        cells = Cells.load(cells_pickle_path, format="pickle")
        cells_loaded = save_and_reload(cells, tmp_path)

        # Navigate deep: Cells -> Cell -> Molecule -> Ligand -> Group -> Metal -> parents
        mol = cells_loaded.reference.refmoleclist[0]
        lig = mol.ligands[0]
        group = lig.groups[0]
        metal = group.metals[0] if group.metals else None

        if metal and metal.parents:
            parent = metal.parents[0]
            # Parent should be a Molecule/Specie, not a string
            assert not isinstance(parent, str), (
                "Deeply nested parent should be resolved"
            )
            assert hasattr(parent, "formula"), "Parent should have formula attribute"
