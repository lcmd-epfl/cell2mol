"""Tests for the Ref type."""

import pytest
from pydantic import BaseModel as PydanticBaseModel

from cell2mol.utils.ref import Ref


class MockBaseModel:
    """Mock BaseModel for testing."""

    def __init__(self, id: str):
        self.id = id

    def __repr__(self) -> str:
        return f"MockBaseModel(id={self.id!r})"


class TestRef:
    """Tests for Ref class."""

    def test_create_from_object(self):
        """Test creating a Ref from an actual object."""
        obj = MockBaseModel("test-uuid")
        ref = Ref(obj)

        assert ref.id == "test-uuid"
        assert ref.is_resolved()
        assert ref.get() is obj

    def test_create_from_uuid(self):
        """Test creating a Ref from a UUID string."""
        ref = Ref("test-uuid")

        assert ref.id == "test-uuid"
        assert not ref.is_resolved()

    def test_get_unresolved_raises(self):
        """Test that getting an unresolved ref raises."""
        ref = Ref("test-uuid")

        with pytest.raises(ValueError, match="not yet resolved"):
            ref.get()

    def test_resolve(self):
        """Test resolving a ref from a registry."""
        obj = MockBaseModel("test-uuid")
        ref = Ref("test-uuid")

        assert not ref.is_resolved()

        ref.resolve({"test-uuid": obj})

        assert ref.is_resolved()
        assert ref.get() is obj

    def test_equality(self):
        """Test that refs with same UUID are equal."""
        ref1 = Ref("same-uuid")
        ref2 = Ref("same-uuid")
        ref3 = Ref("different-uuid")

        assert ref1 == ref2
        assert ref1 != ref3

    def test_hash(self):
        """Test that refs can be used in sets/dicts."""
        ref1 = Ref("same-uuid")
        ref2 = Ref("same-uuid")

        s = {ref1, ref2}
        assert len(s) == 1

    def test_repr_resolved(self):
        """Test repr of resolved ref."""
        obj = MockBaseModel("test-uuid")
        ref = Ref(obj)

        assert "MockBaseModel" in repr(ref)

    def test_repr_unresolved(self):
        """Test repr of unresolved ref."""
        ref = Ref("test-uuid")

        assert "unresolved" in repr(ref)
        assert "test-uuid" in repr(ref)


class TestRefPydanticIntegration:
    """Tests for Ref integration with Pydantic."""

    def test_ref_field_in_model(self):
        """Test that Ref can be used as a field type in Pydantic models."""

        class Target(PydanticBaseModel):
            id: str
            name: str

        class Container(PydanticBaseModel):
            model_config = {"arbitrary_types_allowed": True}
            target_ref: Ref[Target]

        target = Target(id="target-uuid", name="test")
        container = Container(target_ref=Ref(target))

        assert container.target_ref.id == "target-uuid"
        assert container.target_ref.get() is target

    def test_ref_serializes_to_uuid(self):
        """Test that Ref serializes to UUID string."""

        class Target(PydanticBaseModel):
            id: str
            name: str

        class Container(PydanticBaseModel):
            model_config = {"arbitrary_types_allowed": True}
            target_ref: Ref[Target]

        target = Target(id="target-uuid", name="test")
        container = Container(target_ref=Ref(target))

        dumped = container.model_dump()
        assert dumped["target_ref"] == "target-uuid"

    def test_ref_accepts_string_uuid(self):
        """Test that Ref field accepts UUID string directly."""

        class Container(PydanticBaseModel):
            model_config = {"arbitrary_types_allowed": True}
            target_ref: Ref[MockBaseModel]

        container = Container(target_ref="some-uuid")

        assert container.target_ref.id == "some-uuid"
        assert not container.target_ref.is_resolved()
