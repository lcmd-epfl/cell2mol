import uuid
from typing_extensions import deprecated
import pydantic
from abc import ABC, abstractmethod


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


def serialize_circular_references(
    value: list[BaseModel] | BaseModel | None,
) -> list[str] | str | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        print(f"Serializing circular reference: {value.id}")
        return value.id
    return [item.id for item in value]
