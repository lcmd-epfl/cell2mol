from typing_extensions import deprecated
import pydantic
from abc import ABC, abstractmethod


class BaseModel(pydantic.BaseModel, ABC):
    """
    Wrapper class for pydantic.BaseModel with some additional methods.
    """

    @classmethod
    @deprecated(
        "The constructor using the keyword arguments should be used instead, this method is not safe."
    )
    @abstractmethod
    def from_positional(cls, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by the subclass")
