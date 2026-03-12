# models/registry.py
from typing import Type

from .base import BaseModel

MODEL_REGISTRY: dict[str, Type[BaseModel]] = {}


def register(cls: Type[BaseModel]) -> Type[BaseModel]:
    """Decorator to auto-register a model class."""
    MODEL_REGISTRY[cls.name] = cls
    return cls


"""
To use the registry in the model files:
    # models/regression/linear.py
    from ..registry import register
    from ..base import BaseModel

    @register
    class LinearRegressionModel(BaseModel):
        name = "linear"
        category = "regression"
        ...
"""
