"""Research tone-mapping operators for the original Modular Neural ISP."""
from .config import ModelConfig

__all__ = ['ModelConfig', 'TMModel']


def __getattr__(name):
    if name == 'TMModel':
        from .model import TMModel
        return TMModel
    raise AttributeError(name)
