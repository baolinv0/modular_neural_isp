"""Lazy operator registry; importing the package never loads external weights."""
from importlib import import_module

OPERATORS = {
    'region_curves': ('region_curves', 'RegionCurveOperator'),
    'spatial_grid': ('spatial_grid', 'SpatialGridOperator'),
    'gain_residual': ('gain_residual', 'GainResidualOperator'),
    'exposure_fusion': ('exposure_fusion', 'ExposureFusionOperator'),
    'base_detail': ('base_detail', 'BaseDetailOperator'),
}


def build_operator(config):
    if config.algorithm not in OPERATORS:
        raise ValueError(f'no candidate operator for {config.algorithm!r}')
    module, name = OPERATORS[config.algorithm]
    return getattr(import_module(f'{__name__}.{module}'), name)(config)


def __getattr__(name):
    for module, class_name in OPERATORS.values():
        if name == class_name:
            return getattr(import_module(f'{__name__}.{module}'), class_name)
    raise AttributeError(name)
