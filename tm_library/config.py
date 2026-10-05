"""Validated, checkpoint-serializable model settings."""
from dataclasses import asdict, dataclass, fields
import math
from numbers import Real
from typing import Mapping

ALGORITHMS = ('baseline', 'gtm', 'region_curves', 'spatial_grid', 'gain_residual', 'exposure_fusion', 'base_detail')
SEMANTIC_MODES = ('none', 'train_only', 'explicit')


@dataclass
class ModelConfig:
    architecture_version: int = 2
    algorithm: str = 'baseline'
    base_algorithm: str | None = None
    correction_algorithm: str = 'none'
    correction_strength: float = 1.0
    correction_max_delta: float = .25
    correction_semantic_channel: int | None = None
    post_mode: str = 'adaptive'
    semantic_mode: str = 'none'
    semantic_channels: int = 3
    width: int = 16
    analysis_size: int = 64
    curve_bins: int = 16
    num_experts: int = 4
    grid_size: int = 8
    grid_depth: int = 8
    # Use the same 16-EV magnitude limit as synthetic exposures; larger
    # unrestricted log2 gains can overflow float32 during RGB rendering.
    max_ev: float = 1.0
    exposures: tuple[float, ...] = (-2., 0., 2.)
    pyramid_levels: int = 3
    filter_radius: int = 3
    filter_eps: float = .01
    use_3d_lut: bool = False
    freeze_backbone: bool = False

    def __post_init__(self):
        self.validate()

    @property
    def effective_base(self):
        return self.base_algorithm if self.base_algorithm is not None else self.algorithm

    def validate(self):
        if type(self.architecture_version) is not int or self.architecture_version != 2:
            raise ValueError('architecture_version must be 2; v1 library weights require migration/retraining')
        if self.base_algorithm is not None and self.base_algorithm not in ALGORITHMS:
            raise ValueError(f'base_algorithm must be one of {ALGORITHMS}')
        if self.base_algorithm is not None and self.algorithm != 'baseline':
            raise ValueError('base_algorithm overrides only the default algorithm=baseline; ambiguous selectors')
        if self.correction_algorithm not in ('none', 'region_curves', 'gain_residual'):
            raise ValueError('correction_algorithm must be none, region_curves or gain_residual')
        for name in ('correction_strength', 'correction_max_delta'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
        if not 0 <= self.correction_strength <= 1:
            raise ValueError('correction_strength must lie in [0,1]')
        if not 0 < self.correction_max_delta <= 1:
            raise ValueError('correction_max_delta must lie in (0,1]')
        if self.correction_semantic_channel is not None:
            if (type(self.correction_semantic_channel) is not int or
                not 0 <= self.correction_semantic_channel < self.semantic_channels or
                self.semantic_mode != 'explicit' or self.correction_algorithm == 'none'):
                raise ValueError('correction_semantic_channel must be valid and requires explicit semantic correction')
        if self.post_mode not in ('adaptive', 'fixed_reference'):
            raise ValueError('post_mode must be adaptive or fixed_reference')
        if self.post_mode == 'fixed_reference' and not self.freeze_backbone:
            raise ValueError('fixed_reference operator diagnostics require freeze_backbone=True')
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f'algorithm must be one of {ALGORITHMS}')
        if self.semantic_mode not in SEMANTIC_MODES:
            raise ValueError(f'semantic_mode must be one of {SEMANTIC_MODES}')
        if self.effective_base in ('baseline', 'gtm') and self.correction_algorithm == 'none' and self.semantic_mode == 'train_only':
            raise ValueError('baseline/gtm without a learned candidate has no train_only auxiliary features')
        minima = {'semantic_channels': 1, 'width': 1, 'analysis_size': 1, 'curve_bins': 2,
                  'num_experts': 1, 'grid_size': 2, 'grid_depth': 2, 'pyramid_levels': 1,
                  'filter_radius': 0}
        for name, minimum in minima.items():
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name, allow_zero in [('max_ev', True), ('filter_eps', False)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
            if value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f'{name} must be {"nonnegative" if allow_zero else "positive"}')
        if self.max_ev > 16:
            raise ValueError('max_ev must be within [0,16] to bound exposure gains')
        for name in ('use_3d_lut', 'freeze_backbone'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if not isinstance(self.exposures, (list, tuple)) or not self.exposures:
            raise ValueError('exposures must be a nonempty list or tuple')
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               or abs(v) > 16 for v in self.exposures):
            raise ValueError('exposures must be finite numeric EV values within [-16,16]')
        if len(set(self.exposures)) != len(self.exposures):
            raise ValueError('exposures must be distinct')
        self.exposures = tuple(float(v) for v in self.exposures)
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping):
        if not isinstance(values, Mapping):
            raise TypeError('model config must be a mapping')
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown model config keys: {sorted(unknown)}')
        return cls(**values)
