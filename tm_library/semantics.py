"""Explicit local, frozen pretrained segmentation adapters.

The provider is external to TM checkpoints/optimizers. ``weights`` is a local
reference; ``to_dict`` records preprocessing, mapping and source policy without
embedding model weights. Inputs are normalized linear sRGB, independent of GT.
Factory loading imports an explicitly named local Python module, constructs its
architecture, and loads tensor weights strictly with ``weights_only=True``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import importlib
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SegmentationConfig:
    backend: str = 'none'
    weights: str | None = None
    factory: str | None = None
    factory_kwargs: dict = field(default_factory=dict)
    state_dict_key: str | None = None
    source: str = 'manifest'
    input_encoding: str = 'linear_srgb'
    input_size: tuple[int, int] | None = None
    mean: tuple[float, float, float] = (0., 0., 0.)
    std: tuple[float, float, float] = (1., 1., 1.)
    output_key: str | None = None
    output_index: int | None = None
    confidence_key: str | None = None
    confidence_index: int | None = None
    output_mode: str = 'softmax_logits'
    class_groups: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self):
        if self.backend not in ('none', 'torchscript', 'factory'):
            raise ValueError('segmentation backend must be none, torchscript or factory')
        if self.source not in ('manifest', 'model', 'prefer_manifest'):
            raise ValueError('segmentation source must be manifest, model or prefer_manifest')
        if self.input_encoding not in ('linear_srgb', 'srgb'):
            raise ValueError('segmentation input_encoding must be linear_srgb or srgb')
        if self.output_mode not in ('softmax_logits', 'sigmoid_logits', 'probabilities'):
            raise ValueError('output_mode must be softmax_logits, sigmoid_logits or probabilities')
        if self.input_size is not None:
            if not isinstance(self.input_size, (list, tuple)) or len(self.input_size) != 2 or any(type(v) is not int or v < 1 for v in self.input_size):
                raise ValueError('input_size must contain positive integer height and width')
            self.input_size = tuple(self.input_size)
        for name in ('mean', 'std'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError(f'{name} must contain three finite numeric values')
            if name == 'std' and any(v <= 0 for v in values):
                raise ValueError('std values must be positive')
            setattr(self, name, tuple(float(v) for v in values))
        for key_name, index_name in (('output_key', 'output_index'), ('confidence_key', 'confidence_index')):
            key, index = getattr(self, key_name), getattr(self, index_name)
            if key is not None and (not isinstance(key, str) or not key):
                raise ValueError(f'{key_name} must be a nonempty string')
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError(f'{index_name} must be a nonnegative integer')
            if key is not None and index is not None:
                raise ValueError(f'choose {key_name} or {index_name}, not both')
        if not isinstance(self.factory_kwargs, Mapping):
            raise TypeError('factory_kwargs must be a mapping')
        self.factory_kwargs = dict(self.factory_kwargs)
        for name in ('weights', 'factory', 'state_dict_key'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f'{name} must be a nonempty string')
        if self.factory is not None and (self.factory.count(':') != 1 or not all(self.factory.split(':'))):
            raise ValueError('factory must explicitly name a local module:function')
        if self.class_groups is not None:
            if not isinstance(self.class_groups, (list, tuple)) or not self.class_groups:
                raise ValueError('class_groups must be a nonempty ordered list of channel groups')
            groups = []
            for group in self.class_groups:
                if not isinstance(group, (list, tuple)) or not group or any(type(v) is not int or v < 0 for v in group) or len(set(group)) != len(group):
                    raise ValueError('each class group must contain distinct nonnegative integer channels')
                groups.append(tuple(group))
            self.class_groups = tuple(groups)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping):
        if not isinstance(values, Mapping):
            raise TypeError('segmentation config must be a mapping')
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown segmentation config keys: {sorted(unknown)}')
        return cls(**values)


class SemanticProvider:
    """Frozen inference with explicit class mapping and external weights.

    Without a configured confidence output, confidence is the maximum original
    model-channel probability. This is an uncalibrated heuristic: independent
    sigmoid channels all near zero receive low trust, including background.
    Supply ``confidence_key``/``confidence_index`` for model-specific trust.
    """
    def __init__(self, config: SegmentationConfig | Mapping, semantic_channels: int = 3, device='cpu'):
        self.config = config if isinstance(config, SegmentationConfig) else SegmentationConfig.from_dict(config)
        if type(semantic_channels) is not int or semantic_channels < 1:
            raise ValueError('semantic_channels must be a positive integer')
        self.semantic_channels = semantic_channels
        self.device = torch.device(device)
        self.model = None
        if self.config.class_groups is not None and len(self.config.class_groups) != semantic_channels:
            raise ValueError('class_groups must map exactly semantic_channels ordered TM channels')
        if self.config.backend == 'none' or self.config.source == 'manifest':
            return
        if self.config.class_groups is None:
            raise ValueError('enabled model segmentation requires explicit class_groups')
        if self.config.weights is None:
            raise ValueError('enabled model segmentation requires a local weights path')
        path = Path(self.config.weights)
        if not path.is_file():
            raise FileNotFoundError(f'segmentation weights do not exist: {path}')
        if self.config.backend == 'torchscript':
            self.model = torch.jit.load(str(path), map_location=self.device)
        else:
            if self.config.factory is None:
                raise ValueError('factory backend requires explicit module:function')
            module, function = self.config.factory.split(':')
            constructor = getattr(importlib.import_module(module), function)
            self.model = constructor(**self.config.factory_kwargs)
            if not isinstance(self.model, nn.Module):
                raise TypeError('segmentation factory must return torch.nn.Module')
            state = torch.load(path, map_location='cpu', weights_only=True)
            if self.config.state_dict_key is not None:
                if not isinstance(state, Mapping) or self.config.state_dict_key not in state:
                    raise ValueError(f'state dict missing key {self.config.state_dict_key!r}')
                state = state[self.config.state_dict_key]
            if not isinstance(state, Mapping) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
                raise ValueError('segmentation checkpoint must contain a tensor state_dict')
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def enabled(self):
        return self.model is not None

    def preprocess(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3 or min(image.shape[0], *image.shape[-2:]) < 1 or not image.is_floating_point():
            raise ValueError('segmentation input must be floating B3HW normalized linear sRGB')
        _probability(image, 'segmentation input')
        value = image.detach().to(device=self.device, dtype=torch.float32)
        if self.config.input_encoding == 'srgb':
            value = torch.where(value <= .0031308, 12.92 * value, 1.055 * value.clamp_min(.0031308).pow(1/2.4) - .055)
        if self.config.input_size is not None and value.shape[-2:] != self.config.input_size:
            value = F.interpolate(value, size=self.config.input_size, mode='bilinear', align_corners=False, antialias=True)
        mean = value.new_tensor(self.config.mean).view(1,3,1,1)
        std = value.new_tensor(self.config.std).view(1,3,1,1)
        return (value - mean) / std

    @staticmethod
    def _select(result, key, index, name):
        if key is not None:
            if not isinstance(result, Mapping) or key not in result:
                raise ValueError(f'segmentation {name} requires configured dict key {key!r}')
            result = result[key]
        elif index is not None:
            if not isinstance(result, (list, tuple)) or index >= len(result):
                raise ValueError(f'segmentation {name} requires configured tuple index {index}')
            result = result[index]
        if not isinstance(result, torch.Tensor):
            raise ValueError(f'segmentation {name} must be tensor; configure its dict key or tuple index')
        return result

    @torch.no_grad()
    def __call__(self, image):
        if not self.enabled:
            raise ValueError('segmentation model backend is disabled or source is manifest')
        self.model.eval()
        raw = self.model(self.preprocess(image))
        value = self._select(raw, self.config.output_key, self.config.output_index, 'output')
        if value.ndim != 4 or value.shape[0] != image.shape[0] or min(value.shape[1:]) < 1 or not value.is_floating_point() or not value.isfinite().all():
            raise ValueError('segmentation output must be finite floating BCHW with matching batch')
        if self.config.output_mode == 'softmax_logits':
            probabilities = value.softmax(1)
        elif self.config.output_mode == 'sigmoid_logits':
            probabilities = value.sigmoid()
        else:
            _probability(value, 'segmentation output probabilities')
            probabilities = value
        if max(max(group) for group in self.config.class_groups) >= probabilities.shape[1]:
            raise ValueError('class_groups index exceeds segmentation output channels')
        # Softmax groups are exclusive-class unions. Independent sigmoid/probability
        # channels use the explicitly documented capped sum, including overlap.
        masks = torch.cat([probabilities[:,group].sum(1, keepdim=True).clamp(max=1)
                           for group in self.config.class_groups], 1)
        if self.config.confidence_key is not None or self.config.confidence_index is not None:
            confidence = self._select(raw, self.config.confidence_key, self.config.confidence_index, 'confidence')
            if confidence.ndim != 4 or confidence.shape[0] != image.shape[0] or confidence.shape[1] not in (1,self.semantic_channels) or confidence.shape[-2:] != masks.shape[-2:]:
                raise ValueError('segmentation confidence must have aligned B1HW or TM-channel BCHW shape')
            _probability(confidence, 'segmentation confidence')
        else:
            confidence = probabilities.max(1, keepdim=True).values
        return {'semantics':masks.to(image).detach(), 'confidence':confidence.to(image).detach()}


def _probability(value, name):
    if not value.isfinite().all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f'{name} must be finite probabilities in [0,1]')


def _align(value, image, channels, name, nearest=False):
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != image.shape[0] or value.shape[1] not in channels or min(value.shape[-2:]) < 1:
        raise ValueError(f'{name} must have matching batch and channels {channels}')
    _probability(value, name)
    value = value.to(image)
    if value.shape[-2:] != image.shape[-2:]:
        kwargs = {} if nearest else {'align_corners':False}
        value = F.interpolate(value, size=image.shape[-2:], mode='nearest' if nearest else 'bilinear', **kwargs)
    return value


def apply_semantic_provider(batch: dict, provider: SemanticProvider | None, semantic_mode: str, training: bool):
    """Choose annotations/runtime masks without requiring a target image.

    S0 and S1 inference do not execute the provider. S1 training and S2 align
    masks to input resolution for existing loss/metric APIs. Annotation validity
    is availability, independent of runtime confidence. ``semantic_supervision_weight``
    retains manual validity, and weights pseudo-label BCE by model confidence.
    Partial per-channel manual annotations use per-channel runtime confidence.
    ``backend=none`` explicitly disables model sourcing and retains annotations.
    """
    if semantic_mode not in ('none', 'train_only', 'explicit'):
        raise ValueError('semantic_mode must be none, train_only or explicit')
    result = dict(batch)
    if semantic_mode == 'none' or (semantic_mode == 'train_only' and not training):
        return result
    image = batch['input']
    channels = provider.semantic_channels if provider is not None else (batch['semantics'].shape[1] if 'semantics' in batch else 3)
    zero = image.new_zeros((image.shape[0],channels,*image.shape[-2:]))
    labels = _align(batch.get('semantics',zero), image, (channels,), 'semantics')
    default_valid = torch.ones_like(zero) if 'semantics' in batch else zero
    valid = _align(batch.get('semantic_valid',default_valid), image, (1,channels), 'semantic_valid', nearest=True).expand_as(labels)
    default_confidence = torch.ones_like(zero[:,:1]) if 'semantics' in batch else zero[:,:1]
    confidence = _align(batch.get('confidence',default_confidence), image, (1,channels), 'confidence')
    source = provider.config.source if provider is not None and provider.enabled else 'manifest'
    if source == 'manifest' or (source == 'prefer_manifest' and bool((valid > 0).all())):
        result.update(semantics=labels, confidence=confidence, semantic_valid=valid,
                      semantic_supervision_weight=valid)
        return result
    predicted = provider(image)
    pseudo = _align(predicted['semantics'], image, (channels,), 'model semantics')
    trust = _align(predicted['confidence'], image, (1,channels), 'model confidence')
    if source == 'model':
        result.update(semantics=pseudo, confidence=trust, semantic_valid=torch.ones_like(pseudo),
                      semantic_supervision_weight=trust.expand_as(pseudo))
    else:
        manual = valid > 0
        result.update(semantics=torch.where(manual,labels,pseudo),
                      confidence=torch.where(manual,confidence.expand_as(labels),trust.expand_as(pseudo)),
                      semantic_valid=torch.where(manual,valid,torch.ones_like(pseudo)),
                      semantic_supervision_weight=torch.where(manual,valid,trust.expand_as(pseudo)))
    return result
