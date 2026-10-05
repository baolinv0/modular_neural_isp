"""Explicit paired manifests and precision-safe, synchronized image loading.

Paths are relative to the manifest. ``input_encoding`` is required and is one
of linear_srgb, srgb, camera_rgb. Targets are aligned sRGB in [0, 1]. Camera
RGB is demosaiced RGB; metadata is a dict or JSON path with cam_illum and ccm.
NPY/NPZ store HWC or CHW floats; NPZ has the single image payload key ``image``.
If both first and last dimensions equal the channel count, specify the field's
``<field>_layout`` as HWC/CHW. Raster RGB is always HWC. uint8/uint16 rasters
are normalized by 255/65535, without dropping precision. Float TIFF is valid.
Semantics are independent probabilities (person/skin/sky), never class IDs.
``semantic_valid`` may be a shared HW/1HW mask or S-channel validity map.
Missing semantics produce zero semantics, confidence, and validity. Labelled
masks default to unit confidence and validity. An inference sample lacking a
target contains a zero target placeholder, never its unaligned reference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ENCODINGS = {'linear_srgb', 'srgb', 'camera_rgb'}
IMAGE_SUFFIXES = {'.npy', '.npz', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= .04045, image / 12.92, ((image + .055) / 1.055) ** 2.4).astype(np.float32)


def camera_to_linear(image: np.ndarray, metadata: dict) -> np.ndarray:
    """Match utils.img_utils.raw_to_lsrgb's NumPy WB+CCM, including clipping."""
    try:
        illum = np.asarray(metadata['cam_illum'], dtype=np.float32)
        ccm = np.asarray(metadata['ccm'], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('camera_rgb metadata requires cam_illum and ccm') from exc
    if illum.shape != (3,) or ccm.shape != (3, 3):
        raise ValueError('camera_rgb metadata cam_illum/ccm shape must be (3,)/(3,3)')
    if not np.isfinite(illum).all() or not np.isfinite(ccm).all() or (illum <= 0).any():
        raise ValueError('camera_rgb metadata must be finite with positive cam_illum')
    matrix = ccm @ np.diag(illum[1] / illum)
    return np.clip(image.reshape(-1, 3) @ matrix.T, 0, 1).reshape(image.shape).astype(np.float32)


def _read_array(path: Path) -> tuple[np.ndarray, bool]:
    suffix = path.suffix.lower()
    if suffix == '.npy':
        return np.load(path, allow_pickle=False), False
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if 'image' not in archive.files:
                raise ValueError(f'{path}: NPZ requires fixed key image')
            return archive['image'], False
    if suffix in {'.tif', '.tiff'}:
        import tifffile
        return tifffile.imread(path), True
    if suffix in IMAGE_SUFFIXES:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'{path}: cannot decode image')
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image[..., ::-1]  # OpenCV BGR -> RGB, preserving uint16.
        return image, True
    raise ValueError(f'{path}: unsupported image extension {suffix}')


def _normalized(array: np.ndarray, name: str) -> np.ndarray:
    if array.dtype in (np.dtype('uint8'), np.dtype('uint16')):
        array = array.astype(np.float32) / np.iinfo(array.dtype).max
    elif np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        array = array.astype(np.float32)
    else:
        raise ValueError(f'{name}: numeric pixels required')
    if not np.isfinite(array).all():
        raise ValueError(f'{name}: pixels must be finite')
    if array.size == 0 or (array < 0).any() or (array > 1).any():
        raise ValueError(f'{name}: pixels outside normalized [0,1] range')
    return array


def _hwc(array: np.ndarray, channels: int, layout: str | None, raster: bool, name: str,
         allow_single: bool = False) -> np.ndarray:
    if layout not in (None, 'HWC', 'CHW'):
        raise ValueError(f'{name}: layout must be HWC or CHW')
    permitted = {channels, 1} if allow_single else {channels}
    if array.ndim == 2 and (channels == 1 or allow_single):
        array = array[..., None]
    elif array.ndim != 3:
        raise ValueError(f'{name}: expected RGB/probability channels, got shape {array.shape}')
    else:
        first, last = array.shape[0] in permitted, array.shape[-1] in permitted
        if raster:
            if layout == 'CHW':
                raise ValueError(f'{name}: raster layout is HWC')
        elif layout is None:
            if first and last:
                raise ValueError(f'{name}: ambiguous shape {array.shape}; specify layout')
            if first:
                array = array.transpose(1, 2, 0)
        elif layout == 'CHW':
            array = array.transpose(1, 2, 0)
    if array.shape[-1] not in permitted or min(array.shape[:2]) == 0:
        raise ValueError(f'{name}: wrong channels/shape {array.shape}; expected {channels}')
    return array


class PairedImageDataset(Dataset):
    def __init__(self, manifest, image_size=None, augment=False, semantic_channels=3, require_target=True):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.image_size = image_size
        self.augment = bool(augment)
        self.semantic_channels = semantic_channels
        self.require_target = bool(require_target)
        if image_size is not None and (not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1):
            raise ValueError('image_size must be a positive integer or None')
        if not isinstance(semantic_channels, int) or isinstance(semantic_channels, bool) or semantic_channels < 1:
            raise ValueError('semantic_channels must be a positive integer')
        text = self.manifest.read_text()
        if self.manifest.suffix.lower() == '.json':
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(records, list) or not records:
            raise ValueError('manifest must contain a nonempty JSON list or JSONL rows')
        self.records: list[dict[str, Any]] = records
        ids = set()
        for row in self.records:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
                raise ValueError('each manifest row requires a nonempty string id')
            if row['id'] in ids:
                raise ValueError(f'duplicate id: {row["id"]}')
            ids.add(row['id'])
            if row.get('input_encoding') not in ENCODINGS:
                raise ValueError(f'{row["id"]}: input_encoding must explicitly name {sorted(ENCODINGS)}')
            if not row.get('input'):
                raise ValueError(f'{row["id"]}: input required')
            if require_target and not row.get('target'):
                raise ValueError(f'{row["id"]}: aligned target required; reference is not a target')
            if row.get('target') and row.get('target_aligned') is not True:
                raise ValueError(f'{row["id"]}: target_aligned:true is required')
            if row.get('target_encoding', 'srgb') != 'srgb':
                raise ValueError(f'{row["id"]}: target_encoding must be srgb')
            for field in ('input', 'target', 'semantics', 'confidence', 'semantic_valid'):
                if row.get(field) is not None:
                    self._path(row[field], field)
            if row.get('metadata') is not None and not isinstance(row['metadata'], dict):
                self._path(row['metadata'], 'metadata')

    def _path(self, value, name):
        if not isinstance(value, (str, Path)):
            raise ValueError(f'{name}: expected file path')
        path = self.root / value
        if not path.is_file():
            raise FileNotFoundError(f'{name}: missing file {path}')
        return path

    def _load(self, row, field, channels, allow_single=False):
        array, raster = _read_array(self._path(row[field], field))
        array = _hwc(array, channels, row.get(f'{field}_layout'), raster, field, allow_single)
        return _normalized(array, field)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = self._load(row, 'input', 3)
        if row['input_encoding'] == 'srgb':
            image = srgb_to_linear(image)
        elif row['input_encoding'] == 'camera_rgb':
            metadata = row.get('metadata')
            if not isinstance(metadata, dict):
                if metadata is None:
                    raise ValueError('camera_rgb requires metadata with cam_illum and ccm')
                metadata = json.loads(self._path(metadata, 'metadata').read_text())
            image = camera_to_linear(image, metadata)
        height, width = image.shape[:2]
        target = self._load(row, 'target', 3) if row.get('target') else np.zeros_like(image)
        if row.get('semantics'):
            semantics = self._load(row, 'semantics', self.semantic_channels)
            confidence = (self._load(row, 'confidence', 1) if row.get('confidence')
                          else np.ones((height, width, 1), np.float32))
            valid = (self._load(row, 'semantic_valid', self.semantic_channels, allow_single=True)
                     if row.get('semantic_valid') else np.ones_like(semantics))
            if valid.shape[-1] == 1:
                valid = np.repeat(valid, self.semantic_channels, axis=-1)
        else:
            # Validate any supplied optional files even when no annotation is
            # available. They cannot provide semantic conditioning by themselves.
            for field, channels in [('confidence', 1), ('semantic_valid', self.semantic_channels)]:
                if row.get(field):
                    supplied = self._load(row, field, channels, allow_single=(field == 'semantic_valid'))
                    if supplied.shape[:2] != (height, width):
                        raise ValueError(f'{row["id"]}: {field} shape does not match input')
            semantics = np.zeros((height, width, self.semantic_channels), np.float32)
            confidence = np.zeros((height, width, 1), np.float32)
            valid = np.zeros_like(semantics)
        arrays = {'input':image, 'target':target, 'semantics':semantics,
                  'confidence':confidence, 'semantic_valid':valid}
        for name, array in arrays.items():
            if array.shape[:2] != (height, width):
                raise ValueError(f'{row["id"]}: {name} shape {array.shape[:2]} does not match input {(height,width)}')
        tensors = {key:torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1))).float()
                   for key,array in arrays.items()}
        if self.image_size is not None:
            for key, tensor in tensors.items():
                mode = 'nearest' if key == 'semantic_valid' else 'bilinear'
                kwargs = {} if mode == 'nearest' else {'align_corners':False}
                tensors[key] = F.interpolate(tensor[None], size=(self.image_size,self.image_size),
                                             mode=mode, **kwargs)[0]
        if self.augment:
            flip_h = bool(torch.rand(()) < .5)
            flip_v = bool(torch.rand(()) < .5)
            # Independent quarter turns would swap rectangular H/W and break
            # collation even when all source samples have identical shapes.
            # Square images retain all rotations; rectangles retain 0/180.
            square = tensors['input'].shape[-2] == tensors['input'].shape[-1]
            rotation = int(torch.randint(0, 4 if square else 2, ()).item())
            if not square:
                rotation *= 2
            for key, tensor in tensors.items():
                if flip_h:
                    tensor = tensor.flip(-1)
                if flip_v:
                    tensor = tensor.flip(-2)
                tensors[key] = torch.rot90(tensor, rotation, (-2,-1)).contiguous()
        tensors.update(id=row['id'], scene=str(row.get('scene','unknown')), camera=str(row.get('camera','unknown')))
        return tensors
