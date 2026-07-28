import hashlib
import json

import pytest
import torch

from photofinishing.eval_unpaired_style import _read_adapted_run_config, build_parser
from photofinishing.unpaired_chroma_heads import ChromaHead


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_affine_checkpoint(path):
    torch.save({
        "_lut_net.matrix_raw": torch.zeros(2, 2),
        "_lut_net.bias_raw": torch.zeros(2),
        "_lut_net.base_lut_net._base_lut": torch.zeros(1, 2, 4, 4),
    }, path)


def test_eval_cli_accepts_explicit_adapted_run_config():
    args = build_parser().parse_args([
        "--manifest", "manifest.csv",
        "--baseline-load", "baseline.pth",
        "--adapted-load", "adapted.pth",
        "--adapted-run-config", "run_config.json",
        "--output", "metrics.json",
    ])
    assert args.adapted_run_config == "run_config.json"


def test_missing_adapted_run_config_preserves_legacy_full_lut(tmp_path):
    checkpoint = tmp_path / "best.pth"
    torch.save({"_lut_net._base_lut": torch.zeros(1, 2, 4, 4)}, checkpoint)
    config, path = _read_adapted_run_config(str(checkpoint), None)
    assert path is None
    assert config["chroma_head"] == ChromaHead.FULL_LUT.value


def test_missing_adapted_run_config_rejects_affine_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pth"
    _write_affine_checkpoint(checkpoint)
    with pytest.raises(FileNotFoundError, match="affine checkpoint requires"):
        _read_adapted_run_config(str(checkpoint), None)


def test_affine_adapted_run_config_restores_head_limits_and_binds_hash(tmp_path):
    checkpoint = tmp_path / "best.pth"
    _write_affine_checkpoint(checkpoint)
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "affine_residual",
        "affine_matrix_limit": 0.15,
        "affine_bias_limit": 0.05,
        "best_checkpoint_sha256": _sha256(checkpoint),
        "last_checkpoint_sha256": "0" * 64,
    }), encoding="utf-8")
    config, path = _read_adapted_run_config(str(checkpoint), None)
    assert path == run_config.resolve()
    assert config["chroma_head"] == ChromaHead.AFFINE_RESIDUAL.value
    assert config["affine_matrix_limit"] == 0.15
    assert config["affine_bias_limit"] == 0.05


def test_affine_run_config_requires_positive_limits(tmp_path):
    checkpoint = tmp_path / "best.pth"
    _write_affine_checkpoint(checkpoint)
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "affine_residual",
        "affine_matrix_limit": 0.0,
        "affine_bias_limit": 0.05,
        "best_checkpoint_sha256": _sha256(checkpoint),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="positive affine limits"):
        _read_adapted_run_config(str(checkpoint), None)


def test_affine_run_config_rejects_unbound_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pth"
    _write_affine_checkpoint(checkpoint)
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "affine_residual",
        "affine_matrix_limit": 0.15,
        "affine_bias_limit": 0.05,
        "best_checkpoint_sha256": "0" * 64,
        "last_checkpoint_sha256": "1" * 64,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _read_adapted_run_config(str(checkpoint), None)


def test_invalid_chroma_head_in_run_config_fails_closed(tmp_path):
    checkpoint = tmp_path / "best.pth"
    torch.save({"_lut_net._base_lut": torch.zeros(1, 2, 4, 4)}, checkpoint)
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "unknown",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown chroma_head"):
        _read_adapted_run_config(str(checkpoint), None)
