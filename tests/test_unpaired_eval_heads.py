import json

import pytest

from photofinishing.eval_unpaired_style import _read_adapted_run_config, build_parser
from photofinishing.unpaired_chroma_heads import ChromaHead


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
    checkpoint.write_bytes(b"checkpoint")
    config, path = _read_adapted_run_config(str(checkpoint), None)
    assert path is None
    assert config["chroma_head"] == ChromaHead.FULL_LUT.value


def test_affine_adapted_run_config_restores_head_limits(tmp_path):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "affine_residual",
        "affine_matrix_limit": 0.15,
        "affine_bias_limit": 0.05,
    }), encoding="utf-8")
    config, path = _read_adapted_run_config(str(checkpoint), None)
    assert path == run_config.resolve()
    assert config["chroma_head"] == ChromaHead.AFFINE_RESIDUAL.value
    assert config["affine_matrix_limit"] == 0.15
    assert config["affine_bias_limit"] == 0.05


def test_affine_run_config_requires_positive_limits(tmp_path):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "affine_residual",
        "affine_matrix_limit": 0.0,
        "affine_bias_limit": 0.05,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="positive affine limits"):
        _read_adapted_run_config(str(checkpoint), None)


def test_invalid_chroma_head_in_run_config_fails_closed(tmp_path):
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "stage": "chroma",
        "chroma_head": "unknown",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown chroma_head"):
        _read_adapted_run_config(str(checkpoint), None)
