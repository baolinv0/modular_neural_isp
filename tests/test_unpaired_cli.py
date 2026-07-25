import pytest

from photofinishing.eval_unpaired_style import build_parser as build_eval_parser
from photofinishing.train_unpaired_style import _validate_arguments, build_parser


def test_cli_requires_explicit_stage_and_checkpoint():
    parser = build_parser()
    args = parser.parse_args([
        "--stage", "luminance",
        "--manifest", "manifest.csv",
        "--load", "source.pth",
        "--output-dir", "run",
    ])
    assert args.stage == "luminance"
    assert args.input_mode == "linear_srgb"


def test_cli_accepts_chroma_stage():
    parser = build_parser()
    args = parser.parse_args([
        "--stage", "chroma",
        "--manifest", "manifest.csv",
        "--load", "stage1.pth",
        "--output-dir", "run",
    ])
    assert args.stage == "chroma"


def test_evaluation_cli_requires_baseline_and_adapted_models():
    args = build_eval_parser().parse_args([
        "--manifest", "manifest.csv",
        "--baseline-load", "source.pth",
        "--adapted-load", "adapted.pth",
        "--output", "metrics.json",
    ])
    assert args.baseline_load == "source.pth"
    assert args.adapted_load == "adapted.pth"


def test_cli_validation_rejects_zero_style_supervision():
    parser = build_parser()
    args = parser.parse_args([
        "--stage", "luminance",
        "--manifest", "m.csv",
        "--load", "m.pth",
        "--output-dir", "run",
        "--exposure-weight", "0",
        "--percentile-weight", "0",
        "--luma-distribution-weight", "0",
    ])
    with pytest.raises(ValueError, match="at least one style loss"):
        _validate_arguments(args)


def test_stage1_provenance_binds_loaded_checkpoint(tmp_path):
    import hashlib
    import json
    from photofinishing.train_unpaired_style import _validate_stage1_provenance

    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    config = tmp_path / "run_config.json"
    config.write_text(json.dumps({
        "stage": "luminance",
        "best_checkpoint_sha256": digest,
        "last_checkpoint_sha256": "0" * 64,
    }), encoding="utf-8")
    assert _validate_stage1_provenance(str(checkpoint), str(config)) == config.resolve()


def test_stage1_provenance_rejects_unbound_checkpoint(tmp_path):
    import json
    from photofinishing.train_unpaired_style import _validate_stage1_provenance

    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"tampered")
    config = tmp_path / "run_config.json"
    config.write_text(json.dumps({
        "stage": "luminance",
        "best_checkpoint_sha256": "0" * 64,
        "last_checkpoint_sha256": "1" * 64,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _validate_stage1_provenance(str(checkpoint), str(config))
