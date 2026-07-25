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
