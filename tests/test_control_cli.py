import pytest

from photofinishing.train_brightness_control import CONTROL_METHODS, build_parser


def test_parser_accepts_all_control_methods(tmp_path):
    common = [
        "--input-training-dir", str(tmp_path),
        "--gt-training-root", str(tmp_path),
        "--input-validation-dir", str(tmp_path),
        "--gt-validation-root", str(tmp_path),
        "--baseline-checkpoint", str(tmp_path / "base.pth"),
        "--output-dir", str(tmp_path / "out"),
    ]
    for method in CONTROL_METHODS:
        args = build_parser().parse_args(common + ["--control-method", method])
        assert args.control_method == method


def test_parser_rejects_unknown_method(tmp_path):
    common = [
        "--input-training-dir", str(tmp_path),
        "--gt-training-root", str(tmp_path),
        "--input-validation-dir", str(tmp_path),
        "--gt-validation-root", str(tmp_path),
        "--baseline-checkpoint", str(tmp_path / "base.pth"),
        "--output-dir", str(tmp_path / "out"),
        "--control-method", "unknown",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(common)
