import numpy as np

from photofinishing.evaluate.metrics import linear_to_srgb
from photofinishing.evaluate.run_evaluation import build_parser, evaluate_output_set


def test_cli_accepts_multi_group_config_and_manifest():
    args = build_parser().parse_args([
        "--config", "experiments.json",
        "--manifest", "manifest.csv",
        "--output-dir", "evaluation",
        "--split", "test",
        "--device", "cpu",
    ])
    assert args.config == "experiments.json"
    assert args.manifest == "manifest.csv"
    assert args.split == "test"
    assert args.panel_limit == 50


def test_evaluate_output_set_returns_all_models_and_reference_noise():
    reference = linear_to_srgb(np.full((16, 16, 3), 0.2, np.float32))
    repeat = linear_to_srgb(np.full((16, 16, 3), 0.21, np.float32))
    pretrained = linear_to_srgb(np.full((16, 16, 3), 0.4, np.float32))
    stage1 = linear_to_srgb(np.full((16, 16, 3), 0.22, np.float32))
    stage2 = stage1.copy()
    stage2[..., 0] = np.clip(stage2[..., 0] - 0.01, 0, 1)
    metrics, noise = evaluate_output_set(
        {"pretrained": pretrained, "stage1": stage1, "stage2/affine": stage2},
        reference,
        reference_repeat=repeat,
    )
    assert set(metrics) == {"pretrained", "stage1", "stage2/affine"}
    assert metrics["stage1"]["absolute_ev_error"] < metrics["pretrained"]["absolute_ev_error"]
    assert noise is not None
    assert noise["absolute_ev_error"] > 0
