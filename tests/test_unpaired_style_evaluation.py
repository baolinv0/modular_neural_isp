import torch

from photofinishing.unpaired_style.eval_cli import build_parser
from photofinishing.unpaired_style.evaluation import evaluate_batch


def test_adapted_style_improvement_is_positive_for_closer_output():
  baseline = torch.full((2, 3, 16, 16), 0.2)
  adapted = torch.full((2, 3, 16, 16), 0.5)
  reference = torch.full((2, 3, 16, 16), 0.6)
  metrics = evaluate_batch(baseline, adapted, reference)
  assert torch.all(metrics["luminance_style_improvement"] > 0)
  assert torch.isfinite(metrics["content_edge_drift"]).all()
  assert torch.all(metrics["adapted_highlight_clip"] <= metrics["baseline_highlight_clip"])


def test_evaluation_cli_requires_source_and_adapted_checkpoints():
  args = build_parser().parse_args([
    "--manifest", "test.jsonl",
    "--source-checkpoint", "source.pth",
    "--adapted-checkpoint", "adapted.pth",
    "--output", "report.json",
  ])
  assert args.source_checkpoint == "source.pth"
  assert args.adapted_checkpoint == "adapted.pth"
