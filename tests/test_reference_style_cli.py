from photofinishing.evaluate_reference_style import build_parser as build_eval_parser
from photofinishing.train_reference_style import build_parser as build_train_parser


def test_training_parser_exposes_two_stage_configuration():
  args = build_train_parser().parse_args([
    "--manifest", "pairs.jsonl",
    "--base-checkpoint", "base.pth",
    "--output-dir", "run",
    "--tone-epochs", "3",
    "--chroma-epochs", "4",
  ])
  assert args.tone_epochs == 3
  assert args.chroma_epochs == 4
  assert not args.skip_chroma


def test_evaluation_parser_requires_three_checkpoints():
  args = build_eval_parser().parse_args([
    "--manifest", "pairs.jsonl",
    "--base-checkpoint", "base.pth",
    "--tone-checkpoint", "tone.pth",
    "--chroma-checkpoint", "chroma.pth",
    "--output-json", "metrics.json",
  ])
  assert args.split == "test"
