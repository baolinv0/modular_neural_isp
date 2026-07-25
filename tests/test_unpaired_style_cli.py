from photofinishing.unpaired_style.cli import build_parser


def test_cli_requires_explicit_manifests_and_checkpoint():
  args = build_parser().parse_args([
    "--train-manifest", "train.jsonl",
    "--validation-manifest", "val.jsonl",
    "--checkpoint", "model.pth",
    "--output-dir", "run",
    "--dry-run",
  ])
  assert args.dry_run is True
  assert args.batch_size == 4
