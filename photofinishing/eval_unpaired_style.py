"""Compatibility entry point for unpaired style holdout evaluation."""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from photofinishing.unpaired_style.eval_cli import main


if __name__ == "__main__":
  raise SystemExit(main())
