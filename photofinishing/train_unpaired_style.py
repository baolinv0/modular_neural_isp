"""Compatibility entry point for unpaired same-scene style adaptation."""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from photofinishing.unpaired_style.cli import main


if __name__ == "__main__":
  raise SystemExit(main())
