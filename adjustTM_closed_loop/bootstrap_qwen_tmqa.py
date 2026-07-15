from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def extract_qwen_tmqa_archive(archive: str | Path, destination: str | Path) -> Path:
    archive = Path(archive)
    destination = Path(destination)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.namelist()
        handle.extractall(destination)
    roots = sorted({Path(name).parts[0] for name in members if Path(name).parts})
    candidates = [destination / root for root in roots if (destination / root / "pyproject.toml").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one Qwen-TMQA project root in archive, found {candidates}")
    return candidates[0]


def install_qwen_tmqa_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    python_executable: str = sys.executable,
    extras: str = "dev",
) -> Path:
    project_root = extract_qwen_tmqa_archive(archive, destination)
    target = f"{project_root}[{extras}]" if extras else str(project_root)
    subprocess.run([python_executable, "-m", "pip", "install", "-e", target], check=True)
    return project_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and install the verified Qwen-TMQA source archive")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", default="adjustTM_closed_loop/vendor")
    parser.add_argument("--extras", default="dev")
    args = parser.parse_args()
    root = install_qwen_tmqa_archive(args.archive, args.destination, extras=args.extras)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
