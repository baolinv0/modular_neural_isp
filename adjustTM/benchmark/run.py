from __future__ import annotations

import argparse
import json
import shlex
import hashlib
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a resumable adjustTM benchmark command plan")
    parser.add_argument("--config", required=True, help="JSON containing commands: {stage: [args...] or string}")
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--state", default="benchmark_state.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    commands = config.get("commands", config)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": [], "config_hash": config_hash}
    if state.get("config_hash") != config_hash and not args.force:
        raise RuntimeError("Run-state config hash differs; use a new state file or --force")
    if args.force:
        state = {"completed": [], "config_hash": config_hash}
    for stage in args.stages:
        if stage in state["completed"] and not args.force:
            print(f"skip completed stage: {stage}")
            continue
        if stage not in commands:
            raise KeyError(f"No command configured for stage: {stage}")
        command = commands[stage]
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        subprocess.run(argv, check=True)
        if stage not in state["completed"]:
            state["completed"].append(stage)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
