#!/usr/bin/env python3
"""Small, model-agnostic command recorder. Not an agent or GPU scheduler."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def validate_task(task: dict) -> None:
    if not isinstance(task, dict) or not re.fullmatch(r'[A-Za-z0-9_-]+', str(task.get('id', ''))):
        raise ValueError('task.id must contain only letters, numbers, underscore or hyphen')
    budget = task.get('timeout_seconds')
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('timeout_seconds must be positive and finite')
    if not isinstance(task.get('steps'), list) or not task['steps']:
        raise ValueError('steps must be a nonempty list')
    for step in task['steps']:
        if not isinstance(step, dict) or not step.get('name'):
            raise ValueError('each step needs a name')
        argv = step.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError('argv must be a nonempty string array; shell strings are unsupported')


def git_context(root: Path) -> dict:
    def get(*args):
        try:
            p = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=5)
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    status = get('status', '--porcelain')
    return {'commit': get('rev-parse', 'HEAD'), 'branch': get('branch', '--show-current'),
            'dirty': None if status is None else bool(status),
            'changed_paths': [] if status is None else status.splitlines()}


def stop_process(proc: subprocess.Popen) -> None:
    # Stop subprocess groups too: training children must not survive a timeout.
    if os.name == 'posix':
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait()


def run_task(task: dict, repo_root: Path, run_dir: Path) -> dict:
    validate_task(task)
    repo_root, run_dir = Path(repo_root).resolve(), Path(run_dir).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'task.json', task)
    started = time.monotonic()
    record = {'task_id': task['id'], 'started_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'git': git_context(repo_root), 'steps': [],
              'scientific_claim_verified': False,
              'meaning': 'Process completion is not CUDA training, data validity, or scientific success.'}
    write_json(run_dir / 'run.json', record)
    substitutions = {'{python}': sys.executable, '{repo}': str(repo_root), '{run_dir}': str(run_dir)}
    try:
        for i, step in enumerate(task['steps']):
            remaining = float(task['timeout_seconds']) - (time.monotonic() - started)
            if remaining <= 0:
                record['status'] = 'timed_out'
                break
            argv = list(step['argv'])
            for j, arg in enumerate(argv):
                for key, value in substitutions.items():
                    arg = arg.replace(key, value)
                argv[j] = arg
            logname = f'{i:02d}.log'
            row = {'name': step['name'], 'argv': argv, 'log': logname}
            t0, proc = time.monotonic(), None
            with (run_dir / logname).open('w', encoding='utf-8') as logfile:
                try:
                    proc = subprocess.Popen(argv, cwd=repo_root, stdout=logfile, stderr=subprocess.STDOUT,
                                            start_new_session=(os.name == 'posix'))
                    row['returncode'] = proc.wait(timeout=remaining)
                    row['status'] = 'completed' if row['returncode'] == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    stop_process(proc)
                    row.update(status='timed_out', returncode=proc.returncode)
                except OSError as error:
                    logfile.write(f'{type(error).__name__}: {error}\n')
                    row.update(status='failed', returncode=127, error_type=type(error).__name__)
                except KeyboardInterrupt:
                    if proc is not None:
                        stop_process(proc)
                    row.update(status='interrupted', returncode=None if proc is None else proc.returncode)
                row['seconds'] = round(time.monotonic() - t0, 4)
            record['steps'].append(row)
            write_json(run_dir / 'run.json', record)
            if row['status'] != 'completed':
                record['status'] = row['status']
                break
        else:
            record['status'] = 'completed'
    finally:
        if record['status'] == 'running':
            record['status'] = 'failed'
        record['seconds'] = round(time.monotonic() - started, 4)
        write_json(run_dir / 'run.json', record)
    return record


def make_packet(run_dir: Path, output: Path) -> dict:
    run_dir, output = Path(run_dir), Path(output)
    record = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    # Deliberately omit command arguments, logs, absolute paths, and environment variables.
    packet = {'task_id': record['task_id'], 'status': record['status'],
              'seconds': record['seconds'], 'code_commit': record['git']['commit'],
              'code_dirty': record['git']['dirty'], 'scientific_claim_verified': False,
              'steps': [{k: row.get(k) for k in ('name','status','returncode','seconds')} for row in record['steps']]}
    source = run_dir / 'preflight.json'
    if source.exists():
        preflight = json.loads(source.read_text(encoding='utf-8'))
        packet['preflight'] = {k: preflight.get(k) for k in ('source','runtime','s24','readiness')}
    write_json(output / 'summary.json', packet)
    rows = '\n'.join(f"| {x['name']} | {x['status']} | {x['returncode']} | {x['seconds']} |" for x in packet['steps'])
    text = f"""# GPU端回传：{packet['task_id']}

状态：**{packet['status']}**；耗时：{packet['seconds']} s。
代码提交：`{packet['code_commit']}`；执行时有未提交文件：`{packet['code_dirty']}`。

| 步骤 | 执行状态 | 退出码 | 秒 |
|---|---|---:|---:|
{rows}

## 证据边界

本文件由记录器生成，仅陈述进程记录。AE–TM画质收益、真实S24训练、CUDA训练能力：**未验证**。
预检详情见 `summary.json`。即使退出码为0，也必须检查readiness/data的blocked字段。
原始完整日志只留GPU端；需人工审查后附最小错误片段。没有自动上传日志、数据或密钥。

## 执行者补充（提交前填写）

- 事实：哪些命令运行了，哪些没有；结果文件的相对位置。
- 解释：哪些是假设，哪些证据支持；无数据时不要补数字。
- 变更：代码提交/PR；是否修改了研究合同、目标或评价方式。
- 阻碍：最小复现、已尝试修复、需要决策者回答的一个问题。
- 下一步建议：最小区分实验，而不是直接启动全量训练。

## 决策者下一轮

先核对 `START_HERE.md`、任务卡、提交diff、summary与必要日志，再决定继续/修复/缩题。
"""
    (output / 'reply.md').write_text(text, encoding='utf-8')
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run')
    run.add_argument('task', type=Path)
    run.add_argument('--repo-root', type=Path, default=Path('.'))
    run.add_argument('--output', type=Path)
    packet = sub.add_parser('packet')
    packet.add_argument('run_dir', type=Path)
    packet.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.cmd == 'packet':
            make_packet(args.run_dir, args.output)
            print(args.output)
            return 0
        task = json.loads(args.task.read_text(encoding='utf-8'))
        validate_task(task)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
        output = args.output or args.repo_root / 'runs/ae_tm_bridge' / f"{task['id']}-{stamp}"
        result = run_task(task, args.repo_root, output)
        print(json.dumps({'status':result['status'], 'run_dir':str(output)}, ensure_ascii=False))
        return 0 if result['status'] == 'completed' else 2
    except (OSError, ValueError, KeyError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
