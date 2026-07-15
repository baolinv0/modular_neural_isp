from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from .policy import RidgeAlphaPolicy
from .schemas import TeacherRecord
from .teacher_manifest import load_teacher_manifest


class _Runner(Protocol):
    def predict(self, image: Any, alpha: float) -> Mapping[str, Any]: ...


def _scene_output_path(root: Path, scene_id: str) -> Path:
    path = root / scene_id
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def render_policy_with_runner(
    records: Iterable[TeacherRecord],
    *,
    policy: RidgeAlphaPolicy,
    runner: _Runner,
    output_dir: str | Path,
    read_linear: Callable[..., Any],
    fit_pad: Callable[..., tuple[Any, Mapping[str, Any]]],
    unpad: Callable[[Any, Mapping[str, Any]], Any],
    write_srgb: Callable[[str | Path, Any], None],
    max_side: int | None = 512,
    multiple: int = 16,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Render baseline and automatic-policy outputs with an existing TM runner.

    Dependencies are injectable to keep the policy layer independently testable.
    The production entry point wires these arguments to ``adjustTM.benchmark``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.scene_id):
        prediction = policy.predict_path(record.input_path)
        image = read_linear(record.input_path, device=device)
        padded, geometry = fit_pad(image, max_side=max_side, multiple=multiple)
        if hasattr(runner, "zero_reference"):
            baseline = runner.zero_reference(padded)
        else:
            baseline = runner.predict(padded, 0.0)["output"]
        if prediction.alpha == 0.0:
            # Exact fail-closed behavior: do not invoke a non-zero control path.
            student = baseline
        else:
            result = runner.predict(padded, prediction.alpha)
            if "output" not in result:
                raise KeyError("TM runner result must contain 'output'")
            student = result["output"]
        baseline = unpad(baseline, geometry)
        student = unpad(student, geometry)
        baseline_path = _scene_output_path(output_dir / "baseline", record.scene_id)
        student_path = _scene_output_path(output_dir / "student", record.scene_id)
        write_srgb(baseline_path, baseline)
        write_srgb(student_path, student)
        rows.append({
            "scene_id": record.scene_id,
            "split": record.split,
            "status": record.status,
            "teacher_alpha": record.selected_alpha,
            "predicted_alpha": prediction.alpha,
            "raw_alpha": prediction.raw_alpha,
            "in_domain": prediction.in_domain,
            "domain_distance": prediction.domain_distance,
            "policy_reason": prediction.reason,
            "input_path": record.input_path,
            "baseline_path": str(baseline_path),
            "student_path": str(student_path),
        })
    records_path = output_dir / "render_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def render_policy_outputs(
    *,
    teacher_manifest: str | Path,
    policy_path: str | Path,
    methods_config: str | Path,
    method_name: str,
    output_dir: str | Path,
    device: str = "cpu",
    max_side: int | None = 512,
    multiple: int = 16,
) -> list[dict[str, Any]]:
    """Production integration with the existing controllable TM runner."""

    from adjustTM.benchmark.image_io import (
        fit_pad_tensor,
        read_linear_png16,
        unpad_tensor,
        write_srgb_png16,
    )
    from adjustTM.benchmark.methods import load_runners

    runners = load_runners(methods_config, device)
    if method_name not in runners:
        raise KeyError(f"Method {method_name!r} is not present in {methods_config}")
    return render_policy_with_runner(
        load_teacher_manifest(teacher_manifest),
        policy=RidgeAlphaPolicy.load(policy_path),
        runner=runners[method_name],
        output_dir=output_dir,
        read_linear=read_linear_png16,
        fit_pad=fit_pad_tensor,
        unpad=unpad_tensor,
        write_srgb=write_srgb_png16,
        max_side=max_side,
        multiple=multiple,
        device=device,
    )
