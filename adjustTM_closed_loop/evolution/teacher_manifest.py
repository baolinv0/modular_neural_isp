from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schemas import DistributionStatus, TeacherRecord, TeacherSelection


def _rank(scene_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _holdout_counts(size: int, validation_fraction: float, test_fraction: float) -> tuple[int, int]:
    # Preserve at least one training scene and, when a stratum is large enough,
    # at least one example in each requested holdout split.
    if size < 3:
        return 0, 0
    test_count = int(round(size * test_fraction)) if test_fraction > 0 else 0
    validation_count = int(round(size * validation_fraction)) if validation_fraction > 0 else 0
    if test_fraction > 0:
        test_count = max(1, test_count)
    if validation_fraction > 0:
        validation_count = max(1, validation_count)
    while test_count + validation_count >= size:
        if test_count >= validation_count and test_count > 1:
            test_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            validation_count -= 1
    return validation_count, test_count


def _stratified_splits(
    selections: list[TeacherSelection],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    groups: dict[str, list[TeacherSelection]] = defaultdict(list)
    for selection in selections:
        groups[selection.status].append(selection)
    assignments: dict[str, str] = {}
    for status, items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: (_rank(item.scene_id, seed), item.scene_id))
        validation_count, test_count = _holdout_counts(len(ordered), validation_fraction, test_fraction)
        for index, item in enumerate(ordered):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            assignments[item.scene_id] = split
    return assignments


def build_teacher_manifest(
    selections: Iterable[TeacherSelection], *, input_dir: str | Path, output_path: str | Path,
    split_seed: int = 42, validation_fraction: float = 0.15, test_fraction: float = 0.15,
    validate_output_paths: bool = False,
) -> Path:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum to less than one")
    input_dir, output_path = Path(input_dir), Path(output_path)
    selections = sorted(list(selections), key=lambda item: item.scene_id)
    if not selections:
        raise ValueError("At least one teacher selection is required")
    scene_ids = [selection.scene_id for selection in selections]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Teacher selections contain duplicate scene IDs")
    splits = _stratified_splits(
        selections,
        seed=split_seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    records = []
    for selection in selections:
        input_path = input_dir / selection.scene_id
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if validate_output_paths:
            for candidate_path in (selection.baseline_path, selection.selected_path):
                if not Path(candidate_path).is_file():
                    raise FileNotFoundError(candidate_path)
        records.append(TeacherRecord(
            scene_id=selection.scene_id, input_path=str(input_path), baseline_path=selection.baseline_path,
            target_path=selection.selected_path, selected_alpha=selection.selected_alpha,
            selected_level=selection.selected_level, baseline_score=selection.baseline_score,
            selected_score=selection.selected_score, score_delta=selection.score_delta,
            confidence=selection.confidence, sample_weight=selection.sample_weight,
            status=selection.status, split=splits[selection.scene_id],
            distribution_status=selection.distribution_status, reason=selection.reason,
            metadata={
                "selection": selection.to_dict(),
                **({"teacher_evaluator_id": selection.metadata["teacher_evaluator_id"]}
                   if selection.metadata.get("teacher_evaluator_id") else {}),
            },
        ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def load_teacher_manifest(path: str | Path) -> list[TeacherRecord]:
    path = Path(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records.append(TeacherRecord(
            scene_id=str(item["scene_id"]), input_path=str(item["input_path"]), baseline_path=str(item["baseline_path"]),
            target_path=str(item["target_path"]), selected_alpha=float(item["selected_alpha"]),
            selected_level=str(item["selected_level"]), baseline_score=float(item["baseline_score"]),
            selected_score=float(item["selected_score"]), score_delta=float(item["score_delta"]),
            confidence=float(item["confidence"]), sample_weight=float(item["sample_weight"]), status=str(item["status"]),
            split=str(item["split"]), distribution_status=DistributionStatus.coerce(item.get("distribution_status")),
            reason=str(item["reason"]), metadata=item.get("metadata", {}),
        ))
    return records
