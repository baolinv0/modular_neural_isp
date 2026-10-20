"""JSONL dataset for same-scene unaligned source/reference pairs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_rgb(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class UnalignedReferenceDataset(Dataset):
    """Rows: {sample_id, source, reference}. Images may have different sizes."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.rows = []
        seen = set()
        for line_number, line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("sample_id", "source", "reference"):
                if not row.get(field):
                    raise ValueError(f"line {line_number}: missing {field}")
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
            for field in ("source", "reference"):
                if not Path(row[field]).is_file():
                    raise FileNotFoundError(row[field])
            self.rows.append(row)
        if not self.rows:
            raise ValueError("empty manifest")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "source": _load_rgb(row["source"]),
            "reference": _load_rgb(row["reference"]),
        }


def unaligned_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("unaligned reference training currently requires batch_size=1")
    item = batch[0]
    return {
        "sample_id": item["sample_id"],
        "source": item["source"].unsqueeze(0),
        "reference": item["reference"].unsqueeze(0),
    }
