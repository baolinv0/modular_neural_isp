from __future__ import annotations

from dataclasses import dataclass, asdict
import base64
import io
import json
from urllib import request
import numpy as np
from PIL import Image

from .config import VLMConfig
from .segmentation import SemanticMasks

FAILURE_LABELS = {
    "FACE_TOO_FLAT",
    "OVER_HDR_LOOK",
    "UNNATURAL_FACE_LIFT",
    "LIGHTING_STRUCTURE_BROKEN",
    "UNNATURAL_TONE",
    "SEVERE_TM_ARTIFACT",
}


@dataclass
class VLMReview:
    decision: str
    failures: list[str]
    confidence: float
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + data


def make_overlay(rgb: np.ndarray, masks: SemanticMasks) -> Image.Image:
    base = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    overlay = base.astype(np.float32)
    alpha = 0.35
    colors = [
        (masks.background, (40, 80, 220)),
        (masks.face, (230, 180, 30)),
        (masks.skin, (230, 70, 70)),
    ]
    for mask, color in colors:
        c = np.asarray(color, dtype=np.float32)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * c
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def build_prompt(evidence: dict) -> str:
    return f"""You are reviewing front-camera portrait Tone Mapping only.
The first image is the candidate. The second image is a semantic overlay: red=skin, yellow=face, blue=background.
Use the numeric evidence as support, not as a rubric to mechanically restate.
Judge only these perceptual TM failures: FACE_TOO_FLAT, OVER_HDR_LOOK, UNNATURAL_FACE_LIFT, LIGHTING_STRUCTURE_BROKEN, UNNATURAL_TONE, SEVERE_TM_ARTIFACT.
Ignore focus, noise, texture, beautification, identity, and bokeh.
Return one JSON object only:
{{"decision":"ACCEPT|REVIEW|REJECT","failures":[],"confidence":0.0,"summary":"short reason"}}
Use REJECT only for an obvious severe TM failure. REVIEW is for noticeable but non-catastrophic issues.
Evidence: {json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}
"""


def parse_review(text: str) -> VLMReview:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen response does not contain JSON")
    obj = json.loads(text[start:end + 1])
    decision = str(obj.get("decision", "REVIEW")).upper()
    if decision not in {"ACCEPT", "REVIEW", "REJECT"}:
        decision = "REVIEW"
    failures = [str(x) for x in obj.get("failures", []) if str(x) in FAILURE_LABELS]
    confidence = float(np.clip(float(obj.get("confidence", 0.5)), 0.0, 1.0))
    summary = str(obj.get("summary", ""))[:300]
    return VLMReview(decision, failures, confidence, summary)


class QwenVLMClient:
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg

    def build_payload(self, rgb: np.ndarray, masks: SemanticMasks, evidence: dict) -> dict:
        image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        overlay = make_overlay(rgb, masks)
        return {
            "model": self.cfg.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(evidence)},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    {"type": "image_url", "image_url": {"url": _image_data_url(overlay)}},
                ],
            }],
            "temperature": self.cfg.temperature,
            "max_completion_tokens": self.cfg.max_tokens,
        }

    def review(self, rgb: np.ndarray, masks: SemanticMasks, evidence: dict) -> VLMReview:
        payload = self.build_payload(rgb, masks, evidence)
        req = request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        text = obj["choices"][0]["message"]["content"]
        return parse_review(text)
