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

SCENE_TYPES = {
    "NORMAL", "BACKLIGHT", "LOW_LIGHT", "SIDE_LIGHT", "HIGH_DR",
    "BRIGHT_BACKGROUND", "DARK_BACKGROUND", "MIXED_LIGHT",
}
INTENT_LEVELS = {"LOW", "MEDIUM", "HIGH"}
SEVERITIES = {"NONE", "MINOR", "MAJOR"}
NATURALNESS_LABELS = {
    "FACE_TOO_FLAT",
    "FACE_OVER_LIFTED",
    "OVER_HDR_LOOK",
    "SHADOW_OVER_LIFTED",
    "HIGHLIGHT_OVER_COMPRESSED",
    "LIGHTING_CAUSALITY_BROKEN",
    "FACE_BACKGROUND_DISCONNECTED",
    "UNNATURAL_GLOBAL_TONE",
}
TM_ONLY = {"PASS", "SUSPICIOUS", "FAIL"}
SEMANTIC_QUALITY = {"GOOD", "ACCEPTABLE", "POOR"}
PAIRWISE_PREFS = {"A_BETTER", "B_BETTER", "EQUIVALENT"}


@dataclass
class SceneIntent:
    face_lift_needed: bool
    background_preservation: str
    shadow_atmosphere: str
    highlight_priority: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticReview:
    scene_type: str
    scene_intent: SceneIntent
    naturalness: dict[str, str]
    tm_only: str
    semantic_quality: str
    confidence: float
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairwiseReview:
    preference: str
    primary_reason: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneAnalysis:
    scene_type: str
    scene_intent: SceneIntent
    confidence: float
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))


def make_overlay(rgb: np.ndarray, masks: SemanticMasks) -> Image.Image:
    overlay = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.float32)
    alpha = 0.35
    for mask, color in (
        (masks.background, (40, 80, 220)),
        (masks.face, (230, 180, 30)),
        (masks.skin, (230, 70, 70)),
    ):
        c = np.asarray(color, dtype=np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * c
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _extract_json(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen response does not contain JSON")
    return json.loads(text[start:end + 1])


def _level(value: object, allowed: set[str], default: str) -> str:
    value = str(value).upper()
    return value if value in allowed else default


def parse_semantic_review(text: str) -> SemanticReview:
    obj = _extract_json(text)
    intent_obj = obj.get("scene_intent", {}) or {}
    intent = SceneIntent(
        face_lift_needed=bool(intent_obj.get("face_lift_needed", False)),
        background_preservation=_level(intent_obj.get("background_preservation"), INTENT_LEVELS, "MEDIUM"),
        shadow_atmosphere=_level(intent_obj.get("shadow_atmosphere"), INTENT_LEVELS, "MEDIUM"),
        highlight_priority=_level(intent_obj.get("highlight_priority"), INTENT_LEVELS, "MEDIUM"),
    )
    naturalness_obj = obj.get("naturalness", {}) or {}
    naturalness = {
        label: _level(naturalness_obj.get(label, "NONE"), SEVERITIES, "NONE")
        for label in NATURALNESS_LABELS
    }
    return SemanticReview(
        scene_type=_level(obj.get("scene_type"), SCENE_TYPES, "NORMAL"),
        scene_intent=intent,
        naturalness=naturalness,
        tm_only=_level(obj.get("tm_only"), TM_ONLY, "SUSPICIOUS"),
        semantic_quality=_level(obj.get("semantic_quality"), SEMANTIC_QUALITY, "ACCEPTABLE"),
        confidence=float(np.clip(float(obj.get("confidence", 0.5)), 0.0, 1.0)),
        summary=str(obj.get("summary", ""))[:500],
    )


def parse_scene_analysis(text: str) -> SceneAnalysis:
    """Parse the one Qwen scene call shared by all candidates of a source."""
    obj = _extract_json(text)
    intent_obj = obj.get("scene_intent", {}) or {}
    return SceneAnalysis(
        scene_type=_level(obj.get("scene_type"), SCENE_TYPES, "NORMAL"),
        scene_intent=SceneIntent(
            face_lift_needed=bool(intent_obj.get("face_lift_needed", False)),
            background_preservation=_level(intent_obj.get("background_preservation"), INTENT_LEVELS, "MEDIUM"),
            shadow_atmosphere=_level(intent_obj.get("shadow_atmosphere"), INTENT_LEVELS, "MEDIUM"),
            highlight_priority=_level(intent_obj.get("highlight_priority"), INTENT_LEVELS, "MEDIUM"),
        ),
        confidence=float(np.clip(float(obj.get("confidence", 0.5)), 0.0, 1.0)),
        summary=str(obj.get("summary", ""))[:500],
    )


def parse_pairwise_review(text: str) -> PairwiseReview:
    obj = _extract_json(text)
    return PairwiseReview(
        preference=_level(obj.get("preference"), PAIRWISE_PREFS, "EQUIVALENT"),
        primary_reason=str(obj.get("primary_reason", ""))[:400],
        confidence=float(np.clip(float(obj.get("confidence", 0.5)), 0.0, 1.0)),
    )


def build_semantic_prompt(evidence: dict) -> str:
    return f"""You are the semantic judgment branch of a front-camera portrait Tone Mapping PGT evaluator.
Image 1 = SOURCE before the candidate Tone Mapping/edit.
Image 2 = CANDIDATE.
Image 3 = CANDIDATE semantic overlay: red=skin, yellow=face, blue=background.

Judge Tone Mapping, not generic image aesthetics. Use the source to understand original lighting and to check TM-only preservation. Use numeric evidence as factual support, not as a mechanical rubric.

Return one JSON object only with:
- scene_type: NORMAL|BACKLIGHT|LOW_LIGHT|SIDE_LIGHT|HIGH_DR|BRIGHT_BACKGROUND|DARK_BACKGROUND|MIXED_LIGHT
- scene_intent: face_lift_needed boolean; background_preservation/shadow_atmosphere/highlight_priority = LOW|MEDIUM|HIGH
- naturalness: each of FACE_TOO_FLAT, FACE_OVER_LIFTED, OVER_HDR_LOOK, SHADOW_OVER_LIFTED, HIGHLIGHT_OVER_COMPRESSED, LIGHTING_CAUSALITY_BROKEN, FACE_BACKGROUND_DISCONNECTED, UNNATURAL_GLOBAL_TONE as NONE|MINOR|MAJOR
- tm_only: PASS|SUSPICIOUS|FAIL. FAIL only when content/identity/geometry/texture/color changes clearly go beyond plausible Tone Mapping.
- semantic_quality: GOOD|ACCEPTABLE|POOR
- confidence: 0..1
- summary: one concise reason

TM-only is about whether Source->Candidate stays within a plausible tone-rendering operation. Do not penalize normal exposure, local contrast, highlight, or shadow changes merely because they are large.
Evidence: {json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}
"""


def build_pairwise_prompt(a_evidence: dict, b_evidence: dict) -> str:
    return f"""You are choosing the better pseudo-GT for front-camera portrait Tone Mapping.
Image 1 = SOURCE.
Image 2 = Candidate A.
Image 3 = Candidate A semantic overlay.
Image 4 = Candidate B.
Image 5 = Candidate B semantic overlay.

Both candidates have already passed deterministic eligibility checks. Compare them for the current scene, prioritizing: natural face exposure, preserved face tonal structure, reasonable highlight/shadow allocation, natural face-background relation, preserved lighting causality, and TM-only fidelity to Source.
Do not prefer a brighter image simply because it is brighter. Do not use generic aesthetics, sharpness, noise, beauty, or bokeh.
Return one JSON object only:
{{"preference":"A_BETTER|B_BETTER|EQUIVALENT","primary_reason":"short reason","confidence":0.0}}
Use EQUIVALENT when the perceptual TM difference is too small to justify a reliable winner.
A evidence: {json.dumps(a_evidence, ensure_ascii=False, separators=(',', ':'))}
B evidence: {json.dumps(b_evidence, ensure_ascii=False, separators=(',', ':'))}
"""


def build_scene_prompt() -> str:
    return """You are the scene-understanding branch of a front-camera portrait Tone Mapping PGT evaluator.
Image 1 is the unedited SOURCE and Image 2 is its semantic overlay: red=skin, yellow=face, blue=background.
Identify only the source lighting intent. Do not judge beauty, detail, noise, focus, or a candidate image.
Return one JSON object only:
{"scene_type":"NORMAL|BACKLIGHT|LOW_LIGHT|SIDE_LIGHT|HIGH_DR|BRIGHT_BACKGROUND|DARK_BACKGROUND|MIXED_LIGHT","scene_intent":{"face_lift_needed":false,"background_preservation":"LOW|MEDIUM|HIGH","shadow_atmosphere":"LOW|MEDIUM|HIGH","highlight_priority":"LOW|MEDIUM|HIGH"},"confidence":0.0,"summary":"short reason"}"""


def build_candidate_prompt(evidence: dict, scene: SceneAnalysis) -> str:
    return f"""You are the candidate naturalness and TM-only branch of a front-camera portrait Tone Mapping PGT evaluator.
Image 1 = SOURCE. Image 2 = CANDIDATE. Image 3 = candidate semantic overlay: red=skin, yellow=face, blue=background.
The source scene was already analyzed as {json.dumps(scene.to_dict(), ensure_ascii=False, separators=(',', ':'))}. Reuse that context; do not relabel the scene.
Judge only Tone Mapping: face exposure/structure, highlight-shadow allocation, face-background relationship, lighting causality, and whether the candidate stays TM-only.
Do not judge sharpness, noise, bokeh, beautification, or generic aesthetics.
Return one JSON object only with naturalness (FACE_TOO_FLAT, FACE_OVER_LIFTED, OVER_HDR_LOOK, SHADOW_OVER_LIFTED, HIGHLIGHT_OVER_COMPRESSED, LIGHTING_CAUSALITY_BROKEN, FACE_BACKGROUND_DISCONNECTED, UNNATURAL_GLOBAL_TONE; each NONE|MINOR|MAJOR), tm_only (PASS|SUSPICIOUS|FAIL), semantic_quality (GOOD|ACCEPTABLE|POOR), confidence (0..1), summary.
TM-only FAIL means clear geometry/identity/content/texture hallucination or large implausible hue change, not merely a substantial exposure change.
Evidence: {json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"""


class QwenSemanticJudge:
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg

    def _payload(self, prompt: str, images: list[Image.Image]) -> dict:
        content = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(image)}}
            for image in images
        )
        return {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.cfg.temperature,
            "max_completion_tokens": self.cfg.max_tokens,
        }

    def build_semantic_payload(self, source_rgb: np.ndarray, candidate_rgb: np.ndarray, masks: SemanticMasks, evidence: dict) -> dict:
        return self._payload(
            build_semantic_prompt(evidence),
            [_to_image(source_rgb), _to_image(candidate_rgb), make_overlay(candidate_rgb, masks)],
        )

    def build_scene_payload(self, source_rgb: np.ndarray, masks: SemanticMasks) -> dict:
        return self._payload(build_scene_prompt(), [_to_image(source_rgb), make_overlay(source_rgb, masks)])

    def build_candidate_payload(self, source_rgb: np.ndarray, candidate_rgb: np.ndarray, masks: SemanticMasks, evidence: dict, scene: SceneAnalysis) -> dict:
        return self._payload(
            build_candidate_prompt(evidence, scene),
            [_to_image(source_rgb), _to_image(candidate_rgb), make_overlay(candidate_rgb, masks)],
        )

    def build_pairwise_payload(self, source_rgb: np.ndarray, a_rgb: np.ndarray, a_masks: SemanticMasks, a_evidence: dict, b_rgb: np.ndarray, b_masks: SemanticMasks, b_evidence: dict) -> dict:
        return self._payload(
            build_pairwise_prompt(a_evidence, b_evidence),
            [_to_image(source_rgb), _to_image(a_rgb), make_overlay(a_rgb, a_masks), _to_image(b_rgb), make_overlay(b_rgb, b_masks)],
        )

    def _call(self, payload: dict) -> str:
        req = request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]

    def review(self, source_rgb: np.ndarray, candidate_rgb: np.ndarray, masks: SemanticMasks, evidence: dict) -> SemanticReview:
        return parse_semantic_review(self._call(self.build_semantic_payload(source_rgb, candidate_rgb, masks, evidence)))

    def analyze_scene(self, source_rgb: np.ndarray, masks: SemanticMasks) -> SceneAnalysis:
        return parse_scene_analysis(self._call(self.build_scene_payload(source_rgb, masks)))

    def review_with_scene(self, source_rgb: np.ndarray, candidate_rgb: np.ndarray, masks: SemanticMasks, evidence: dict, scene: SceneAnalysis) -> SemanticReview:
        review = parse_semantic_review(self._call(self.build_candidate_payload(source_rgb, candidate_rgb, masks, evidence, scene)))
        review.scene_type = scene.scene_type
        review.scene_intent = scene.scene_intent
        return review

    def compare(self, source_rgb: np.ndarray, a_rgb: np.ndarray, a_masks: SemanticMasks, a_evidence: dict, b_rgb: np.ndarray, b_masks: SemanticMasks, b_evidence: dict) -> PairwiseReview:
        payload = self.build_pairwise_payload(source_rgb, a_rgb, a_masks, a_evidence, b_rgb, b_masks, b_evidence)
        return parse_pairwise_review(self._call(payload))
