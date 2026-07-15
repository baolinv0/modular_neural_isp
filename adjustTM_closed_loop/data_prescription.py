from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .failure_taxonomy import classify_failure_reasons
from .schemas import DataPrescription, DataTask, FailureCode, QwenSceneResult


_TASK_LIBRARY: dict[FailureCode, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {
    FailureCode.BRIGHTNESS_UNDER: (
        "Gain/GTM control", "insufficient positive-alpha response",
        ("target log-luminance", "pairwise target preference", "alpha direction"),
        ("positive-alpha brightness improves", "zero-point remains unchanged", "normal scenes do not over-brighten"),
    ),
    FailureCode.BRIGHTNESS_OVER: (
        "Gain/GTM control", "excessive positive-alpha response",
        ("target log-luminance", "over-bright hard-negative label", "alpha response curve"),
        ("over-bright rate decreases", "endpoint range remains sufficient", "zero-point remains unchanged"),
    ),
    FailureCode.HIGHLIGHT_CLIPPING: (
        "Gain/GTM control", "positive-alpha control consumes highlight headroom",
        ("highlight mask", "clipping ratio", "brightness-matched preference pair"),
        ("highlight clipping does not increase", "target brightness remains reachable", "normal scenes do not darken"),
    ),
    FailureCode.SHADOW_CRUSH: (
        "Gain/GTM control", "negative-alpha control destroys shadow visibility",
        ("shadow mask", "deep-shadow ratio", "local contrast preference"),
        ("deep-shadow ratio decreases", "negative endpoint remains dark enough", "midtone ordering remains monotonic"),
    ),
    FailureCode.CHROMA_DRIFT: (
        "Gain/GTM control", "brightness control changes chromaticity",
        ("rg chromaticity drift", "skin/sky region labels", "brightness-matched color preference"),
        ("chroma drift decreases", "luminance control is preserved", "alpha-zero is identical to baseline"),
    ),
    FailureCode.CONTROL_CURVE: (
        "Control encoder and monotonic loss", "alpha response is non-monotonic, discontinuous or saturated",
        ("dense-alpha trajectory", "adjacent-step target", "dead-zone and saturation labels"),
        ("dense violation rate reaches zero", "step error decreases", "endpoint range remains sufficient"),
    ),
    FailureCode.REGIONAL_INCONSISTENCY: (
        "GTM/LTM coordination", "global alpha causes inconsistent semantic-region response",
        ("fixed semantic masks", "regional luminance trajectories", "region-coherence preference"),
        ("regional coherence improves", "global monotonicity remains", "no new face/sky artifacts"),
    ),
    FailureCode.STRUCTURAL_ARTIFACT: (
        "LTM boundary and output regularization", "brightness control introduces halo, banding or structure drift",
        ("edge/gradient maps", "halo/banding labels", "tone-normalized fidelity"),
        ("artifact recall decreases", "brightness target remains matched", "baseline fidelity is preserved"),
    ),
}


def build_data_prescription(scenes: Mapping[str, QwenSceneResult]) -> DataPrescription:
    grouped: dict[FailureCode, list[str]] = defaultdict(list)
    keep_scenes = sorted(name for name, result in scenes.items() if result.action == "KEEP")
    for name, result in scenes.items():
        if result.action == "KEEP":
            continue
        for code in classify_failure_reasons(result.failure_reasons):
            grouped[code].append(name)

    tasks: list[DataTask] = []
    for code in sorted(grouped, key=lambda item: item.value):
        target_module, capability_gap, supervision, gates = _TASK_LIBRARY[code]
        positives = tuple(sorted(set(grouped[code])))
        review = tuple(sorted(name for name in positives if scenes[name].action == "REVIEW"))
        hard_negative = tuple(sorted(name for name in positives if scenes[name].action in {"REJECT", "REGENERATE"}))
        tasks.append(
            DataTask(
                failure_code=code,
                target_module=target_module,
                capability_gap=capability_gap,
                positive_scenes=positives,
                boundary_scenes=review,
                hard_negative_scenes=hard_negative,
                regression_anchor_scenes=tuple(keep_scenes),
                required_supervision=supervision,
                acceptance_gates=gates,
            )
        )
    return DataPrescription(tasks=tuple(tasks))
