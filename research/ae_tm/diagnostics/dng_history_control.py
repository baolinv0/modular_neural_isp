"""Local DNG H-information control: classical WLS and spatial Wiener, no Bayes claim.

Private pixels are read only. Outputs are JSON/CSV identities, risks and statistics.
The source/target/actions/noise/splits are inherited unchanged from DNG_PROTOCOL.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.ndimage import uniform_filter

try:
    import dng_relative_experiment as legacy
except ModuleNotFoundError:
    # Legacy module uses same-directory imports. Keep that original file untouched.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import dng_relative_experiment as legacy
    finally:
        sys.path.pop(0)

BACKENDS = ("H_blind_inverse", "H_blind_Wiener",
            "H_aware_classical_fusion", "H_aware_fusion_Wiener")
WINDOWS = {
    "H_blind_inverse": (1,), "H_blind_Wiener": legacy.WINDOWS,
    "H_aware_classical_fusion": (1,), "H_aware_fusion_Wiener": legacy.WINDOWS,
}


def _codes(value, name):
    x = np.asarray(value)
    if np.any(~np.isfinite(x)) or np.any(x != np.rint(x)) or np.any((x < 0) | (x > legacy.ADC_MAX)):
        raise ValueError(f"{name} must contain integer ADC codes in 0..4095")
    return x.astype(np.int16, copy=False)


def normalized_measurement_variance(xhat, exposure):
    """Prescribed approximate unclipped shot+read+ADC normalized variance."""
    x = np.asarray(xhat, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("history-only xhat must be finite and in [0,1]")
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("known exposure must be finite and positive")
    return ((legacy.FW*exposure*x + legacy.READ_SD**2)/(legacy.FW**2*exposure**2)
            + 1/(12*legacy.ADC_MAX**2*exposure**2))


def classical_fusion_estimate(history_codes, future_codes, exposure):
    """H=(3,patch,y,x), Y=(repeat,patch,y,x). Each repeat uses only H and its Y."""
    h, y = _codes(history_codes, "H"), _codes(future_codes, "future")
    if h.ndim != 4 or h.shape[0] != 3 or any(n < 1 for n in h.shape[1:]):
        raise ValueError("H must have shape (3,positive_patches,positive_y,positive_x)")
    if y.ndim != 4 or y.shape[0] < 1 or y.shape[1:] != h.shape[1:]:
        raise ValueError("future must have shape (positive_repeats,patch,y,x) matching H")
    signal_h = h.astype(np.float64)/legacy.ADC_MAX/.25
    xhat = np.clip(signal_h.mean(axis=0), 0, 1)
    var_h = normalized_measurement_variance(xhat, .25)
    var_y = normalized_measurement_variance(xhat, exposure)
    # Zero codes retain weights; only observed top ADC codes are dropped.
    weight_h = np.where(h == legacy.ADC_MAX, 0., 1/var_h[None])
    weight_y = np.where(y == legacy.ADC_MAX, 0., 1/var_y[None])
    total = weight_h.sum(axis=0)[None] + weight_y
    numerator = (weight_h*signal_h).sum(axis=0)[None]
    numerator = numerator + weight_y*(y.astype(np.float64)/legacy.ADC_MAX/exposure)
    no_weight = total == 0
    estimate = np.ones_like(total)
    variance = np.ones_like(total)  # bounded squared-range fallback placeholder
    np.divide(numerator, total, out=estimate, where=~no_weight)
    np.divide(1., total, out=variance, where=~no_weight)
    return np.clip(estimate, 0, 1), variance, no_weight


def fusion_wiener_target(estimate, noise_variance, window, no_weight=None):
    """Wiener with matching local 2D mean/variance/noise; no patch/repeat mixing."""
    signal, noise = np.asarray(estimate, dtype=float), np.asarray(noise_variance, dtype=float)
    if signal.ndim != 4 or signal.shape != noise.shape or any(n < 1 for n in signal.shape):
        raise ValueError("estimate/noise must have matching (repeat,patch,y,x) shapes")
    if np.any(~np.isfinite(signal)) or np.any((signal < 0) | (signal > 1)) or np.any(~np.isfinite(noise)) or np.any(noise < 0):
        raise ValueError("bounded finite estimate and finite nonnegative noise required")
    if window not in legacy.WINDOWS:
        raise ValueError("window must be one of 1/3/5/7")
    if no_weight is not None and np.asarray(no_weight).shape != signal.shape:
        raise ValueError("fallback mask shape mismatch")
    if window == 1:
        filtered = signal
    else:
        shape = (1, 1, window, window)
        mean = uniform_filter(signal, size=shape, mode="reflect")
        variance = np.maximum(0., uniform_filter(signal*signal, size=shape, mode="reflect")-mean*mean)
        local_noise = uniform_filter(noise, size=shape, mode="reflect")
        gain = np.clip(1-local_noise/np.maximum(variance, 1e-20), 0, 1)
        filtered = mean+gain*(signal-mean)
    if no_weight is not None:
        filtered = np.where(no_weight, 1., filtered)
    return legacy.tone(np.clip(filtered, 0, 1))


def preview_history_and_features(x, rng):
    """Exact old H capture and feature arithmetic, returning H for renderer reuse."""
    previews = legacy.capture_codes(np.broadcast_to(x, (3,)+x.shape), .25, rng)
    values = previews.astype(np.float64)/legacy.ADC_MAX/.25
    mean = values.mean(axis=0)
    gradient = np.concatenate((np.diff(mean, axis=1).ravel(), np.diff(mean, axis=2).ravel()))
    features = np.array([mean.mean(), np.quantile(mean, .99),
        np.sqrt(values.var(axis=0, ddof=1).mean()), np.mean(np.abs(gradient))])
    return previews, features


def source_risks(x, history_codes, repeats, rng):
    """Same old future RNG call order; inference never receives the evaluation target."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or any(n < 1 for n in x.shape) or np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("source must have bounded finite (patch,y,x) shape")
    if repeats < 1 or int(repeats) != repeats:
        raise ValueError("positive integer future repeat count required")
    if np.asarray(history_codes).shape != (3,)+x.shape:
        raise ValueError("H/source patch shapes mismatch")
    target = legacy.tone(x)[None]
    costs = {name: np.empty((len(WINDOWS[name]), len(legacy.EXPOSURES))) for name in BACKENDS}
    mc_se = {name: np.empty_like(costs[name]) for name in BACKENDS}
    zero_counts = np.zeros(len(legacy.EXPOSURES), dtype=np.int64)
    for i, exposure in enumerate(legacy.EXPOSURES):
        codes = legacy.capture_codes(np.broadcast_to(x, (repeats,)+x.shape), exposure, rng)
        inverse = legacy.tone(np.clip(codes.astype(float)/legacy.ADC_MAX/exposure, 0, 1))
        losses = ((inverse-target)**2).mean(axis=(1, 2, 3))
        # Preserve the exact legacy inverse reduction order.
        costs["H_blind_inverse"][0, i] = losses.mean()
        mc_se["H_blind_inverse"][0, i] = losses.std(ddof=1)/np.sqrt(repeats) if repeats > 1 else 0.
        fused, variance, no_weight = classical_fusion_estimate(history_codes, codes, float(exposure))
        zero_counts[i] = np.count_nonzero(no_weight)
        for widx, window in enumerate(legacy.WINDOWS):
            blind = legacy.wiener_target(codes, exposure, window)
            aware = fusion_wiener_target(fused, variance, window, no_weight)
            for name, prediction in [("H_blind_Wiener", blind), ("H_aware_fusion_Wiener", aware)]:
                # Preserve legacy spatial reduction order, including window=1.
                costs[name][widx, i] = ((prediction-target)**2).mean()
                loss = ((prediction-target)**2).mean(axis=(1, 2, 3))
                mc_se[name][widx, i] = loss.std(ddof=1)/np.sqrt(repeats) if repeats > 1 else 0.
            if window == 1:
                costs["H_aware_classical_fusion"][0, i] = costs["H_aware_fusion_Wiener"][0, i]
                mc_se["H_aware_classical_fusion"][0, i] = mc_se["H_aware_fusion_Wiener"][0, i]
    return {"costs": costs, "future_mc_se": mc_se,
            "all_weight_zero_counts_by_action": zero_counts,
            "pixel_observations_per_action": int(repeats*x.size)}


def backend_costs(records, backend, window_indices):
    return np.stack([row["risks"]["costs"][backend][window_indices, np.arange(len(legacy.EXPOSURES))]
                     for row in records])


def freeze_selectors(train, development):
    tf = np.stack([row["features"] for row in train])
    df = np.stack([row["features"] for row in development])
    selectors = {}
    for name in BACKENDS:
        window_risk = legacy.group_mean(np.stack([row["risks"]["costs"][name] for row in development]), development)
        wi = window_risk.argmin(axis=0)
        costs = backend_costs(development, name, wi)
        fixed = int(legacy.group_mean(costs, development).argmin())
        selectors[name] = {
            "renderer": name, "window_indices": wi.tolist(),
            "dev_selected_windows_by_action": [WINDOWS[name][int(i)] for i in wi],
            "dev_fixed_action_index": fixed, "dev_fixed_exposure": float(legacy.EXPOSURES[fixed]),
            "dev_group_balanced_risk_by_action": legacy.group_mean(costs, development).tolist(),
            "frozen_rules": {feature: legacy.choose_rule(tf, df, costs, development, i)
                             for i, feature in enumerate(legacy.FEATURE_NAMES)},
        }
    return selectors


def paired_cluster_ci(difference, records, rng, bootstraps=2000, direction=""):
    values = np.asarray(difference, dtype=float)
    if values.shape != (len(records),) or np.any(~np.isfinite(values)) or bootstraps < 2:
        raise ValueError("finite one-difference-per-source and >=2 bootstrap draws required")
    groups = sorted({r["group_id"] for r in records})
    if not groups:
        raise ValueError("nonempty source groups required")
    means = np.array([values[[i for i, row in enumerate(records) if row["group_id"] == g]].mean()
                      for g in groups])
    draws = means[rng.integers(0, len(groups), (bootstraps, len(groups)))].mean(axis=1)
    return {"mean_difference": float(means.mean()),
            "descriptive_paired_cluster_percentile_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "group_mean_differences": dict(zip(groups, means.tolist())),
            "diagnostic_group_count": len(groups), "bootstrap_replicates": bootstraps,
            "difference_direction": direction,
            "scope": "descriptive, one session; conditional on frozen dev selection; groups are bootstrap units"}


def cross_evaluate(costs, features, records, selectors, seed, cam_id, bootstraps=2000):
    """4 frozen selector rows x 4 frozen renderer columns; source fixed e retained."""
    n = len(records)
    if np.asarray(features).shape != (n, len(legacy.FEATURE_NAMES)) or any(
            np.asarray(costs[name]).shape != (n, len(legacy.EXPOSURES)) for name in BACKENDS):
        raise ValueError("cross inputs have mismatched source/action/feature dimensions")
    matrix = {}
    for sid, selector_name in enumerate(BACKENDS):
        selector = selectors[selector_name]
        actions = {name: legacy.policy_actions(features, rule)
                   for name, rule in selector["frozen_rules"].items()}
        matrix[selector_name] = {}
        fixed = selector["dev_fixed_action_index"]
        for rid, renderer in enumerate(BACKENDS):
            c = costs[renderer]
            fixed_cost = c[:, fixed]
            selected = {feature: c[np.arange(n), actions[feature]] for feature in legacy.FEATURE_NAMES}
            brightness = selected["mean_brightness"]
            children = np.random.SeedSequence([seed, cam_id, sid, rid, 9081]).spawn(2*len(legacy.FEATURE_NAMES))
            policies = {}
            for fid, feature in enumerate(legacy.FEATURE_NAMES):
                value = selected[feature]
                policies[feature] = {
                    "group_balanced_risk": float(legacy.group_mean(value, records)),
                    "source_action_counts": np.bincount(actions[feature], minlength=len(legacy.EXPOSURES)).tolist(),
                    "paired_rule_minus_selector_fixed": paired_cluster_ci(
                        value-fixed_cost, records, np.random.default_rng(children[2*fid]), bootstraps,
                        "rule minus source selector's dev fixed; negative favors rule"),
                    "paired_rule_minus_mean_brightness": paired_cluster_ci(
                        value-brightness, records, np.random.default_rng(children[2*fid+1]), bootstraps,
                        "rule minus source selector's mean-brightness rule; negative favors this rule"),
                }
            matrix[selector_name][renderer] = {
                "source_selector": selector_name, "evaluated_renderer": renderer,
                "renderer_windows": selectors[renderer]["dev_selected_windows_by_action"],
                "source_selector_dev_fixed_action_index": fixed,
                "source_selector_dev_fixed_exposure": float(legacy.EXPOSURES[fixed]),
                "group_balanced_risk_by_action": legacy.group_mean(c, records).tolist(),
                "fixed_group_balanced_risk": float(legacy.group_mean(fixed_cost, records)),
                "diagnostic_group_count": len({r["group_id"] for r in records}),
                "policies": policies,
            }
    return matrix


def _validate_audit_and_cache(cache, audit):
    manifest = json.loads((cache/"cache_manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != audit["protocol_sha256"]:
        raise ValueError("Cache/audit protocol mismatch")
    sources = audit["sources"]
    if len(sources) != 68 or len({s["source_id"] for s in sources}) != 68:
        raise ValueError("This fixed DNG control requires the same 68 unique audited sources")
    cached = {s["source_id"]: s for s in manifest["sources"]}
    if set(cached) != {s["source_id"] for s in sources}:
        raise ValueError("Cache/audit source IDs differ")
    for s in sources:
        if s["camera"] not in ("iphone", "s25") or s["split"] not in ("train", "development", "diagnostic"):
            raise ValueError("Unknown camera or source split")
        for key in ("pair_id", "camera", "group_id", "split", "sha256"):
            if cached[s["source_id"]][key] != s[key]:
                raise ValueError(f"Cache/audit source metadata mismatch: {key}")
    return sources


def run(cache, audit_path, output, seeds=(19, 37, 73), repeats=4, bootstraps=2000, smoke=False):
    """Local metrics only. Caller must choose a new output directory."""
    cache, audit_path, output = Path(cache), Path(audit_path), Path(output)
    if output.exists():
        raise FileExistsError("Output exists; preserve results and use a new directory")
    if repeats < 1 or int(repeats) != repeats or bootstraps < 2 or not seeds:
        raise ValueError("positive repeats/nonempty seeds and >=2 bootstrap draws required")
    if smoke and len(seeds) != 1:
        raise ValueError("smoke must have exactly one seed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sources = _validate_audit_and_cache(cache, audit)
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    repository = Path(__file__).resolve().parents[3]
    report = {
        "status": "completed_DNG_history_control_software_smoke" if smoke else "completed_DNG_history_classical_diagnostic",
        "new_protocol": "DNG_HISTORY_CONTROL.md; FROZEN 2026-10-09",
        "new_protocol_sha256": legacy.file_sha(Path(__file__).with_name("DNG_HISTORY_CONTROL.md")),
        "original_protocol_id": audit["protocol_id"], "original_protocol_sha256": audit["protocol_sha256"],
        "source_manifest_sha256": audit["manifest_sha256"],
        "cache_sha256": {s["source_id"]: legacy.file_sha(cache/f"{s['source_id']}.npz") for s in sources},
        "source_count": len(sources), "data_domain": audit["source_domain"],
        "independence_limit": audit["independence_limit"],
        "target": "sqrt(x/(x+.5)) of same bounded/noisy processed green surrogate; not expert GT",
        "actions": legacy.EXPOSURES.tolist(),
        "config": {"seeds": list(seeds), "future_repeats": repeats, "past_preview_frames": 3,
                   "past_preview_exposure": .25, "windows": {k: list(v) for k, v in WINDOWS.items()},
                   "bootstrap_replicates": bootstraps, "bayesian_nodes": None},
        "sensor_assumptions": {"full_well": legacy.FW, "read_sd": legacy.READ_SD,
                               "adc_max": legacy.ADC_MAX, "camera_calibrated": False},
        "source_rng": "SeedSequence([seed,pair_id,cam_id]).spawn(2); old H and future e call order",
        "ci_rng": "SeedSequence([seed,cam_id,selector_index,renderer_index,9081]).spawn(8)",
        "backend_scope": "future-only inverse/Wiener versus approximate H+one-future WLS/Wiener; no neural/Bayes/strong-image claim",
        "fusion_variance": "(1000*e*xhat+4^2)/(1000^2*e^2)+1/(12*4095^2*e^2); xhat only H, clip[0,1]",
        "fusion_bias": "unclipped independent-noise variance; H-dependent weights, zero/top censoring and clipping bias; not calibrated posterior uncertainty",
        "fusion_wiener_noise": "local uniform_filter of fusion noisevar with same (1,1,w,w) window as signal moments",
        "all_zero_weight_fallback": "normalized estimate1, variance1 finite bounded-range placeholder; spatial preserves estimate1, count reported",
        "selection": "camera/backend-specific train quantiles and group-balanced dev windows/fixed/rules; freeze before 4x4 diagnostic replay",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip()),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "replicates": [],
    }
    csv_rows = []
    for seed in seeds:
        seed_begin = time.perf_counter()
        records = []
        for s in sources:
            x = legacy.load_patch(cache, s)
            if x.shape != (8, 64, 64):
                raise ValueError("Expected the same eight 64x64 DNG green patches")
            cam_id = 0 if s["camera"] == "iphone" else 1
            streams = np.random.SeedSequence([seed, s["pair_id"], cam_id]).spawn(2)
            history, features = preview_history_and_features(x, np.random.default_rng(streams[0]))
            risks = source_risks(x, history, repeats, np.random.default_rng(streams[1]))
            records.append({**{k: s[k] for k in ("source_id", "camera", "pair_id", "group_id", "split")},
                            "features": features, "risks": risks})
        replicate = {
            "seed": int(seed), "capture_risk_seconds": time.perf_counter()-seed_begin, "cameras": {},
            "all_weight_zero_counts_by_action": np.sum([r["risks"]["all_weight_zero_counts_by_action"] for r in records], axis=0).tolist(),
            "pixel_observations_per_action": sum(r["risks"]["pixel_observations_per_action"] for r in records),
        }
        for cam_id, camera in enumerate(("iphone", "s25")):
            selection_begin = time.perf_counter()
            sets = {split: [r for r in records if r["camera"] == camera and r["split"] == split]
                    for split in ("train", "development", "diagnostic")}
            train, dev, diag = (sets[name] for name in ("train", "development", "diagnostic"))
            if any(not rows for rows in (train, dev, diag)):
                raise ValueError("Each camera needs original train/development/diagnostic sources")
            selectors = freeze_selectors(train, dev)
            features = np.stack([r["features"] for r in diag])
            costs = {name: backend_costs(diag, name, np.array(selectors[name]["window_indices"])) for name in BACKENDS}
            matrix = cross_evaluate(costs, features, diag, selectors, int(seed), cam_id, bootstraps)
            replicate["cameras"][camera] = {
                "source_counts_by_split": {name: len(rows) for name, rows in sets.items()},
                "diagnostic_group_ids": sorted({r["group_id"] for r in diag}),
                "frozen_selectors": selectors, "backend_cross_matrix": matrix,
                "selection_and_summary_seconds": time.perf_counter()-selection_begin,
            }
            for selector_name, selector in selectors.items():
                actions = {name: legacy.policy_actions(features, rule) for name, rule in selector["frozen_rules"].items()}
                for renderer in BACKENDS:
                    for i, row in enumerate(diag):
                        csv_rows.append({
                            "seed": int(seed), "camera": camera, "source_selector": selector_name,
                            "renderer": renderer, "source_id": row["source_id"], "group_id": row["group_id"],
                            "split": row["split"], "source_selector_dev_fixed_action": selector["dev_fixed_action_index"],
                            "dev_fixed_action": selector["dev_fixed_action_index"],
                            **{f"risk_e{e:g}": float(costs[renderer][i, j]) for j, e in enumerate(legacy.EXPOSURES)},
                            **{f"policy_action_{name}": int(actions[name][i]) for name in legacy.FEATURE_NAMES},
                        })
        replicate["elapsed_seconds"] = time.perf_counter()-seed_begin
        report["replicates"].append(replicate)
        print(json.dumps({"seed": int(seed), "sources": len(records), "status": "done",
                          "all_weight_zero_total": int(sum(replicate["all_weight_zero_counts_by_action"]))}), flush=True)
    with (output/"source_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report["source_metrics_sha256"] = legacy.file_sha(output/"source_metrics.csv")
    report["elapsed_seconds"] = time.perf_counter()-begin
    report["limitations"] = [
        "same processed/noisy/clipped DNG surrogate, not clean RAW/expert GT/HDR or actual recapture",
        "one session, only three diagnostic background groups; crop/camera/noise copies are not new independent scenes",
        "variance-weighted fusion is approximate WLS with data-dependent/censored weights, not a Poisson posterior",
        "local Wiener is a classical simplicity control, not a strong unrestricted image backend",
        "descriptive group bootstrap is conditional on frozen two-group development selections",
        "no original catalog, neural training, camera calibration, motion, or S24 formal test",
    ]
    with (output/"results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[19, 37, 73])
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.smoke:
        a.seeds, a.repeats, a.bootstraps = [19], 2, 128
    result = run(a.cache, a.audit, a.out, a.seeds, a.repeats, a.bootstraps, a.smoke)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "git_commit": result["git_commit"], "source_count": result["source_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
