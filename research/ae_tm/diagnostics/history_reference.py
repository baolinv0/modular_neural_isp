"""Full-history known-prior conditional reference and frozen backend cross replay.

Independent N0/N1 simulation only. No S24/formal test or original CaptureTM run.
CLI: python research/ae_tm/diagnostics/history_reference.py --smoke --out PATH
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

if __package__:
    from . import continuous_reference as base
else:
    import continuous_reference as base

BACKENDS = ("H_blind", "H_aware")
SCOPES = {
    "H_blind": "E_train[T(L)|future_ADC,do(e)]; ignores H and observational action-selection information",
    "H_aware": "E_train[T(L)|all past preview_ADC,future_ADC,do(e)]; known training prior, not unknown-prior optimality",
}
STREAMS = {
    "train_sources": 0, "train_history": 1, "dev_sources": 2, "dev_history": 3,
    "dev_future_noise": 4, "N0_sources": 5, "N0_history": 6, "N0_future_noise": 7,
    "N1_sources": 8, "N1_history": 9, "N1_future_noise": 10,
    "N0_source_bootstrap": 11, "N1_source_bootstrap": 12,
}


@dataclass(frozen=True)
class _HistoryFactor:
    log_probabilities: np.ndarray
    source_frame_indices: np.ndarray


@dataclass(frozen=True)
class IndexedHistory:
    """Only observation lookup/index storage, not source-by-repeat node tiles."""
    owner: int
    source_count: int
    frame_count: int
    exposures: np.ndarray
    axes: tuple


def _adc_codes(value, name):
    x = np.asarray(value)
    if np.any(~np.isfinite(x)) or np.any(x != np.rint(x)):
        raise ValueError(f"{name} must contain finite integer ADC codes")
    if np.any((x < 0) | (x > base.ADC_MAX)):
        raise ValueError(f"{name} ADC codes must be in 0..4095")
    return x.astype(np.int16, copy=False)


class HistoryReference(base.ContinuousReference):
    """Explicit product likelihood E_train[T|H,Y_future,do(e)], with shared nodes."""

    def _history_axis(self, codes, pixel, latent, exposures):
        factors = []
        for exposure in np.unique(exposures):
            frames = np.flatnonzero(exposures == exposure)
            values = codes[:, frames, pixel]
            unique, inverse = np.unique(values, return_inverse=True)
            probabilities = self.likelihood.probabilities(
                unique, base.FW * float(exposure) * latent, self.likelihood_chunk)
            with np.errstate(divide="ignore"):
                logs = np.log(probabilities)
            factors.append(_HistoryFactor(logs, inverse.reshape(values.shape)))
        return tuple(factors)

    def prepare_history(self, history_codes, history_exposures=.25):
        """Prepare (source,frame,2) observed ADC; the runner supplies exactly 3 frames."""
        codes = _adc_codes(history_codes, "history")
        if codes.ndim != 3 or codes.shape[2] != 2 or codes.shape[0] < 1:
            raise ValueError("history must have shape (positive_sources,frames,2)")
        exposures = np.asarray(history_exposures, dtype=np.float64)
        if exposures.ndim == 0:
            if not np.isfinite(exposures) or exposures <= 0:
                raise ValueError("history exposure must be finite and positive")
            exposures = np.full(codes.shape[1], float(exposures))
        if exposures.shape != (codes.shape[1],) or np.any(~np.isfinite(exposures)) or np.any(exposures <= 0):
            raise ValueError("history exposures must be positive scalar or one per frame")
        a = self._history_axis(codes, 0, self.a, exposures)
        b = self._history_axis(codes, 1, self.b, exposures)
        ca = a if np.array_equal(self.a, self.ca) else self._history_axis(codes, 0, self.ca, exposures)
        cb = b if np.array_equal(self.b, self.cb) else self._history_axis(codes, 1, self.cb, exposures)
        return IndexedHistory(id(self), len(codes), codes.shape[1], exposures.copy(), (a, b, ca, cb))

    @staticmethod
    def _product_log(table, inverse, factors, source_ids):
        with np.errstate(divide="ignore"):
            logs = np.log(table[inverse])
        for factor in factors:
            rows = factor.source_frame_indices[source_ids]
            for frame in range(rows.shape[1]):
                logs += factor.log_probabilities[rows[:, frame]]
        return logs

    def posterior(self, adc_pairs, exposure, history=None, *,
                  history_exposures=.25, source_indices=None):
        """Input future shape (source,2) or (source,repeat,2); return matching means.

        source_indices maps the future first axis to a prepared history source.
        No target, latent, family ID, policy identity, or unobserved future input.
        """
        codes = _adc_codes(adc_pairs, "future")
        if codes.ndim not in (2, 3) or codes.shape[-1] != 2 or codes.shape[0] < 1:
            raise ValueError("future must have shape (sources,2) or (sources,repeats,2)")
        if codes.ndim == 3 and codes.shape[1] < 1:
            raise ValueError("future needs at least one repeat")
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("known future exposure must be finite and positive")
        rows, repeats = len(codes), (codes.shape[1] if codes.ndim == 3 else 1)
        flat = codes.reshape(-1, 2)
        if history is None:
            if source_indices is not None:
                raise ValueError("source_indices requires history")
            mean, gamma = super().posterior(flat, exposure)
            return mean.reshape(codes.shape), gamma.reshape(codes.shape[:-1])
        prepared = history if isinstance(history, IndexedHistory) else self.prepare_history(history, history_exposures)
        if prepared.owner != id(self):
            raise ValueError("prepared history belongs to a different reference/node grid")
        if source_indices is None:
            if rows != prepared.source_count:
                raise ValueError("future source count differs from history; supply source_indices")
            row_sources = np.arange(rows)
        else:
            raw = np.asarray(source_indices)
            if raw.shape != (rows,) or np.any(~np.isfinite(raw)) or np.any(raw != np.rint(raw)):
                raise ValueError("source_indices must be one integer per future source row")
            row_sources = raw.astype(np.int64)
            if np.any((row_sources < 0) | (row_sources >= prepared.source_count)):
                raise ValueError("source_indices outside prepared history")
        if prepared.frame_count == 0:
            mean, gamma = super().posterior(flat, exposure)
            return mean.reshape(codes.shape), gamma.reshape(codes.shape[:-1])
        tables = [self._tables(flat[:, 0], self.a, exposure),
                  self._tables(flat[:, 1], self.b, exposure)]
        tables.append(tables[0] if np.array_equal(self.a, self.ca) else self._tables(flat[:, 0], self.ca, exposure))
        tables.append(tables[1] if np.array_equal(self.b, self.cb) else self._tables(flat[:, 1], self.cb, exposure))
        wa, wb = self.weights * base.tone(self.a), self.weights * base.tone(self.b)
        wca, wcb = self.cweights * base.tone(self.ca), self.cweights * base.tone(self.cb)
        mean, gamma = np.empty_like(flat, dtype=np.float64), np.empty(len(flat))
        for start in range(0, len(flat), self.chunk):
            end = min(start + self.chunk, len(flat))
            # Index H per chunk. Never repeat/tile a sources*nodes history over future repeats.
            source_ids = row_sources[np.arange(start, end) // repeats]
            logs = [self._product_log(table, inverse[start:end], factors, source_ids)
                    for (table, inverse), factors in zip(tables, prepared.axes)]
            offset_a = np.maximum(logs[0].max(axis=1), logs[2].max(axis=1))
            offset_b = np.maximum(logs[1].max(axis=1), logs[3].max(axis=1))
            if np.any(~np.isfinite(offset_a)) or np.any(~np.isfinite(offset_b)):
                raise FloatingPointError("all likelihood nodes underflow for a history/future pixel")
            # Common scaling across mixture components is essential: separate component
            # normalization here would incorrectly erase posterior mixture evidence.
            pa, pb = np.exp(logs[0]-offset_a[:, None]), np.exp(logs[1]-offset_b[:, None])
            pca, pcb = np.exp(logs[2]-offset_a[:, None]), np.exp(logs[3]-offset_b[:, None])
            za, zb = pa @ self.weights, pb @ self.weights
            independent = za * zb
            paired = pca * pcb
            correlated = paired @ self.cweights
            w = self.prior.weight
            evidence = (1-w)*independent + w*correlated
            if np.any(evidence <= 0) or np.any(~np.isfinite(evidence)):
                raise FloatingPointError("zero/nonfinite product evidence; no fallback")
            mean[start:end, 0] = ((1-w)*(pa @ wa)*zb + w*(paired @ wca)) / evidence
            mean[start:end, 1] = ((1-w)*(pb @ wb)*za + w*(paired @ wcb)) / evidence
            gamma[start:end] = w*correlated / evidence
        return mean.reshape(codes.shape), gamma.reshape(codes.shape[:-1])

    def predict(self, adc_pairs, exposure, history=None, *,
                history_exposures=.25, source_indices=None):
        return self.posterior(adc_pairs, exposure, history,
            history_exposures=history_exposures, source_indices=source_indices)[0]


def evaluate_backends(reference, latents, history_codes, future_cube):
    """Evaluation-only target access; both inference backends see the same future ADC."""
    latents = np.asarray(latents)
    cube = np.asarray(future_cube)
    if latents.ndim != 2 or latents.shape[1] != 2:
        raise ValueError("evaluation latents must have shape (sources,2)")
    if cube.ndim != 4 or cube.shape[0] != len(latents) or cube.shape[1] != len(base.EXPOSURES) or cube.shape[-1] != 2 or cube.shape[2] < 1:
        raise ValueError("future cube must have shape (sources,7,positive_repeats,2)")
    if np.asarray(history_codes).shape != (len(latents), 3, 2):
        raise ValueError("controlled diagnostic requires exactly three past preview frames")
    prepared = reference.prepare_history(history_codes)
    target = base.tone(latents)[:, None, :]
    repeats = cube.shape[2]
    costs = {name: np.empty((len(latents), len(base.EXPOSURES))) for name in BACKENDS}
    noise_se = {name: [] for name in BACKENDS}
    for i, exposure in enumerate(base.EXPOSURES):
        observed = cube[:, i]
        for name in BACKENDS:
            pred = reference.predict(observed, float(exposure),
                                     history=prepared if name == "H_aware" else None)
            losses = ((pred-target)**2).mean(axis=2)
            costs[name][:, i] = losses.mean(axis=1)
            se = float(np.sqrt(losses.var(axis=1, ddof=1).mean()/len(latents)/repeats)) if repeats > 1 else None
            noise_se[name].append(se)
    return costs, {"future_mc_se_by_backend_and_action": noise_se}


def freeze_selectors(train_features, dev_features, dev_costs):
    return {name: {
        "trained_against_renderer": name,
        "fixed_action_index": int(dev_costs[name].mean(axis=0).argmin()),
        "rules": {feature: base.select_threshold(train_features[feature], dev_features[feature], dev_costs[name])
                  for feature in base.FEATURES},
        "dev_expected_cost_by_action": dev_costs[name].mean(axis=0).tolist(),
    } for name in BACKENDS}


def selector_actions(features, selectors):
    return {name: {feature: base.apply_rule(features[feature], selector["rules"][feature])
                   for feature in base.FEATURES}
            for name, selector in selectors.items()}


def _ci(difference, seed, bootstraps, direction):
    result = base.paired_bootstrap(difference, np.random.default_rng(seed), bootstraps)
    result["difference_direction"] = direction
    return result


def cross_evaluate(costs, features, selectors, bootstrap_seed, bootstraps=1000, include_ci=True):
    """Frozen selector rows x renderer columns. No selector training in this function."""
    actions = selector_actions(features, selectors)
    n = len(next(iter(costs.values())))
    if any(np.asarray(c).shape != (n, len(base.EXPOSURES)) for c in costs.values()):
        raise ValueError("cross-evaluation costs must have identical source/action dimensions")
    indices = np.arange(n)
    matrix = {}
    for selector_name, selector in selectors.items():
        matrix[selector_name] = {}
        fixed_action = selector["fixed_action_index"]
        for renderer in BACKENDS:
            c = costs[renderer]
            fixed_cost = c[:, fixed_action]
            policies, selected = {}, {}
            for feature in base.FEATURES:
                action = actions[selector_name][feature]
                value = c[indices, action]
                selected[feature] = value
                policies[feature] = {
                    "expected_cost": float(value.mean()),
                    "action_counts": np.bincount(action, minlength=len(base.EXPOSURES)).tolist(),
                    "mean_difference_vs_selector_fixed": float((value-fixed_cost).mean()),
                }
                if include_ci:
                    policies[feature]["paired_ci_vs_selector_fixed"] = _ci(
                        value-fixed_cost, bootstrap_seed, bootstraps,
                        "rule minus this source-selector's dev-selected fixed; negative favors rule")
            delta = selected["known_linear_residual"] - selected["global_brightness"]
            residual = {"mean_difference": float(delta.mean()),
                        "difference_direction": "known residual minus brightness; negative favors residual"}
            if include_ci:
                residual = _ci(delta, bootstrap_seed, bootstraps,
                               "known residual minus brightness; negative favors residual")
            matrix[selector_name][renderer] = {
                "selector_trained_against": selector_name, "evaluated_renderer": renderer,
                "fixed_action_index_from_selector_dev": fixed_action,
                "fixed_exposure_from_selector_dev": float(base.EXPOSURES[fixed_action]),
                "fixed_expected_cost": float(fixed_cost.mean()), "policies": policies,
                "paired_residual_minus_brightness": residual,
            }
    return matrix


def _node_difference(costs, highest_costs, features, selectors):
    indices = np.arange(len(next(iter(costs.values()))))
    actions = selector_actions(features, selectors)
    report = {"per_renderer": {}, "cross_policy_risk_difference": {}}
    for renderer in BACKENDS:
        diff = costs[renderer] - highest_costs[renderer]
        report["per_renderer"][renderer] = {
            "risk_difference_by_action_vs_highest_nodes": diff.mean(axis=0).tolist(),
            "max_abs_action_risk_difference": float(np.max(np.abs(diff.mean(axis=0)))),
            "max_abs_source_action_cost_difference": float(np.max(np.abs(diff))),
        }
    for selector_name, selector in selectors.items():
        report["cross_policy_risk_difference"][selector_name] = {}
        for renderer in BACKENDS:
            diff = costs[renderer] - highest_costs[renderer]
            values = {feature: float(diff[indices, actions[selector_name][feature]].mean())
                      for feature in base.FEATURES}
            values["fixed"] = float(diff[:, selector["fixed_action_index"]].mean())
            report["cross_policy_risk_difference"][selector_name][renderer] = values
    return report


def _write_source_arrays(path, seed, population, node_counts, all_costs, history, cube, features, selectors):
    n = len(history)
    actions = selector_actions(features, selectors)
    source_ids = np.array([f"{population}/seed_{seed}/source_{i:06d}" for i in range(n)])
    costs = np.stack([np.stack([all_costs[count][name] for name in BACKENDS]) for count in node_counts])
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, source_id=source_ids, source_index=np.arange(n),
            history_adc_codes=history, history_exposures=np.full(3, .25), future_adc_codes=cube,
            expected_costs_by_nodes_backend_source_action=costs,
            cost_axis_names=np.array(["nodes", "renderer", "source", "action"]),
            nodes=np.array(node_counts), renderer_names=np.array(BACKENDS), selector_names=np.array(BACKENDS),
            feature_names=np.array(list(base.FEATURES)),
            preview_features=np.column_stack([features[f] for f in base.FEATURES]),
            frozen_policy_actions_by_selector_source_feature=np.stack([
                np.column_stack([actions[name][f] for f in base.FEATURES]) for name in BACKENDS]),
            fixed_action_indices_by_selector=np.array([selectors[name]["fixed_action_index"] for name in BACKENDS]),
            exposures=base.EXPOSURES)
    return Path(path).name


def run_diagnostic(*, seeds=(19, 37, 73), train_sources=1024, dev_sources=1024,
                   diagnostic_sources=2048, repeats=16, nodes=(128, 256),
                   selection_nodes=None, train_prior=base.Prior(), heldout_prior=base.Prior(copula="reverse"),
                   chunk=256, likelihood_chunk=64, bootstraps=1000,
                   source_metrics_dir=None, smoke=False):
    if min(train_sources, dev_sources, diagnostic_sources) < 2 or repeats < 1 or bootstraps < 2:
        raise ValueError("need >=2 train/dev/diagnostic sources, >=1 repeat, >=2 bootstrap replicates")
    node_counts = sorted(set(int(n) for n in nodes))
    if not node_counts or min(node_counts) < 2:
        raise ValueError("integration nodes must be >=2")
    selection_nodes = max(node_counts) if selection_nodes is None else int(selection_nodes)
    if source_metrics_dir is not None:
        source_metrics_dir = Path(source_metrics_dir)
        for seed in seeds:
            for population in ("N0_diagnostic", "N1_dependence_diagnostic"):
                if (source_metrics_dir/f"seed_{seed}_{population}.npz").exists():
                    raise FileExistsError("source metric file exists; preserve evidence and use a new directory")
        source_metrics_dir.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    likelihood = base.ADCLikelihood()
    report = {
        "status": "completed_history_software_numerical_smoke" if smoke else "completed_history_simulated_diagnostic",
        "protocol": "research/ae_tm/diagnostics/HISTORY_PROTOCOL.md; FROZEN 2026-10-09",
        "scope": "independent N0/N1 toy simulation; preserves old results; no S24/real-camera/original CaptureTM evidence",
        "backend_conditional_scopes": SCOPES,
        "action_conditioning": "do(e); conditional on H the deterministic selected action supplies no extra information",
        "config": {"seeds": list(seeds), "train_sources": train_sources, "dev_sources": dev_sources,
                   "sources_per_diagnostic": diagnostic_sources, "future_noise_repeats": repeats,
                   "nodes": node_counts, "selection_nodes": selection_nodes, "chunk": chunk,
                   "likelihood_chunk": likelihood_chunk, "bootstraps": bootstraps},
        "training_prior": asdict(train_prior), "N0_true_dependence": asdict(train_prior),
        "N1_true_dependence": asdict(heldout_prior), "N1_note": "both renderers retain training prior, not prior-shift Bayes optimality",
        "independent_stream_children": STREAMS, "policy_features": base.FEATURES,
        "exposures": base.EXPOSURES.tolist(),
        "physical_chain": {"full_well_electrons": base.FW, "read_sd_electrons": base.READ_SD,
                           "adc_max_code": base.ADC_MAX, "target": "sqrt(l/(l+.5))",
                           "shot_noise": "Poisson", "past_preview_frames": 3, "past_preview_exposure": .25},
        "likelihood_audit": likelihood.audit(),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "source_metric_directory": source_metrics_dir.name if source_metrics_dir is not None else None,
        "replicates": [],
    }
    for seed in seeds:
        start = time.perf_counter()
        children = np.random.SeedSequence(seed).spawn(13)
        rng = [np.random.default_rng(s) for s in children]
        train = base.sample_sources(train_sources, rng[0], train_prior)
        _, train_features = base.previews_and_features(train, rng[1])
        dev = base.sample_sources(dev_sources, rng[2], train_prior)
        dev_history, dev_features = base.previews_and_features(dev, rng[3])
        dev_cube = base.future_codes(dev, repeats, rng[4])
        frozen = {}
        for population, prior, si, hi, fi in [
            ("N0_diagnostic", train_prior, 5, 6, 7),
            ("N1_dependence_diagnostic", heldout_prior, 8, 9, 10)]:
            latents = base.sample_sources(diagnostic_sources, rng[si], prior)
            history, features = base.previews_and_features(latents, rng[hi])
            frozen[population] = (latents, history, features, base.future_codes(latents, repeats, rng[fi]))
        selection_start = time.perf_counter()
        selector_reference = HistoryReference(train_prior, selection_nodes, chunk, likelihood_chunk, likelihood)
        dev_costs, _ = evaluate_backends(selector_reference, dev, dev_history, dev_cube)
        selectors = freeze_selectors(train_features, dev_features, dev_costs)
        row = {"seed": int(seed), "frozen_selectors": selectors,
               "selection_seconds": time.perf_counter()-selection_start,
               "diagnostics_by_nodes": {}, "nodes_convergence": {}, "source_metrics_files": {}}
        all_costs = {population: {} for population in frozen}
        for count in node_counts:
            node_start = time.perf_counter()
            reference = HistoryReference(train_prior, count, chunk, likelihood_chunk, likelihood)
            node_report = {}
            for population, (latents, history, features, cube) in frozen.items():
                costs, noise = evaluate_backends(reference, latents, history, cube)
                all_costs[population][count] = costs
                node_report[population] = {
                    "expected_cost_by_renderer_and_action": {name: costs[name].mean(axis=0).tolist() for name in BACKENDS},
                    "backend_cross_matrix": cross_evaluate(costs, features, selectors,
                        children[11 if population == "N0_diagnostic" else 12], bootstraps,
                        include_ci=(count == max(node_counts))),
                    **noise,
                }
            node_report["elapsed_seconds"] = time.perf_counter()-node_start
            row["diagnostics_by_nodes"][str(count)] = node_report
        for count in node_counts:
            row["nodes_convergence"][str(count)] = {
                population: _node_difference(all_costs[population][count],
                    all_costs[population][max(node_counts)], features, selectors)
                for population, (_, _, features, _) in frozen.items()}
        if source_metrics_dir is not None:
            for population, (_, history, features, cube) in frozen.items():
                path = source_metrics_dir/f"seed_{seed}_{population}.npz"
                row["source_metrics_files"][population] = _write_source_arrays(path, seed, population,
                    node_counts, all_costs[population], history, cube, features, selectors)
        row["elapsed_seconds"] = time.perf_counter()-start
        report["replicates"].append(row)
    report["elapsed_seconds"] = time.perf_counter()-begin
    report["limitations"] = [
        "known explicit training prior and known residual relation; neither backend claims unknown-prior optimality",
        "H_blind intentionally omits H and observational selection information; H_aware conditions on all three observed previews",
        "N1 is a true dependence shift with the training prior frozen; no prior-shift Bayes optimality claim",
        "finite nodes and future-noise repetitions; CI conditional on frozen policy and repeated captures",
        "shared ADC kernel tail approximation/rare likelihood underflow; no Gaussian fallback",
        "no S24 formal test, real camera, full-image prior, neural training, or motion evaluation",
    ]
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-metrics-dir", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[19, 37, 73])
    p.add_argument("--train-sources", type=int, default=1024)
    p.add_argument("--dev-sources", type=int, default=1024)
    p.add_argument("--diagnostic-sources", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--nodes", type=int, nargs="+", default=[128, 256])
    p.add_argument("--selection-nodes", type=int)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--likelihood-chunk", type=int, default=64)
    p.add_argument("--bootstraps", type=int, default=1000)
    for prefix, copula in (("train", "linear"), ("heldout", "reverse")):
        p.add_argument(f"--{prefix}-w", type=float, default=.5)
        p.add_argument(f"--{prefix}-copula", choices=["linear", "reverse", "rotate"], default=copula)
        p.add_argument(f"--{prefix}-shift", type=float, default=.5)
    a = p.parse_args(argv)
    if a.out.exists():
        p.error("output exists; preserve evidence and use a new output path")
    if a.smoke:
        a.seeds, a.train_sources, a.dev_sources, a.diagnostic_sources = [19], 32, 24, 32
        a.repeats, a.nodes, a.selection_nodes, a.bootstraps = 2, [64, 128], 128, 64
    metrics_dir = a.source_metrics_dir if a.source_metrics_dir is not None else a.out.parent/(a.out.stem+"_sources")
    result = run_diagnostic(seeds=a.seeds, train_sources=a.train_sources, dev_sources=a.dev_sources,
        diagnostic_sources=a.diagnostic_sources, repeats=a.repeats, nodes=a.nodes,
        selection_nodes=a.selection_nodes, train_prior=base.Prior(a.train_w, a.train_copula, a.train_shift),
        heldout_prior=base.Prior(a.heldout_w, a.heldout_copula, a.heldout_shift),
        chunk=a.chunk, likelihood_chunk=a.likelihood_chunk, bootstraps=a.bootstraps,
        source_metrics_dir=metrics_dir, smoke=a.smoke)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "seeds": result["config"]["seeds"], "nodes": result["config"]["nodes"],
                      "source_metric_files": [r["source_metrics_files"] for r in result["replicates"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
