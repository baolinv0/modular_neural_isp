"""N0/N1 numerical physical-chain diagnostic; no S24 or original CaptureTM test.

Run from repository root:
  python research/ae_tm/diagnostics/continuous_reference.py --smoke --out PATH
Arrays are transient. The only output is an explicitly requested JSON report.
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
from scipy import sparse
from scipy.special import ndtr
from scipy.stats import poisson

EXPOSURES = np.array([.125, .25, .5, 1., 2., 4., 8.])
FW = 1000.
READ_SD = 4.
ADC_MAX = 4095
READ_SIGMAS = 8.
READ_TAIL_BOUND = float(3 * ndtr(-READ_SIGMAS))
FEATURES = {
    "known_linear_residual": "abs(mean(b_preview/.25)-(4*mean(a_preview/.25)+.6))",
    "global_brightness": "mean of both exposure-compensated preview pixel means",
    "temporal_noise": "RMS of per-pixel temporal sample SD from three compensated previews",
}


def tone(radiance):
    x = np.asarray(radiance, dtype=np.float64)
    return np.sqrt(x / (x + .5))


@dataclass(frozen=True)
class Prior:
    """Uniform marginals plus a measure-preserving 1D copula component."""
    weight: float = .5
    copula: str = "linear"
    shift: float = .5

    def __post_init__(self):
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("prior weight must be in [0,1]")
        if self.copula not in ("linear", "reverse", "rotate"):
            raise ValueError("copula must be linear, reverse, or rotate")
        if not np.isfinite(self.shift) or not 0 <= self.shift < 1:
            raise ValueError("rotation shift must be in [0,1)")

    def map_uniform(self, u):
        if self.copula == "linear":
            return np.asarray(u)
        if self.copula == "reverse":
            return 1 - np.asarray(u)
        return (np.asarray(u) + self.shift) % 1


def quadrature(nodes: int, breaks=(0., 1.)):
    """Composite Gauss-Legendre rule with exactly nodes total uniform weights."""
    if nodes < 2 or nodes < len(breaks) - 1:
        raise ValueError("need at least two nodes and one per interval")
    intervals = np.diff(breaks)
    allocation = np.maximum(1, np.floor(nodes * intervals).astype(int))
    while allocation.sum() < nodes:
        i = int(np.argmax(nodes * intervals - allocation))
        allocation[i] += 1
    while allocation.sum() > nodes:
        choices = np.where(allocation > 1, allocation - nodes * intervals, -np.inf)
        allocation[int(np.argmax(choices))] -= 1
    xs, ws = [], []
    for low, high, count in zip(breaks[:-1], breaks[1:], allocation):
        x, w = np.polynomial.legendre.leggauss(int(count))
        xs.append(low + (x + 1) * (high - low) / 2)
        ws.append(w * (high - low) / 2)
    return np.concatenate(xs), np.concatenate(ws)


class ADCLikelihood:
    """Sparse read/ADC kernel times the *Poisson* PMF, with complete upper tail.

    ADC 0 corresponds to (-inf, FW*.5/ADC_MAX); ADC_MAX corresponds to
    [FW*(ADC_MAX-.5)/ADC_MAX, inf). The finite integer-count kernel retains
    all cells intersecting +/-8 read sigmas. Counts above its final integer
    are included in ADC_MAX by a complete Poisson survival function. Only
    tiny Gaussian read-tail terms are approximated, not the Poisson tails.
    """
    def __init__(self):
        self.last_count = int(np.ceil(FW + READ_SIGMAS * READ_SD))
        self.counts = np.arange(self.last_count + 1, dtype=np.float64)
        row_parts, col_parts, values = [], [], []
        for code in range(ADC_MAX + 1):
            low = FW * (code - .5) / ADC_MAX
            high = FW * (code + .5) / ADC_MAX
            if code == 0:
                counts = np.arange(0, int(np.floor(high + READ_SIGMAS * READ_SD)) + 1)
                probs = ndtr((high - counts) / READ_SD)
            elif code == ADC_MAX:
                first = max(0, int(np.ceil(low - READ_SIGMAS * READ_SD)))
                counts = np.arange(first, self.last_count + 1)
                probs = ndtr((counts - low) / READ_SD)
            else:
                first = max(0, int(np.ceil(low - READ_SIGMAS * READ_SD)))
                last = min(self.last_count, int(np.floor(high + READ_SIGMAS * READ_SD)))
                counts = np.arange(first, last + 1)
                zlo, zhi = (low - counts) / READ_SD, (high - counts) / READ_SD
                # Survival differences avoid cancellation in the positive tail.
                probs = np.where(zlo >= 0, ndtr(-zlo) - ndtr(-zhi),
                                 ndtr(zhi) - ndtr(zlo))
            row_parts.append(np.full(len(counts), code, dtype=np.int32))
            col_parts.append(counts.astype(np.int32))
            values.append(probs)
        self.kernel = sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(ADC_MAX + 1, self.last_count + 1))

    def probabilities(self, codes, poisson_means, chunk=64):
        """Return P(ADC=codes | Poisson means), shape (codes, means)."""
        raw_codes = np.asarray(codes)
        if raw_codes.ndim != 1 or np.any(raw_codes != np.rint(raw_codes)):
            raise ValueError("ADC codes must be a one-dimensional integer array")
        codes = raw_codes.astype(np.int64)
        means = np.asarray(poisson_means, dtype=np.float64)
        if means.ndim != 1 or np.any(~np.isfinite(means)) or np.any(means < 0):
            raise ValueError("Poisson means must be one-dimensional, finite and nonnegative")
        if np.any((codes < 0) | (codes > ADC_MAX)) or chunk < 1:
            raise ValueError("ADC code or likelihood chunk out of range")
        out = np.empty((len(codes), len(means)), dtype=np.float64)
        kernel = self.kernel[codes]
        top = codes == ADC_MAX
        for start in range(0, len(means), chunk):
            mu = means[start:start + chunk]
            pmf = poisson.pmf(self.counts[:, None], mu[None, :])
            value = kernel @ pmf
            if np.any(top):
                value[top] += poisson.sf(self.last_count, mu)
            out[:, start:start + chunk] = value
        return out

    def audit(self):
        means = np.array([0., 1., 100., 500., 999., 1000., 1500., 14400.])
        total = self.probabilities(np.arange(ADC_MAX + 1), means).sum(axis=0)
        return {
            "poisson_means": means.tolist(),
            "probability_sum": total.tolist(),
            "max_normalization_abs_error": float(np.max(np.abs(total - 1))),
            "read_kernel_sigmas": READ_SIGMAS,
            "per_cell_absolute_read_tail_bound": READ_TAIL_BOUND,
            "last_explicit_poisson_count": self.last_count,
            "upper_poisson_tail": "complete scipy.stats.poisson.sf(last_count, mean)",
        }


class ContinuousReference:
    """Conditional mean of the common tone target under a known uniform prior."""
    def __init__(self, prior=Prior(), nodes=128, chunk=256,
                 likelihood_chunk=64, likelihood=None):
        if chunk < 1 or likelihood_chunk < 1:
            raise ValueError("chunks must be positive")
        self.prior, self.nodes = prior, int(nodes)
        self.chunk, self.likelihood_chunk = int(chunk), int(likelihood_chunk)
        self.likelihood = likelihood if likelihood is not None else ADCLikelihood()
        u, self.weights = quadrature(self.nodes)
        self.a, self.b = .05 + .25 * u, .8 + u
        if prior.copula == "rotate" and prior.shift > 0:
            uc, self.cweights = quadrature(self.nodes, (0., 1 - prior.shift, 1.))
        else:
            uc, self.cweights = u, self.weights
        self.ca, self.cb = .05 + .25 * uc, .8 + prior.map_uniform(uc)

    def _tables(self, codes, latent, exposure):
        unique, inverse = np.unique(codes, return_inverse=True)
        p = self.likelihood.probabilities(unique, FW * exposure * latent,
                                          self.likelihood_chunk)
        return p, inverse

    def posterior(self, adc_pairs, exposure):
        """Return (two target means, posterior copula probability); no latent input."""
        codes = np.asarray(adc_pairs)
        if codes.ndim != 2 or codes.shape[1] != 2:
            raise ValueError("observation must have shape (n,2)")
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("known exposure must be positive")
        la, ia = self._tables(codes[:, 0], self.a, exposure)
        lb, ib = self._tables(codes[:, 1], self.b, exposure)
        if np.array_equal(self.a, self.ca):
            lca, ica = la, ia
        else:
            lca, ica = self._tables(codes[:, 0], self.ca, exposure)
        if np.array_equal(self.b, self.cb):
            lcb, icb = lb, ib
        else:
            lcb, icb = self._tables(codes[:, 1], self.cb, exposure)
        wa, wb = self.weights * tone(self.a), self.weights * tone(self.b)
        wca, wcb = self.cweights * tone(self.ca), self.cweights * tone(self.cb)
        out = np.empty((len(codes), 2), dtype=np.float64)
        gamma = np.empty(len(codes), dtype=np.float64)
        for start in range(0, len(codes), self.chunk):
            end = start + self.chunk
            pa, pb = la[ia[start:end]], lb[ib[start:end]]
            za, zb = pa @ self.weights, pb @ self.weights
            independent = za * zb
            paired = lca[ica[start:end]] * lcb[icb[start:end]]
            correlated = paired @ self.cweights
            z = (1 - self.prior.weight) * independent + self.prior.weight * correlated
            if np.any(z <= 0) or np.any(~np.isfinite(z)):
                raise FloatingPointError("zero/nonfinite evidence: likelihood underflow or quadrature miss")
            na = ((1 - self.prior.weight) * (pa @ wa) * zb +
                  self.prior.weight * (paired @ wca))
            nb = ((1 - self.prior.weight) * (pb @ wb) * za +
                  self.prior.weight * (paired @ wcb))
            out[start:end, 0], out[start:end, 1] = na / z, nb / z
            gamma[start:end] = self.prior.weight * correlated / z
        return out, gamma

    def predict(self, adc_pairs, exposure):
        return self.posterior(adc_pairs, exposure)[0]


def sample_sources(n, rng, prior=Prior()):
    """IID source pairs; true component IDs are deliberately not returned."""
    if n < 1:
        raise ValueError("need at least one source")
    u, v = rng.random(n), rng.random(n)
    component = rng.random(n) < prior.weight
    v[component] = prior.map_uniform(u[component])
    return np.column_stack((.05 + .25 * u, .8 + v))


def capture_codes(radiance, exposure, rng):
    x = np.asarray(radiance, dtype=np.float64)
    electrons = rng.poisson(FW * exposure * x) + rng.normal(0, READ_SD, x.shape)
    return np.rint(np.clip(electrons, 0, FW) * ADC_MAX / FW).astype(np.int16)


def previews_and_features(latents, rng):
    codes = capture_codes(np.broadcast_to(latents[:, None, :], (len(latents), 3, 2)),
                          .25, rng)
    compensated = codes.astype(np.float64) / ADC_MAX / .25
    means = compensated.mean(axis=1)
    features = {
        "known_linear_residual": np.abs(means[:, 1] - (4 * means[:, 0] + .6)),
        "global_brightness": means.mean(axis=1),
        "temporal_noise": np.sqrt(compensated.var(axis=1, ddof=1).mean(axis=1)),
    }
    return codes, features


def future_codes(latents, repeats, rng):
    if repeats < 1:
        raise ValueError("future repetitions must be positive")
    # Source first: all captures of one source remain one bootstrap unit.
    cube = np.empty((len(latents), len(EXPOSURES), repeats, 2), dtype=np.int16)
    radiance = np.broadcast_to(latents[:, None, :], (len(latents), repeats, 2))
    for i, e in enumerate(EXPOSURES):
        cube[:, i] = capture_codes(radiance, e, rng)
    return cube


def evaluate_costs(reference, latents, cube):
    if cube.shape[:2] != (len(latents), len(EXPOSURES)) or cube.shape[-1] != 2:
        raise ValueError("future observation cube shape mismatch")
    repeats = cube.shape[2]
    target = tone(latents)[:, None, :]
    losses = np.empty(cube.shape[:-1], dtype=np.float64)
    for i, e in enumerate(EXPOSURES):
        pred = reference.predict(cube[:, i].reshape(-1, 2), float(e))
        pred = pred.reshape(len(latents), repeats, 2)
        losses[:, i] = ((pred - target) ** 2).mean(axis=2)
    costs = losses.mean(axis=2)
    if repeats > 1:
        mc_se = np.sqrt(losses.var(axis=2, ddof=1).mean(axis=0) / len(latents) / repeats)
    else:
        mc_se = np.full(len(EXPOSURES), np.nan)
    diagnostics = {
        "future_mc_se_by_action": [float(x) if np.isfinite(x) else None for x in mc_se],
        "hindsight_min_realized_future_cost": float(losses.min(axis=1).mean()),
        "hindsight_note": "uses unavailable realized future noise; not an oracle or causal bound",
    }
    return costs, diagnostics


def select_threshold(train_feature, dev_feature, dev_costs):
    candidates = np.unique(np.quantile(train_feature, np.arange(.1, 1., .1)))
    best = None
    for threshold in candidates:
        left = dev_feature <= threshold
        if not np.any(left) or np.all(left):
            continue
        li, ri = int(dev_costs[left].mean(axis=0).argmin()), int(dev_costs[~left].mean(axis=0).argmin())
        score = float(np.where(left, dev_costs[:, li], dev_costs[:, ri]).mean())
        # Stable enumeration breaks ties without looking at diagnostic outcomes.
        if best is None or score < best["dev_expected_cost"]:
            best = {"threshold": float(threshold), "left_action_index": li,
                    "right_action_index": ri, "dev_expected_cost": score,
                    "candidate_thresholds_from_train": candidates.tolist()}
    if best is None:
        fixed = int(dev_costs.mean(axis=0).argmin())
        best = {"threshold": float(np.median(train_feature)),
                "left_action_index": fixed, "right_action_index": fixed,
                "dev_expected_cost": float(dev_costs[:, fixed].mean()),
                "candidate_thresholds_from_train": candidates.tolist(),
                "degenerate_note": "no nonempty dev partition; same fixed action on both sides"}
    best["actions"] = [float(EXPOSURES[best["left_action_index"]]),
                       float(EXPOSURES[best["right_action_index"]])]
    return best


def apply_rule(feature, rule):
    return np.where(feature <= rule["threshold"], rule["left_action_index"],
                    rule["right_action_index"])


def paired_bootstrap(difference, rng, replicates=1000, chunk=64):
    x = np.asarray(difference, dtype=np.float64)
    if len(x) < 2 or replicates < 2 or chunk < 1:
        raise ValueError("bootstrap requires >=2 sources/replicates and positive chunk")
    sampled = np.empty(replicates)
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        idx = rng.integers(0, len(x), (count, len(x)))
        sampled[start:start + count] = x[idx].mean(axis=1)
    return {"mean_difference": float(x.mean()),
            "paired_independent_source_percentile_ci95": np.quantile(sampled, [.025, .975]).tolist(),
            "bootstrap_replicates": replicates, "independent_source_count": len(x),
            "difference_direction": "policy minus dev-selected fixed; negative favors policy"}


def summarize(costs, features, rules, fixed, rng, bootstraps):
    indices = np.arange(len(costs))
    fixed_cost = costs[:, fixed]
    policies = {}
    for name, rule in rules.items():
        actions = apply_rule(features[name], rule)
        policy_cost = costs[indices, actions]
        policies[name] = {
            "expected_cost": float(policy_cost.mean()),
            "action_counts": np.bincount(actions, minlength=len(EXPOSURES)).tolist(),
            "relative_reduction_vs_fixed": float(1 - policy_cost.mean() / fixed_cost.mean()),
            "paired_comparison": paired_bootstrap(policy_cost - fixed_cost, rng, bootstraps),
        }
    return {"expected_cost_by_action": costs.mean(axis=0).tolist(),
            "dev_selected_fixed_exposure": float(EXPOSURES[fixed]),
            "fixed_expected_cost": float(fixed_cost.mean()), "policies": policies}


def run_diagnostic(*, seeds=(19, 37, 73), train_sources=1024, dev_sources=1024,
                   diagnostic_sources=2048, repeats=16, nodes=(32, 64, 128, 256),
                   selection_nodes=None, train_prior=Prior(), heldout_prior=Prior(copula="reverse"),
                   chunk=256, likelihood_chunk=64, bootstraps=1000, smoke=False):
    if min(train_sources, dev_sources, diagnostic_sources) < 2:
        raise ValueError("train/dev/diagnostic need at least two independent sources")
    if bootstraps < 2:
        raise ValueError("need at least two bootstrap replicates")
    node_counts = sorted(set(int(n) for n in nodes))
    if not node_counts or min(node_counts) < 2:
        raise ValueError("need integration nodes >=2")
    if selection_nodes is None:
        selection_nodes = max(node_counts)
    likelihood = ADCLikelihood()
    begin = time.perf_counter()
    report = {
        "status": "completed_software_numerical_smoke" if smoke else "completed_simulated_diagnostic",
        "scope": "independent N0/N1 two-pixel simulation; not S24, real camera or original CaptureTM evaluation",
        "protocol": "research/ae_tm/diagnostics/N0_N1_PROTOCOL.md; FROZEN 2026-10-09",
        "exposures": EXPOSURES.tolist(),
        "physical_chain": {"full_well_electrons": FW, "read_sd_electrons": READ_SD,
                           "adc_max_code": ADC_MAX, "target": "sqrt(l/(l+.5))",
                           "shot_noise": "Poisson; no Gaussian shot-noise approximation",
                           "metadata": "known exposure; analog and digital gain 1"},
        "config": {"seeds": list(seeds), "train_sources": train_sources,
                   "dev_sources": dev_sources, "sources_per_diagnostic": diagnostic_sources,
                   "future_noise_repeats": repeats, "nodes": node_counts,
                   "selection_nodes": selection_nodes, "chunk": chunk,
                   "likelihood_chunk": likelihood_chunk, "bootstraps": bootstraps},
        "training_renderer_prior": asdict(train_prior),
        "N0_true_dependence": asdict(train_prior),
        "N1_true_dependence": asdict(heldout_prior),
        "N1_changes_actual_copula": (heldout_prior.copula != train_prior.copula or
            (heldout_prior.copula == "rotate" and heldout_prior.shift != train_prior.shift)),
        "known_prior_note": "renderer knows explicit training uniform mixture; residual policy knows b=4a+.6",
        "policy_features": FEATURES,
        "likelihood_audit": likelihood.audit(),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "replicates": [],
    }
    for seed in seeds:
        seed_begin = time.perf_counter()
        # Independent named streams for sources, previews, future captures, and bootstrap.
        seed_streams = np.random.SeedSequence(seed).spawn(13)
        streams = [np.random.default_rng(s) for s in seed_streams]
        train = sample_sources(train_sources, streams[0], train_prior)
        _, train_features = previews_and_features(train, streams[1])
        dev = sample_sources(dev_sources, streams[2], train_prior)
        _, dev_features = previews_and_features(dev, streams[3])
        dev_cube = future_codes(dev, repeats, streams[4])
        populations = {
            "N0_diagnostic": (sample_sources(diagnostic_sources, streams[5], train_prior), streams[6], streams[7]),
            "N1_dependence_diagnostic": (sample_sources(diagnostic_sources, streams[8], heldout_prior), streams[9], streams[10]),
        }
        frozen = {}
        for name, (latents, preview_rng, future_rng) in populations.items():
            _, features = previews_and_features(latents, preview_rng)
            frozen[name] = (latents, features, future_codes(latents, repeats, future_rng))
        selection_begin = time.perf_counter()
        selector_reference = ContinuousReference(train_prior, selection_nodes, chunk,
                                                 likelihood_chunk, likelihood)
        dev_costs, _ = evaluate_costs(selector_reference, dev, dev_cube)
        fixed = int(dev_costs.mean(axis=0).argmin())
        rules = {name: select_threshold(train_features[name], dev_features[name], dev_costs)
                 for name in FEATURES}
        row = {"seed": int(seed), "frozen_rules": rules,
               "dev_expected_cost_by_action": dev_costs.mean(axis=0).tolist(),
               "dev_selected_fixed_action_index": fixed,
               "selection_seconds": time.perf_counter() - selection_begin,
               "diagnostics_by_nodes": {}, "nodes_convergence": {}}
        all_costs = {}
        for count in node_counts:
            node_begin = time.perf_counter()
            reference = ContinuousReference(train_prior, count, chunk, likelihood_chunk, likelihood)
            node_row, all_costs[count] = {}, {}
            for name, (latents, features, cube) in frozen.items():
                costs, noise_diagnostics = evaluate_costs(reference, latents, cube)
                # Same independent-source resample indices across node counts.
                brng = np.random.default_rng(seed_streams[11 if name == "N0_diagnostic" else 12])
                summary = summarize(costs, features, rules, fixed, brng, bootstraps)
                summary.update(noise_diagnostics)
                node_row[name] = summary
                all_costs[count][name] = costs
            node_row["elapsed_seconds"] = time.perf_counter() - node_begin
            row["diagnostics_by_nodes"][str(count)] = node_row
        highest = max(node_counts)
        for count in node_counts:
            convergence = {}
            for name, (_, features, _) in frozen.items():
                diff = all_costs[count][name] - all_costs[highest][name]
                idx = np.arange(len(diff))
                convergence[name] = {
                    "risk_difference_by_action_vs_highest_nodes": diff.mean(axis=0).tolist(),
                    "max_abs_action_risk_difference": float(np.max(np.abs(diff.mean(axis=0)))),
                    "max_abs_source_action_cost_difference": float(np.max(np.abs(diff))),
                    "policy_risk_difference_vs_highest_nodes": {
                        feature: float(diff[idx, apply_rule(features[feature], rule)].mean())
                        for feature, rule in rules.items()},
                }
            row["nodes_convergence"][str(count)] = convergence
        row["node_reference_note"] = "highest requested node count is a numerical reference, not proof of exact convergence"
        row["elapsed_seconds"] = time.perf_counter() - seed_begin
        report["replicates"].append(row)
    report["elapsed_seconds"] = time.perf_counter() - begin
    report["limitations"] = [
        "finite Gauss-Legendre integration and finite future-noise Monte Carlo",
        "paired CI resamples independent sources conditional on frozen rule and finite repeated captures",
        "known toy training prior and known linear residual relation; not unrestricted image Bayes",
        "no real-camera, S24 formal test, unknown full-image prior, motion or neural training validation",
        "extremely rare observations may underflow likelihood/evidence and raise; no Gaussian fallback",
    ]
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[19, 37, 73])
    p.add_argument("--train-sources", type=int, default=1024)
    p.add_argument("--dev-sources", type=int, default=1024)
    p.add_argument("--diagnostic-sources", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--nodes", type=int, nargs="+", default=[32, 64, 128, 256])
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
        p.error("output already exists; use a new output path to preserve evidence")
    if a.smoke:
        a.seeds, a.train_sources, a.dev_sources, a.diagnostic_sources = [19], 32, 24, 32
        a.repeats, a.nodes, a.selection_nodes, a.bootstraps = 2, [32, 64], 64, 64
    train = Prior(a.train_w, a.train_copula, a.train_shift)
    heldout = Prior(a.heldout_w, a.heldout_copula, a.heldout_shift)
    result = run_diagnostic(seeds=a.seeds, train_sources=a.train_sources,
        dev_sources=a.dev_sources, diagnostic_sources=a.diagnostic_sources,
        repeats=a.repeats, nodes=a.nodes, selection_nodes=a.selection_nodes,
        train_prior=train, heldout_prior=heldout, chunk=a.chunk,
        likelihood_chunk=a.likelihood_chunk, bootstraps=a.bootstraps, smoke=a.smoke)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a concurrent run from overwriting a report.
    with a.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "seeds": result["config"]["seeds"], "nodes": result["config"]["nodes"],
                      "max_likelihood_normalization_error":
                          result["likelihood_audit"]["max_normalization_abs_error"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
