"""Necessary local tests for DNG H-information control; no external review/pixels."""
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.ndimage import uniform_filter

D = Path(__file__).resolve().parents[1]/"diagnostics"
sys.path.insert(0, str(D))
import dng_history_control as ctl

old = ctl.legacy


class FusionTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(19)
        h = rng.integers(0, 1100, (3, 2, 6, 6), dtype=np.int16)
        y = rng.integers(0, 4096, (3, 2, 6, 6), dtype=np.int16)
        h[0, 0, 0, 0], y[1, 0, 0, 0] = 4095, 4095
        return h, y

    def test_variance_formula_and_fusion_match_independent_explicit_wls(self):
        h, y = self.data()
        exposure = 2.
        estimate, variance, no_weight = ctl.classical_fusion_estimate(h, y, exposure)
        xhat = np.clip((h.astype(float)/4095/.25).mean(axis=0), 0, 1)
        for e in [.125, .25, 2., 8.]:
            exact = (1000*e*xhat+16)/(1000000*e*e)+1/(12*4095*4095*e*e)
            np.testing.assert_allclose(ctl.normalized_measurement_variance(xhat, e), exact, rtol=1e-15, atol=0)
        for repeat in range(len(y)):
            observations = np.concatenate([h, y[repeat:repeat+1]], axis=0)
            es = np.array([.25, .25, .25, exposure])[:, None, None, None]
            signals = observations.astype(float)/4095/es
            variances = (1000*es*xhat[None]+16)/(1000000*es*es)+1/(12*4095*4095*es*es)
            weights = np.where(observations == 4095, 0., 1/variances)
            expected = np.clip((weights*signals).sum(axis=0)/weights.sum(axis=0), 0, 1)
            np.testing.assert_allclose(estimate[repeat], expected, atol=2e-16, rtol=1e-15)
            np.testing.assert_allclose(variance[repeat], 1/weights.sum(axis=0), atol=2e-18, rtol=1e-15)
        self.assertFalse(no_weight.any())

    def test_other_future_repeat_changes_do_not_change_this_prediction(self):
        h, y = self.data()
        estimate, noise, mask = ctl.classical_fusion_estimate(h, y, 2.)
        expected = ctl.fusion_wiener_target(estimate, noise, 7, mask)
        modified = y.copy()
        modified[2] = 4095
        second = ctl.classical_fusion_estimate(h, modified, 2.)
        actual = ctl.fusion_wiener_target(*second[:2], 7, second[2])
        np.testing.assert_array_equal(actual[:2], expected[:2])
        alone = ctl.classical_fusion_estimate(h, y[:1], 2.)
        np.testing.assert_array_equal(ctl.fusion_wiener_target(*alone[:2], 7, alone[2])[0], expected[0])

    def test_observed_top_drop_zero_keep_and_finite_fallback(self):
        h = np.zeros((3, 1, 2, 2), dtype=np.int16)
        y = np.full((1, 1, 2, 2), 100, dtype=np.int16)
        estimate, noise, mask = ctl.classical_fusion_estimate(h, y, 1.)
        self.assertTrue(np.all(estimate > 0))
        self.assertTrue(np.all(estimate < 100/4095))
        self.assertFalse(mask.any())
        top = np.full_like(y, 4095)
        drop, _, _ = ctl.classical_fusion_estimate(h, top, 1.)
        np.testing.assert_array_equal(drop, 0.)
        top_h = np.full_like(h, 4095)
        fallback, variance, missing = ctl.classical_fusion_estimate(top_h, top, 1.)
        np.testing.assert_array_equal(fallback, 1.)
        np.testing.assert_array_equal(variance, 1.)
        self.assertTrue(missing.all())
        pred = ctl.fusion_wiener_target(fallback, variance, 7, missing)
        np.testing.assert_allclose(pred, old.tone(1.), atol=0, rtol=0)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_local_noise_uses_same_2d_window_and_not_whole_patch_mean(self):
        signal = np.linspace(.1, .9, 64).reshape(1, 1, 8, 8)
        noise = np.zeros_like(signal)
        noise[:, :, :, 4:] = .02
        shape = (1, 1, 3, 3)
        mean = uniform_filter(signal, size=shape, mode="reflect")
        variance = np.maximum(0, uniform_filter(signal*signal, size=shape, mode="reflect")-mean*mean)
        local = uniform_filter(noise, size=shape, mode="reflect")
        gain = np.clip(1-local/np.maximum(variance, 1e-20), 0, 1)
        expected = old.tone(np.clip(mean+gain*(signal-mean), 0, 1))
        actual = ctl.fusion_wiener_target(signal, noise, 3)
        np.testing.assert_array_equal(actual, expected)
        global_gain = np.clip(1-noise.mean()/np.maximum(variance, 1e-20), 0, 1)
        global_pred = old.tone(np.clip(mean+global_gain*(signal-mean), 0, 1))
        self.assertGreater(np.max(np.abs(actual-global_pred)), 1e-5)

    def test_spatial_filters_do_not_cross_patch_or_repeat(self):
        signal = np.zeros((2, 2, 8, 8))
        signal[1, 0] = .4
        signal[0, 1] = .8
        noise = np.full_like(signal, .01)
        pred = ctl.fusion_wiener_target(signal, noise, 7)
        np.testing.assert_array_equal(pred[0, 0], 0.)
        np.testing.assert_array_equal(pred[1, 1], 0.)
        np.testing.assert_allclose(pred[1, 0], old.tone(.4), atol=1e-14)
        changed = signal.copy()
        changed[1, 1] = np.eye(8)
        actual = ctl.fusion_wiener_target(changed, noise, 7)
        np.testing.assert_array_equal(actual[0], pred[0])
        np.testing.assert_array_equal(actual[1, 0], pred[1, 0])

    def test_public_fusion_inputs_are_observations_only(self):
        self.assertEqual(set(inspect.signature(ctl.classical_fusion_estimate).parameters),
                         {"history_codes", "future_codes", "exposure"})
        self.assertEqual(set(inspect.signature(ctl.preview_history_and_features).parameters), {"x", "rng"})

    def test_legal_input_validation_and_bounded_outputs(self):
        h, y = self.data()
        for bad in [h[:2], h.astype(float)+.1, np.full_like(h, -1)]:
            with self.assertRaises(ValueError):
                ctl.classical_fusion_estimate(bad, y, 1.)
        for exposure in [0., -1., np.inf]:
            with self.assertRaises(ValueError):
                ctl.classical_fusion_estimate(h, y, exposure)
        with self.assertRaises(ValueError):
            ctl.classical_fusion_estimate(h, y[:, :1], 1.)
        with self.assertRaises(ValueError):
            ctl.normalized_measurement_variance([-.1], 1.)
        estimate, noise, mask = ctl.classical_fusion_estimate(h, y, .125)
        self.assertTrue(np.all((estimate >= 0) & (estimate <= 1)))
        for w in old.WINDOWS:
            pred = ctl.fusion_wiener_target(estimate, noise, w, mask)
            self.assertTrue(np.all(np.isfinite(pred)))
            self.assertTrue(np.all((pred >= 0) & (pred <= old.tone(1.))))
        with self.assertRaises(ValueError):
            ctl.fusion_wiener_target(estimate, noise, 2)
        with self.assertRaises(ValueError):
            ctl.fusion_wiener_target(estimate, -noise, 3)


class LegacyFlowTests(unittest.TestCase):
    def test_original_H_features_inverse_Wiener_costs_and_rng_are_identical(self):
        x = np.random.default_rng(73).uniform(0, 1, (2, 8, 8))
        children = np.random.SeedSequence([19, 3, 1]).spawn(2)
        preview_rng = np.random.default_rng(children[0])
        history, features = ctl.preview_history_and_features(x, preview_rng)
        old_preview_rng = np.random.default_rng(children[0])
        expected_features = old.preview_features(x, old_preview_rng)
        np.testing.assert_array_equal(features, expected_features)
        self.assertEqual(preview_rng.integers(2**30), old_preview_rng.integers(2**30))
        new_future = np.random.default_rng(children[1])
        old_future = np.random.default_rng(children[1])
        dummy = np.zeros((7, 4096))
        expected = old.source_risks(x, 3, old_future, dummy, dummy)
        actual = ctl.source_risks(x, history, 3, new_future)
        np.testing.assert_array_equal(actual["costs"]["H_blind_inverse"][0], expected["inverse"])
        np.testing.assert_array_equal(actual["costs"]["H_blind_Wiener"], expected["spatial_windows"])
        self.assertEqual(new_future.integers(2**30), old_future.integers(2**30))

    def test_exact_future_capture_order_and_no_extra_draws(self):
        x = np.linspace(0, 1, 64).reshape(1, 8, 8)
        h, _ = ctl.preview_history_and_features(x, np.random.default_rng(19))
        with mock.patch.object(old, "capture_codes", wraps=old.capture_codes) as calls:
            ctl.source_risks(x, h, 2, np.random.default_rng(37))
        self.assertEqual(calls.call_count, 7)
        np.testing.assert_array_equal([call.args[1] for call in calls.call_args_list], old.EXPOSURES)
        self.assertTrue(all(call.args[0].shape == (2, 1, 8, 8) for call in calls.call_args_list))


class FrozenCrossTests(unittest.TestCase):
    def fixture(self):
        records = [{"group_id": g, "source_id": f"s{i}"} for i, g in enumerate(["g04", "g04", "g06", "g06", "g10", "g10"])]
        features = np.column_stack([np.linspace(.1, .6, 6), np.linspace(.6, .1, 6),
                                    np.linspace(.2, .7, 6), np.linspace(.7, .2, 6)])
        costs = {name: (.01*np.arange(6)[:, None]+.001*(bid+1)*np.arange(7)[None, :]) for bid, name in enumerate(ctl.BACKENDS)}
        selectors = {}
        for bid, name in enumerate(ctl.BACKENDS):
            selectors[name] = {"dev_fixed_action_index": bid, "dev_selected_windows_by_action": [1]*7,
                "frozen_rules": {feature: {"feature": feature, "threshold": .35,
                                          "left": (bid+fid)%7, "right": (bid+fid+3)%7}
                                 for fid, feature in enumerate(old.FEATURE_NAMES)}}
        return records, features, costs, selectors

    def test_all_sixteen_cells_keep_source_fixed_and_direct_brightness_ci(self):
        records, features, costs, selectors = self.fixture()
        before = copy.deepcopy(selectors)
        matrix = ctl.cross_evaluate(costs, features, records, selectors, 19, 0, bootstraps=64)
        self.assertEqual(selectors, before)
        self.assertEqual(sum(len(x) for x in matrix.values()), 16)
        for selector_name, columns in matrix.items():
            selector = selectors[selector_name]
            bright_actions = old.policy_actions(features, selector["frozen_rules"]["mean_brightness"])
            for renderer, cell in columns.items():
                fixed = selector["dev_fixed_action_index"]
                self.assertEqual(cell["source_selector_dev_fixed_action_index"], fixed)
                expected_fixed = old.group_mean(costs[renderer][:, fixed], records)
                self.assertAlmostEqual(cell["fixed_group_balanced_risk"], expected_fixed)
                for feature, result in cell["policies"].items():
                    actions = old.policy_actions(features, selector["frozen_rules"][feature])
                    selected = costs[renderer][np.arange(6), actions]
                    bright = costs[renderer][np.arange(6), bright_actions]
                    expected = old.group_mean(selected-bright, records)
                    ci = result["paired_rule_minus_mean_brightness"]
                    self.assertAlmostEqual(ci["mean_difference"], expected)
                    self.assertEqual(ci["diagnostic_group_count"], 3)
                    self.assertEqual(set(ci["group_mean_differences"]), {"g04", "g06", "g10"})
                    if feature == "mean_brightness":
                        np.testing.assert_array_equal(ci["descriptive_paired_cluster_percentile_ci95"], [0., 0.])

    def test_group_balanced_ci_resamples_groups_not_crops_or_sources(self):
        records = [{"group_id": g} for g in ["a"]*10+["b"]+["c"]]
        difference = np.array([1.]*10+[2., 3.])
        result = ctl.paired_cluster_ci(difference, records, np.random.default_rng(19), 128)
        self.assertEqual(result["diagnostic_group_count"], 3)
        self.assertAlmostEqual(result["mean_difference"], 2.)
        self.assertNotAlmostEqual(result["mean_difference"], difference.mean())

    def test_freeze_selectors_dev_windows_and_train_only_thresholds(self):
        train, dev = [], []
        desired = np.array([0, 1, 2, 3, 0, 1, 2])
        for i in range(4):
            costs = {}
            for name in ctl.BACKENDS:
                array = np.ones((len(ctl.WINDOWS[name]), 7))
                selected = desired if len(array) > 1 else np.zeros(7, dtype=int)
                array[selected, np.arange(7)] = .01+.001*np.arange(7)
                costs[name] = array
            row = {"group_id": f"g{i//2}", "features": np.full(4, .1*(i+1)),
                   "risks": {"costs": costs}}
            train.append(copy.deepcopy(row))
            dev.append(copy.deepcopy(row))
        frozen = ctl.freeze_selectors(train, dev)
        for name, selector in frozen.items():
            expected = desired if len(ctl.WINDOWS[name]) > 1 else np.zeros(7, dtype=int)
            np.testing.assert_array_equal(selector["window_indices"], expected)
            for i, feature in enumerate(old.FEATURE_NAMES):
                rule = selector["frozen_rules"][feature]
                candidates = np.unique(np.quantile(np.array([r["features"][i] for r in train]), np.arange(.1, 1, .1)))
                np.testing.assert_array_equal(rule["candidate_thresholds_train_only"], candidates)

    def test_fixed_cache_identity_and_output_refuse_overwrite(self):
        sources = []
        for i in range(68):
            sources.append({"source_id": f"source_{i}", "pair_id": i//2,
                "camera": "iphone" if i%2 == 0 else "s25", "group_id": f"g{i//8}",
                "split": "diagnostic", "sha256": f"digest{i}"})
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            audit = {"protocol_sha256": "same", "sources": sources}
            (p/"cache_manifest.json").write_text(json.dumps({"protocol_sha256": "same", "sources": sources}))
            self.assertEqual(len(ctl._validate_audit_and_cache(p, audit)), 68)
            bad = copy.deepcopy(audit)
            bad["sources"][0]["group_id"] = "wrong"
            with self.assertRaises(ValueError):
                ctl._validate_audit_and_cache(p, bad)
            with self.assertRaises(FileExistsError):
                ctl.run(p, p/"missing_audit.json", p, [19], repeats=2, smoke=True)


if __name__ == "__main__":
    unittest.main()
