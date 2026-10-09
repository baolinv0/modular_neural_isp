"""Numerical/causal tests for the new history reference, not S24 formal test."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.ae_tm.diagnostics import history_reference as hr

base = hr.base


class HistoryPosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.likelihood = base.ADCLikelihood()

    def reference(self, **kwargs):
        return hr.HistoryReference(likelihood=self.likelihood, **kwargs)

    def observations(self, n=8, repeats=3):
        rng = np.random.default_rng(19)
        latents = base.sample_sources(n, rng)
        history, _ = base.previews_and_features(latents, rng)
        future = base.capture_codes(np.broadcast_to(latents[:, None, :], (n, repeats, 2)), .5, rng)
        return latents, history, future

    def test_no_history_and_empty_history_equal_existing_reference(self):
        _, history, future = self.observations()
        ref = self.reference(nodes=128)
        old = base.ContinuousReference(nodes=128, likelihood=self.likelihood)
        expected, gamma = old.posterior(future.reshape(-1, 2), .5)
        for past in [None, history[:, :0]]:
            actual, actual_gamma = ref.posterior(future, .5, past)
            np.testing.assert_array_equal(actual.reshape(-1, 2), expected)
            np.testing.assert_array_equal(actual_gamma.ravel(), gamma)

    def test_explicit_history_product_and_full_independent_grid_integration(self):
        ref = self.reference(nodes=32)
        history = np.array([[[133, 1147], [140, 1160], [126, 1125]]])
        future = np.array([[2200, 4095]])
        exposure = 4.
        products = []
        for pixel, latent in [(0, ref.a), (1, ref.b), (0, ref.ca), (1, ref.cb)]:
            p = self.likelihood.probabilities(history[0, :, pixel], base.FW*.25*latent).prod(axis=0)
            p *= self.likelihood.probabilities(future[:, pixel], base.FW*exposure*latent)[0]
            products.append(p)
        pa, pb, pca, pcb = products
        independent_grid = np.outer(pa, pb)*np.outer(ref.weights, ref.weights)
        correlated_grid = pca*pcb*ref.cweights
        w = ref.prior.weight
        evidence = (1-w)*independent_grid.sum()+w*correlated_grid.sum()
        expected_a = ((1-w)*(independent_grid*base.tone(ref.a)[:, None]).sum() +
                      w*(correlated_grid*base.tone(ref.ca)).sum())/evidence
        expected_b = ((1-w)*(independent_grid*base.tone(ref.b)[None, :]).sum() +
                      w*(correlated_grid*base.tone(ref.cb)).sum())/evidence
        pred, gamma = ref.posterior(future, exposure, history)
        np.testing.assert_allclose(pred[0], [expected_a, expected_b], atol=3e-14, rtol=0)
        self.assertAlmostEqual(gamma[0], w*correlated_grid.sum()/evidence, places=13)

    def test_history_updates_posterior_component_not_just_target_mean(self):
        ref = self.reference(nodes=128)
        history = np.array([[[133, 1147], [133, 1147], [133, 1147]]])
        future = np.array([[2100, 4095]])
        _, blind_gamma = ref.posterior(future, 4.)
        _, aware_gamma = ref.posterior(future, 4., history)
        self.assertGreater(aware_gamma[0], blind_gamma[0]+.1)

    def test_fully_saturated_history_and_future_restore_prior_means(self):
        ref = self.reference(nodes=128)
        history = np.full((3, 3, 2), 4095)
        future = np.full((3, 2, 2), 4095)
        pred, gamma = ref.posterior(future, 1000., history, history_exposures=1000.)
        expected = [ref.weights @ base.tone(ref.a), ref.weights @ base.tone(ref.b)]
        np.testing.assert_allclose(pred, np.broadcast_to(expected, pred.shape), atol=2e-14, rtol=0)
        np.testing.assert_allclose(gamma, .5, atol=2e-14, rtol=0)

    def test_low_boundary_history_log_products_are_stable(self):
        ref = self.reference(nodes=128)
        pred, gamma = ref.posterior(np.zeros((2, 2), dtype=int), .5,
                                    np.zeros((2, 3, 2), dtype=int), history_exposures=.5)
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertTrue(np.all(np.isfinite(gamma)))
        self.assertTrue(np.all(pred[:, 0] >= base.tone(.05)))
        self.assertTrue(np.all(pred[:, 1] >= base.tone(.8)))

    def test_history_permutation_with_matching_known_exposures(self):
        rng = np.random.default_rng(37)
        latents = base.sample_sources(4, rng)
        exposures = np.array([.125, .25, .5])
        history = np.stack([base.capture_codes(latents, float(e), rng) for e in exposures], axis=1)
        future = base.capture_codes(latents, 2., rng)
        ref = self.reference(nodes=128)
        expected = ref.posterior(future, 2., history, history_exposures=exposures)
        p = np.array([2, 0, 1])
        actual = ref.posterior(future, 2., history[:, p], history_exposures=exposures[p])
        np.testing.assert_allclose(actual[0], expected[0], atol=2e-14, rtol=0)
        np.testing.assert_allclose(actual[1], expected[1], atol=2e-14, rtol=0)

    def test_source_repeat_shape_and_indexed_history_match_single_repeat_calls(self):
        _, history, future = self.observations(n=5, repeats=3)
        ref = self.reference(nodes=128, chunk=4)
        prepared = ref.prepare_history(history)
        expected = ref.predict(future, .5, prepared)
        for repeat in range(3):
            np.testing.assert_allclose(ref.predict(future[:, repeat], .5, prepared),
                                       expected[:, repeat], atol=2e-14, rtol=0)
        order = np.array([3, 1, 3])
        actual = ref.predict(future[order], .5, prepared, source_indices=order)
        np.testing.assert_allclose(actual, expected[order], atol=2e-14, rtol=0)
        flattened = ref.predict(future.reshape(-1, 2), .5, prepared,
                                source_indices=np.repeat(np.arange(5), 3))
        np.testing.assert_allclose(flattened.reshape(future.shape), expected, atol=2e-14, rtol=0)

    def test_quadrature_and_chunk_convergence_with_history(self):
        rng = np.random.default_rng(73)
        latents = base.sample_sources(16, rng)
        history, _ = base.previews_and_features(latents, rng)
        r128 = self.reference(nodes=128, chunk=7, likelihood_chunk=11)
        r256 = self.reference(nodes=256, chunk=13, likelihood_chunk=23)
        alternate = self.reference(nodes=128, chunk=64, likelihood_chunk=64)
        worst = 0.
        for e in [.5, 2., 8.]:
            future = base.capture_codes(latents, e, rng)
            pred = r128.predict(future, e, history)
            np.testing.assert_allclose(pred, alternate.predict(future, e, history), atol=3e-14, rtol=0)
            worst = max(worst, float(np.max(np.abs(pred-r256.predict(future, e, history)))))
        self.assertLess(worst, 2e-7)

    def test_observation_only_signature_and_compact_history_storage(self):
        names = set(inspect.signature(hr.HistoryReference.predict).parameters)
        self.assertEqual(names, {"self", "adc_pairs", "exposure", "history",
                                "history_exposures", "source_indices"})
        _, history, _ = self.observations(n=8, repeats=3)
        ref = self.reference(nodes=128)
        prepared = ref.prepare_history(history)
        self.assertEqual(prepared.source_count, 8)
        self.assertEqual(prepared.frame_count, 3)
        for axis in prepared.axes:
            for factor in axis:
                self.assertEqual(factor.source_frame_indices.shape, (8, 3))
                self.assertEqual(factor.log_probabilities.ndim, 2)
                self.assertEqual(factor.log_probabilities.shape[1], 128)

    def test_invalid_shapes_metadata_and_other_node_grid_rejected(self):
        _, history, future = self.observations(n=3, repeats=2)
        ref = self.reference(nodes=128)
        for bad in [history[:, 0], np.zeros((3, 3, 3)), history.astype(float)+.1]:
            with self.assertRaises(ValueError):
                ref.prepare_history(bad)
        with self.assertRaises(ValueError):
            ref.predict(future[:2], .5, history)
        with self.assertRaises(ValueError):
            ref.predict(future[:2], .5, history, source_indices=[0, 3])
        with self.assertRaises(ValueError):
            ref.prepare_history(history, history_exposures=[.25, -.25, .25])
        with self.assertRaises(ValueError):
            ref.predict(future, .5, self.reference(nodes=64).prepare_history(history))
        with self.assertRaises(ValueError):
            ref.predict(np.zeros((3, 0, 2)), .5, history)

    def test_controlled_evaluation_requires_three_past_frames(self):
        latents, history, _ = self.observations(n=3)
        cube = base.future_codes(latents, 2, np.random.default_rng(7301))
        ref = self.reference(nodes=64)
        costs, noise = hr.evaluate_backends(ref, latents, history, cube)
        self.assertEqual(set(costs), set(hr.BACKENDS))
        self.assertTrue(all(c.shape == (3, 7) for c in costs.values()))
        legacy = base.ContinuousReference(nodes=64, likelihood=self.likelihood)
        legacy_costs, _ = base.evaluate_costs(legacy, latents, cube)
        np.testing.assert_array_equal(costs["H_blind"], legacy_costs)
        with self.assertRaises(ValueError):
            hr.evaluate_backends(ref, latents, history[:, :2], cube)
        with self.assertRaises(ValueError):
            hr.evaluate_backends(ref, latents, history, cube[:, :6])


class ControlledReplayTests(unittest.TestCase):
    def fixture(self):
        features = {name: np.array([0., .25, .75, 1.]) for name in base.FEATURES}
        costs = {"H_blind": np.arange(7)[None, :]+10*np.arange(4)[:, None],
                 "H_aware": (7-np.arange(7)[None, :])**2+10*np.arange(4)[:, None]}
        selectors = {}
        for name, fixed, left, right in [("H_blind", 0, 1, 3), ("H_aware", 6, 4, 5)]:
            rules = {f: {"threshold": .5, "left_action_index": left, "right_action_index": right}
                     for f in base.FEATURES}
            rules["global_brightness"] = {"threshold": .5, "left_action_index": 2, "right_action_index": 2}
            selectors[name] = {"fixed_action_index": fixed, "rules": rules}
        return features, costs, selectors

    def test_strict_selector_renderer_matrix_no_refit_and_paired_arithmetic(self):
        features, costs, selectors = self.fixture()
        frozen = copy.deepcopy(selectors)
        matrix = hr.cross_evaluate(costs, features, selectors, np.random.SeedSequence(19), bootstraps=32)
        self.assertEqual(selectors, frozen)
        index = np.arange(4)
        actions = hr.selector_actions(features, selectors)
        for selector in hr.BACKENDS:
            for renderer in hr.BACKENDS:
                cell = matrix[selector][renderer]
                fixed = selectors[selector]["fixed_action_index"]
                self.assertEqual(cell["fixed_action_index_from_selector_dev"], fixed)
                self.assertAlmostEqual(cell["fixed_expected_cost"], costs[renderer][:, fixed].mean())
                residual = costs[renderer][index, actions[selector]["known_linear_residual"]]
                bright = costs[renderer][index, actions[selector]["global_brightness"]]
                self.assertAlmostEqual(cell["paired_residual_minus_brightness"]["mean_difference"],
                                       (residual-bright).mean())
                self.assertEqual(cell["paired_residual_minus_brightness"]["independent_source_count"], 4)
        self.assertAlmostEqual(matrix["H_blind"]["H_aware"]["paired_residual_minus_brightness"]["mean_difference"], 1.)

    def test_source_array_shape_ids_and_same_original_n0_random_streams(self):
        with tempfile.TemporaryDirectory() as temp:
            result = hr.run_diagnostic(seeds=[19], train_sources=16, dev_sources=12,
                diagnostic_sources=8, repeats=2, nodes=[32, 64], selection_nodes=64,
                bootstraps=16, chunk=8, source_metrics_dir=Path(temp))
            self.assertEqual(result["config"]["future_noise_repeats"], 2)
            children = np.random.SeedSequence(19).spawn(13)
            latents = base.sample_sources(8, np.random.default_rng(children[5]), base.Prior())
            expected_h, _ = base.previews_and_features(latents, np.random.default_rng(children[6]))
            expected_future = base.future_codes(latents, 2, np.random.default_rng(children[7]))
            file = Path(temp)/result["replicates"][0]["source_metrics_files"]["N0_diagnostic"]
            with np.load(file, allow_pickle=False) as arrays:
                self.assertEqual(len(set(arrays["source_id"])), 8)
                np.testing.assert_array_equal(arrays["history_adc_codes"], expected_h)
                np.testing.assert_array_equal(arrays["future_adc_codes"], expected_future)
                self.assertEqual(arrays["expected_costs_by_nodes_backend_source_action"].shape, (2, 2, 8, 7))
                self.assertEqual(arrays["frozen_policy_actions_by_selector_source_feature"].shape, (2, 8, 3))
                c = arrays["expected_costs_by_nodes_backend_source_action"][-1, 1]
                action = arrays["frozen_policy_actions_by_selector_source_feature"][0, :, 0]
                point = c[np.arange(8), action].mean()
            recorded = result["replicates"][0]["diagnostics_by_nodes"]["64"]["N0_diagnostic"]["backend_cross_matrix"]
            self.assertAlmostEqual(point, recorded["H_blind"]["H_aware"]["policies"]["known_linear_residual"]["expected_cost"])
            with self.assertRaises(FileExistsError):
                hr.run_diagnostic(seeds=[19], train_sources=16, dev_sources=12, diagnostic_sources=8,
                                  repeats=2, nodes=[32], bootstraps=16, source_metrics_dir=Path(temp))


if __name__ == "__main__":
    unittest.main()
