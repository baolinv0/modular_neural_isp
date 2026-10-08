"""Numerical tests for the independent N0/N1 reference (not S24 formal test)."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.special import ndtr
from scipy.stats import poisson

PATH = Path(__file__).resolve().parents[1] / "diagnostics" / "continuous_reference.py"
SPEC = importlib.util.spec_from_file_location("ae_tm_continuous_reference", PATH)
cr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cr
SPEC.loader.exec_module(cr)


class LikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.likelihood = cr.ADCLikelihood()

    def test_normalization_covers_complete_saturated_poisson_tail(self):
        audit = self.likelihood.audit()
        self.assertLess(audit["max_normalization_abs_error"], 3e-12)
        self.assertLess(cr.READ_TAIL_BOUND, 2e-15)
        p = self.likelihood.probabilities(np.array([0, 2048, 4095]), np.array([14400.]))
        np.testing.assert_array_equal(p[:, 0], [0., 0., 1.])

    def test_zero_code_includes_negative_read_noise_clipping(self):
        p = self.likelihood.probabilities(np.array([0]), np.array([0.]))[0, 0]
        expected = ndtr((cr.FW * .5 / cr.ADC_MAX) / cr.READ_SD)
        self.assertAlmostEqual(p, expected, places=14)
        self.assertGreater(p, .5)

    def test_edges_match_independent_full_poisson_summation(self):
        # Finite summation tail here is <1e-60, unlike truncating at full-well.
        means = np.array([.5, 32., 970., 1000., 1100., 1500.])
        counts = np.arange(2501.)
        pmf = poisson.pmf(counts[:, None], means[None])
        lo = cr.FW * (cr.ADC_MAX - .5) / cr.ADC_MAX
        hi = cr.FW * .5 / cr.ADC_MAX
        direct0 = ndtr((hi - counts) / cr.READ_SD) @ pmf
        directmax = ndtr((counts - lo) / cr.READ_SD) @ pmf
        actual = self.likelihood.probabilities(np.array([0, cr.ADC_MAX]), means)
        np.testing.assert_allclose(actual[0], direct0, rtol=0, atol=2e-15)
        np.testing.assert_allclose(actual[1], directmax, rtol=0, atol=2e-12)
        # At mean 1500 the clipped code has almost all mass, including counts >1032.
        self.assertGreater(actual[1, -1], .999999999)

    def test_internal_adc_cell_matches_full_convolution(self):
        codes = np.array([1, 123, 2048, 4094])
        means = np.array([0., 30., 500., 999.])
        counts = np.arange(1801.)
        pmf = poisson.pmf(counts[:, None], means[None])
        direct = []
        for code in codes:
            lo = cr.FW * (code - .5) / cr.ADC_MAX
            hi = cr.FW * (code + .5) / cr.ADC_MAX
            zlo, zhi = (lo - counts) / cr.READ_SD, (hi - counts) / cr.READ_SD
            kernel = np.where(zlo >= 0, ndtr(-zlo) - ndtr(-zhi), ndtr(zhi) - ndtr(zlo))
            direct.append(kernel @ pmf)
        actual = self.likelihood.probabilities(codes, means, chunk=1)
        np.testing.assert_allclose(actual, np.array(direct), rtol=0, atol=2e-15)

    def test_capture_chain_boundary_frequency_matches_likelihood(self):
        rng = np.random.default_rng(73019)
        for radiance, code in [(0., 0), (1., cr.ADC_MAX)]:
            codes = cr.capture_codes(np.full(80000, radiance), 1., rng)
            expected = self.likelihood.probabilities(np.array([code]), np.array([1000*radiance]))[0, 0]
            self.assertLess(abs(np.mean(codes == code) - expected), .009)

    def test_input_validation(self):
        for code in [[-1], [4096], [1.5]]:
            with self.assertRaises(ValueError):
                self.likelihood.probabilities(code, [1.])
        with self.assertRaises(ValueError):
            self.likelihood.probabilities([0], [-1.])


class PosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.likelihood = cr.ADCLikelihood()

    def reference(self, **kwargs):
        return cr.ContinuousReference(likelihood=self.likelihood, **kwargs)

    def test_independent_prior_posterior_separates(self):
        obs = np.array([[200, 1300], [200, 1700], [280, 1300]])
        pred, gamma = self.reference(prior=cr.Prior(weight=0), nodes=128).posterior(obs, .25)
        self.assertAlmostEqual(pred[0, 0], pred[1, 0], places=13)
        self.assertAlmostEqual(pred[0, 1], pred[2, 1], places=13)
        np.testing.assert_array_equal(gamma, 0.)
        self.assertTrue(np.all(pred[:, 0] >= cr.tone(.05)))
        self.assertTrue(np.all(pred[:, 1] <= cr.tone(1.8)))

    def test_fully_clipped_observation_returns_uniform_prior_target_means(self):
        ref = self.reference(nodes=128)
        pred, gamma = ref.posterior(np.array([[4095, 4095]]), 1000.)
        u, weights = cr.quadrature(512)
        expected = [weights @ cr.tone(.05 + .25*u), weights @ cr.tone(.8 + u)]
        np.testing.assert_allclose(pred[0], expected, atol=2e-13, rtol=0)
        self.assertAlmostEqual(gamma[0], .5, places=13)

    def test_partial_clipping_posterior_uses_full_sensor_likelihood(self):
        ref = self.reference(nodes=128)
        pred, gamma = ref.posterior(np.array([[4095, 4095]]), 8.)
        likelihood_a = self.likelihood.probabilities([4095], cr.FW*8*ref.a)[0]
        # Bright likelihood equals one to floating precision in this regime.
        norm = likelihood_a @ ref.weights
        expected_a = likelihood_a @ (ref.weights*cr.tone(ref.a)) / norm
        correlated_b = likelihood_a @ (ref.weights*cr.tone(ref.b)) / norm
        independent_b = ref.weights @ cr.tone(ref.b)
        np.testing.assert_allclose(pred[0], [expected_a, .5*(correlated_b+independent_b)],
                                   rtol=0, atol=2e-13)
        self.assertGreater(pred[0, 0], ref.weights @ cr.tone(ref.a))
        self.assertAlmostEqual(gamma[0], .5, places=12)

    def test_chunk_invariance_and_nodes_128_256_convergence(self):
        rng = np.random.default_rng(19)
        latents = cr.sample_sources(32, rng)
        r128 = self.reference(nodes=128, chunk=7, likelihood_chunk=11)
        r256 = self.reference(nodes=256, chunk=13, likelihood_chunk=23)
        alternate = self.reference(nodes=128, chunk=32, likelihood_chunk=64)
        worst = 0.
        for exposure in [.125, .5, 2., 8.]:
            observed = cr.capture_codes(latents, exposure, rng)
            pred128 = r128.predict(observed, exposure)
            np.testing.assert_allclose(pred128, alternate.predict(observed, exposure), atol=2e-13, rtol=0)
            worst = max(worst, float(np.max(np.abs(pred128-r256.predict(observed, exposure)))))
        self.assertLess(worst, 2e-7)

    def test_extremely_rare_evidence_underflow_raises_without_fallback(self):
        with self.assertRaises(FloatingPointError):
            self.reference(nodes=128).predict([[4095, 4095]], .25)

    def test_copula_maps_preserve_marginals_and_rotation_quadrature(self):
        u = (np.arange(10000) + .5) / 10000
        for copula in ["linear", "reverse", "rotate"]:
            prior = cr.Prior(copula=copula, shift=.37)
            np.testing.assert_allclose(np.sort(prior.map_uniform(u)), u, atol=2e-16, rtol=0)
        ref = self.reference(prior=cr.Prior(copula="rotate", shift=.37), nodes=128)
        self.assertAlmostEqual(ref.cweights.sum(), 1., places=14)
        self.assertAlmostEqual(ref.cweights @ ((ref.cb-.8)), .5, places=14)
        self.assertTrue(np.all(np.isfinite(ref.predict([[200, 1500]], .25))))

    def test_reverse_dependence_changes_pairing_without_marginal_changes(self):
        linear = cr.sample_sources(2000, np.random.default_rng(37), cr.Prior(weight=1))
        reverse = cr.sample_sources(2000, np.random.default_rng(37), cr.Prior(weight=1, copula="reverse"))
        np.testing.assert_array_equal(linear[:, 0], reverse[:, 0])
        np.testing.assert_allclose(linear[:, 1] + reverse[:, 1], 2.6, atol=1e-15, rtol=0)
        self.assertGreater(np.corrcoef(linear.T)[0, 1], .999)
        self.assertLess(np.corrcoef(reverse.T)[0, 1], -.999)


class PolicyTests(unittest.TestCase):
    def test_preview_policy_features_use_only_three_observed_frames(self):
        latents = np.array([[.12, 1.08], [.2, 1.0]])
        codes, features = cr.previews_and_features(latents, np.random.default_rng(19))
        self.assertEqual(codes.shape, (2, 3, 2))
        preview = codes.astype(float)/4095/.25
        means = preview.mean(axis=1)
        np.testing.assert_array_equal(features["known_linear_residual"],
                                     np.abs(means[:, 1]-(4*means[:, 0]+.6)))

    def test_dev_selection_two_actions_and_frozen_application(self):
        train_feature = np.linspace(0, 1, 100)
        dev_feature = np.array([.1, .2, .8, .9])
        costs = np.ones((4, 7))
        costs[:2, 1], costs[2:, 5] = .1, .2
        rule = cr.select_threshold(train_feature, dev_feature, costs)
        np.testing.assert_array_equal(cr.apply_rule(dev_feature, rule), [1, 1, 5, 5])
        self.assertAlmostEqual(rule["dev_expected_cost"], .15)
        # Application is only feature + frozen rule, with no future costs/latents.
        np.testing.assert_array_equal(cr.apply_rule(np.array([0., 1.]), rule), [1, 5])

    def test_bootstrap_resamples_paired_source_differences(self):
        result = cr.paired_bootstrap(np.full(16, -.002), np.random.default_rng(73), 128, chunk=11)
        self.assertEqual(result["independent_source_count"], 16)
        np.testing.assert_allclose(result["paired_independent_source_percentile_ci95"], [-.002, -.002])
        self.assertAlmostEqual(result["mean_difference"], -.002)


if __name__ == "__main__":
    unittest.main()
