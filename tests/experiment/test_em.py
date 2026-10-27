import numpy as np
from catella.experiment._contexts import GCH, A
from catella.experiment._em import (
    _calibrate_from_data,
    _calibrate_from_profiles,
    _channel_tests,
    _em_starts,
    _log_binom_coef_sum,
    _mixture_loglik,
    _run_em,
)


def _mixture(pi, rates_prot, rates_acc, n_win=4000, m=20, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.random(n_win) < pi
    N = np.full((len(rates_prot), n_win), float(m))
    K = np.stack(
        [
            rng.binomial(m, np.where(z, tp, ta)).astype(float)
            for tp, ta in zip(rates_prot, rates_acc)
        ]
    )
    return K, N


class TestCalibrateFromData:
    def test_recovers_shared_pi_and_per_channel_rates(self):
        K, N = _mixture(0.3, [0.1, 0.2], [0.9, 0.7])
        ctx = np.array([A, GCH], dtype=np.int8)
        tp, ta, _, diag = _calibrate_from_data(K, N, ctx, [A, GCH])
        np.testing.assert_allclose(diag["theta_prot"], [0.1, 0.2], atol=0.02)
        np.testing.assert_allclose(diag["theta_acc"], [0.9, 0.7], atol=0.02)
        assert abs(diag["frac_protected"] - 0.3) < 0.03
        assert tp[0] < ta[0] and tp[1] < ta[1]

    def test_label_swap_flips_frac_protected(self):
        K, N = _mixture(0.3, [0.1], [0.9])
        ctx = np.array([A], dtype=np.int8)
        _, _, _, diag = _calibrate_from_data(
            K, N, ctx, [A], init_prot=0.9, init_acc=0.1
        )
        assert diag["theta_prot"][0] < diag["theta_acc"][0]
        assert abs(diag["frac_protected"] - 0.3) < 0.03

    def test_symmetric_user_start_still_recovers_truth(self):
        K, N = _mixture(0.6, [0.1, 0.05], [0.8, 0.4])
        ctx = np.array([A, GCH], dtype=np.int8)
        _, _, _, diag = _calibrate_from_data(
            K, N, ctx, [A, GCH], init_prot=0.5, init_acc=0.5
        )
        np.testing.assert_allclose(diag["theta_prot"], [0.1, 0.05], atol=0.02)
        np.testing.assert_allclose(diag["theta_acc"], [0.8, 0.4], atol=0.02)
        assert abs(diag["frac_protected"] - 0.6) < 0.03


_LO, _HI = np.full(2, 1e-3), np.full(2, 1 - 1e-3)


def _profile_mixture(pi, tau, kappa, n_win=4000, m=20, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.random(n_win) < pi
    f = rng.uniform(0.01, 0.05, n_win)
    d = rng.uniform(0.6, 0.9, n_win)
    p = np.where(z, f + tau * d, f + kappa * d)
    K = rng.binomial(m, p).astype(float)[None]
    N = np.full((1, n_win), float(m))
    return K, N, m * f[None], m * d[None]


class TestCalibrateFromProfiles:
    _ctx = np.array([A], dtype=np.int8)
    _fpr = np.array([0.03])
    _acc = np.array([0.8])

    def test_constant_profiles_match_no_controls_fit(self):
        K, N = _mixture(0.3, [0.1], [0.9])
        _, _, _, ref = _calibrate_from_data(K, N, self._ctx, [A])
        _, _, _, diag = _calibrate_from_profiles(
            K,
            N,
            0.0 * N,
            N.copy(),
            [A],
            self._ctx,
            np.zeros(1),
            np.ones(1),
            np.ones(1, dtype=bool),
            fit_kappa=True,
        )
        np.testing.assert_allclose(diag["tau"], ref["theta_prot"], atol=1e-4)
        np.testing.assert_allclose(diag["kappa"], ref["theta_acc"], atol=1e-4)
        assert abs(diag["frac_protected"] - ref["frac_protected"]) < 1e-4

    def test_recovers_tau_with_kappa_fixed(self):
        K, N, SF, SD = _profile_mixture(0.4, 0.2, 1.0)
        tp, ta, _, diag = _calibrate_from_profiles(
            K,
            N,
            SF,
            SD,
            [A],
            self._ctx,
            self._fpr,
            self._acc,
            np.ones(1, dtype=bool),
        )
        assert abs(diag["tau"][0] - 0.2) < 0.02
        assert diag["kappa"][0] == 1.0
        assert abs(diag["frac_protected"] - 0.4) < 0.03
        np.testing.assert_allclose(tp, 0.03 + diag["tau"] * 0.77)
        np.testing.assert_array_equal(ta, self._acc)

    def test_fit_kappa_recovers_both(self):
        K, N, SF, SD = _profile_mixture(0.4, 0.15, 0.7)
        _, ta, _, diag = _calibrate_from_profiles(
            K,
            N,
            SF,
            SD,
            [A],
            self._ctx,
            self._fpr,
            self._acc,
            np.ones(1, dtype=bool),
            fit_kappa=True,
            init_prot=0.9,
            init_acc=0.1,
        )
        assert abs(diag["tau"][0] - 0.15) < 0.03
        assert abs(diag["kappa"][0] - 0.7) < 0.03
        assert abs(diag["frac_protected"] - 0.4) < 0.03
        np.testing.assert_allclose(ta, 0.03 + diag["kappa"] * 0.77)


class TestChannelTests:
    _lo, _hi = np.full(1, 1e-3), np.full(1, 1 - 1e-3)

    def test_planted_contrast_passes(self):
        K, N = _mixture(0.4, [0.1], [0.8])
        tp, ta = np.array([0.1]), np.array([0.8])
        bic, rel = _channel_tests(K, N, None, tp, ta, 0.4, self._lo, self._hi)
        assert bic[0] > 0
        assert abs(rel[0] - 0.875) < 1e-9

    def test_null_channel_fails_bic(self):
        rng = np.random.default_rng(3)
        N = rng.poisson(4, (1, 5000)).astype(float)
        K = rng.binomial(N.astype(int), 0.3).astype(float)
        _, _, _, diag = _calibrate_from_data(
            K, N, np.array([A], dtype=np.int8), [A]
        )
        assert diag["bic_margin"][0] < 0

    def test_profiles_planted_contrast_passes(self):
        K, N, SF, SD = _profile_mixture(0.4, 0.2, 1.0)
        tp, ta = np.array([0.2]), np.array([1.0])
        bic, rel = _channel_tests(
            K, N, (SF, SD), tp, ta, 0.4, np.zeros(1), np.ones(1)
        )
        assert bic[0] > 0
        assert abs(rel[0] - 0.8) < 1e-9

    def test_failing_channel_is_uninformative(self):
        rng = np.random.default_rng(4)
        K_a, N_a = _mixture(0.4, [0.1], [0.8], n_win=3000)
        N_b = np.full((1, 3000), 20.0)
        K_b = rng.binomial(20, 0.3, (1, 3000)).astype(float)
        K, N = np.vstack([K_a, K_b]), np.vstack([N_a, N_b])
        ctx = np.array([A, A, GCH, GCH], dtype=np.int8)
        _, _, informative, diag = _calibrate_from_data(K, N, ctx, [A, GCH])
        np.testing.assert_array_equal(informative, [True, True, False, False])
        assert diag["bic_margin"][0] > 0 > diag["bic_margin"][1]


class TestEmStarts:
    def test_user_start_then_ordered_data_starts(self):
        K, N = _mixture(0.5, [0.1, 0.05], [0.8, 0.4])
        starts = _em_starts(K, N, None, 0.05, 0.95, True, _LO, _HI)
        assert [s[2] for s in starts] == [0.75, 0.25, 0.5, 0.75]
        np.testing.assert_array_equal(starts[0][0], [0.05, 0.05])
        np.testing.assert_array_equal(starts[0][1], [0.95, 0.95])
        for tp, ta, _ in starts[1:]:
            assert np.all(tp < ta)

    def test_channel_without_sites_gets_valid_rates(self):
        K, N = _mixture(0.5, [0.1, 0.1], [0.8, 0.8])
        K[1], N[1] = 0.0, 0.0
        for tp, ta, _ in _em_starts(K, N, None, 0.05, 0.95, True, _LO, _HI):
            assert np.all((tp > 0) & (tp < 1) & (ta > 0) & (ta < 1))


class TestRunEm:
    def test_reports_loglik_at_returned_parameters(self):
        K, N = _mixture(0.3, [0.1], [0.9])
        const = _log_binom_coef_sum(K, N)
        tp, ta, pi, ll, n_iter = _run_em(
            K,
            N,
            None,
            np.array([0.2]),
            np.array([0.7]),
            0.5,
            const,
            fit_acc=True,
            lo=_LO[:1],
            hi=_HI[:1],
            max_iters=5,
            tol=1e-8,
        )
        assert 1 <= n_iter <= 5
        loglik = _mixture_loglik(K, N, tp[:, None], ta[:, None], pi, const)
        assert ll == loglik[0]
