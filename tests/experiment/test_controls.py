import numpy as np
from catella.experiment._contexts import GCH, NONE, A
from catella.experiment._controls import (
    _NU_BOUNDS,
    _NU_FALLBACK,
    _beta_posterior,
    _context_prior_rate,
    _control_profiles,
    _fit_channel_nu,
    _fit_nu,
    _prob_greater,
)


def _beta_binomial(nu, m, n_sites=3000, depth=30, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.beta(nu * m, nu * (1 - m), n_sites)
    k = rng.binomial(depth, theta).astype(float)
    return k, np.full(n_sites, float(depth))


class TestFitNu:
    def test_recovers_planted_concentration(self):
        k, n = _beta_binomial(20.0, 0.3)
        nu, m = _fit_nu(k, n)
        assert 20.0 / 1.5 < nu < 20.0 * 1.5
        assert abs(m - 0.3) < 0.01

    def test_homogeneous_rates_hit_upper_bound(self):
        rng = np.random.default_rng(1)
        n = np.full(3000, 30.0)
        k = rng.binomial(30, 0.3, 3000).astype(float)
        nu, _ = _fit_nu(k, n)
        assert nu > _NU_BOUNDS[1] / 10

    def test_too_few_sites_fall_back(self):
        nu, m = _fit_nu(np.array([1.0, 2.0]), np.array([4.0, 4.0]))
        assert nu == _NU_FALLBACK
        assert m == 3.0 / 8.0

    def test_unread_sites_ignored_and_k_clipped(self):
        k, n = _beta_binomial(20.0, 0.3)
        nu_ref, _ = _fit_nu(k, n)
        k_bad = np.append(k, [5.0, 3.0])
        n_bad = np.append(n, [0.0, 2.0])
        nu, _ = _fit_nu(k_bad, n_bad)
        assert np.isfinite(nu)
        assert abs(np.log(nu / nu_ref)) < 0.05


class TestFitChannelNu:
    def test_fits_each_channel_separately(self):
        k_a, n_a = _beta_binomial(5.0, 0.3, seed=2)
        k_g, n_g = _beta_binomial(100.0, 0.6, seed=3)
        ctx = np.concatenate(
            [np.full(len(k_a), A), np.full(len(k_g), GCH), [NONE]]
        ).astype(np.int8)
        k = np.concatenate([k_a, k_g, [0.0]])
        n = np.concatenate([n_a, n_g, [0.0]])
        nu, m = _fit_channel_nu(k, n, ctx)
        assert set(nu) == {A, GCH}
        assert nu[A] < 10.0 < nu[GCH]
        assert abs(m[GCH] - 0.6) < 0.01


def _arr(*v):
    return np.array(v, dtype=float)


class TestProbGreater:
    def test_matches_monte_carlo_for_small_parameters(self):
        rng = np.random.default_rng(0)
        cases = [(5, 95, 1, 99), (0.6, 0.4, 0.03, 0.97), (2, 19, 1.2, 20)]
        for a_x, b_x, a_y, b_y in cases:
            mc = np.mean(
                rng.beta(a_x, b_x, 1000000) > rng.beta(a_y, b_y, 1000000)
            )
            p = _prob_greater(_arr(a_x), _arr(b_x), _arr(a_y), _arr(b_y))
            assert abs(p[0] - mc) < 3e-3

    def test_clear_cases_and_symmetry(self):
        a_x, b_x = _arr(40, 10, 20), _arr(15, 40, 20)
        a_y, b_y = _arr(2, 40, 20), _arr(60, 15, 20)
        p = _prob_greater(a_x, b_x, a_y, b_y)
        assert p[0] > 0.999
        assert p[1] < 0.001
        assert abs(p[2] - 0.5) < 1e-3

    def test_near_uses_exact_value_close_to_threshold(self):
        args = (_arr(5), _arr(95), _arr(1), _arr(99))
        exact = _prob_greater(*args)
        approx_far = _prob_greater(*args, near=0.2)
        near = _prob_greater(*args, near=0.95)
        assert near[0] == exact[0]
        assert approx_far[0] != exact[0]

    def test_non_positive_parameter_gives_nan(self):
        p = _prob_greater(_arr(0.0), _arr(1.0), _arr(1.0), _arr(1.0))
        assert np.isnan(p[0])


class TestBetaPosterior:
    def test_prior_plus_counts_with_clipping(self):
        ctx = np.array([A, A, NONE], dtype=np.int8)
        a, b = _beta_posterior(
            _arr(3.0, 5.0, 1.0), _arr(4.0, 4.0, 1.0), ctx, 10.0, {A: 0.5}
        )
        np.testing.assert_allclose(a[:2], [8.0, 9.0])
        np.testing.assert_allclose(b[:2], [6.0, 5.0])
        assert np.isnan(a[2]) and np.isnan(b[2])


class TestControlProfilesInformative:
    def _profiles(self, meth_k, meth_n, unmeth_k, unmeth_n, nu):
        ctx = np.full(len(meth_k), A, dtype=np.int8)
        return _control_profiles(
            meth_k, meth_n, unmeth_k, unmeth_n, ctx, nu=nu
        )[2]

    def test_deep_contrast_kept_inverted_dropped(self):
        meth_k, meth_n = _arr(*[30] * 20, 2), _arr(*[40] * 21)
        unmeth_k, unmeth_n = _arr(*[1] * 20, 30), _arr(*[40] * 21)
        informative = self._profiles(meth_k, meth_n, unmeth_k, unmeth_n, 1.0)
        assert informative[:20].all()
        assert not informative[20]

    def test_unread_site_judged_by_channel_prior(self):
        meth_k, meth_n = _arr(*[12] * 20, 0), _arr(*[40] * 20, 0)
        unmeth_k, unmeth_n = _arr(*[6] * 20, 0), _arr(*[40] * 20, 0)
        strong = self._profiles(meth_k, meth_n, unmeth_k, unmeth_n, 100.0)
        weak = self._profiles(meth_k, meth_n, unmeth_k, unmeth_n, 2.0)
        assert strong[20]
        assert not weak[20]


class TestContextPriorRateDict:
    def test_dict_nu_matches_per_channel_float(self):
        ctx = np.array([A, A, GCH, GCH], dtype=np.int8)
        k = np.array([3.0, 0.0, 1.0, 4.0])
        n = np.array([4.0, 2.0, 5.0, 5.0])
        both = _context_prior_rate(k, n, ctx, nu={A: 2.0, GCH: 8.0})
        a = _context_prior_rate(k, n, ctx, nu=2.0)
        g = _context_prior_rate(k, n, ctx, nu=8.0)
        np.testing.assert_array_equal(both[:2], a[:2])
        np.testing.assert_array_equal(both[2:], g[2:])
