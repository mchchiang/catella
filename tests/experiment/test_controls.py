import numpy as np
from catella.experiment._contexts import GCH, NONE, A
from catella.experiment._controls import (
    _NU_BOUNDS,
    _NU_FALLBACK,
    _context_prior_rate,
    _control_profiles,
    _fit_channel_nu,
    _fit_nu,
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
