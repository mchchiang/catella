import numpy as np
from catella.experiment._contexts import GCH, A
from catella.experiment._em import (
    _calibrate_from_data,
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


class TestEmStarts:
    def test_user_start_then_ordered_data_starts(self):
        K, N = _mixture(0.5, [0.1, 0.05], [0.8, 0.4])
        starts = _em_starts(K, N, 0.05, 0.95, 1e-3)
        assert [s[2] for s in starts] == [0.75, 0.25, 0.5, 0.75]
        np.testing.assert_array_equal(starts[0][0], [0.05, 0.05])
        np.testing.assert_array_equal(starts[0][1], [0.95, 0.95])
        for tp, ta, _ in starts[1:]:
            assert np.all(tp < ta)

    def test_channel_without_sites_gets_valid_rates(self):
        K, N = _mixture(0.5, [0.1, 0.1], [0.8, 0.8])
        K[1], N[1] = 0.0, 0.0
        for tp, ta, _ in _em_starts(K, N, 0.05, 0.95, 1e-3):
            assert np.all((tp > 0) & (tp < 1) & (ta > 0) & (ta < 1))


class TestRunEm:
    def test_reports_loglik_at_returned_parameters(self):
        K, N = _mixture(0.3, [0.1], [0.9])
        const = _log_binom_coef_sum(K, N)
        tp, ta, pi, ll, n_iter = _run_em(
            K,
            N,
            np.array([0.2]),
            np.array([0.7]),
            0.5,
            const,
            max_iters=5,
            eps=1e-3,
            tol=1e-8,
        )
        assert 1 <= n_iter <= 5
        assert ll == _mixture_loglik(K, N, tp, ta, pi, const)[0]
