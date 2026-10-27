import numpy as np
from catella.experiment._contexts import GCH, A
from catella.experiment._em import _calibrate_from_data


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
