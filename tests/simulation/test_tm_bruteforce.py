# test_tm_bruteforce.py

import math

import numpy as np
import pytest
from catella_cpp import NucPosTM


def _erep(gap, nucbp, llink, elink):
    i = gap - nucbp
    if i >= llink:
        return 0.0
    sr6 = (llink / (i + 1.0)) ** 6 - 1.0
    return elink * sr6 * sr6


def _brute_force(nucbp, llink, elink, mu, eseq, temp=1.0):
    """
    Enumerate every valid (non-overlapping) nucleosome placement on a
    small fiber by direct summation, using the same Hamiltonian as
    NucPosModel/NucPosTM. Returns (logz, start_prob, occup, mean_n,
    var_n), each computed exactly.
    """
    nbp = len(eseq)
    npos = nbp - nucbp
    total_weight = 0.0
    start_weight = [0.0] * npos
    mean_n_num = 0.0
    mean_n2_num = 0.0

    def recurse(idx, placed):
        nonlocal total_weight, mean_n_num, mean_n2_num
        if idx == npos:
            w = 1.0
            for i, x in enumerate(placed):
                w *= math.exp(-(eseq[x] - mu) / temp)
                if i > 0:
                    gap = x - placed[i - 1]
                    w *= math.exp(-_erep(gap, nucbp, llink, elink) / temp)
            total_weight += w
            for x in placed:
                start_weight[x] += w
            n = len(placed)
            mean_n_num += w * n
            mean_n2_num += w * n * n
            return
        recurse(idx + 1, placed)
        if not placed or idx - placed[-1] >= nucbp:
            recurse(idx + 1, [*placed, idx])

    recurse(0, [])

    start_prob = [w / total_weight for w in start_weight]
    occup = [0.0] * nbp
    for x in range(npos):
        for y in range(x, min(x + nucbp, nbp)):
            occup[y] += start_prob[x]
    mean_n = mean_n_num / total_weight
    var_n = mean_n2_num / total_weight - mean_n * mean_n
    logz = math.log(total_weight)
    return logz, start_prob, occup, mean_n, var_n


def _tm(nucbp, llink, elink, mu, eseq, temp=1.0, seed=1):
    tm = NucPosTM(nucbp, len(eseq), llink, elink, mu, seed)
    tm.setEnergy(list(eseq), 1e9)
    tm.solve(temp)
    return tm


class TestCoreQuantities:
    @pytest.mark.parametrize(
        "nucbp,llink,elink,mu,nbp,seed",
        [
            (2, 2, 1.0, 0.5, 10, 0),
            (3, 2, 0.7, -0.3, 14, 1),
            (2, 3, 2.0, 1.2, 12, 2),
            (4, 1, 0.5, 0.0, 13, 3),
        ],
    )
    def test_logz_startprob_occup_meann(
        self, nucbp, llink, elink, mu, nbp, seed
    ):
        rng = np.random.default_rng(seed)
        eseq = rng.normal(size=nbp).tolist()
        logz_bf, start_bf, occup_bf, meann_bf, _ = _brute_force(
            nucbp, llink, elink, mu, eseq
        )
        tm = _tm(nucbp, llink, elink, mu, eseq)

        assert tm.getLogZ() == pytest.approx(logz_bf, rel=1e-8)
        np.testing.assert_allclose(tm.getStartProb(), start_bf, atol=1e-8)
        np.testing.assert_allclose(tm.getOccup(), occup_bf, atol=1e-8)
        assert tm.getMeanN() == pytest.approx(meann_bf, rel=1e-8)

    def test_consistency_identity_z_equals_sum_start_weight(self):
        # Z = sum_s v_x[s] u_x[s] for any x; in particular the marginals
        # must sum to <N>, and occupancy must lie in [0, 1] everywhere.
        rng = np.random.default_rng(42)
        nbp = 16
        eseq = rng.normal(size=nbp).tolist()
        tm = _tm(3, 2, 1.0, 0.2, eseq)
        start_prob = np.asarray(tm.getStartProb())
        occup = np.asarray(tm.getOccup())
        assert start_prob.sum() == pytest.approx(tm.getMeanN(), rel=1e-8)
        assert np.all(occup >= -1e-12) and np.all(occup <= 1 + 1e-12)


class TestUncertainty:
    def test_var_n_matches_brute_force(self):
        rng = np.random.default_rng(7)
        nucbp, llink, elink, mu, nbp = 3, 2, 1.0, 0.4, 14
        eseq = rng.normal(size=nbp).tolist()
        _, _, _, _, var_n_bf = _brute_force(nucbp, llink, elink, mu, eseq)
        tm = _tm(nucbp, llink, elink, mu, eseq)
        assert tm.getVarN() == pytest.approx(var_n_bf, rel=1e-4)

    def test_var_occup_matches_bernoulli_closed_form(self):
        rng = np.random.default_rng(8)
        nucbp, llink, elink, mu, nbp = 2, 3, 0.8, -0.2, 13
        eseq = rng.normal(size=nbp).tolist()
        tm = _tm(nucbp, llink, elink, mu, eseq)
        occup = np.asarray(tm.getOccup())
        expected = occup * (1.0 - occup)
        np.testing.assert_allclose(tm.getVarOccup(), expected, atol=1e-12)

    def test_var_n_insensitive_to_step_size(self):
        rng = np.random.default_rng(9)
        eseq = rng.normal(size=12).tolist()
        tm = _tm(2, 2, 1.0, 0.1, eseq)
        v1 = tm.getVarN(dmu=1e-3)
        v2 = tm.getVarN(dmu=1e-5)
        assert v1 == pytest.approx(v2, rel=1e-3)


class TestSampling:
    def test_sampled_configurations_match_exact_marginals(self):
        rng = np.random.default_rng(11)
        nucbp, llink, elink, mu, nbp = 2, 2, 1.0, 0.5, 10
        eseq = rng.normal(size=nbp).tolist()
        tm = _tm(nucbp, llink, elink, mu, eseq, seed=123)
        exact_start = np.asarray(tm.getStartProb())

        n_samples = 20000
        npos = nbp - nucbp
        counts = np.zeros(npos)
        for _ in range(n_samples):
            positions, _energy = tm.sampleConfig()
            for x in positions:
                counts[x] += 1
        empirical_start = counts / n_samples

        np.testing.assert_allclose(empirical_start, exact_start, atol=0.03)

    def test_sample_energy_matches_hamiltonian(self):
        rng = np.random.default_rng(12)
        nucbp, llink, elink, mu, nbp = 3, 2, 1.0, 0.3, 14
        eseq = rng.normal(size=nbp).tolist()
        tm = _tm(nucbp, llink, elink, mu, eseq, seed=5)
        for _ in range(20):
            positions, energy = tm.sampleConfig()
            expected = 0.0
            for i, x in enumerate(positions):
                expected += eseq[x]
                if i > 0:
                    expected += _erep(
                        x - positions[i - 1], nucbp, llink, elink
                    )
            expected -= mu * len(positions)
            assert energy == pytest.approx(expected, rel=1e-8, abs=1e-8)


class TestRequireSolve:
    def test_query_before_solve_raises(self):
        tm = NucPosTM(2, 10, 2, 1.0, 0.5, seed=1)
        tm.setEnergy([0.0] * 10, 1e9)
        with pytest.raises(RuntimeError):
            tm.getOccup()

    def test_var_n_does_not_require_solve(self):
        tm = NucPosTM(2, 10, 2, 1.0, 0.5, seed=1)
        tm.setEnergy([0.0] * 10, 1e9)
        # Should not raise, even though solve() was never called.
        tm.getVarN()
