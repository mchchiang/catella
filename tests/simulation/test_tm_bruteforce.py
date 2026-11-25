# test_tm_bruteforce.py

import itertools
import math

import numpy as np
import pytest
from catella_cpp import NucPosTM


def _erep(gap, lnuc, llink, elink):
    i = gap - lnuc
    if i >= llink:
        return 0.0
    sr6 = (llink / (i + 1.0)) ** 6 - 1.0
    return elink * sr6 * sr6


def _brute_force(lnuc, llink, elink, mu, edata, temp=1.0):
    """
    Enumerate every valid (non-overlapping) nucleosome placement on a
    small fiber by direct summation, using the same Hamiltonian as
    NucPosModel/NucPosTM. Returns (logz, start_prob, occup, mean_n,
    var_n), each computed exactly.
    """
    nbp = len(edata)
    npos = nbp - lnuc + 1
    total_weight = 0.0
    start_weight = [0.0] * npos
    mean_n_num = 0.0
    mean_n2_num = 0.0

    def recurse(idx, placed):
        nonlocal total_weight, mean_n_num, mean_n2_num
        if idx == npos:
            w = 1.0
            for i, x in enumerate(placed):
                w *= math.exp(-(edata[x] - mu) / temp)
                if i > 0:
                    gap = x - placed[i - 1]
                    w *= math.exp(-_erep(gap, lnuc, llink, elink) / temp)
            total_weight += w
            for x in placed:
                start_weight[x] += w
            n = len(placed)
            mean_n_num += w * n
            mean_n2_num += w * n * n
            return
        recurse(idx + 1, placed)
        if not placed or idx - placed[-1] >= lnuc:
            recurse(idx + 1, [*placed, idx])

    recurse(0, [])

    start_prob = [w / total_weight for w in start_weight]
    occup = [0.0] * nbp
    for x in range(npos):
        for y in range(x, min(x + lnuc, nbp)):
            occup[y] += start_prob[x]
    mean_n = mean_n_num / total_weight
    var_n = mean_n2_num / total_weight - mean_n * mean_n
    logz = math.log(total_weight)
    return logz, start_prob, occup, mean_n, var_n


def _brute_force_log(lnuc, llink, elink, mu, edata, temp):
    nbp = len(edata)
    npos = nbp - lnuc + 1
    configs = []

    def recurse(idx, placed):
        if idx == npos:
            configs.append(placed)
            return
        recurse(idx + 1, placed)
        if not placed or idx - placed[-1] >= lnuc:
            recurse(idx + 1, [*placed, idx])

    recurse(0, [])
    logw = []
    for placed in configs:
        energy = sum(edata[x] - mu for x in placed)
        energy += sum(
            _erep(b - a, lnuc, llink, elink)
            for a, b in itertools.pairwise(placed)
        )
        logw.append(-energy / temp)
    logw = np.asarray(logw)
    logz = logw.max() + math.log(np.exp(logw - logw.max()).sum())
    probs = np.exp(logw - logz)
    start_prob = np.zeros(npos)
    for p, placed in zip(probs, configs):
        start_prob[placed] += p
    ground = configs[int(np.argmax(logw))]
    return logz, start_prob, ground


def _tm(lnuc, llink, elink, mu, edata, temp=1.0, seed=1):
    tm = NucPosTM(lnuc, len(edata), llink, elink, mu, seed)
    tm.setEnergy(list(edata), 1e9)
    tm.solve(temp)
    return tm


class TestCoreQuantities:
    @pytest.mark.parametrize(
        "lnuc,llink,elink,mu,nbp,seed",
        [
            (2, 2, 1.0, 0.5, 10, 0),
            (3, 2, 0.7, -0.3, 14, 1),
            (2, 3, 2.0, 1.2, 12, 2),
            (4, 1, 0.5, 0.0, 13, 3),
            (3, 0, 1.0, 0.5, 14, 4),
        ],
    )
    def test_logz_startprob_occup_meann(
        self, lnuc, llink, elink, mu, nbp, seed
    ):
        rng = np.random.default_rng(seed)
        edata = rng.normal(size=nbp).tolist()
        logz_bf, start_bf, occup_bf, meann_bf, _ = _brute_force(
            lnuc, llink, elink, mu, edata
        )
        tm = _tm(lnuc, llink, elink, mu, edata)

        assert tm.getLogZ() == pytest.approx(logz_bf, rel=1e-8)
        np.testing.assert_allclose(tm.getStartProb(), start_bf, atol=1e-8)
        np.testing.assert_allclose(tm.getOccup(), occup_bf, atol=1e-8)
        assert tm.getMeanN() == pytest.approx(meann_bf, rel=1e-8)

    def test_consistency_identity_z_equals_sum_start_weight(self):
        # Z = sum_s v_x[s] u_x[s] for any x; in particular the marginals
        # must sum to <N>, and occupancy must lie in [0, 1] everywhere.
        rng = np.random.default_rng(42)
        nbp = 16
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(3, 2, 1.0, 0.2, edata)
        start_prob = np.asarray(tm.getStartProb())
        occup = np.asarray(tm.getOccup())
        assert start_prob.sum() == pytest.approx(tm.getMeanN(), rel=1e-8)
        assert np.all(occup >= -1e-12) and np.all(occup <= 1 + 1e-12)


class TestUncertainty:
    def test_var_n_matches_brute_force(self):
        rng = np.random.default_rng(7)
        lnuc, llink, elink, mu, nbp = 3, 2, 1.0, 0.4, 14
        edata = rng.normal(size=nbp).tolist()
        _, _, _, _, var_n_bf = _brute_force(lnuc, llink, elink, mu, edata)
        tm = _tm(lnuc, llink, elink, mu, edata)
        assert tm.getVarN() == pytest.approx(var_n_bf, rel=1e-4)

    def test_var_occup_matches_bernoulli_closed_form(self):
        rng = np.random.default_rng(8)
        lnuc, llink, elink, mu, nbp = 2, 3, 0.8, -0.2, 13
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata)
        occup = np.asarray(tm.getOccup())
        expected = occup * (1.0 - occup)
        np.testing.assert_allclose(tm.getVarOccup(), expected, atol=1e-12)

    def test_var_n_insensitive_to_step_size(self):
        rng = np.random.default_rng(9)
        edata = rng.normal(size=12).tolist()
        tm = _tm(2, 2, 1.0, 0.1, edata)
        v1 = tm.getVarN(dmu=1e-3)
        v2 = tm.getVarN(dmu=1e-5)
        assert v1 == pytest.approx(v2, rel=1e-3)


class TestSampling:
    def test_sampled_configurations_match_exact_marginals(self):
        rng = np.random.default_rng(11)
        lnuc, llink, elink, mu, nbp = 2, 2, 1.0, 0.5, 10
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, seed=123)
        exact_start = np.asarray(tm.getStartProb())

        n_samples = 20000
        npos = nbp - lnuc + 1
        counts = np.zeros(npos)
        for _ in range(n_samples):
            positions, _energy = tm.sampleConfig()
            for x in positions:
                counts[x] += 1
        empirical_start = counts / n_samples

        np.testing.assert_allclose(empirical_start, exact_start, atol=0.03)

    def test_sample_energy_matches_hamiltonian(self):
        rng = np.random.default_rng(12)
        lnuc, llink, elink, mu, nbp = 3, 2, 1.0, 0.3, 14
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, seed=5)
        for _ in range(20):
            positions, energy = tm.sampleConfig()
            expected = 0.0
            for i, x in enumerate(positions):
                expected += edata[x]
                if i > 0:
                    expected += _erep(
                        x - positions[i - 1], lnuc, llink, elink
                    )
            expected -= mu * len(positions)
            assert energy == pytest.approx(expected, rel=1e-8, abs=1e-8)

    def test_set_seed_matches_fresh_model(self):
        rng = np.random.default_rng(13)
        lnuc, llink, elink, mu, nbp = 3, 2, 1.0, 0.3, 40
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, seed=1)
        for s in (7, 8, 9):
            tm.setSeed(s)
            got = tm.sampleConfig()
            fresh = _tm(lnuc, llink, elink, mu, edata, seed=s)
            expected = fresh.sampleConfig()
            assert list(got[0]) == list(expected[0])
            assert got[1] == expected[1]

    def test_set_seed_repeats_sample(self):
        rng = np.random.default_rng(14)
        lnuc, llink, elink, mu, nbp = 3, 2, 1.0, 0.3, 40
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, seed=1)
        tm.setSeed(42)
        first = tm.sampleConfig()
        tm.setSeed(42)
        second = tm.sampleConfig()
        assert list(first[0]) == list(second[0])
        assert first[1] == second[1]

    def test_reset_returns_to_set_seed(self):
        rng = np.random.default_rng(15)
        lnuc, llink, elink, mu, nbp = 3, 2, 1.0, 0.3, 40
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, seed=1)
        tm.setSeed(42)
        first = tm.sampleConfig()
        tm.sampleConfig()
        tm.reset()
        assert list(tm.sampleConfig()[0]) == list(first[0])


class TestLowTemperature:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_log_space_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        lnuc, llink, elink, mu, nbp, temp = 3, 2, 1.0, 0.5, 16, 0.01
        edata = (10.0 * rng.normal(size=nbp)).tolist()
        logz_bf, start_bf, _ = _brute_force_log(
            lnuc, llink, elink, mu, edata, temp
        )
        tm = _tm(lnuc, llink, elink, mu, edata, temp=temp)

        assert tm.getLogZ() == pytest.approx(logz_bf, rel=1e-8)
        np.testing.assert_allclose(tm.getStartProb(), start_bf, atol=1e-8)
        occup = np.asarray(tm.getOccup())
        assert np.all(occup >= -1e-12) and np.all(occup <= 1 + 1e-12)
        assert np.isfinite(tm.getVarN())

    def test_samples_ground_state(self):
        rng = np.random.default_rng(3)
        lnuc, llink, elink, mu, nbp, temp = 3, 2, 1.0, 0.5, 16, 0.01
        edata = (-10.0 + rng.normal(size=nbp)).tolist()
        _, _, ground = _brute_force_log(lnuc, llink, elink, mu, edata, temp)
        tm = _tm(lnuc, llink, elink, mu, edata, temp=temp, seed=4)
        for _ in range(10):
            positions, _energy = tm.sampleConfig()
            assert list(positions) == ground

    @pytest.mark.parametrize("scale", [1.0, 10.0])
    def test_long_fiber_marginals_are_valid(self, scale):
        rng = np.random.default_rng(5)
        edata = np.clip(scale * rng.normal(size=2000) - 5.0, -10.0, 10.0)
        tm = _tm(147, 50, 1.0, 0.0, edata.tolist(), temp=0.01)
        start_prob = np.asarray(tm.getStartProb())
        occup = np.asarray(tm.getOccup())

        assert np.isfinite(tm.getLogZ())
        assert np.all(start_prob >= 0.0)
        assert np.all(start_prob <= 1 + 1e-9)
        assert np.all(occup >= -1e-12) and np.all(occup <= 1 + 1e-9)
        assert start_prob.sum() == pytest.approx(tm.getMeanN(), rel=1e-8)
        assert tm.getMeanN() > 0.0


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


def _config_energy(lnuc, llink, elink, mu, edata, placed):
    energy = sum(edata[x] - mu for x in placed)
    energy += sum(
        _erep(b - a, lnuc, llink, elink)
        for a, b in itertools.pairwise(placed)
    )
    return energy


PARAM_SETS = [
    (2, 2, 1.0, 0.5, 10),
    (3, 2, 0.7, -0.3, 14),
    (2, 3, 2.0, 1.2, 12),
    (4, 1, 0.5, 0.0, 13),
    (3, 0, 1.0, 0.5, 14),
]


class TestMostLikelyConfig:
    @pytest.mark.parametrize("lnuc,llink,elink,mu,nbp", PARAM_SETS)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_brute_force_ground_state(
        self, lnuc, llink, elink, mu, nbp, seed
    ):
        rng = np.random.default_rng(seed)
        edata = rng.normal(size=nbp).tolist()
        _, _, ground = _brute_force_log(lnuc, llink, elink, mu, edata, 1.0)
        tm = _tm(lnuc, llink, elink, mu, edata)

        positions, energy = tm.getMostLikelyConfig()

        assert list(positions) == ground
        assert energy == pytest.approx(
            _config_energy(lnuc, llink, elink, mu, edata, ground)
        )

    def test_does_not_require_solve(self):
        tm = NucPosTM(2, 10, 2, 1.0, 0.5, seed=1)
        tm.setEnergy([-1.0] * 10, 1e9)
        positions, energy = tm.getMostLikelyConfig()
        assert len(positions) > 0
        assert np.isfinite(energy)

    def test_empty_when_placement_is_unfavorable(self):
        tm = NucPosTM(3, 12, 2, 1.0, 0.0, seed=1)
        tm.setEnergy([5.0] * 12, 1e9)
        positions, energy = tm.getMostLikelyConfig()
        assert list(positions) == []
        assert energy == 0.0

    def test_independent_of_solve_temperature(self):
        rng = np.random.default_rng(7)
        edata = rng.normal(size=14).tolist()
        cold = _tm(3, 2, 1.0, 0.5, edata, temp=0.1).getMostLikelyConfig()
        hot = _tm(3, 2, 1.0, 0.5, edata, temp=5.0).getMostLikelyConfig()
        assert list(cold[0]) == list(hot[0])

    def test_long_fiber_is_valid(self):
        rng = np.random.default_rng(5)
        edata = np.clip(rng.normal(size=2000) - 1.0, -10.0, 10.0).tolist()
        tm = NucPosTM(147, 2000, 50, 1.0, 0.0, seed=1)
        tm.setEnergy(edata, 1e9)
        positions, energy = tm.getMostLikelyConfig()
        positions = np.asarray(positions)

        assert len(positions) > 0
        assert np.all(np.diff(positions) >= 147)
        assert positions.min() >= 0 and positions.max() <= 2000 - 147
        assert energy == pytest.approx(
            _config_energy(147, 50, 1.0, 0.0, edata, positions.tolist())
        )


def _valid_configs(lnuc, npos):
    configs = []

    def recurse(idx, placed):
        if idx == npos:
            configs.append(placed)
            return
        recurse(idx + 1, placed)
        if not placed or idx - placed[-1] >= lnuc:
            recurse(idx + 1, [*placed, idx])

    recurse(0, [])
    return configs


def _footprint_gain(occup, lnuc, gamma):
    c = (1.0 + gamma) * np.asarray(occup) - 1.0
    npos = len(occup) - lnuc + 1
    return np.array([c[x : x + lnuc].sum() for x in range(npos)])


def _repulsion(placed, lnuc, llink, elink):
    return sum(
        _erep(b - a, lnuc, llink, elink)
        for a, b in itertools.pairwise(placed)
    )


class TestBestMatchConfig:
    @pytest.mark.parametrize("lnuc,llink,elink,mu,nbp", PARAM_SETS)
    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_matches_brute_force_expected_accuracy(
        self, lnuc, llink, elink, mu, nbp, gamma
    ):
        rng = np.random.default_rng(nbp)
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata)
        gain = _footprint_gain(tm.getOccup(), lnuc, gamma)
        configs = _valid_configs(lnuc, nbp - lnuc + 1)
        scores = [
            gain[c].sum() - _repulsion(c, lnuc, llink, elink) if c else 0.0
            for c in configs
        ]
        best = configs[int(np.argmax(scores))]

        positions, energy = tm.getBestMatchConfig(gamma)
        positions = list(positions)

        assert positions == best
        score = gain[positions].sum() - _repulsion(
            positions, lnuc, llink, elink
        )
        assert score == pytest.approx(max(scores))
        assert energy == pytest.approx(
            _config_energy(lnuc, llink, elink, mu, edata, best)
        )

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_no_gap_with_repulsion_above_max_gain(self, gamma):
        lnuc, llink, elink = 20, 6, 1.0
        rng = np.random.default_rng(3)
        edata = (rng.normal(size=600) - 2.0).tolist()
        tm = _tm(lnuc, llink, elink, 0.0, edata)

        positions, _ = tm.getBestMatchConfig(gamma)

        gaps = np.diff(positions)
        assert len(gaps) > 0
        assert all(
            _erep(g, lnuc, llink, elink) <= gamma * lnuc for g in gaps
        )

    def test_flat_high_occupancy_not_packed(self):
        lnuc, llink = 147, 25
        tm = _tm(lnuc, llink, 1.0, 0.0, [-5.0] * 3000)

        positions, _ = tm.getBestMatchConfig()

        assert len(positions) > 1
        assert np.diff(positions).min() > lnuc

    def test_no_repulsion_matches_unpenalized(self):
        lnuc, nbp = 3, 14
        rng = np.random.default_rng(4)
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, 0, 1.0, 0.5, edata)
        gain = _footprint_gain(tm.getOccup(), lnuc, 1.0)
        configs = _valid_configs(lnuc, nbp - lnuc + 1)
        scores = [gain[c].sum() if c else 0.0 for c in configs]

        positions, _ = tm.getBestMatchConfig()

        assert list(positions) == configs[int(np.argmax(scores))]

    def test_default_gamma_covers_sites_above_half(self):
        tm = NucPosTM(3, 8, 0, 0.0, 0.0, seed=1)
        tm.setEnergy([-3.0, 2.0, 2.0, 5.0, -3.0, 2.0, 2.0, 5.0], 1e9)
        tm.solve()
        occup = np.asarray(tm.getOccup())
        positions, _ = tm.getBestMatchConfig()
        occupied = np.zeros(8, dtype=bool)
        for x in positions:
            occupied[x : x + 3] = True
        np.testing.assert_array_equal(occupied, occup > 0.5)

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_occupied_occupancy_grows_with_gamma(self, seed):
        rng = np.random.default_rng(seed)
        edata = rng.normal(size=14).tolist()
        tm = _tm(3, 2, 1.0, 0.5, edata)
        occup = np.asarray(tm.getOccup())

        def occupied_mass(gamma):
            occupied = np.zeros(14, dtype=bool)
            for x in tm.getBestMatchConfig(gamma)[0]:
                occupied[x : x + 3] = True
            return occup[occupied].sum()

        masses = [occupied_mass(g) for g in (0.25, 1.0, 4.0)]
        assert masses[0] <= masses[1] + 1e-12
        assert masses[1] <= masses[2] + 1e-12

    def test_requires_solve(self):
        tm = NucPosTM(2, 10, 2, 1.0, 0.5, seed=1)
        tm.setEnergy([0.0] * 10, 1e9)
        with pytest.raises(RuntimeError):
            tm.getBestMatchConfig()

    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_nonpositive_gamma_raises(self, gamma):
        tm = _tm(2, 2, 1.0, 0.5, [0.0] * 10)
        with pytest.raises(ValueError, match="gamma must be positive"):
            tm.getBestMatchConfig(gamma)

    def test_long_fiber_is_valid(self):
        rng = np.random.default_rng(5)
        edata = np.clip(rng.normal(size=2000) - 1.0, -10.0, 10.0).tolist()
        tm = _tm(147, 50, 1.0, 0.0, edata)
        positions, energy = tm.getBestMatchConfig()
        positions = np.asarray(positions)

        assert len(positions) > 0
        assert np.all(np.diff(positions) >= 147)
        assert positions.min() >= 0 and positions.max() <= 2000 - 147
        assert energy == pytest.approx(
            _config_energy(147, 50, 1.0, 0.0, edata, positions.tolist())
        )


def _linker_hist_bf(lnuc, llink, elink, mu, edata, temp, lmax, xlo, xhi):
    nbp = len(edata)
    configs = _valid_configs(lnuc, nbp - lnuc + 1)
    logw = np.array(
        [
            -_config_energy(lnuc, llink, elink, mu, edata, c) / temp
            for c in configs
        ]
    )
    probs = np.exp(logw - logw.max())
    probs /= probs.sum()
    hist = np.zeros(lmax + 2)
    for p, c in zip(probs, configs):
        for a, b in itertools.pairwise(c):
            if a < xlo or b + lnuc > xhi:
                continue
            length = b - a - lnuc
            hist[min(length, lmax + 1)] += p
    return hist


class TestLinkerHist:
    @pytest.mark.parametrize("lnuc,llink,elink,mu,nbp", PARAM_SETS)
    @pytest.mark.parametrize("temp", [1.0, 2.0])
    @pytest.mark.parametrize("trim", [0, 1])
    def test_matches_brute_force(
        self, lnuc, llink, elink, mu, nbp, temp, trim
    ):
        rng = np.random.default_rng(nbp)
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, temp=temp)
        lmax = 2
        xlo, xhi = trim, nbp - trim
        expected = _linker_hist_bf(
            lnuc, llink, elink, mu, edata, temp, lmax, xlo, xhi
        )
        got = tm.getLinkerHist(lmax, xlo, xhi)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("lnuc,llink,elink,mu,nbp", PARAM_SETS)
    def test_sum_rule(self, lnuc, llink, elink, mu, nbp):
        rng = np.random.default_rng(nbp + 1)
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata)
        hist = np.asarray(tm.getLinkerHist(nbp))
        expected = tm.getMeanN() - 1.0 + math.exp(-tm.getLogZ())
        assert hist.sum() == pytest.approx(expected, rel=1e-10)
        assert hist[-1] == pytest.approx(0.0, abs=1e-15)

    def test_default_window_is_full_fiber(self):
        rng = np.random.default_rng(3)
        edata = rng.normal(size=14).tolist()
        tm = _tm(3, 2, 1.0, 0.5, edata)
        np.testing.assert_array_equal(
            tm.getLinkerHist(2), tm.getLinkerHist(2, 0, 14)
        )

    def test_short_fiber_gives_zeros(self):
        tm = _tm(3, 2, 1.0, 0.5, [-1.0] * 5)
        assert list(tm.getLinkerHist(4)) == [0.0] * 6

    def test_lmax_zero_has_one_bin_and_tail(self):
        rng = np.random.default_rng(4)
        tm = _tm(2, 2, 1.0, 0.5, rng.normal(size=10).tolist())
        assert len(tm.getLinkerHist(0)) == 2

    def test_empty_window_gives_zeros(self):
        rng = np.random.default_rng(5)
        tm = _tm(2, 2, 1.0, 0.5, rng.normal(size=10).tolist())
        assert list(tm.getLinkerHist(3, 4, 4)) == [0.0] * 5

    def test_negative_lmax_raises(self):
        tm = _tm(2, 2, 1.0, 0.5, [0.0] * 10)
        with pytest.raises(ValueError, match="lmax must be nonnegative"):
            tm.getLinkerHist(-1)

    @pytest.mark.parametrize("xlo,xhi", [(-1, 10), (5, 3), (0, 11)])
    def test_invalid_window_raises(self, xlo, xhi):
        tm = _tm(2, 2, 1.0, 0.5, [0.0] * 10)
        with pytest.raises(ValueError, match="window must satisfy"):
            tm.getLinkerHist(3, xlo, xhi)

    def test_requires_solve(self):
        tm = NucPosTM(2, 10, 2, 1.0, 0.5, seed=1)
        tm.setEnergy([0.0] * 10, 1e9)
        with pytest.raises(RuntimeError):
            tm.getLinkerHist(3)

    def test_matches_sampled_histogram(self):
        rng = np.random.default_rng(6)
        lnuc, llink, elink, mu, nbp = 2, 2, 1.0, 0.5, 30
        edata = rng.normal(size=nbp).tolist()
        tm = _tm(lnuc, llink, elink, mu, edata, seed=9)
        lmax = 5
        exact = np.asarray(tm.getLinkerHist(lmax))
        n_samples = 20000
        counts = np.zeros(lmax + 2)
        for _ in range(n_samples):
            positions, _energy = tm.sampleConfig()
            for a, b in itertools.pairwise(positions):
                counts[min(b - a - lnuc, lmax + 1)] += 1
        np.testing.assert_allclose(counts / n_samples, exact, atol=0.05)

    def test_long_fiber_is_finite_and_obeys_sum_rule(self):
        rng = np.random.default_rng(7)
        edata = np.clip(rng.normal(size=2000) - 1.0, -10.0, 10.0).tolist()
        tm = _tm(147, 50, 1.0, 0.0, edata)
        hist = np.asarray(tm.getLinkerHist(1853))
        assert np.all(np.isfinite(hist)) and np.all(hist >= 0.0)
        expected = tm.getMeanN() - 1.0 + math.exp(-tm.getLogZ())
        assert hist.sum() == pytest.approx(expected, rel=1e-8)
        short = np.asarray(tm.getLinkerHist(100))
        np.testing.assert_allclose(short[:101], hist[:101])
        assert short[-1] == pytest.approx(hist[101:].sum(), rel=1e-8)
