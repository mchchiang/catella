# test_exact.py

import itertools

import numpy as np
import pandas as pd
import pytest
from catella.h5_array import H5Array
from catella.simulation._exact import (
    TMStat,
    _mean_nnuc,
    _reject_time,
    _require_edata,
    _require_tm,
    _solved_tm,
    _spanned_fraction,
    _spanned_rows,
    _tm_configs,
    _tm_per_bp,
    _tm_per_mol,
    _tm_stats,
    _var_nnuc,
)
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset
from catella_cpp import NucPosTM

from tests.simulation.test_tm_bruteforce import _erep


def _make_tm_settings(**overrides):
    defaults = {
        "lnuc": 3,
        "llink": 2,
        "elink": 1.0,
        "mu": 0.5,
        "end_temp": 0.8,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_tm_dataset(tmp_path, *, nmol=3, nbp=12, **kwargs):
    rng = np.random.default_rng(4)
    return SimManager(nworker=1, verbose=False).run(
        regions="chr1",
        nsim=0,
        method="tm",
        settings=_make_tm_settings(),
        meth_prob=rng.uniform(0.05, 0.95, (nmol, nbp)),
        out_dir=tmp_path / "tm",
        seed=1,
        **kwargs,
    )


def _make_mcmc_dataset(tmp_path):
    return SimDataset.create(
        regions=["chr1"],
        nmol={"chr1": 1},
        nsim=1,
        nbp={"chr1": 12},
        settings=_make_tm_settings(),
        out_dir=tmp_path / "mcmc",
    )


class TestRequireEdata:
    def test_passes_with_edata(self, tmp_path):
        _require_edata(_make_tm_dataset(tmp_path))

    def test_raises_without_edata(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, store_edata=False)
        with pytest.raises(ValueError, match="store_edata"):
            _require_edata(dataset)


class TestRequireTM:
    def test_passes_for_tm_with_edata(self, tmp_path):
        _require_tm(_make_tm_dataset(tmp_path), "f")

    def test_mcmc_raises_with_caller_name(self, tmp_path):
        with pytest.raises(ValueError, match="my_method requires"):
            _require_tm(_make_mcmc_dataset(tmp_path), "my_method")

    def test_tm_without_edata_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, store_edata=False)
        with pytest.raises(ValueError, match="store_edata"):
            _require_tm(dataset, "f")


class TestRejectTime:
    def test_passes_without_time(self):
        _reject_time(None, False)

    @pytest.mark.parametrize(
        ("time", "record_time"), [(0, False), (10, False), (None, True)]
    )
    def test_raises_with_time(self, time, record_time):
        with pytest.raises(ValueError):
            _reject_time(time, record_time)


class TestSolvedTM:
    def test_matches_manual_model(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        tm = _solved_tm(dataset, "chr1", 1)
        ref = NucPosTM(3, 12, 2, 1.0, 0.5, 1)
        ref.setEnergy(dataset.edata["chr1"][1].tolist(), 5.0)
        ref.solve(0.8)
        np.testing.assert_allclose(tm.getOccup(), ref.getOccup())
        assert tm.getLogZ() == pytest.approx(ref.getLogZ())


class TestTMPerMol:
    def test_stores_one_value_per_molecule(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _tm_per_mol(dataset, ["chr1"], "m", NucPosTM.getMeanN)
        df = dataset.analysis["chr1"]["m"]
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 1)
        for mol in range(3):
            expected = _solved_tm(dataset, "chr1", mol).getMeanN()
            assert df.iloc[mol, 0] == pytest.approx(expected)


class TestTMPerBp:
    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    def test_stores_profile_per_molecule(self, tmp_path, batch_size):
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=12)
        _tm_per_bp(dataset, ["chr1"], "o", batch_size, NucPosTM.getOccup)
        arr = dataset.analysis["chr1"]["o"]
        assert isinstance(arr, H5Array)
        assert arr.shape == (3, 12)
        for mol in range(3):
            expected = _solved_tm(dataset, "chr1", mol).getOccup()
            np.testing.assert_allclose(arr.to_numpy()[mol], expected)


def _make_tm_dataset_with_gap(tmp_path):
    rng = np.random.default_rng(5)
    meth_prob = rng.uniform(0.05, 0.95, (3, 12))
    meth_prob[1] = np.nan
    return SimManager(nworker=1, verbose=False).run(
        regions="chr1",
        nsim=0,
        method="tm",
        settings=_make_tm_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "tm_gap",
        seed=1,
    )


class TestTMStatsFixedWidth:
    def test_stores_fixed_width_rows(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=12)
        stat = TMStat("h", "getLinkerHist", {"lmax": 2}, ncol=4)
        _tm_stats(dataset, ["chr1"], [stat], batch_size=2)
        arr = dataset.analysis["chr1"]["h"]
        assert isinstance(arr, H5Array)
        assert arr.shape == (3, 4)
        for mol in range(3):
            expected = _solved_tm(dataset, "chr1", mol).getLinkerHist(2)
            np.testing.assert_allclose(arr.to_numpy()[mol], expected)

    def test_column_labels(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        labels = ("0", "1", "2", "tail")
        stat = TMStat(
            "h", "getLinkerHist", {"lmax": 2}, ncol=4, columns=labels
        )
        _tm_stats(dataset, ["chr1"], [stat])
        cols = dataset.analysis["chr1"]["h"].columns
        assert list(cols) == list(labels)

    def test_no_data_molecule_gives_nan_row(self, tmp_path):
        dataset = _make_tm_dataset_with_gap(tmp_path)
        stat = TMStat("h", "getLinkerHist", {"lmax": 2}, ncol=4)
        _tm_stats(dataset, ["chr1"], [stat])
        arr = dataset.analysis["chr1"]["h"].to_numpy()
        assert np.isnan(arr[1]).all()
        assert np.isfinite(arr[[0, 2]]).all()

    def test_mixed_with_scalar_stat(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        stats = [
            TMStat("h", "getLinkerHist", {"lmax": 1}, ncol=3),
            TMStat("n", "getMeanN"),
        ]
        _tm_stats(dataset, ["chr1"], stats)
        analysis = dataset.analysis["chr1"]
        assert analysis["h"].shape == (3, 3)
        assert isinstance(analysis["n"], pd.DataFrame)
        for mol in range(3):
            tm = _solved_tm(dataset, "chr1", mol)
            assert analysis["n"].iloc[mol, 0] == pytest.approx(tm.getMeanN())

    def test_per_bp_and_ncol_raise(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TMStat("h", "getOccup", per_bp=True, ncol=3)

    @pytest.mark.parametrize(
        ("stat", "expected"),
        [
            (TMStat("a", "getMeanN"), None),
            (TMStat("b", "getOccup", per_bp=True), 12),
            (TMStat("c", "getLinkerHist", ncol=5), 5),
        ],
    )
    def test_width(self, stat, expected):
        assert stat.width(12) == expected


def _interval_count(tm, intervals, settings):
    return -1.0 if intervals is None else float(len(intervals))


def _first_start(tm, intervals, settings, shift=0):
    return float(intervals[0, 0] + shift)


def _mean_n_and_intervals(tm, intervals, settings):
    return tm.getMeanN() + len(intervals) + settings.lnuc


def _make_partial_tm_dataset(tmp_path):
    rng = np.random.default_rng(6)
    meth_prob = rng.uniform(0.05, 0.95, (4, 12))
    meth_prob[0, :4] = np.nan
    meth_prob[1, 3:7] = np.nan
    meth_prob[3] = np.nan
    return SimManager(nworker=1, verbose=False).run(
        regions="chr1",
        nsim=0,
        method="tm",
        settings=_make_tm_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "tm_partial",
        seed=1,
        fill_nan=0.5,
    )


class TestTMStatsPythonGetter:
    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    @pytest.mark.parametrize("nworker", [1, 2])
    def test_gets_each_molecule_intervals(self, tmp_path, batch_size, nworker):
        dataset = _make_partial_tm_dataset(tmp_path)
        _tm_stats(
            dataset,
            ["chr1"],
            [TMStat("k", _interval_count)],
            batch_size=batch_size,
            nworker=nworker,
        )
        got = dataset.analysis["chr1"]["k"].to_numpy().ravel()
        np.testing.assert_array_equal(got[:3], [1, 2, 1])
        assert np.isnan(got[3])

    def test_kwargs_are_passed(self, tmp_path):
        dataset = _make_partial_tm_dataset(tmp_path)
        stat = TMStat("s", _first_start, {"shift": 100})
        _tm_stats(dataset, ["chr1"], [stat], batch_size=2)
        got = dataset.analysis["chr1"]["s"].to_numpy().ravel()
        np.testing.assert_array_equal(got[:3], [106, 100, 100])

    def test_intervals_none_when_all_spanned(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _tm_stats(dataset, ["chr1"], [TMStat("k", _interval_count)])
        got = dataset.analysis["chr1"]["k"].to_numpy().ravel()
        np.testing.assert_array_equal(got, [-1, -1, -1])

    def test_mixed_with_cpp_getter(self, tmp_path):
        dataset = _make_partial_tm_dataset(tmp_path)
        stats = [
            TMStat("n", "getMeanN"),
            TMStat("m", _mean_n_and_intervals),
        ]
        _tm_stats(dataset, ["chr1"], stats, batch_size=3)
        analysis = dataset.analysis["chr1"]
        n = analysis["n"].to_numpy().ravel()
        m = analysis["m"].to_numpy().ravel()
        np.testing.assert_allclose(m[:3], n[:3] + [1, 2, 1] + 3)


def _solved(edata, settings):
    tm = NucPosTM(
        settings.lnuc,
        len(edata),
        settings.llink,
        settings.elink,
        settings.mu,
        1,
    )
    tm.setEnergy(list(edata), settings.emax)
    tm.solve(settings.end_temp)
    return tm


def _brute_fraction_moments(edata, settings, frac):
    lnuc, npos = settings.lnuc, len(frac)
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
        energy = sum(edata[x] - settings.mu for x in placed)
        energy += sum(
            _erep(b - a, lnuc, settings.llink, settings.elink)
            for a, b in itertools.pairwise(placed)
        )
        logw.append(-energy / settings.end_temp)
    logw = np.asarray(logw)
    prob = np.exp(logw - logw.max())
    prob /= prob.sum()
    count = np.array([frac[placed].sum() for placed in configs])
    mean = float(prob @ count)
    return mean, float(prob @ (count - mean) ** 2)


class TestSpannedRows:
    _SPANNED = (np.array([[1, 3], [4, 6], [0, 2]]), np.array([0, 2, 2, 3]))

    def test_rows_in_given_order(self):
        got = _spanned_rows(self._SPANNED, [2, 0, 1], 6)
        np.testing.assert_array_equal(
            got,
            [
                [True, True, False, False, False, False],
                [False, True, True, False, True, True],
                [False] * 6,
            ],
        )

    def test_repeated_molecule(self):
        got = _spanned_rows(self._SPANNED, [2, 2], 6)
        assert got.shape == (2, 6)
        np.testing.assert_array_equal(got[0], got[1])

    def test_none(self):
        assert _spanned_rows(None, [0, 1], 6) is None


class TestSpannedFraction:
    def test_hand_example(self):
        got = _spanned_fraction(np.array([[2, 5]]), 8, 3)
        np.testing.assert_allclose(got, [1, 2, 3, 2, 1, 0] / np.float64(3))

    def test_two_intervals(self):
        got = _spanned_fraction(np.array([[0, 2], [4, 6]]), 6, 2)
        np.testing.assert_allclose(got, [1, 0.5, 0, 0.5, 1])


class TestCountGetters:
    _EDATA = np.random.default_rng(8).normal(0, 1.5, 10)

    def test_none_intervals_match_cpp(self):
        settings = _make_tm_settings()
        tm = _solved(self._EDATA, settings)
        assert _mean_nnuc(tm, None, settings) == tm.getMeanN()
        assert _var_nnuc(tm, None, settings) == tm.getVarN()

    @pytest.mark.parametrize("emax", [5.0, 0.7])
    def test_full_interval_matches_cpp(self, emax):
        settings = _make_tm_settings(emax=emax)
        tm = _solved(self._EDATA, settings)
        full = np.array([[0, 10]])
        assert _mean_nnuc(tm, full, settings) == pytest.approx(tm.getMeanN())
        assert _var_nnuc(tm, full, settings) == pytest.approx(
            tm.getVarN(), rel=1e-5
        )

    @pytest.mark.parametrize(
        "intervals", [[[2, 7]], [[0, 3], [6, 10]], [[4, 5]]]
    )
    def test_matches_brute_force(self, intervals):
        settings = _make_tm_settings()
        tm = _solved(self._EDATA, settings)
        intervals = np.array(intervals)
        frac = _spanned_fraction(intervals, 10, settings.lnuc)
        mean, var = _brute_fraction_moments(self._EDATA, settings, frac)
        assert _mean_nnuc(tm, intervals, settings) == pytest.approx(mean)
        assert _var_nnuc(tm, intervals, settings) == pytest.approx(
            var, rel=1e-5
        )


class TestTMConfigs:
    def test_stores_configs_energy_and_log_prob(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _tm_configs(dataset, ["chr1"], "ml", NucPosTM.getMostLikelyConfig)
        analysis = dataset.analysis["chr1"]
        flat = analysis["ml_position_flat"]["position"].to_numpy()
        offset = analysis["ml_position_offset"]["offset"].to_numpy()
        energy = analysis["ml_energy"]
        assert len(offset) == 4
        for mol in range(3):
            tm = _solved_tm(dataset, "chr1", mol)
            pos, e = tm.getMostLikelyConfig()
            got = flat[int(offset[mol]) : int(offset[mol + 1])]
            np.testing.assert_array_equal(got, pos)
            assert energy["energy"].iloc[mol] == pytest.approx(e)
            assert energy["log_prob"].iloc[mol] == pytest.approx(
                -e / 0.8 - tm.getLogZ()
            )
