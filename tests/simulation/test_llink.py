# test_llink.py

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from catella.h5_array import H5Array
from catella.simulation._llink import _estimate_llink, _llink_evidence
from catella.simulation.config import SimSettings
from scipy.special import expit

from tests.simulation.test_tm_bruteforce import _brute_force


def _settings(**overrides):
    values = {
        "nucbp": 20,
        "llink": 7,
        "elink": 25.0,
        "mu": 0.0,
        "end_temp": 1.0,
        "emax": 100.0,
    }
    values.update(overrides)
    return SimSettings(**values)


def _exp(arrays):
    return SimpleNamespace(
        chroms=tuple(arrays),
        analysis={c: {"meth_prob": a} for c, a in arrays.items()},
    )


def _planted(nmol, linker, nucbp=20, nbp=600, seed=0):
    rng = np.random.default_rng(seed)
    energy = np.full((nmol, nbp), 5.0) + rng.normal(0, 0.1, (nmol, nbp))
    for i in range(nmol):
        start = int(rng.integers(0, nucbp + linker))
        energy[i, start : nbp - nucbp + 1 : nucbp + linker] = -20.0
    return expit(energy)


def _random_probs(nmol=8, nbp=60, seed=1):
    return np.random.default_rng(seed).uniform(0.05, 0.95, (nmol, nbp))


class TestEvidence:
    @pytest.mark.parametrize("llink", [0, 1, 2, 3])
    def test_matches_enumeration(self, llink):
        rng = np.random.default_rng(llink)
        nucbp, elink, mu = 3, 1.5, 0.5
        energies = rng.normal(0, 1, (3, 14))
        table = _llink_evidence(
            _exp({"c": expit(energies)}),
            _settings(nucbp=nucbp, elink=elink, mu=mu),
            llinks=[llink],
        )
        zero = [0.0] * 14
        want = sum(
            _brute_force(nucbp, llink, elink, mu, e.tolist())[0]
            - _brute_force(nucbp, llink, elink, mu, zero)[0]
            for e in energies
        )
        assert table["evidence"].iloc[0] == pytest.approx(want)

    def test_zero_energy_gives_zero_evidence(self):
        table = _llink_evidence(
            _exp({"c": np.full((3, 200), 0.5)}), _settings()
        )
        np.testing.assert_allclose(table["evidence"], 0.0, atol=1e-8)

    def test_columns_and_rows(self):
        table = _llink_evidence(
            _exp({"a": _random_probs(), "b": _random_probs(seed=2)}),
            _settings(),
            llinks=[0, 5, 10],
        )
        assert list(table.columns) == ["chrom", "llink", "evidence", "nmol"]
        assert len(table) == 6
        assert set(table["chrom"]) == {"a", "b"}
        assert (table["nmol"] == 8).all()

    def test_ignores_settings_llink(self):
        exp = _exp({"c": _random_probs()})
        pd.testing.assert_frame_equal(
            _llink_evidence(exp, _settings(llink=1), llinks=[0, 5]),
            _llink_evidence(exp, _settings(llink=99), llinks=[0, 5]),
        )

    def test_rejects_too_large_llink(self):
        table = _llink_evidence(
            _exp({"c": _planted(10, linker=30)}),
            _settings(),
            llinks=[30, 60],
        ).set_index("llink")
        assert table.loc[30, "evidence"] > table.loc[60, "evidence"]

    def test_chroms_selects(self):
        exp = _exp({"a": _random_probs(), "b": _random_probs(seed=2)})
        for chroms in ("b", ["b"]):
            table = _llink_evidence(
                exp, _settings(), llinks=[0], chroms=chroms
            )
            assert list(table["chrom"]) == ["b"]

    def test_all_nan_rows_are_skipped(self):
        probs = _random_probs()
        with_nan = np.vstack([probs, np.full((2, probs.shape[1]), np.nan)])
        got = _llink_evidence(_exp({"c": with_nan}), _settings(), llinks=[5])
        want = _llink_evidence(_exp({"c": probs}), _settings(), llinks=[5])
        assert got["nmol"].iloc[0] == 8
        assert got["evidence"].iloc[0] == pytest.approx(
            want["evidence"].iloc[0]
        )

    def test_mask_excludes_molecules(self):
        probs = _random_probs()
        exp = _exp({"c": probs})
        keep = np.array([True, False] * 4)
        exp.analysis["c"]["test_dm"] = pd.DataFrame({"keep": keep})
        got = _llink_evidence(exp, _settings(), llinks=[5], mask_name="dm")
        want = _llink_evidence(
            _exp({"c": probs[keep]}), _settings(), llinks=[5]
        )
        assert got["nmol"].iloc[0] == 4
        assert got["evidence"].iloc[0] == pytest.approx(
            want["evidence"].iloc[0]
        )

    def test_missing_mask_raises(self):
        with pytest.raises(KeyError):
            _llink_evidence(
                _exp({"c": _random_probs()}), _settings(), mask_name="dm"
            )

    def test_missing_prob_name_raises(self):
        with pytest.raises(KeyError):
            _llink_evidence(
                _exp({"c": _random_probs()}), _settings(), prob_name="nope"
            )

    def test_no_usable_molecules_raises(self):
        with pytest.raises(ValueError, match="no usable molecules"):
            _llink_evidence(
                _exp({"c": np.full((2, 60), np.nan)}), _settings(), llinks=[0]
            )

    def test_dataframe_and_h5array_inputs(self, tmp_path):
        probs = _random_probs()
        want = _llink_evidence(_exp({"c": probs}), _settings(), llinks=[5])
        stored = H5Array.create(probs.shape, dtype=np.float64, dir=tmp_path)
        stored.write_batch(0, probs.shape[0], probs)
        for arr in (pd.DataFrame(probs), stored):
            got = _llink_evidence(_exp({"c": arr}), _settings(), llinks=[5])
            assert got["evidence"].iloc[0] == pytest.approx(
                want["evidence"].iloc[0]
            )

    def test_batch_size_does_not_change_result(self):
        exp = _exp({"c": _random_probs()})
        a = _llink_evidence(exp, _settings(), llinks=[5], batch_size=3)
        b = _llink_evidence(exp, _settings(), llinks=[5])
        assert a["evidence"].iloc[0] == pytest.approx(b["evidence"].iloc[0])

    def test_settings_from_file(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "nucbp": 20,
                    "llink": 7,
                    "elink": 25.0,
                    "mu": 0.0,
                    "end_temp": 1.0,
                    "emax": 100.0,
                }
            )
        )
        exp = _exp({"c": _random_probs()})
        pd.testing.assert_frame_equal(
            _llink_evidence(exp, path, llinks=[5]),
            _llink_evidence(exp, _settings(), llinks=[5]),
        )

    def test_invalid_settings_raises(self):
        with pytest.raises(TypeError, match="settings"):
            _llink_evidence(_exp({"c": _random_probs()}), 1)

    @pytest.mark.parametrize(
        ("llinks", "match"),
        [
            ([], "must not be empty"),
            ([-1], "non-negative integers"),
            ([1.5], "non-negative integers"),
            ([True], "non-negative integers"),
        ],
    )
    def test_invalid_llinks_raises(self, llinks, match):
        with pytest.raises(ValueError, match=match):
            _llink_evidence(
                _exp({"c": _random_probs()}), _settings(), llinks=llinks
            )

    def test_does_not_modify_experiment(self):
        probs = _random_probs()
        before = probs.copy()
        exp = _exp({"c": probs})
        _llink_evidence(exp, _settings(), llinks=[5])
        np.testing.assert_array_equal(probs, before)
        assert set(exp.analysis["c"]) == {"meth_prob"}


class TestMaxNmol:
    def _table(self, probs=None, **kwargs):
        probs = _random_probs(nmol=20) if probs is None else probs
        return _llink_evidence(
            _exp({"c": probs}), _settings(), llinks=[5], **kwargs
        )

    def test_caps_per_chromosome(self):
        table = _llink_evidence(
            _exp({"a": _random_probs(nmol=20), "b": _random_probs(nmol=3)}),
            _settings(),
            llinks=[5],
            max_nmol=5,
            seed=1,
        ).set_index("chrom")
        assert table.loc["a", "nmol"] == 5
        assert table.loc["b", "nmol"] == 3

    def test_same_seed_repeats(self):
        a = self._table(max_nmol=5, seed=3)["evidence"].iloc[0]
        b = self._table(max_nmol=5, seed=3)["evidence"].iloc[0]
        assert a == b

    def test_different_seed_changes_sample(self):
        a = self._table(max_nmol=5, seed=3)["evidence"].iloc[0]
        b = self._table(max_nmol=5, seed=4)["evidence"].iloc[0]
        assert a != b

    def test_sample_is_not_first_molecules(self):
        probs = _random_probs(nmol=20)
        sampled = self._table(probs, max_nmol=5, seed=3)["evidence"].iloc[0]
        first = self._table(probs[:5])["evidence"].iloc[0]
        assert sampled != pytest.approx(first)

    def test_above_count_matches_all(self):
        a = self._table(max_nmol=1000, seed=3)["evidence"].iloc[0]
        b = self._table()["evidence"].iloc[0]
        assert a == pytest.approx(b)

    def test_applies_after_skipping_all_nan(self):
        probs = np.vstack([np.full((6, 60), np.nan), _random_probs(nmol=4)])
        assert self._table(probs, max_nmol=4, seed=1)["nmol"].iloc[0] == 4

    @pytest.mark.parametrize("value", [0, -1])
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError, match="max_nmol must be positive"):
            self._table(max_nmol=value)


class TestNworker:
    def _exp(self):
        return _exp(
            {
                "a": _random_probs(nmol=10),
                "b": _random_probs(nmol=7, seed=2),
            }
        )

    def test_parallel_matches_serial(self, monkeypatch):
        monkeypatch.setattr("catella.simulation._llink._CHUNK_NMOL", 3)
        exp = self._exp()
        serial = _llink_evidence(exp, _settings(), llinks=[0, 5, 10])
        parallel = _llink_evidence(
            exp, _settings(), llinks=[0, 5, 10], nworker=2
        )
        pd.testing.assert_frame_equal(serial, parallel)

    def test_chunk_size_does_not_change_result(self, monkeypatch):
        exp = self._exp()
        whole = _llink_evidence(exp, _settings(), llinks=[0, 5])
        monkeypatch.setattr("catella.simulation._llink._CHUNK_NMOL", 2)
        chunked = _llink_evidence(exp, _settings(), llinks=[0, 5])
        pd.testing.assert_frame_equal(whole, chunked)

    def test_parallel_with_sampling_matches_serial(self):
        exp = self._exp()
        kwargs = {"llinks": [5], "max_nmol": 4, "seed": 3}
        pd.testing.assert_frame_equal(
            _llink_evidence(exp, _settings(), **kwargs),
            _llink_evidence(exp, _settings(), nworker=2, **kwargs),
        )

    def test_parallel_with_h5array(self, tmp_path):
        probs = _random_probs()
        stored = H5Array.create(probs.shape, dtype=np.float64, dir=tmp_path)
        stored.write_batch(0, probs.shape[0], probs)
        got = _llink_evidence(
            _exp({"c": stored}), _settings(), llinks=[5], nworker=2
        )
        want = _llink_evidence(_exp({"c": probs}), _settings(), llinks=[5])
        pd.testing.assert_frame_equal(got, want)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError, match="nworker must be a positive"):
            _llink_evidence(
                _exp({"c": _random_probs()}), _settings(), nworker=value
            )


class TestEstimate:
    def test_each_chromosome_gets_its_own_maximizer(self):
        exp = _exp(
            {
                "short": _planted(50, linker=10, seed=1),
                "long": _planted(50, linker=40, seed=2),
            }
        )
        best = _estimate_llink(exp, _settings(), llinks=[0, 10, 40, 60])
        table = _llink_evidence(exp, _settings(), llinks=[0, 10, 40, 60])
        assert best == {"short": 10, "long": 40}
        for chrom, sub in table.groupby("chrom"):
            top = sub.loc[sub["evidence"].idxmax(), "llink"]
            assert best[chrom] == top

    def test_no_warning_for_a_clear_estimate(self):
        exp = _exp({"c": _planted(50, linker=30)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _estimate_llink(
                exp, _settings(), llinks=[0, 30, 60]
            ) == {"c": 30}

    def test_warns_at_largest_candidate(self):
        exp = _exp({"c": _planted(50, linker=30)})
        with pytest.warns(UserWarning, match="largest candidate"):
            assert _estimate_llink(
                exp, _settings(), llinks=[0, 10]
            ) == {"c": 10}

    def test_warns_with_few_molecules(self):
        exp = _exp({"c": _planted(5, linker=30)})
        with pytest.warns(UserWarning, match="from 5 molecules"):
            _estimate_llink(exp, _settings(), llinks=[0, 30, 60])

    def test_few_molecules_warning_uses_count_after_sampling(self):
        exp = _exp({"c": _planted(60, linker=30)})
        with pytest.warns(UserWarning, match="from 10 molecules"):
            _estimate_llink(
                exp, _settings(), llinks=[0, 30, 60], max_nmol=10, seed=1
            )

    def test_warns_when_end_temp_is_not_one(self):
        exp = _exp({"c": _planted(50, linker=30)})
        with pytest.warns(UserWarning, match="end_temp is ignored"):
            _estimate_llink(
                exp, _settings(end_temp=0.5), llinks=[0, 30, 60]
            )
