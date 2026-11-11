# test_exact.py

import numpy as np
import pandas as pd
import pytest
from catella.h5_array import H5Array
from catella.simulation._exact import (
    _reject_time,
    _require_eseq,
    _require_tm,
    _solved_tm,
    _tm_configs,
    _tm_per_bp,
    _tm_per_mol,
)
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset
from catella_cpp import NucPosTM


def _make_tm_settings(**overrides):
    defaults = {
        "nucbp": 3,
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
        chroms="chr1",
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
        chroms=["chr1"],
        nmol={"chr1": 1},
        nsim=1,
        nbp={"chr1": 12},
        settings=_make_tm_settings(),
        out_dir=tmp_path / "mcmc",
    )


class TestRequireEseq:
    def test_passes_with_eseq(self, tmp_path):
        _require_eseq(_make_tm_dataset(tmp_path))

    def test_raises_without_eseq(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, store_eseq=False)
        with pytest.raises(ValueError, match="store_eseq"):
            _require_eseq(dataset)


class TestRequireTM:
    def test_passes_for_tm_with_eseq(self, tmp_path):
        _require_tm(_make_tm_dataset(tmp_path), "f")

    def test_mcmc_raises_with_caller_name(self, tmp_path):
        with pytest.raises(ValueError, match="my_method requires"):
            _require_tm(_make_mcmc_dataset(tmp_path), "my_method")

    def test_tm_without_eseq_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, store_eseq=False)
        with pytest.raises(ValueError, match="store_eseq"):
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
        ref.setEnergy(dataset.eseq["chr1"][1].tolist(), 5.0)
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
