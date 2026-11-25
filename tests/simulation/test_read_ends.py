# test_read_ends.py

import h5py
import numpy as np
import pytest
from catella.simulation._inputs import _extent, _set_energy
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset
from catella_cpp import NucPosTM


def _tm_settings():
    return SimSettings(
        lnuc=3, llink=2, elink=1.0, mu=0.5, end_temp=1.0, emax=2.0
    )


def _mcmc_settings():
    return SimSettings(
        lnuc=3,
        llink=2,
        elink=1.0,
        mu=0.5,
        start_temp=1.0,
        end_temp=1.0,
        cool_option="linear",
        nsweep=20,
        print_freq=5,
        emax=2.0,
    )


def _meth_prob():
    meth_prob = np.random.default_rng(2).uniform(0.02, 0.98, (3, 20))
    meth_prob[0, :6] = np.nan
    meth_prob[1, 8:12] = np.nan
    meth_prob[2, 14:] = np.nan
    return meth_prob


def _run(tmp_path, method="tm", nsim=0, occupy_read_ends=False, name="ds"):
    settings = _tm_settings() if method == "tm" else _mcmc_settings()
    return SimManager(nworker=1, verbose=False).run(
        regions="chr1",
        nsim=nsim,
        method=method,
        settings=settings,
        meth_prob=_meth_prob(),
        out_dir=tmp_path / name,
        seed=3,
        fill_nan=0.5,
        occupy_read_ends=occupy_read_ends,
    )


_EXTENTS = [(8, 20), (0, 20), (0, 14)]


class TestExtent:
    def test_first_to_last_spanned_bp(self):
        spanned = (np.array([[2, 5], [8, 12], [0, 3]]), np.array([0, 2, 2, 3]))
        assert _extent(spanned, 0) == (2, 12)
        assert _extent(spanned, 1) is None
        assert _extent(spanned, 2) == (0, 3)

    def test_none_without_spanned(self):
        assert _extent(None, 0) is None


class TestSetEnergy:
    def _model(self, nbp=10):
        return NucPosTM(3, nbp, 2, 1.0, 0.5, 1)

    def test_without_extent_matches_set_energy(self):
        edata = np.linspace(-5, 5, 10)
        got, ref = self._model(), self._model()
        _set_energy(got, edata, 2.0, 3)
        ref.setEnergy(edata.tolist(), 2.0)
        assert got.getDataEnergy() == ref.getDataEnergy()

    def test_extent_forbids_starts_outside(self):
        edata = np.linspace(-5, 5, 10)
        model = self._model()
        _set_energy(model, edata, 2.0, 3, extent=(2, 8))
        energy = np.asarray(model.getDataEnergy())
        assert np.isposinf(energy[[0, 1, 6, 7, 8, 9]]).all()
        np.testing.assert_allclose(energy[2:6], np.clip(edata[2:6], -2, 2))


class TestRunOccupyReadEnds:
    def test_flag_is_stored_and_reloaded(self, tmp_path):
        dataset = _run(tmp_path)
        assert dataset.occupy_read_ends is False
        assert SimDataset.load(dataset._dataset_file).occupy_read_ends is False

    def test_default_is_true(self, tmp_path):
        dataset = _run(tmp_path, occupy_read_ends=True)
        assert dataset.occupy_read_ends is True

    def test_old_dataset_defaults_to_true(self, tmp_path):
        dataset = _run(tmp_path)
        with h5py.File(dataset._dataset_file, "a") as f:
            del f["metadata"].attrs["occupy_read_ends"]
        assert SimDataset.load(dataset._dataset_file).occupy_read_ends is True

    def test_tm_occupancy_matches_cropped_molecule(self, tmp_path):
        dataset = _run(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        settings = _tm_settings()
        for mol, (lo, hi) in enumerate(_EXTENTS):
            tm = NucPosTM(3, hi - lo, 2, 1.0, 0.5, 1)
            tm.setEnergy(dataset.edata["chr1"][mol, lo:hi].tolist(), 2.0)
            tm.solve(settings.end_temp)
            cropped = np.asarray(tm.getOccup())
            got = occup[mol, lo:hi]
            keep = ~np.isnan(got)
            np.testing.assert_allclose(got[keep], cropped[keep])

    def test_true_matches_default_run(self, tmp_path):
        on = _run(tmp_path, occupy_read_ends=True, name="on")
        SimAnalysis().compute_occup(dataset=on)
        ref = SimManager(nworker=1, verbose=False).run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=_tm_settings(),
            meth_prob=_meth_prob(),
            out_dir=tmp_path / "ref",
            seed=3,
            fill_nan=0.5,
        )
        SimAnalysis().compute_occup(dataset=ref)
        np.testing.assert_array_equal(
            on.analysis["chr1"]["occup"].to_numpy(),
            ref.analysis["chr1"]["occup"].to_numpy(),
        )

    def test_tm_samples_stay_in_extent(self, tmp_path):
        dataset = _run(tmp_path, nsim=20)
        for mol, (lo, hi) in enumerate(_EXTENTS):
            for run in range(20):
                pos = np.asarray(dataset.raw["chr1", mol, run].position)
                assert ((pos >= lo) & (pos + 3 <= hi)).all()

    @pytest.mark.parametrize("name", ["most_likely", "best_match"])
    def test_tm_configs_stay_in_extent(self, tmp_path, name):
        dataset = _run(tmp_path)
        ana = SimAnalysis()
        if name == "most_likely":
            ana.compute_most_likely_config(dataset=dataset, name=name)
        else:
            ana.compute_best_match_config(dataset=dataset, name=name)
        configs = ana.extract_config(dataset, region="chr1", name=name)
        for (lo, hi), pos in zip(_EXTENTS, configs, strict=True):
            assert ((pos >= lo) & (pos + 3 <= hi)).all()

    def test_mcmc_positions_stay_in_extent(self, tmp_path):
        dataset = _run(tmp_path, method="mcmc", nsim=3)
        for mol, (lo, hi) in enumerate(_EXTENTS):
            for run in range(3):
                for pos in dataset.raw["chr1", mol, run].position:
                    pos = np.asarray(pos)
                    assert ((pos >= lo) & (pos + 3 <= hi)).all()

    def test_mcmc_rerun_keeps_extent(self, tmp_path):
        dataset = _run(tmp_path, method="mcmc", nsim=2)
        dataset.mcmc_file("chr1", 0, 1).unlink()
        SimManager(nworker=1, verbose=False).rerun(
            dataset=dataset, meth_prob=_meth_prob()
        )
        for pos in dataset.raw["chr1", 0, 1].position:
            pos = np.asarray(pos)
            assert ((pos >= 8) & (pos + 3 <= 20)).all()
