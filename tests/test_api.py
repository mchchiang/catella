# test_api.py

import numpy as np
import pandas as pd
import pytest

import nucmc
from nucmc import utils
from nucmc.experiment.methdata import MethPrintData, MethPrintExperiment
from nucmc.simulation.analysis import SimAnalysis
from nucmc.simulation.config import SimSettings
from nucmc.simulation.engine import SimManager


def _settings():
    return SimSettings(nucbp=5, llink=2, mu=-1.0, start_temp=1.0,
                       end_temp=0.1, cool_option="linear", nsweep=2,
                       print_freq=1, emax=5.0)


def _make_dataset(tmp_path, nmol=6, nbp=10, seed=1):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(chroms="chr1", nsim=2, settings=_settings(),
                       meth_prob=meth_prob, out_dir=tmp_path / "sim",
                       seed=seed)


def _make_experiment(nmol=6, nbp=10, seed=3):
    rng = np.random.default_rng(seed)
    rows = [(m, p, "+", float(rng.random()), 0) for m in range(nmol)
           for p in rng.choice(nbp, size=min(6, nbp), replace=False)]
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                     "mod_qual", "mod_code"])
    mol_id = np.array([f"m{m}" for m in range(nmol)], dtype=object)
    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, test_mol_id=mol_id, test_data=df,
        meth_mol_id=None, meth_data=None, unmeth_mol_id=None,
        unmeth_data=None)
    return MethPrintExperiment._create(_raw_data={"chr1": raw})


class TestDownsample:
    def test_matches_utils_downsample(self):
        data = np.random.default_rng(0).random((10, 4))
        expected = utils.downsample(data, 3, how="mean")
        got = nucmc.downsample(data=data, max_rows=3, how="mean")
        np.testing.assert_allclose(got, expected)


class TestSortByLinkage:
    def test_raises_if_neither_given(self):
        with pytest.raises(ValueError):
            nucmc.sort_by_linkage()

    def test_raises_if_both_given(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        exp = _make_experiment()
        with pytest.raises(ValueError):
            nucmc.sort_by_linkage(dataset=dataset, exp=exp)

    def test_dataset_dispatch_sorts_and_stores_link_mat(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        order, ref_link = utils.compute_linkage(occup)

        nucmc.sort_by_linkage(dataset=dataset)

        sorted_arr = dataset.analysis["chr1"]["occup_sorted"].to_numpy()
        link_mat = dataset.analysis["chr1"]["occup_linkage"].to_numpy()
        np.testing.assert_allclose(sorted_arr, occup[order])
        np.testing.assert_allclose(link_mat, ref_link)

    def test_experiment_dispatch_with_raw_which(self):
        exp = _make_experiment()

        nucmc.sort_by_linkage(exp=exp, raw_which="test")

        assert "test_sorted" in exp.analysis["chr1"]
        assert "test_linkage" in exp.analysis["chr1"]

    def test_store_link_mat_false_skips_persistence(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)

        nucmc.sort_by_linkage(dataset=dataset, store_link_mat=False)

        assert "occup_sorted" in dataset.analysis["chr1"]
        assert "occup_linkage" not in dataset.analysis["chr1"]

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)

        with pytest.raises(ValueError):
            nucmc.sort_by_linkage(dataset=dataset, data_name="occup_nan")

        nucmc.sort_by_linkage(dataset=dataset, data_name="occup_nan",
                              fill_nan="mean")
        link_mat = dataset.analysis["chr1"]["occup_nan_linkage"].to_numpy()
        assert np.isfinite(link_mat).all()
