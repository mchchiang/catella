# test_analysis.py

import numpy as np
import pandas as pd
import pytest

from nucmc.simulation.config import SimSettings
from nucmc.simulation.engine import SimManager
from nucmc.simulation.analysis import SimAnalysis
from nucmc.simulation.results import SimDataset
from nucmc.h5_array import H5Array
from nucmc import utils


def _make_settings(**overrides):
    defaults = dict(nucbp=147, llink=20, mu=-1.0, start_temp=1.0,
                     end_temp=0.1, cool_option="linear", nsweep=5,
                     print_freq=1, emax=5.0)
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_dataset(tmp_path, *, nmol=6, nbp=40, nsim=2, seed=1):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(chroms="chr1", nsim=nsim, settings=_make_settings(),
                       meth_prob=meth_prob, out_dir=tmp_path / "dataset",
                       seed=seed)


class TestComputeOccup:
    def test_output_is_h5array(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=40)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        arr = dataset.analysis["chr1"]["occup"]
        assert isinstance(arr, H5Array)
        assert arr.shape == (5, 40)
        assert ((arr.to_numpy() >= 0) & (arr.to_numpy() <= 1)).all()

    def test_batching_matches_single_batch(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=7, nbp=30)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, name="occup_batched",
                          batch_size=2)
        ana.compute_occup(dataset=dataset, name="occup_full",
                          batch_size=1000)

        batched = dataset.analysis["chr1"]["occup_batched"].to_numpy()
        full = dataset.analysis["chr1"]["occup_full"].to_numpy()
        assert np.allclose(batched, full)

    def test_record_time_appends_suffix(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=20)
        ana = SimAnalysis()
        time = dataset.raw["chr1", 0, 0].time[-1]
        ana.compute_occup(dataset=dataset, record_time=True, batch_size=2)

        assert f"occup_t_{time}" in dataset.analysis["chr1"]


class TestComputeAccess:
    def test_output_is_h5array_and_complement_of_occup(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=25)
        ana = SimAnalysis()
        ana.compute_access(dataset=dataset, batch_size=2)

        occup = dataset.analysis["chr1"]["occup"]
        access = dataset.analysis["chr1"]["access"]
        assert isinstance(occup, H5Array)
        assert isinstance(access, H5Array)
        assert np.allclose(access.to_numpy(), 1.0 - occup.to_numpy())

    def test_batching_matches_single_batch(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=18)
        ana = SimAnalysis()
        ana.compute_access(dataset=dataset, access_name="access_batched",
                           batch_size=2)
        ana.compute_access(dataset=dataset, access_name="access_full",
                           batch_size=1000)

        batched = dataset.analysis["chr1"]["access_batched"].to_numpy()
        full = dataset.analysis["chr1"]["access_full"].to_numpy()
        assert np.allclose(batched, full)

    def test_reuses_existing_occup(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=15)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup_before = dataset.analysis["chr1"]["occup"].to_numpy().copy()

        ana.compute_access(dataset=dataset, batch_size=2)

        occup_after = dataset.analysis["chr1"]["occup"].to_numpy()
        assert np.allclose(occup_before, occup_after)


class TestSaveLoadRoundTrip:
    def test_h5array_analysis_round_trips(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        ana.compute_access(dataset=dataset, batch_size=2)
        ana.compute_mean_nnuc(dataset=dataset)

        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)

        occup = loaded.analysis["chr1"]["occup"]
        access = loaded.analysis["chr1"]["access"]
        mean_nnuc = loaded.analysis["chr1"]["mean_nnuc"]
        assert isinstance(occup, H5Array)
        assert isinstance(access, H5Array)
        assert isinstance(mean_nnuc, pd.DataFrame)
        assert np.allclose(occup.to_numpy(),
                           dataset.analysis["chr1"]["occup"].to_numpy())
        assert np.allclose(access.to_numpy(), 1.0 - occup.to_numpy())

    def test_save_rejects_same_path_as_backing_h5array(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)

        with pytest.raises(ValueError):
            loaded.save(loaded._dataset_file)


class TestSortByLinkage:
    def test_output_is_h5array_and_matches_oracle(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()

        link_mats = ana.sort_by_linkage(dataset=dataset, batch_size=2)
        sorted_arr = dataset.analysis["chr1"]["occup_sorted"]
        assert isinstance(sorted_arr, H5Array)
        assert sorted_arr.shape == occup.shape

        order, ref_link = utils.compute_linkage(occup)
        np.testing.assert_allclose(sorted_arr.to_numpy(), occup[order])
        np.testing.assert_allclose(link_mats["chr1"], ref_link)

    def test_sorted_rows_are_a_permutation_of_original_rows(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=12)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()

        ana.sort_by_linkage(dataset=dataset, batch_size=2)
        sorted_arr = dataset.analysis["chr1"]["occup_sorted"].to_numpy()

        orig_rows = sorted(map(tuple, occup.tolist()))
        got_rows = sorted(map(tuple, sorted_arr.tolist()))
        assert orig_rows == got_rows

    def test_default_and_custom_sorted_name(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        ana.sort_by_linkage(dataset=dataset, batch_size=2)
        assert "occup_sorted" in dataset.analysis["chr1"]

        ana.sort_by_linkage(dataset=dataset, sorted_name="my_sorted",
                            batch_size=2)
        assert "my_sorted" in dataset.analysis["chr1"]

    def test_chroms_filter_only_processes_selected_chromosome(self,
                                                               tmp_path):
        rng = np.random.default_rng(2)
        meth_prob = {"chr1": rng.random((4, 10)), "chr2": rng.random((4, 10))}
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms=["chr1", "chr2"], nsim=2,
                              settings=_make_settings(), meth_prob=meth_prob,
                              out_dir=tmp_path / "dataset", seed=2)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        ana.sort_by_linkage(dataset=dataset, chroms="chr1", batch_size=2)
        assert "occup_sorted" in dataset.analysis["chr1"]
        assert "occup_sorted" not in dataset.analysis["chr2"]

    def test_missing_data_name_raises_key_error(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        ana = SimAnalysis()
        with pytest.raises(KeyError):
            ana.sort_by_linkage(dataset=dataset)

    def test_works_with_legacy_dataframe_source(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=8)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        dataset.analysis["chr1"]["occup_df"] = pd.DataFrame(occup)

        ana.sort_by_linkage(dataset=dataset, data_name="occup_df",
                            batch_size=2)
        sorted_df = dataset.analysis["chr1"]["occup_df_sorted"]
        order, _ = utils.compute_linkage(occup)
        assert isinstance(sorted_df, pd.DataFrame)
        np.testing.assert_allclose(sorted_df.to_numpy(), occup[order])
