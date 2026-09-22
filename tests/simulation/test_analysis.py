# test_analysis.py

import numpy as np
import pandas as pd
import pytest
from catella import utils
from catella.h5_array import H5Array
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset


def _make_settings(**overrides):
    defaults = {
        "nucbp": 147,
        "llink": 20,
        "elink": 1.0,
        "mu": -1.0,
        "start_temp": 1.0,
        "end_temp": 0.1,
        "cool_option": "linear",
        "nsweep": 5,
        "print_freq": 1,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_dataset(tmp_path, *, nmol=6, nbp=40, nsim=2, seed=1):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        chroms="chr1",
        nsim=nsim,
        settings=_make_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "dataset",
        seed=seed,
    )


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
        ana.compute_occup(dataset=dataset, name="occup_batched", batch_size=2)
        ana.compute_occup(dataset=dataset, name="occup_full", batch_size=1000)

        batched = dataset.analysis["chr1"]["occup_batched"].to_numpy()
        full = dataset.analysis["chr1"]["occup_full"].to_numpy()
        assert np.allclose(batched, full)

    def test_record_time_appends_suffix(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=20)
        ana = SimAnalysis()
        time = dataset.raw["chr1", 0, 0].time[-1]
        ana.compute_occup(dataset=dataset, record_time=True, batch_size=2)

        assert f"occup_t_{time}" in dataset.analysis["chr1"]

    def test_missing_molecule_filled_with_nan(self, tmp_path, capsys):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=16, nsim=2)
        for run in range(dataset.nsim):
            dataset.mcmc_file("chr1", 2, run).unlink()

        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        assert occup.shape == (5, 16)
        assert np.isnan(occup[2]).all()
        for mol in (0, 1, 3, 4):
            assert not np.isnan(occup[mol]).any()
        assert "chr1" in capsys.readouterr().out

    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=40)
        ana = SimAnalysis()
        ana.compute_occup(dataset, batch_size=2)
        assert "occup" in dataset.analysis["chr1"]


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
        ana.compute_access(
            dataset=dataset, access_name="access_batched", batch_size=2
        )
        ana.compute_access(
            dataset=dataset, access_name="access_full", batch_size=1000
        )

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

    def test_missing_molecule_stays_nan(self, tmp_path, capsys):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=15, nsim=2)
        for run in range(dataset.nsim):
            dataset.mcmc_file("chr1", 3, run).unlink()

        ana = SimAnalysis()
        ana.compute_access(dataset=dataset, batch_size=2)

        access = dataset.analysis["chr1"]["access"].to_numpy()
        assert np.isnan(access[3]).all()
        for mol in (0, 1, 2, 4):
            assert not np.isnan(access[mol]).any()
        assert "chr1" in capsys.readouterr().out

    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=15)
        ana = SimAnalysis()
        ana.compute_access(dataset, batch_size=2)
        assert "access" in dataset.analysis["chr1"]


class TestComputeMeanNnuc:
    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_mean_nnuc(dataset)
        assert "mean_nnuc" in dataset.analysis["chr1"]


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
        assert np.allclose(
            occup.to_numpy(), dataset.analysis["chr1"]["occup"].to_numpy()
        )
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

    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset, batch_size=2)
        link_mats = ana.sort_by_linkage(dataset, batch_size=2)
        assert "chr1" in link_mats

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

        ana.sort_by_linkage(
            dataset=dataset, sorted_name="my_sorted", batch_size=2
        )
        assert "my_sorted" in dataset.analysis["chr1"]

    def test_chroms_filter_only_processes_selected_chromosome(self, tmp_path):
        rng = np.random.default_rng(2)
        meth_prob = {"chr1": rng.random((4, 10)), "chr2": rng.random((4, 10))}
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms=["chr1", "chr2"],
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=2,
        )
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

        ana.sort_by_linkage(
            dataset=dataset, data_name="occup_df", batch_size=2
        )
        sorted_df = dataset.analysis["chr1"]["occup_df_sorted"]
        order, _ = utils.compute_linkage(occup)
        assert isinstance(sorted_df, pd.DataFrame)
        np.testing.assert_allclose(sorted_df.to_numpy(), occup[order])

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=8)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)

        with pytest.raises(ValueError):
            ana.sort_by_linkage(
                dataset=dataset, data_name="occup_nan", batch_size=2
            )

        link_mats = ana.sort_by_linkage(
            dataset=dataset,
            data_name="occup_nan",
            batch_size=2,
            fill_nan="mean",
        )
        assert np.isfinite(link_mats["chr1"]).all()
        sorted_arr = dataset.analysis["chr1"]["occup_nan_sorted"].to_numpy()
        orig_rows = sorted(map(tuple, np.nan_to_num(occup).tolist()))
        got_rows = sorted(map(tuple, np.nan_to_num(sorted_arr).tolist()))
        assert orig_rows == got_rows


def _make_tm_settings(**overrides):
    defaults = {
        "nucbp": 3,
        "llink": 2,
        "elink": 1.0,
        "mu": 0.5,
        "end_temp": 1.0,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_tm_dataset(tmp_path, *, nmol=3, nbp=20, seed=1, **kwargs):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        chroms="chr1",
        nsim=0,
        method="tm",
        settings=_make_tm_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "tm_dataset",
        seed=seed,
        **kwargs,
    )


def _brute_force_occup_meann(nucbp, llink, elink, mu, eseq, temp=1.0):
    """Minimal brute-force reference, matching the one in
    test_tm_bruteforce.py, duplicated here per this file's existing
    convention of self-contained helpers."""
    import math

    nbp = len(eseq)
    npos = nbp - nucbp + 1

    def erep(gap):
        i = gap - nucbp
        if i >= llink:
            return 0.0
        sr6 = (llink / (i + 1.0)) ** 6 - 1.0
        return elink * sr6 * sr6

    total_weight = 0.0
    start_weight = [0.0] * npos

    def recurse(idx, placed):
        nonlocal total_weight
        if idx == npos:
            w = 1.0
            for i, x in enumerate(placed):
                w *= math.exp(-(eseq[x] - mu) / temp)
                if i > 0:
                    w *= math.exp(-erep(x - placed[i - 1]) / temp)
            total_weight += w
            for x in placed:
                start_weight[x] += w
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
    mean_n = sum(start_prob)
    return np.array(occup), mean_n


class TestComputeOccupTM:
    def test_matches_brute_force(self, tmp_path):
        settings = _make_tm_settings()
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16, seed=3)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset)
        occup_arr = dataset.analysis["chr1"]["occup"].to_numpy()
        for mol in range(2):
            eseq = dataset.eseq["chr1"][mol]
            expected_occup, _ = _brute_force_occup_meann(
                settings.nucbp,
                settings.llink,
                settings.elink,
                settings.mu,
                eseq,
                temp=settings.end_temp,
            )
            np.testing.assert_allclose(
                occup_arr[mol], expected_occup, atol=1e-8
            )

    def test_time_raises_for_tm(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=1, nbp=16)
        ana = SimAnalysis()
        with pytest.raises(ValueError):
            ana.compute_occup(dataset=dataset, time=5)

    def test_requires_eseq(self, tmp_path):
        rng = np.random.default_rng(1)
        meth_prob = rng.random((2, 16))
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=0,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "no_eseq",
            seed=1,
            store_eseq=False,
        )
        ana = SimAnalysis()
        with pytest.raises(ValueError):
            ana.compute_occup(dataset=dataset)


class TestComputeMeanNnucTM:
    def test_matches_brute_force(self, tmp_path):
        settings = _make_tm_settings()
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16, seed=4)
        ana = SimAnalysis()
        ana.compute_mean_nnuc(dataset=dataset)
        df = dataset.analysis["chr1"]["mean_nnuc"]
        for mol in range(2):
            eseq = dataset.eseq["chr1"][mol]
            _, expected_mean = _brute_force_occup_meann(
                settings.nucbp,
                settings.llink,
                settings.elink,
                settings.mu,
                eseq,
                temp=settings.end_temp,
            )
            assert df.iloc[mol, 0] == pytest.approx(expected_mean, rel=1e-6)

    def test_time_raises_for_tm(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=1, nbp=16)
        ana = SimAnalysis()
        with pytest.raises(ValueError):
            ana.compute_mean_nnuc(dataset=dataset, record_time=True)


class TestComputeAccessTM:
    def test_matches_one_minus_occup(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16, seed=5)
        ana = SimAnalysis()
        ana.compute_access(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        access = dataset.analysis["chr1"]["access"].to_numpy()
        np.testing.assert_allclose(access, 1.0 - occup)


class TestComputeVarNnuc:
    def test_matches_brute_force_variance(self, tmp_path):
        import math

        settings = _make_tm_settings()
        dataset = _make_tm_dataset(tmp_path, nmol=1, nbp=14, seed=6)
        ana = SimAnalysis()
        ana.compute_var_nnuc(dataset=dataset)
        df = dataset.analysis["chr1"]["var_nnuc"]

        eseq = dataset.eseq["chr1"][0]
        nucbp, llink, elink, mu = (
            settings.nucbp,
            settings.llink,
            settings.elink,
            settings.mu,
        )
        nbp = len(eseq)
        npos = nbp - nucbp + 1

        def erep(gap):
            i = gap - nucbp
            if i >= llink:
                return 0.0
            sr6 = (llink / (i + 1.0)) ** 6 - 1.0
            return elink * sr6 * sr6

        total_weight = 0.0
        mean_n_num = 0.0
        mean_n2_num = 0.0

        def recurse(idx, placed):
            nonlocal total_weight, mean_n_num, mean_n2_num
            if idx == npos:
                w = 1.0
                for i, x in enumerate(placed):
                    w *= math.exp(-(eseq[x] - mu))
                    if i > 0:
                        w *= math.exp(-erep(x - placed[i - 1]))
                total_weight += w
                n = len(placed)
                mean_n_num += w * n
                mean_n2_num += w * n * n
                return
            recurse(idx + 1, placed)
            if not placed or idx - placed[-1] >= nucbp:
                recurse(idx + 1, [*placed, idx])

        recurse(0, [])
        mean_n = mean_n_num / total_weight
        expected_var = mean_n2_num / total_weight - mean_n * mean_n

        assert df.iloc[0, 0] == pytest.approx(expected_var, rel=1e-4)

    def test_requires_tm_method(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)
        ana = SimAnalysis()
        with pytest.raises(ValueError):
            ana.compute_var_nnuc(dataset=dataset)


class TestComputeVarOccup:
    def test_matches_bernoulli_closed_form(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16, seed=7)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset)
        ana.compute_var_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        var_occup = dataset.analysis["chr1"]["var_occup"].to_numpy()
        expected = occup * (1.0 - occup)
        np.testing.assert_allclose(var_occup, expected, atol=1e-10)

    def test_requires_tm_method(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)
        ana = SimAnalysis()
        with pytest.raises(ValueError):
            ana.compute_var_occup(dataset=dataset)
