# test_analysis.py

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from catella import utils
from catella.h5_array import H5Array
from catella.simulation.analysis import NucFiberMap, SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset
from catella_cpp import NucPosTM


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
        regions="chr1",
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

    def test_nonpositive_batch_size_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=20)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SimAnalysis().compute_occup(dataset=dataset, batch_size=-5)

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

    def test_default_time_with_first_molecule_missing(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=16, nsim=2)
        for run in range(dataset.nsim):
            dataset.mcmc_file("chr1", 0, run).unlink()

        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        assert np.isnan(occup[0]).all()
        assert not np.isnan(occup[1:]).any()

    def test_default_time_with_first_run_missing(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=16, nsim=2)
        dataset.mcmc_file("chr1", 0, 0).unlink()

        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        assert not np.isnan(occup).any()

    def test_no_simulation_files_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=16, nsim=1)
        for mol in range(2):
            dataset.mcmc_file("chr1", mol, 0).unlink()

        with pytest.raises(ValueError):
            SimAnalysis().compute_occup(dataset=dataset)


class TestComputeMeanNnuc:
    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_mean_nnuc(dataset)
        assert "mean_nnuc" in dataset.analysis["chr1"]

    def test_default_time_with_first_molecule_missing(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=20, nsim=2)
        for run in range(dataset.nsim):
            dataset.mcmc_file("chr1", 0, run).unlink()

        SimAnalysis().compute_mean_nnuc(dataset)

        df = dataset.analysis["chr1"]["mean_nnuc"]
        assert len(df) == 3
        assert df.iloc[1:, 0].notna().all()

    def test_no_simulation_files_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20, nsim=1)
        for mol in range(2):
            dataset.mcmc_file("chr1", mol, 0).unlink()

        with pytest.raises(ValueError):
            SimAnalysis().compute_mean_nnuc(dataset)

    def test_only_requested_regions_extracted(self, tmp_path):
        rng = np.random.default_rng(0)
        meth_prob = {r: rng.random((2, 20)) for r in ("chr1", "chr2")}
        dataset = SimManager(nworker=1, verbose=False).run(
            regions=["chr1", "chr2"],
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=1,
        )
        ana = SimAnalysis()
        with patch.object(
            SimDataset,
            "extract",
            autospec=True,
            side_effect=SimDataset.extract,
        ) as spy:
            ana.compute_mean_nnuc(dataset, regions="chr1")

        for call in spy.call_args_list:
            assert list(call.kwargs["regions"]) == ["chr1"]
        assert "mean_nnuc" in dataset.analysis["chr1"]
        assert "mean_nnuc" not in dataset.analysis["chr2"]


class TestSaveLoadRoundTrip:
    def test_h5array_analysis_round_trips(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        ana.compute_mean_nnuc(dataset=dataset)

        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)

        occup = loaded.analysis["chr1"]["occup"]
        mean_nnuc = loaded.analysis["chr1"]["mean_nnuc"]
        assert isinstance(occup, H5Array)
        assert isinstance(mean_nnuc, pd.DataFrame)
        assert np.allclose(
            occup.to_numpy(), dataset.analysis["chr1"]["occup"].to_numpy()
        )

    def test_resave_to_backing_file_keeps_h5array(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)
        want = loaded.analysis["chr1"]["occup"].to_numpy()

        loaded.save(loaded._dataset_file)

        reloaded = SimDataset.load(dataset._dataset_file)
        np.testing.assert_array_equal(
            reloaded.analysis["chr1"]["occup"].to_numpy(), want
        )
        np.testing.assert_array_equal(
            loaded.analysis["chr1"]["occup"].to_numpy(), want
        )


class TestSortByLinkage:
    def test_linkage_matches_oracle(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()

        link_mats = ana.sort_by_linkage(dataset=dataset, batch_size=2)

        _, ref_link = utils.compute_linkage(occup)
        np.testing.assert_allclose(link_mats["chr1"], ref_link)
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["occup_linkage"].to_numpy(), ref_link
        )

    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset, batch_size=2)
        link_mats = ana.sort_by_linkage(dataset, batch_size=2)
        assert "chr1" in link_mats

    def test_stores_only_the_linkage(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=12)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        before = set(dataset.analysis["chr1"])

        ana.sort_by_linkage(dataset=dataset, batch_size=2)
        assert set(dataset.analysis["chr1"]) - before == {"occup_linkage"}

    def test_custom_link_mat_name(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        ana.sort_by_linkage(
            dataset=dataset, link_mat_name="my_link", batch_size=2
        )
        assert "my_link" in dataset.analysis["chr1"]

    def test_regions_filter_only_processes_selected_region(self, tmp_path):
        rng = np.random.default_rng(2)
        meth_prob = {"chr1": rng.random((4, 10)), "chr2": rng.random((4, 10))}
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            regions=["chr1", "chr2"],
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=2,
        )
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        ana.sort_by_linkage(dataset=dataset, regions="chr1", batch_size=2)
        assert "occup_linkage" in dataset.analysis["chr1"]
        assert "occup_linkage" not in dataset.analysis["chr2"]

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

        link_mats = ana.sort_by_linkage(
            dataset=dataset, data_name="occup_df", batch_size=2
        )
        _, want = utils.compute_linkage(occup)
        np.testing.assert_allclose(link_mats["chr1"], want)

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=5, nbp=8)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)

        with pytest.raises(ValueError, match="1 of 5 rows have no data"):
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
        assert len(link_mats["chr1"]) == 4


def _calls_dataset(tmp_path, nmol=8, nbp=30):
    meth_prob = np.random.default_rng(6).random((nmol, nbp))
    meth_prob[2] = np.nan
    dataset = SimManager(nworker=1, verbose=False).run(
        regions="chr1",
        nsim=0,
        method="tm",
        settings=_make_tm_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "calls",
        seed=6,
    )
    SimAnalysis().compute_best_match_config(dataset=dataset, name="bm")
    return dataset


def _footprint_matrix(dataset, name):
    nbp = dataset.nbp["chr1"]
    nuc_map = NucFiberMap(nbp=nbp, nucbp=dataset.settings["nucbp"])
    configs = SimAnalysis().extract_config(dataset, region="chr1", name=name)
    return np.array(
        [
            np.full(nbp, np.nan) if c is None else nuc_map.project(c)
            for c in configs
        ],
        dtype=np.float64,
    )


class TestSortByConfig:
    def test_matches_linkage_of_footprints(self, tmp_path):
        dataset = _calls_dataset(tmp_path)
        link_mats = SimAnalysis().sort_by_linkage(
            dataset, config_name="bm", fill_nan="mean"
        )
        _, want = utils.compute_linkage(
            _footprint_matrix(dataset, "bm"), fill_nan="mean"
        )
        np.testing.assert_allclose(link_mats["chr1"], want)
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["bm_linkage"].to_numpy(), want
        )

    def test_small_batches_give_same_linkage(self, tmp_path):
        dataset = _calls_dataset(tmp_path)
        ana = SimAnalysis()
        full = ana.sort_by_linkage(
            dataset, config_name="bm", fill_nan="mean", store_link_mat=False
        )
        batched = ana.sort_by_linkage(
            dataset,
            config_name="bm",
            fill_nan="mean",
            batch_size=3,
            store_link_mat=False,
        )
        np.testing.assert_allclose(batched["chr1"], full["chr1"])

    def test_no_data_molecule_needs_fill_nan(self, tmp_path):
        dataset = _calls_dataset(tmp_path)
        with pytest.raises(ValueError, match="1 of 8 rows have no data"):
            SimAnalysis().sort_by_linkage(dataset, config_name="bm")

    def test_data_name_and_config_name_raises(self, tmp_path):
        dataset = _calls_dataset(tmp_path)
        with pytest.raises(ValueError, match="cannot both be given"):
            SimAnalysis().sort_by_linkage(
                dataset, data_name="occup", config_name="bm"
            )

    def test_missing_config_name_raises(self, tmp_path):
        dataset = _calls_dataset(tmp_path)
        with pytest.raises(KeyError, match="nope"):
            SimAnalysis().sort_by_linkage(dataset, config_name="nope")

    def test_temporary_file_is_removed(self, tmp_path):
        dataset = _calls_dataset(tmp_path)
        tmp_dir = Path(dataset.resolve_tmp_dir())
        before = set(tmp_dir.glob("*.h5"))
        SimAnalysis().sort_by_linkage(
            dataset, config_name="bm", fill_nan="mean"
        )
        assert set(tmp_dir.glob("*.h5")) == before

    def test_plot_config_rows_follow_the_linkage(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scipy.cluster.hierarchy as sch
        from catella.simulation.plot import SimPlot

        dataset = _calls_dataset(tmp_path)
        link_mats = SimAnalysis().sort_by_linkage(
            dataset, config_name="bm", fill_nan="mean"
        )
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            link_mat=link_mats["chr1"],
            show=False,
        )
        footprints = np.nan_to_num(_footprint_matrix(dataset, "bm"), nan=-1)
        order = sch.leaves_list(link_mats["chr1"])
        image = np.ma.getdata(plt.gcf().axes[0].images[0].get_array())
        np.testing.assert_array_equal(image, footprints[order])
        plt.close("all")


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


def _make_tm_dataset(tmp_path, *, nmol=3, nbp=20, seed=1, nsim=0, **kwargs):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        regions="chr1",
        nsim=nsim,
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


def _make_mcmc_dataset(tmp_path, nmol=3, nsim=2):
    rng = np.random.default_rng(2)
    return SimManager(nworker=1, verbose=False).run(
        regions="chr1",
        nsim=nsim,
        settings=_make_settings(nsweep=4),
        meth_prob=rng.random((nmol, 400)),
        out_dir=tmp_path / "mcmc",
        seed=2,
    )


class TestExtractConfigRaw:
    def test_mcmc_defaults_to_run_zero_final_frame(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        got = SimAnalysis().extract_config(dataset, region="chr1")
        for mol in range(3):
            np.testing.assert_array_equal(
                got[mol], dataset.raw["chr1", mol, 0].position[-1]
            )

    def test_mcmc_run_and_time(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        data = dataset.raw["chr1", 2, 1]
        time = int(data.time[1])
        got = SimAnalysis().extract_config(
            dataset, region="chr1", run=1, time=time, mols=[2]
        )
        np.testing.assert_array_equal(got[0], data.position[1])

    def test_mols_order(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        ana = SimAnalysis()
        full = ana.extract_config(dataset, region="chr1")
        got = ana.extract_config(dataset, region="chr1", mols=[2, 0])
        np.testing.assert_array_equal(got[0], full[2])
        np.testing.assert_array_equal(got[1], full[0])

    def test_missing_file_is_none(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        dataset.mcmc_file("chr1", 1, 0).unlink()
        got = SimAnalysis().extract_config(dataset, region="chr1")
        assert got[1] is None
        assert got[0] is not None

    def test_threads_match_serial(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        ana = SimAnalysis()
        serial = ana.extract_config(dataset, region="chr1")
        threaded = ana.extract_config(dataset, region="chr1", nworker=2)
        for a, b in zip(serial, threaded, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_unrecorded_time_raises(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        with pytest.raises(ValueError, match="time 12345 is not recorded"):
            SimAnalysis().extract_config(dataset, region="chr1", time=12345)

    def test_run_out_of_range_raises(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        with pytest.raises(IndexError, match="run must be in"):
            SimAnalysis().extract_config(dataset, region="chr1", run=2)

    def test_tm_samples(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20, nsim=3)
        got = SimAnalysis().extract_config(dataset, region="chr1", run=2)
        for mol in range(2):
            np.testing.assert_array_equal(
                got[mol], dataset.raw["chr1", mol, 2].position
            )

    def test_tm_without_samples_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20)
        with pytest.raises(ValueError, match="no samples stored"):
            SimAnalysis().extract_config(dataset, region="chr1")

    def test_tm_time_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20, nsim=1)
        with pytest.raises(ValueError, match="time is not meaningful"):
            SimAnalysis().extract_config(dataset, region="chr1", time=0)


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
            regions="chr1",
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


class TestComputeOccupBundle:
    def test_tm_stores_mean_nnuc_matching_compute_mean_nnuc(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16, seed=4)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset)
        ana.compute_mean_nnuc(dataset=dataset, name="direct")
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["mean_nnuc"].to_numpy(),
            dataset.analysis["chr1"]["direct"].to_numpy(),
        )

    def test_mean_nnuc_name_none_skips(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16)
        SimAnalysis().compute_occup(dataset=dataset, mean_nnuc_name=None)
        assert set(dataset.analysis["chr1"]) == {"occup"}

    def test_tm_compute_var_matches_var_methods(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=16, seed=5)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, compute_var=True)
        ana.compute_var_nnuc(dataset=dataset, name="vn")
        ana.compute_var_occup(dataset=dataset, name="vo")
        ana_chr = dataset.analysis["chr1"]
        np.testing.assert_allclose(
            ana_chr["var_nnuc"].to_numpy(), ana_chr["vn"].to_numpy()
        )
        np.testing.assert_allclose(
            np.asarray(ana_chr["var_occup"]), np.asarray(ana_chr["vo"])
        )

    def test_compute_var_on_mcmc_raises_before_work(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError, match="compute_var=True requires"):
            SimAnalysis().compute_occup(dataset=dataset, compute_var=True)
        assert "occup" not in dataset.analysis["chr1"]

    def test_mcmc_stores_mean_nnuc(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset)
        ana.compute_mean_nnuc(dataset=dataset, name="direct")
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["mean_nnuc"].to_numpy(),
            dataset.analysis["chr1"]["direct"].to_numpy(),
        )

    def test_mcmc_record_time_suffixes_both(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        time = dataset.raw["chr1", 0, 0].time[-1]
        SimAnalysis().compute_occup(
            dataset=dataset, time=time, record_time=True
        )
        assert {f"occup_t_{time}", f"mean_nnuc_t_{time}"} <= set(
            dataset.analysis["chr1"]
        )


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


def _direct_tm(dataset, mol, settings):
    tm = NucPosTM(
        settings.nucbp,
        dataset.nbp["chr1"],
        settings.llink,
        settings.elink,
        settings.mu,
        1,
    )
    tm.setEnergy(dataset.eseq["chr1"][mol].tolist(), settings.emax)
    tm.solve(settings.end_temp)
    return tm


class TestComputeConfigs:
    @pytest.mark.parametrize(
        "method,name,call",
        [
            (
                "compute_most_likely_config",
                "most_likely",
                lambda tm: tm.getMostLikelyConfig(),
            ),
            (
                "compute_best_match_config",
                "best_match",
                lambda tm: tm.getBestMatchConfig(1.0),
            ),
        ],
    )
    def test_matches_direct_solver(self, tmp_path, method, name, call):
        settings = _make_tm_settings()
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=20, seed=5)
        ana = SimAnalysis()
        getattr(ana, method)(dataset=dataset)
        energy_df = dataset.analysis["chr1"][f"{name}_energy"]
        configs = ana.extract_config(dataset, region="chr1", name=name)
        for mol in range(3):
            tm = _direct_tm(dataset, mol, settings)
            positions, energy = call(tm)
            np.testing.assert_array_equal(configs[mol], positions)
            assert energy_df["energy"].iloc[mol] == pytest.approx(energy)
            assert energy_df["log_prob"].iloc[mol] == pytest.approx(
                -energy / settings.end_temp - tm.getLogZ()
            )

    def test_flat_offset_layout(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=20, seed=5)
        ana = SimAnalysis()
        ana.compute_most_likely_config(dataset=dataset)
        flat = dataset.analysis["chr1"]["most_likely_position_flat"]
        offset = dataset.analysis["chr1"]["most_likely_position_offset"]
        assert list(flat.columns) == ["position"]
        assert list(offset.columns) == ["offset"]
        assert len(offset) == 4
        assert offset["offset"].iloc[0] == 0
        assert offset["offset"].iloc[-1] == len(flat)

    def test_custom_name_and_gamma(self, tmp_path):
        settings = _make_tm_settings()
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20, seed=5)
        ana = SimAnalysis()
        ana.compute_best_match_config(dataset=dataset, name="calls", gamma=3.0)
        tm = _direct_tm(dataset, 1, settings)
        np.testing.assert_array_equal(
            ana.extract_config(
                dataset, region="chr1", name="calls", mols=[1]
            )[0],
            tm.getBestMatchConfig(3.0)[0],
        )

    def test_round_trips_through_save(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=20, seed=5)
        ana = SimAnalysis()
        ana.compute_best_match_config(dataset=dataset)
        expected = ana.extract_config(
            dataset, region="chr1", name="best_match"
        )
        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)
        got = ana.extract_config(loaded, region="chr1", name="best_match")
        for mol in range(3):
            np.testing.assert_array_equal(got[mol], expected[mol])

    def test_extract_config_out_of_range_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20, seed=5)
        ana = SimAnalysis()
        ana.compute_most_likely_config(dataset=dataset)
        with pytest.raises(IndexError):
            ana.extract_config(
                dataset, region="chr1", name="most_likely", mols=[2]
            )

    def test_extract_config_missing_name_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20, seed=5)
        ana = SimAnalysis()
        with pytest.raises(KeyError, match="most_likely"):
            ana.extract_config(dataset, region="chr1", name="most_likely")

    def test_extract_config_run_with_name_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=2, nbp=20, seed=5)
        ana = SimAnalysis()
        ana.compute_most_likely_config(dataset=dataset)
        with pytest.raises(ValueError, match="cannot be given with name"):
            ana.extract_config(
                dataset, region="chr1", name="most_likely", run=0
            )

    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_nonpositive_gamma_raises(self, tmp_path, gamma):
        dataset = _make_tm_dataset(tmp_path, nmol=1, nbp=20, seed=5)
        ana = SimAnalysis()
        with pytest.raises(ValueError, match="gamma must be positive"):
            ana.compute_best_match_config(dataset=dataset, gamma=gamma)

    @pytest.mark.parametrize(
        "method", ["compute_most_likely_config", "compute_best_match_config"]
    )
    def test_requires_tm_method(self, tmp_path, method):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)
        ana = SimAnalysis()
        with pytest.raises(ValueError, match="requires a method='tm'"):
            getattr(ana, method)(dataset=dataset)

    @pytest.mark.parametrize(
        "method", ["compute_most_likely_config", "compute_best_match_config"]
    )
    def test_requires_eseq(self, tmp_path, method):
        rng = np.random.default_rng(1)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=rng.random((2, 16)),
            out_dir=tmp_path / "no_eseq",
            seed=1,
            store_eseq=False,
        )
        ana = SimAnalysis()
        with pytest.raises(ValueError):
            getattr(ana, method)(dataset=dataset)


def _make_multi_region_tm_dataset(tmp_path):
    rng = np.random.default_rng(7)
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        regions=["chr1", "chr2"],
        nsim=0,
        method="tm",
        settings=_make_tm_settings(),
        meth_prob={
            "chr1": rng.random((5, 20)),
            "chr2": rng.random((3, 16)),
        },
        out_dir=tmp_path / "tm_multi",
        seed=1,
    )


def _as_array(value):
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    return value[:]


TM_STAT_METHODS = [
    "compute_occup",
    "compute_mean_nnuc",
    "compute_var_nnuc",
    "compute_var_occup",
]

TM_CONFIG_METHODS = [
    "compute_most_likely_config",
    "compute_best_match_config",
]


class TestParallelTM:
    @pytest.mark.parametrize("method", TM_STAT_METHODS)
    def test_stat_matches_serial(self, tmp_path, method):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        ana = SimAnalysis()
        getattr(ana, method)(dataset=dataset, name="serial", nworker=1)
        getattr(ana, method)(dataset=dataset, name="par", nworker=2)
        for region in ["chr1", "chr2"]:
            analysis = dataset.analysis[region]
            np.testing.assert_array_equal(
                _as_array(analysis["par"]), _as_array(analysis["serial"])
            )

    @pytest.mark.parametrize("method", TM_CONFIG_METHODS)
    def test_config_matches_serial(self, tmp_path, method):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        ana = SimAnalysis()
        getattr(ana, method)(dataset=dataset, name="serial", nworker=1)
        getattr(ana, method)(dataset=dataset, name="par", nworker=2)
        for region in ["chr1", "chr2"]:
            analysis = dataset.analysis[region]
            for suffix in ["position_flat", "position_offset", "energy"]:
                pd.testing.assert_frame_equal(
                    analysis[f"par_{suffix}"], analysis[f"serial_{suffix}"]
                )

    def test_best_match_gamma_reaches_workers(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        ana = SimAnalysis()
        ana.compute_best_match_config(
            dataset=dataset, name="serial", gamma=3.0, nworker=1
        )
        ana.compute_best_match_config(
            dataset=dataset, name="par", gamma=3.0, nworker=2
        )
        for region in ["chr1", "chr2"]:
            pd.testing.assert_frame_equal(
                dataset.analysis[region]["par_energy"],
                dataset.analysis[region]["serial_energy"],
            )

    @pytest.mark.parametrize("method", ["compute_occup", "compute_var_occup"])
    def test_small_batch_size_matches_default(self, tmp_path, method):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        ana = SimAnalysis()
        getattr(ana, method)(dataset=dataset, name="default")
        getattr(ana, method)(
            dataset=dataset, name="small", batch_size=2, nworker=2
        )
        for region in ["chr1", "chr2"]:
            analysis = dataset.analysis[region]
            np.testing.assert_array_equal(
                analysis["small"][:], analysis["default"][:]
            )

    @pytest.mark.parametrize("method", TM_STAT_METHODS + TM_CONFIG_METHODS)
    def test_nonpositive_nworker_raises(self, tmp_path, method):
        dataset = _make_tm_dataset(tmp_path)
        ana = SimAnalysis()
        with pytest.raises(ValueError, match="nworker must be positive"):
            getattr(ana, method)(dataset=dataset, nworker=0)

    @pytest.mark.parametrize("method", ["compute_occup", "compute_mean_nnuc"])
    def test_parallel_mcmc_raises(self, tmp_path, method):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)
        ana = SimAnalysis()
        with pytest.raises(ValueError, match="nworker > 1 requires"):
            getattr(ana, method)(dataset=dataset, nworker=2)

    @pytest.mark.parametrize("method", ["compute_occup", "compute_mean_nnuc"])
    def test_serial_mcmc_accepts_nworker_one(self, tmp_path, method):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)
        ana = SimAnalysis()
        getattr(ana, method)(dataset=dataset, name="x", nworker=1)
        assert "x" in dataset.analysis["chr1"]


def _linker_dataset_with_gap(tmp_path):
    rng = np.random.default_rng(3)
    meth_prob = rng.random((3, 20))
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


class TestComputeLinkerHist:
    def test_rows_match_direct_solver(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=20)
        SimAnalysis().compute_linker_hist(dataset, lmax=4)
        arr = dataset.analysis["chr1"]["linker_hist"]
        assert isinstance(arr, H5Array)
        assert arr.shape == (3, 6)
        settings = _make_tm_settings()
        for mol in range(3):
            expected = _direct_tm(dataset, mol, settings).getLinkerHist(4)
            np.testing.assert_allclose(arr.to_numpy()[mol], expected)

    def test_column_labels(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        SimAnalysis().compute_linker_hist(dataset, lmax=2)
        cols = dataset.analysis["chr1"]["linker_hist"].columns
        assert list(cols) == ["0", "1", "2", "tail"]

    def test_settings_entry(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nbp=20)
        SimAnalysis().compute_linker_hist(dataset, lmax=3, trim=2)
        settings = dataset.analysis["chr1"]["linker_hist_settings"]
        assert settings.iloc[0].to_dict() == {
            "lmax": 3,
            "xlo": 2,
            "xhi": 18,
            "llink": 2,
        }

    def test_custom_name(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        SimAnalysis().compute_linker_hist(dataset, lmax=2, name="lh")
        assert "lh" in dataset.analysis["chr1"]
        assert "lh_settings" in dataset.analysis["chr1"]

    def test_trim_matches_window(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3, nbp=20)
        ana = SimAnalysis()
        ana.compute_linker_hist(dataset, lmax=3, trim=2, name="a")
        ana.compute_linker_hist(dataset, lmax=3, window=(2, 18), name="b")
        a = dataset.analysis["chr1"]["a"].to_numpy()
        b = dataset.analysis["chr1"]["b"].to_numpy()
        np.testing.assert_allclose(a, b)
        settings = _make_tm_settings()
        expected = _direct_tm(dataset, 0, settings).getLinkerHist(3, 2, 18)
        np.testing.assert_allclose(a[0], expected)

    def test_window_and_trim_combine(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nbp=20)
        SimAnalysis().compute_linker_hist(
            dataset, lmax=3, window=(1, 19), trim=1
        )
        settings = dataset.analysis["chr1"]["linker_hist_settings"]
        assert (settings["xlo"].iloc[0], settings["xhi"].iloc[0]) == (2, 18)

    def test_window_dict_by_region(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        SimAnalysis().compute_linker_hist(
            dataset, lmax=3, window={"chr1": (2, 18), "chrX": (0, 5)}
        )
        s1 = dataset.analysis["chr1"]["linker_hist_settings"].iloc[0]
        s2 = dataset.analysis["chr2"]["linker_hist_settings"].iloc[0]
        assert (s1["xlo"], s1["xhi"]) == (2, 18)
        assert (s2["xlo"], s2["xhi"]) == (0, 16)
        assert dataset.analysis["chr2"]["linker_hist"].shape == (3, 5)

    def test_no_data_molecule_gives_nan_row(self, tmp_path):
        dataset = _linker_dataset_with_gap(tmp_path)
        SimAnalysis().compute_linker_hist(dataset, lmax=3)
        arr = dataset.analysis["chr1"]["linker_hist"].to_numpy()
        assert np.isnan(arr[1]).all()
        assert np.isfinite(arr[[0, 2]]).all()

    def test_parallel_matches_serial(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=5)
        ana = SimAnalysis()
        ana.compute_linker_hist(dataset, lmax=3, name="s")
        ana.compute_linker_hist(
            dataset, lmax=3, name="p", nworker=2, batch_size=2
        )
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["s"].to_numpy(),
            dataset.analysis["chr1"]["p"].to_numpy(),
        )

    def test_window_too_small_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nbp=20)
        with pytest.raises(ValueError, match="two nucleosomes"):
            SimAnalysis().compute_linker_hist(dataset, trim=8)

    def test_window_outside_fiber_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nbp=20)
        with pytest.raises(ValueError, match="window must satisfy"):
            SimAnalysis().compute_linker_hist(dataset, window=(0, 21))

    def test_negative_lmax_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="lmax must be nonnegative"):
            SimAnalysis().compute_linker_hist(dataset, lmax=-1)

    @pytest.mark.parametrize("lmax", [1.5, True])
    def test_non_integer_lmax_raises(self, tmp_path, lmax):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(TypeError, match="lmax must be an integer"):
            SimAnalysis().compute_linker_hist(dataset, lmax=lmax)

    def test_negative_trim_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="trim must be nonnegative"):
            SimAnalysis().compute_linker_hist(dataset, trim=-1)

    @pytest.mark.parametrize(
        "window", ["a", (1,), (0, 2.5), {"chr1": (1,)}]
    )
    def test_invalid_window_raises(self, tmp_path, window):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="window must be a pair"):
            SimAnalysis().compute_linker_hist(dataset, window=window)

    def test_mcmc_dataset_raises(self, tmp_path):
        dataset = _make_mcmc_dataset(tmp_path)
        with pytest.raises(ValueError, match="compute_linker_hist requires"):
            SimAnalysis().compute_linker_hist(dataset)

    def test_warns_when_dataset_used_fill_nan(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, fill_nan=0.5)
        with pytest.warns(UserWarning, match="fill_nan=0.5"):
            SimAnalysis().compute_linker_hist(dataset, lmax=2)

    def test_no_warning_without_fill_nan(self, tmp_path, recwarn):
        dataset = _make_tm_dataset(tmp_path)
        SimAnalysis().compute_linker_hist(dataset, lmax=2)
        assert not [w for w in recwarn if "fill_nan" in str(w.message)]


def _put_hist(dataset, region, values, name="h"):
    values = np.asarray(values, dtype=np.float64)
    columns = [*(str(i) for i in range(values.shape[1] - 1)), "tail"]
    arr = H5Array.create(values.shape, columns=columns)
    arr.write_batch(0, values.shape[0], values)
    dataset.analysis[region][name] = arr
    return values


_HIST = [
    [1.0, 2.0, 0.5],
    [np.nan, np.nan, np.nan],
    [3.0, 0.0, 1.5],
]


class TestPoolLinkerHist:
    def test_sum_skips_nan_rows(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        got = SimAnalysis().pool_linker_hist(dataset, name="h")
        assert list(got.index) == ["0", "1", "tail"]
        assert list(got.columns) == ["chr1"]
        np.testing.assert_allclose(got["chr1"], [4.0, 2.0, 2.0])

    def test_mean_divides_by_molecules_with_data(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        got = SimAnalysis().pool_linker_hist(dataset, name="h", how="mean")
        np.testing.assert_allclose(got["chr1"], [2.0, 1.0, 1.0])

    def test_mols_unsorted_with_duplicates(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        got = SimAnalysis().pool_linker_hist(
            dataset, name="h", mols=[2, 0, 2]
        )
        np.testing.assert_allclose(got["chr1"], [4.0, 2.0, 2.0])
        only = SimAnalysis().pool_linker_hist(dataset, name="h", mols=[2])
        np.testing.assert_allclose(only["chr1"], [3.0, 0.0, 1.5])

    def test_small_batch_size_matches(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        ana = SimAnalysis()
        full = ana.pool_linker_hist(dataset, name="h")
        small = ana.pool_linker_hist(dataset, name="h", batch_size=1)
        pd.testing.assert_frame_equal(full, small)

    def test_empty_selection(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        ana = SimAnalysis()
        total = ana.pool_linker_hist(dataset, name="h", mols=[1])
        mean = ana.pool_linker_hist(dataset, name="h", mols=[1], how="mean")
        np.testing.assert_allclose(total["chr1"], 0.0)
        assert mean["chr1"].isna().all()

    def test_dict_mols_by_region(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        v1 = _put_hist(dataset, "chr1", np.arange(15).reshape(5, 3))
        v2 = _put_hist(dataset, "chr2", np.ones((3, 3)))
        got = SimAnalysis().pool_linker_hist(
            dataset, name="h", mols={"chr1": [0, 4], "chrX": [0]}
        )
        np.testing.assert_allclose(got["chr1"], v1[[0, 4]].sum(axis=0))
        np.testing.assert_allclose(got["chr2"], v2.sum(axis=0))

    def test_non_dict_mols_with_several_regions_raises(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        _put_hist(dataset, "chr1", np.ones((5, 3)))
        _put_hist(dataset, "chr2", np.ones((3, 3)))
        with pytest.raises(ValueError, match="dict keyed by region"):
            SimAnalysis().pool_linker_hist(dataset, name="h", mols=[0])

    def test_different_bins_raise(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        _put_hist(dataset, "chr1", np.ones((5, 3)))
        _put_hist(dataset, "chr2", np.ones((3, 4)))
        with pytest.raises(ValueError, match="different bins"):
            SimAnalysis().pool_linker_hist(dataset, name="h")

    def test_mols_out_of_range_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        with pytest.raises(ValueError, match="mols must be in"):
            SimAnalysis().pool_linker_hist(dataset, name="h", mols=[3])

    def test_invalid_how_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        with pytest.raises(ValueError, match="how must be"):
            SimAnalysis().pool_linker_hist(dataset, name="h", how="max")

    def test_missing_name_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(KeyError, match="run compute_linker_hist first"):
            SimAnalysis().pool_linker_hist(dataset)

    def test_pools_computed_histogram(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=4)
        ana = SimAnalysis()
        ana.compute_linker_hist(dataset, lmax=3)
        got = ana.pool_linker_hist(dataset)
        rows = dataset.analysis["chr1"]["linker_hist"].to_numpy()
        np.testing.assert_allclose(got["chr1"], rows.sum(axis=0))
        assert list(got.index) == ["0", "1", "2", "3", "tail"]


class TestExtractLinkerStats:
    def test_per_molecule_values(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        got = SimAnalysis().extract_linker_stats(dataset, name="h")
        assert list(got.columns) == ["mean", "sd", "nlink"]
        assert got.index.names == ["region", "mol"]
        assert list(got.index) == [("chr1", 0), ("chr1", 1), ("chr1", 2)]
        np.testing.assert_allclose(
            got.loc[("chr1", 0)], [2 / 3, np.sqrt(2 / 9), 3.0]
        )
        np.testing.assert_allclose(got.loc[("chr1", 2)], [0.0, 0.0, 3.0])
        assert got.loc[("chr1", 1)].isna().all()

    def test_pooled_values(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        got = SimAnalysis().extract_linker_stats(
            dataset, name="h", pooled=True
        )
        assert got.index.name == "region"
        np.testing.assert_allclose(
            got.loc["chr1"], [1 / 3, np.sqrt(2 / 9), 6.0]
        )

    def test_pooled_sd_is_not_mean_of_per_molecule_sd(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        ana = SimAnalysis()
        per_mol = ana.extract_linker_stats(dataset, name="h")["sd"].mean()
        pooled = ana.extract_linker_stats(dataset, name="h", pooled=True)["sd"]
        assert pooled.iloc[0] != pytest.approx(per_mol)

    def test_tail_is_ignored(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        values = np.array(_HIST)
        values[[0, 2], -1] = 100.0
        _put_hist(dataset, "chr1", values)
        _put_hist(dataset, "chr1", _HIST, name="ref")
        ana = SimAnalysis()
        pd.testing.assert_frame_equal(
            ana.extract_linker_stats(dataset, name="h"),
            ana.extract_linker_stats(dataset, name="ref"),
        )

    def test_no_linkers_gives_nan_mean_and_sd(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", [[0, 0, 2], [1, 0, 0], [0, 0, 0]])
        got = SimAnalysis().extract_linker_stats(dataset, name="h")
        assert np.isnan(got.loc[("chr1", 0), "mean"])
        assert np.isnan(got.loc[("chr1", 0), "sd"])
        assert got.loc[("chr1", 0), "nlink"] == 0.0

    def test_mols_subset(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        ana = SimAnalysis()
        got = ana.extract_linker_stats(dataset, name="h", mols=[2, 0, 2])
        assert list(got.index) == [("chr1", 0), ("chr1", 2)]
        pooled = ana.extract_linker_stats(
            dataset, name="h", mols=[2], pooled=True
        )
        np.testing.assert_allclose(pooled.loc["chr1"], [0.0, 0.0, 3.0])

    def test_small_batch_size_matches(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        ana = SimAnalysis()
        pd.testing.assert_frame_equal(
            ana.extract_linker_stats(dataset, name="h"),
            ana.extract_linker_stats(dataset, name="h", batch_size=1),
        )

    def test_several_regions_with_different_bins(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        _put_hist(dataset, "chr1", np.ones((5, 3)))
        _put_hist(dataset, "chr2", np.ones((3, 4)))
        ana = SimAnalysis()
        per_mol = ana.extract_linker_stats(dataset, name="h")
        assert len(per_mol) == 8
        pooled = ana.extract_linker_stats(dataset, name="h", pooled=True)
        assert list(pooled.index) == ["chr1", "chr2"]
        np.testing.assert_allclose(pooled["mean"], [0.5, 1.0])

    def test_dict_mols_by_region(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        _put_hist(dataset, "chr1", np.ones((5, 3)))
        _put_hist(dataset, "chr2", np.ones((3, 3)))
        got = SimAnalysis().extract_linker_stats(
            dataset, name="h", mols={"chr1": [1, 3]}
        )
        assert list(got.loc["chr1"].index) == [1, 3]
        assert list(got.loc["chr2"].index) == [0, 1, 2]

    def test_empty_selection(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=3)
        _put_hist(dataset, "chr1", _HIST)
        got = SimAnalysis().extract_linker_stats(dataset, name="h", mols=[])
        assert got.empty
        assert list(got.columns) == ["mean", "sd", "nlink"]

    def test_non_dict_mols_with_several_regions_raises(self, tmp_path):
        dataset = _make_multi_region_tm_dataset(tmp_path)
        _put_hist(dataset, "chr1", np.ones((5, 3)))
        _put_hist(dataset, "chr2", np.ones((3, 3)))
        with pytest.raises(ValueError, match="dict keyed by region"):
            SimAnalysis().extract_linker_stats(dataset, name="h", mols=[0])

    def test_missing_name_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(KeyError, match="run compute_linker_hist first"):
            SimAnalysis().extract_linker_stats(dataset)

    def test_matches_computed_histogram(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path, nmol=4)
        ana = SimAnalysis()
        ana.compute_linker_hist(dataset, lmax=5)
        got = ana.extract_linker_stats(dataset)
        rows = dataset.analysis["chr1"]["linker_hist"].to_numpy()[:, :-1]
        lengths = np.arange(6)
        for mol in range(4):
            c = rows[mol]
            mean = (c * lengths).sum() / c.sum()
            sd = np.sqrt((c * lengths**2).sum() / c.sum() - mean**2)
            np.testing.assert_allclose(
                got.loc[("chr1", mol)], [mean, sd, c.sum()]
            )
