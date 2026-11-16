# test_plot.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from catella.simulation.analysis import NucFiberMap, SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.plot import SimPlot


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


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotOccupMolsSlicing:
    def test_mols_none_shows_all_molecules(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(chrom="chr1", dataset=dataset, show=False)

        img = plt.gcf().axes[0].images[0].get_array()
        assert img.shape[0] == 6

    def test_mols_restricts_displayed_rows(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()

        mols = [0, 2, 5]
        SimPlot().plot_occup(
            chrom="chr1", dataset=dataset, mols=mols, show=False
        )

        img = plt.gcf().axes[0].images[0].get_array()
        assert img.shape[0] == len(mols)
        np.testing.assert_allclose(np.asarray(img), occup[mols])

    def test_mols_with_plot_eseq_does_not_raise(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(
            chrom="chr1",
            dataset=dataset,
            mols=[1, 3],
            plot_eseq=True,
            show=False,
        )

        img = plt.gcf().axes[0].images[0].get_array()
        assert img.shape[0] == 2

    def test_plot_eseq_handles_nan_from_missing_coverage(self, tmp_path):
        # A genomic position with no coverage (NaN in meth_prob) propagates
        # NaN into eseq at that position for every molecule; the y-limit
        # calculation must tolerate that instead of crashing.
        rng = np.random.default_rng(3)
        meth_prob = rng.random((6, 20))
        meth_prob[:, 5] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=3,
        )
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        assert np.isnan(dataset.eseq["chr1"][:, 5]).all()

        SimPlot().plot_occup(
            chrom="chr1", dataset=dataset, plot_eseq=True, show=False
        )

    def test_plot_eseq_skips_ylim_when_all_nan(self, tmp_path):
        # If every selected molecule has no coverage at all, the eseq
        # average is all-NaN; set_ylim must be skipped instead of
        # raising "Axis limits cannot be NaN or Inf".
        rng = np.random.default_rng(4)
        meth_prob = rng.random((6, 20))
        meth_prob[0, :] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=4,
        )
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        assert np.isnan(dataset.eseq["chr1"][0, :]).all()

        SimPlot().plot_occup(
            chrom="chr1", dataset=dataset, mols=[0], plot_eseq=True, show=False
        )

        seq_ax = plt.gcf().axes[2]
        assert np.isfinite(seq_ax.get_ylim()).all()

    def test_plot_eseq_ylim_matches_full_data_range(self, tmp_path):
        # emin/emax must span the actual eseq range rather than a
        # narrower window, otherwise real energy values get clipped
        # off the visible axis.
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(
            chrom="chr1", dataset=dataset, plot_eseq=True, show=False
        )

        seq_ax = plt.gcf().axes[2]
        eseq_mean = np.nanmean(dataset.eseq["chr1"], axis=0)
        np.testing.assert_allclose(
            seq_ax.get_ylim(), (np.nanmin(eseq_mean), np.nanmax(eseq_mean))
        )


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotOccupCmapAndNan:
    def test_nan_cells_do_not_raise(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, 0] = np.nan
        dataset.analysis["chr1"]["occup"] = pd.DataFrame(occup)

        SimPlot().plot_occup(chrom="chr1", dataset=dataset, show=False)

    def test_cmap_override_does_not_mutate_instance(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        simplot = SimPlot()
        simplot.plot_occup(
            chrom="chr1", dataset=dataset, cmap="viridis", show=False
        )
        assert simplot.cmap == "OrRd"


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPositionalDatasetArg:
    def test_plot_occup_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(dataset, chrom="chr1", show=False)

        assert plt.gcf().axes[0].images

    def test_plot_energy_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_energy(dataset, chrom="chr1", mol=0, run=0, show=False)

        assert plt.gcf().axes

    def test_plot_samples_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_samples(dataset, chrom="chr1", mol=0, show=False)

        assert plt.gcf().axes


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotSamplesCmapOverride:
    def test_cmap_override_does_not_mutate_instance(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        simplot = SimPlot()
        simplot.plot_samples(
            dataset, chrom="chr1", mol=0, cmap="viridis", show=False
        )
        assert simplot.cmap == "OrRd"


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotSamplesEseq:
    def test_plot_eseq_handles_nan_from_missing_coverage(self, tmp_path):
        # A genomic position with no coverage (NaN in meth_prob) propagates
        # NaN into eseq for that molecule; the y-limit calculation must
        # tolerate that instead of crashing.
        rng = np.random.default_rng(5)
        meth_prob = rng.random((2, 20))
        meth_prob[:, 5] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=5,
        )
        assert np.isnan(dataset.eseq["chr1"][:, 5]).all()

        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, plot_eseq=True, show=False
        )

    def test_no_data_molecule_raises_over_time(self, tmp_path):
        rng = np.random.default_rng(6)
        meth_prob = rng.random((2, 20))
        meth_prob[0, :] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=6,
        )
        assert dataset.no_data["chr1"][0]

        with pytest.raises(ValueError, match="molecule 0 of chrom 'chr1'"):
            SimPlot().plot_samples(
                dataset, chrom="chr1", mol=0, plot_eseq=True, show=False
            )

    def test_plot_eseq_ylim_matches_full_data_range(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, plot_eseq=True, show=False
        )

        seq_ax = plt.gcf().axes[2]
        eseq = dataset.eseq["chr1"][0, :]
        np.testing.assert_allclose(
            seq_ax.get_ylim(), (np.nanmin(eseq), np.nanmax(eseq))
        )


def _mcmc_dataset(tmp_path, nmol=2, nsim=3):
    meth_prob = np.random.default_rng(7).random((nmol, 400))
    meth_prob[1] = np.nan
    return SimManager(nworker=1, verbose=False).run(
        chroms="chr1",
        nsim=nsim,
        settings=_make_settings(nsweep=4),
        meth_prob=meth_prob,
        out_dir=tmp_path / "mcmc",
        seed=7,
    )


def _tm_dataset(tmp_path, nsim=4, name="tm"):
    settings = SimSettings(
        nucbp=3, llink=2, elink=1.0, mu=0.5, end_temp=1.0, emax=5.0
    )
    meth_prob = np.random.default_rng(8).random((2, 20))
    meth_prob[1] = np.nan
    return SimManager(nworker=1, verbose=False).run(
        chroms="chr1",
        nsim=nsim,
        method="tm",
        settings=settings,
        meth_prob=meth_prob,
        out_dir=tmp_path / name,
        seed=8,
    )


def _image():
    return np.ma.getdata(plt.gcf().axes[0].images[0].get_array())


def _footprint(dataset, starts):
    nuc_map = NucFiberMap(
        nbp=dataset.nbp["chr1"], nucbp=dataset.settings["nucbp"]
    )
    return nuc_map.project(np.asarray(starts))


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotSamples:
    def test_mcmc_over_time_rows_are_frames(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, runs=2, show=False
        )
        data = dataset.raw["chr1", 0, 2]
        img = _image()
        assert img.shape == (len(data.time), 400)
        for i, pos in enumerate(data.position):
            np.testing.assert_array_equal(img[i], _footprint(dataset, pos))

    def test_mcmc_over_runs_rows_are_runs(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, over="runs", show=False
        )
        img = _image()
        assert img.shape == (3, 400)
        ana = SimAnalysis()
        for r in range(3):
            starts = ana.extract_config(
                dataset, chrom="chr1", run=r, mols=[0]
            )[0]
            np.testing.assert_array_equal(img[r], _footprint(dataset, starts))

    def test_runs_order(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, over="runs", runs=[2, 0], show=False
        )
        last = dataset.raw["chr1", 0, 2].position[-1]
        np.testing.assert_array_equal(_image()[0], _footprint(dataset, last))

    def test_missing_run_is_no_data_row(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        dataset.mcmc_file("chr1", 0, 1).unlink()
        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, over="runs", show=False
        )
        img = _image()
        assert (img[1] == -1).all()
        assert (img[0] >= 0).all()

    def test_tm_default_rows_are_samples(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(dataset, chrom="chr1", mol=0, show=False)
        img = _image()
        assert img.shape == (4, 20)
        for r in range(4):
            pos = dataset.raw["chr1", 0, r].position
            np.testing.assert_array_equal(img[r], _footprint(dataset, pos))

    def test_tm_no_data_molecule_rows(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(dataset, chrom="chr1", mol=1, show=False)
        assert (_image() == -1).all()

    def test_max_rows_stride_keeps_every_kth_run(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, chrom="chr1", mol=0, max_rows=2, show=False
        )
        img = _image()
        assert img.shape == (2, 20)
        pos = dataset.raw["chr1", 0, 2].position
        np.testing.assert_array_equal(img[1], _footprint(dataset, pos))

    @pytest.mark.parametrize("how", ["mean", "max"])
    def test_max_rows_merges_runs(self, tmp_path, how):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset,
            chrom="chr1",
            mol=0,
            max_rows=2,
            downsample_how=how,
            show=False,
        )
        rows = np.array(
            [
                _footprint(dataset, dataset.raw["chr1", 0, r].position)
                for r in range(4)
            ]
        )
        want = getattr(rows[:2], how)(axis=0)
        np.testing.assert_allclose(_image()[0], want)

    def test_size_warning(self, tmp_path, monkeypatch):
        monkeypatch.setattr("catella._plotting._RENDER_WARN_BYTES", 1)
        dataset = _tm_dataset(tmp_path)
        with pytest.warns(UserWarning, match="max_rows"):
            SimPlot().plot_samples(dataset, chrom="chr1", mol=0, show=False)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"over": "frames"}, "over must be"),
            ({"over": "runs", "tstart": 0}, "only used with over='time'"),
            ({"runs": [0, 1]}, "shows one run"),
            ({"time": 1}, "only used with over='runs'"),
            ({"max_rows": 0}, "max_rows must be positive"),
            ({"downsample_how": "min"}, "downsample_how must be"),
        ],
    )
    def test_mcmc_argument_errors(self, tmp_path, kwargs, match):
        dataset = _mcmc_dataset(tmp_path)
        with pytest.raises(ValueError, match=match):
            SimPlot().plot_samples(
                dataset, chrom="chr1", mol=0, show=False, **kwargs
            )

    def test_tm_over_time_raises(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="no time axis"):
            SimPlot().plot_samples(
                dataset, chrom="chr1", mol=0, over="time", show=False
            )

    def test_tm_without_samples_raises(self, tmp_path):
        dataset = _tm_dataset(tmp_path, nsim=0)
        with pytest.raises(ValueError, match="no samples stored"):
            SimPlot().plot_samples(dataset, chrom="chr1", mol=0, show=False)

    def test_plot_eseq_without_stored_energy_raises(self, tmp_path):
        dataset = SimManager(nworker=1, verbose=False).run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=np.random.default_rng(1).random((1, 20)),
            out_dir=tmp_path / "no_eseq",
            seed=1,
            store_eseq=False,
        )
        with pytest.raises(ValueError, match="store_eseq=True"):
            SimPlot().plot_samples(
                dataset, chrom="chr1", mol=0, plot_eseq=True, show=False
            )


def _tm_calls_dataset(tmp_path, nmol=5, nsim=0):
    settings = SimSettings(
        nucbp=3, llink=2, elink=1.0, mu=0.5, end_temp=1.0, emax=5.0
    )
    meth_prob = np.random.default_rng(9).random((nmol, 20))
    meth_prob[1] = np.nan
    dataset = SimManager(nworker=1, verbose=False).run(
        chroms="chr1",
        nsim=nsim,
        method="tm",
        settings=settings,
        meth_prob=meth_prob,
        out_dir=tmp_path / "tm_calls",
        seed=9,
    )
    SimAnalysis().compute_best_match_config(dataset=dataset, name="bm")
    return dataset


def _rows(dataset, configs):
    return np.array(
        [
            np.full(dataset.nbp["chr1"], -1, dtype=np.int8)
            if c is None
            else _footprint(dataset, c)
            for c in configs
        ]
    )


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotConfig:
    def test_stored_calls(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(dataset, chrom="chr1", name="bm", show=False)
        configs = SimAnalysis().extract_config(
            dataset, chrom="chr1", name="bm"
        )
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))
        assert (_image()[1] == -1).all()

    def test_mcmc_raw_run(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_config(dataset, chrom="chr1", run=1, show=False)
        configs = SimAnalysis().extract_config(dataset, chrom="chr1", run=1)
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))

    def test_tm_sample(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nsim=3)
        SimPlot().plot_config(dataset, chrom="chr1", run=2, show=False)
        pos = dataset.raw["chr1", 0, 2].position
        np.testing.assert_array_equal(_image()[0], _footprint(dataset, pos))

    def test_mols_order(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, chrom="chr1", name="bm", mols=[4, 0], show=False
        )
        configs = SimAnalysis().extract_config(
            dataset, chrom="chr1", name="bm", mols=[4, 0]
        )
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))

    def test_max_rows_stride_picks_molecules(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, chrom="chr1", name="bm", max_rows=3, show=False
        )
        configs = SimAnalysis().extract_config(
            dataset, chrom="chr1", name="bm", mols=[0, 2, 4]
        )
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))

    def test_max_rows_mean_ignores_no_data(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nmol=4)
        SimPlot().plot_config(
            dataset,
            chrom="chr1",
            name="bm",
            max_rows=2,
            downsample_how="mean",
            show=False,
        )
        configs = SimAnalysis().extract_config(
            dataset, chrom="chr1", name="bm"
        )
        rows = _rows(dataset, configs)
        np.testing.assert_allclose(_image()[0], rows[0])
        np.testing.assert_allclose(_image()[1], rows[2:].mean(axis=0))

    def test_max_rows_max(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nmol=4)
        SimPlot().plot_config(
            dataset,
            chrom="chr1",
            name="bm",
            max_rows=2,
            downsample_how="max",
            show=False,
        )
        configs = SimAnalysis().extract_config(
            dataset, chrom="chr1", name="bm"
        )
        rows = _rows(dataset, configs)
        np.testing.assert_array_equal(_image()[1], rows[2:].max(axis=0))

    def test_plot_eseq_averages_shown_molecules(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            chrom="chr1",
            name="bm",
            mols=[0, 2],
            plot_eseq=True,
            show=False,
        )
        assert len(plt.gcf().axes) == 4

    def test_size_warning(self, tmp_path, monkeypatch):
        monkeypatch.setattr("catella._plotting._RENDER_WARN_BYTES", 1)
        dataset = _tm_calls_dataset(tmp_path)
        with pytest.warns(UserWarning, match="'max_rows' or 'mols'"):
            SimPlot().plot_config(
                dataset, chrom="chr1", name="bm", show=False
            )

    def test_missing_name_raises(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        with pytest.raises(KeyError, match="nope"):
            SimPlot().plot_config(
                dataset, chrom="chr1", name="nope", show=False
            )

    def test_bad_downsample_how_raises(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        with pytest.raises(ValueError, match="downsample_how must be"):
            SimPlot().plot_config(
                dataset,
                chrom="chr1",
                name="bm",
                max_rows=2,
                downsample_how="sum",
                show=False,
            )
