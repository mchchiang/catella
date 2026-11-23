# test_plot.py

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
from catella import utils
from catella.h5_array import H5Array
from catella.simulation.analysis import NucFiberMap, SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.plot import SimPlot, _merge_rows


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


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotOccupMolsSlicing:
    def test_link_mat_orders_rows(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        dataset.analysis["chr1"]["occup"] = pd.DataFrame(
            np.random.default_rng(3).random((6, 20))
        )
        link_mats = SimAnalysis().sort_by_linkage(dataset=dataset)
        SimPlot().plot_occup(
            dataset,
            regions="chr1",
            link_mat=dataset.analysis["chr1"]["occup_linkage"],
            show=False,
        )
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        order = sch.leaves_list(link_mats["chr1"])
        img = plt.gcf().axes[0].images[0].get_array()
        np.testing.assert_allclose(np.ma.getdata(img), occup[order])

    def test_link_mat_with_mols_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        SimAnalysis().compute_occup(dataset=dataset)
        link_mat = utils.compute_linkage(np.random.rand(6, 3))[1]
        with pytest.raises(ValueError, match="mols cannot be combined"):
            SimPlot().plot_occup(
                dataset,
                regions="chr1",
                mols=[0, 1],
                link_mat=link_mat,
                show=False,
            )

    def test_size_warning_counts_shown_molecules(self, tmp_path, monkeypatch):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        SimAnalysis().compute_occup(dataset=dataset)
        monkeypatch.setattr(
            "catella._plotting._RENDER_WARN_BYTES", 3 * 20 * 80
        )
        with pytest.warns(UserWarning, match="6 rows x 20 bp"):
            SimPlot().plot_occup(dataset, regions="chr1", show=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SimPlot().plot_occup(
                dataset, regions="chr1", mols=[0, 1, 2], show=False
            )

    def test_mols_none_shows_all_molecules(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(regions="chr1", dataset=dataset, show=False)

        img = plt.gcf().axes[0].images[0].get_array()
        assert img.shape[0] == 6

    def test_mols_restricts_displayed_rows(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()

        mols = [0, 2, 5]
        SimPlot().plot_occup(
            regions="chr1", dataset=dataset, mols=mols, show=False
        )

        img = plt.gcf().axes[0].images[0].get_array()
        assert img.shape[0] == len(mols)
        np.testing.assert_allclose(np.asarray(img), occup[mols])

    def test_mols_with_plot_eseq_does_not_raise(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(
            regions="chr1",
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
            regions="chr1",
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
            regions="chr1", dataset=dataset, plot_eseq=True, show=False
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
            regions="chr1",
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
            regions="chr1", dataset=dataset, mols=[0], plot_eseq=True,
            show=False,
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
            regions="chr1", dataset=dataset, plot_eseq=True, show=False
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

        SimPlot().plot_occup(regions="chr1", dataset=dataset, show=False)

    def test_cmap_override_does_not_mutate_instance(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        simplot = SimPlot()
        simplot.plot_occup(
            regions="chr1", dataset=dataset, cmap="viridis", show=False
        )
        assert simplot.cmap == "OrRd"


def _make_two_region_dataset(tmp_path, seed=5):
    rng = np.random.default_rng(seed)
    manager = SimManager(nworker=1, verbose=False)
    dataset = manager.run(
        regions=["chr1", "chr2"],
        nsim=2,
        settings=_make_settings(),
        meth_prob={"chr1": rng.random((6, 20)), "chr2": rng.random((4, 30))},
        out_dir=tmp_path / "dataset",
        seed=seed,
    )
    for region, shape in (("chr1", (6, 20)), ("chr2", (4, 30))):
        dataset.analysis[region]["occup"] = pd.DataFrame(rng.random(shape))
    return dataset


def _make_region_dataset(tmp_path, nregion, seed=6):
    rng = np.random.default_rng(seed)
    regions = [f"chr{i + 1}" for i in range(nregion)]
    dataset = SimManager(nworker=1, verbose=False).run(
        regions=regions,
        nsim=2,
        settings=_make_settings(),
        meth_prob={r: rng.random((4, 20)) for r in regions},
        out_dir=tmp_path / "dataset_n",
        seed=seed,
    )
    for region in regions:
        dataset.analysis[region]["occup"] = pd.DataFrame(rng.random((4, 20)))
    return dataset


def _heatmap_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def _boxes(axes):
    axes[0].figure.canvas.draw()
    return [ax.get_position() for ax in axes]


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotOccupMultiRegion:
    def test_one_panel_per_region(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        out_file = tmp_path / "occup.png"
        SimPlot().plot_occup(
            dataset, regions=["chr1", "chr2"], out_file=out_file, show=False
        )
        assert out_file.exists()
        hm_axes = _heatmap_axes(plt.gcf())
        assert [ax.images[0].get_array().shape for ax in hm_axes] == [
            (6, 20),
            (4, 30),
        ]
        assert [ax.get_title() for ax in hm_axes] == ["chr1", "chr2"]

    def test_one_shared_colorbar(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(dataset, regions=["chr1", "chr2"], show=False)
        assert len(plt.gcf().axes) == 3

    def test_panels_side_by_side(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset,
            regions=["chr1", "chr2"],
            mols={"chr2": [0, 1, 2, 3]},
            show=False,
        )
        boxes = _boxes(_heatmap_axes(plt.gcf()))
        assert boxes[0].y0 == pytest.approx(boxes[1].y0)
        assert boxes[0].y1 == pytest.approx(boxes[1].y1)
        assert boxes[0].x1 < boxes[1].x0

    def test_four_per_row_then_wrap(self, tmp_path):
        dataset = _make_region_dataset(tmp_path, 5)
        SimPlot().plot_occup(dataset, regions=dataset.regions, show=False)
        fig = plt.gcf()
        hm_axes = _heatmap_axes(fig)
        boxes = _boxes(hm_axes)
        assert len(hm_axes) == 5
        assert all(b.y0 == pytest.approx(boxes[0].y0) for b in boxes[:4])
        assert boxes[4].y1 < boxes[0].y0
        assert boxes[4].x0 == pytest.approx(boxes[0].x0)
        assert boxes[4].x1 == pytest.approx(boxes[0].x1)
        assert sum(not ax.get_visible() for ax in fig.axes) == 3

    def test_colorbar_next_to_top_right_heatmap(self, tmp_path):
        dataset = _make_region_dataset(tmp_path, 5)
        SimPlot().plot_occup(dataset, regions=dataset.regions, show=False)
        fig = plt.gcf()
        hm_axes = _heatmap_axes(fig)
        cbar_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Occupancy")
        cbar, top_right = _boxes([cbar_ax, hm_axes[3]])
        assert cbar.y0 == pytest.approx(top_right.y0)
        assert cbar.y1 == pytest.approx(top_right.y1)
        assert cbar.x0 > top_right.x1

    def test_ylabel_on_leftmost_panels_only(self, tmp_path):
        dataset = _make_region_dataset(tmp_path, 5)
        SimPlot().plot_occup(
            dataset, regions=dataset.regions, plot_eseq=True, show=False
        )
        fig = plt.gcf()
        labels = [ax.get_ylabel() for ax in _heatmap_axes(fig)]
        assert labels == ["Molecule index", "", "", "", "Molecule index"]
        seq_labels = [ax.get_ylabel() for ax in fig.axes if ax.lines]
        assert [bool(s) for s in seq_labels] == [
            True, False, False, False, True
        ]

    def test_xlim_per_region_by_default(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset, regions=["chr1", "chr2"], xscale=1, show=False
        )
        xlims = [ax.get_xlim() for ax in _heatmap_axes(plt.gcf())]
        assert xlims == [(0, 20), (0, 30)]

    def test_sharex_uses_longest_region(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset,
            regions=["chr1", "chr2"],
            xscale=1,
            plot_eseq=True,
            sharex=True,
            show=False,
        )
        fig = plt.gcf()
        hm_axes = _heatmap_axes(fig)
        assert [ax.get_xlim() for ax in hm_axes] == [(0, 30), (0, 30)]
        seq_axes = [ax for ax in fig.axes if ax.lines]
        assert [ax.get_xlim() for ax in seq_axes] == [(0, 30), (0, 30)]

    def test_link_mat_dict_sorts_each_region(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        links = SimAnalysis().sort_by_linkage(dataset=dataset)
        SimPlot().plot_occup(
            dataset, regions=["chr1", "chr2"], link_mat=links, show=False
        )
        for region, ax in zip(["chr1", "chr2"], _heatmap_axes(plt.gcf())):
            occup = dataset.analysis[region]["occup"].to_numpy()
            order = sch.leaves_list(links[region])
            np.testing.assert_allclose(
                np.ma.getdata(ax.images[0].get_array()), occup[order]
            )

    def test_link_mat_dict_with_region_subset(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        links = SimAnalysis().sort_by_linkage(dataset=dataset)
        SimPlot().plot_occup(
            dataset, regions="chr2", link_mat=links, show=False
        )
        occup = dataset.analysis["chr2"]["occup"].to_numpy()
        img = _heatmap_axes(plt.gcf())[0].images[0].get_array()
        order = sch.leaves_list(links["chr2"])
        np.testing.assert_allclose(np.ma.getdata(img), occup[order])

    def test_link_mat_dict_missing_region_is_unsorted(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        links = SimAnalysis().sort_by_linkage(dataset=dataset)
        SimPlot().plot_occup(
            dataset,
            regions=["chr1", "chr2"],
            link_mat={"chr2": links["chr2"]},
            show=False,
        )
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        img = _heatmap_axes(plt.gcf())[0].images[0].get_array()
        np.testing.assert_allclose(np.ma.getdata(img), occup)

    @pytest.mark.parametrize("name", ["link_mat", "mols"])
    def test_non_dict_with_several_regions_raises(self, tmp_path, name):
        dataset = _make_two_region_dataset(tmp_path)
        value = np.zeros((3, 4)) if name == "link_mat" else [0, 1]
        with pytest.raises(ValueError, match=f"{name} must be a dict"):
            SimPlot().plot_occup(
                dataset, regions=["chr1", "chr2"], show=False, **{name: value}
            )

    def test_empty_regions_raises(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        with pytest.raises(ValueError, match="regions must not be empty"):
            SimPlot().plot_occup(dataset, regions=[], show=False)

    def test_many_regions_warn(self, tmp_path, monkeypatch):
        dataset = _make_two_region_dataset(tmp_path)
        monkeypatch.setattr("catella._plotting._MAX_PANELS", 1)
        with pytest.warns(UserWarning, match="2 regions in one figure"):
            SimPlot().plot_occup(
                dataset, regions=["chr1", "chr2"], show=False
            )

    def test_size_warning_counts_all_panels(self, tmp_path, monkeypatch):
        dataset = _make_two_region_dataset(tmp_path)
        monkeypatch.setattr(
            "catella._plotting._RENDER_WARN_BYTES", 8 * 30 * 80
        )
        with pytest.warns(UserWarning, match="10 rows x 30 bp"):
            SimPlot().plot_occup(
                dataset, regions=["chr1", "chr2"], show=False
            )


def _mean_axes(fig):
    return [ax for ax in fig.axes if ax.lines and ax.get_ylim() == (0, 1)]


def _mean_line(fig):
    return _mean_axes(fig)[0].lines[0].get_ydata()


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotOccupMean:
    def test_mean_over_all_molecules(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset, regions="chr1", xscale=1, plot_mean=True, show=False
        )
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        np.testing.assert_allclose(_mean_line(plt.gcf()), occup.mean(0))
        mean_ax = _mean_axes(plt.gcf())[0]
        assert mean_ax.get_ylabel() == "Mean"
        assert mean_ax.get_xlim() == (0, 20)

    def test_mean_over_mols_subset(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset, regions="chr1", mols=[0, 2], plot_mean=True, show=False
        )
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        np.testing.assert_allclose(
            _mean_line(plt.gcf()), occup[[0, 2]].mean(0)
        )

    def test_mean_unchanged_by_link_mat(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        links = SimAnalysis().sort_by_linkage(dataset=dataset)
        SimPlot().plot_occup(
            dataset, regions="chr1", link_mat=links, plot_mean=True, show=False
        )
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        np.testing.assert_allclose(_mean_line(plt.gcf()), occup.mean(0))

    def test_nan_column_stays_nan(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        occup = dataset.analysis["chr1"]["occup"].to_numpy().copy()
        occup[:, 3] = np.nan
        occup[0, 5] = np.nan
        dataset.analysis["chr1"]["occup"] = pd.DataFrame(occup)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            SimPlot().plot_occup(
                dataset, regions="chr1", plot_mean=True, show=False
            )
        mean = _mean_line(plt.gcf())
        assert np.isnan(mean[3])
        assert mean[5] == pytest.approx(np.nanmean(occup[:, 5]))

    def test_mean_above_eseq(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset,
            regions="chr1",
            plot_mean=True,
            plot_eseq=True,
            show=False,
        )
        fig = plt.gcf()
        hm_ax = _heatmap_axes(fig)[0]
        mean_ax = _mean_axes(fig)[0]
        seq_ax = next(
            ax for ax in fig.axes if ax.lines and ax is not mean_ax
        )
        hm, mean, seq = _boxes([hm_ax, mean_ax, seq_ax])
        assert hm.y0 > mean.y1 and mean.y0 > seq.y1
        assert not hm_ax.get_xaxis().get_visible()
        assert not mean_ax.get_xaxis().get_visible()
        assert seq_ax.get_xlabel().startswith("Position")

    def test_mean_is_bottom_row_without_eseq(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset, regions="chr1", plot_mean=True, show=False
        )
        fig = plt.gcf()
        assert not _heatmap_axes(fig)[0].get_xaxis().get_visible()
        assert _mean_axes(fig)[0].get_xlabel().startswith("Position")

    def test_several_regions(self, tmp_path):
        dataset = _make_two_region_dataset(tmp_path)
        SimPlot().plot_occup(
            dataset,
            regions=["chr1", "chr2"],
            xscale=1,
            plot_mean=True,
            sharex=True,
            show=False,
        )
        mean_axes = _mean_axes(plt.gcf())
        assert [ax.get_xlim() for ax in mean_axes] == [(0, 30), (0, 30)]
        assert [ax.get_ylabel() for ax in mean_axes] == ["Mean", ""]
        left, right = _boxes(mean_axes)
        assert left.y0 == pytest.approx(right.y0)
        assert left.y1 == pytest.approx(right.y1)


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPositionalDatasetArg:
    def test_plot_occup_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(dataset, regions="chr1", show=False)

        assert plt.gcf().axes[0].images

    def test_plot_energy_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_energy(dataset, region="chr1", mol=0, run=0, show=False)

        assert plt.gcf().axes

    def test_plot_samples_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_samples(dataset, region="chr1", mol=0, show=False)

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
            dataset, region="chr1", mol=0, cmap="viridis", show=False
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
            regions="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=5,
        )
        assert np.isnan(dataset.eseq["chr1"][:, 5]).all()

        SimPlot().plot_samples(
            dataset, region="chr1", mol=0, plot_eseq=True, show=False
        )

    def test_no_data_molecule_raises_over_time(self, tmp_path):
        rng = np.random.default_rng(6)
        meth_prob = rng.random((2, 20))
        meth_prob[0, :] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            regions="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "dataset",
            seed=6,
        )
        assert dataset.no_data["chr1"][0]

        with pytest.raises(ValueError, match="molecule 0 of region 'chr1'"):
            SimPlot().plot_samples(
                dataset, region="chr1", mol=0, plot_eseq=True, show=False
            )

    def test_plot_eseq_ylim_matches_full_data_range(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_samples(
            dataset, region="chr1", mol=0, plot_eseq=True, show=False
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
        regions="chr1",
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
        regions="chr1",
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
            dataset, region="chr1", mol=0, runs=2, show=False
        )
        data = dataset.raw["chr1", 0, 2]
        img = _image()
        assert img.shape == (len(data.time), 400)
        for i, pos in enumerate(data.position):
            np.testing.assert_array_equal(img[i], _footprint(dataset, pos))

    def test_mcmc_over_runs_rows_are_runs(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, region="chr1", mol=0, over="runs", show=False
        )
        img = _image()
        assert img.shape == (3, 400)
        ana = SimAnalysis()
        for r in range(3):
            starts = ana.extract_config(
                dataset, region="chr1", run=r, mols=[0]
            )[0]
            np.testing.assert_array_equal(img[r], _footprint(dataset, starts))

    def test_runs_order(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, region="chr1", mol=0, over="runs", runs=[2, 0], show=False
        )
        last = dataset.raw["chr1", 0, 2].position[-1]
        np.testing.assert_array_equal(_image()[0], _footprint(dataset, last))

    def test_missing_run_is_no_data_row(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        dataset.mcmc_file("chr1", 0, 1).unlink()
        SimPlot().plot_samples(
            dataset, region="chr1", mol=0, over="runs", show=False
        )
        img = _image()
        assert (img[1] == -1).all()
        assert (img[0] >= 0).all()

    def test_tm_default_rows_are_samples(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(dataset, region="chr1", mol=0, show=False)
        img = _image()
        assert img.shape == (4, 20)
        for r in range(4):
            pos = dataset.raw["chr1", 0, r].position
            np.testing.assert_array_equal(img[r], _footprint(dataset, pos))

    def test_tm_no_data_molecule_rows(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(dataset, region="chr1", mol=1, show=False)
        assert (_image() == -1).all()

    def test_max_rows_stride_keeps_every_kth_run(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        SimPlot().plot_samples(
            dataset, region="chr1", mol=0, max_rows=2, show=False
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
            region="chr1",
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
            SimPlot().plot_samples(dataset, region="chr1", mol=0, show=False)

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
                dataset, region="chr1", mol=0, show=False, **kwargs
            )

    def test_tm_over_time_raises(self, tmp_path):
        dataset = _tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="no time axis"):
            SimPlot().plot_samples(
                dataset, region="chr1", mol=0, over="time", show=False
            )

    def test_tm_without_samples_raises(self, tmp_path):
        dataset = _tm_dataset(tmp_path, nsim=0)
        with pytest.raises(ValueError, match="no samples stored"):
            SimPlot().plot_samples(dataset, region="chr1", mol=0, show=False)

    def test_plot_eseq_without_stored_energy_raises(self, tmp_path):
        dataset = SimManager(nworker=1, verbose=False).run(
            regions="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=np.random.default_rng(1).random((1, 20)),
            out_dir=tmp_path / "no_eseq",
            seed=1,
            store_eseq=False,
        )
        with pytest.raises(ValueError, match="store_eseq=True"):
            SimPlot().plot_samples(
                dataset, region="chr1", mol=0, plot_eseq=True, show=False
            )


def _tm_calls_dataset(tmp_path, nmol=5, nsim=0):
    settings = SimSettings(
        nucbp=3, llink=2, elink=1.0, mu=0.5, end_temp=1.0, emax=5.0
    )
    meth_prob = np.random.default_rng(9).random((nmol, 20))
    meth_prob[1] = np.nan
    dataset = SimManager(nworker=1, verbose=False).run(
        regions="chr1",
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
        SimPlot().plot_config(dataset, regions="chr1", name="bm", show=False)
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm"
        )
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))
        assert (_image()[1] == -1).all()

    def test_mcmc_raw_run(self, tmp_path):
        dataset = _mcmc_dataset(tmp_path)
        SimPlot().plot_config(dataset, regions="chr1", run=1, show=False)
        configs = SimAnalysis().extract_config(dataset, region="chr1", run=1)
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))

    def test_tm_sample(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nsim=3)
        SimPlot().plot_config(dataset, regions="chr1", run=2, show=False)
        pos = dataset.raw["chr1", 0, 2].position
        np.testing.assert_array_equal(_image()[0], _footprint(dataset, pos))

    def test_mols_order(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions="chr1", name="bm", mols=[4, 0], show=False
        )
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm", mols=[4, 0]
        )
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))

    def test_max_rows_stride_picks_molecules(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions="chr1", name="bm", max_rows=3, show=False
        )
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm", mols=[0, 2, 4]
        )
        np.testing.assert_array_equal(_image(), _rows(dataset, configs))

    def test_max_rows_mean_ignores_no_data(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nmol=4)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            max_rows=2,
            downsample_how="mean",
            show=False,
        )
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm"
        )
        rows = _rows(dataset, configs)
        np.testing.assert_allclose(_image()[0], rows[0])
        np.testing.assert_allclose(_image()[1], rows[2:].mean(axis=0))

    def test_max_rows_max(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nmol=4)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            max_rows=2,
            downsample_how="max",
            show=False,
        )
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm"
        )
        rows = _rows(dataset, configs)
        np.testing.assert_array_equal(_image()[1], rows[2:].max(axis=0))

    def test_plot_eseq_averages_shown_molecules(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
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
                dataset, regions="chr1", name="bm", show=False
            )

    def test_binary_plot_has_legend(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(dataset, regions="chr1", name="bm", show=False)
        legend = plt.gcf().axes[1].get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == ["Nucleosome", "Linker", "No data"]

    def test_no_data_entry_only_if_present(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions="chr1", name="bm", mols=[0, 2], show=False
        )
        legend = plt.gcf().axes[1].get_legend()
        assert "No data" not in [t.get_text() for t in legend.get_texts()]

    def test_mean_plot_has_colorbar(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path, nmol=4)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            max_rows=2,
            downsample_how="mean",
            show=False,
        )
        cbar_ax = plt.gcf().axes[1]
        assert cbar_ax.get_legend() is None
        assert cbar_ax.get_ylabel() == "Covered share"

    def test_rows_are_centered_with_integer_ticks(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(dataset, regions="chr1", name="bm", show=False)
        hm_ax = plt.gcf().axes[0]
        assert hm_ax.get_ylim() == (-0.5, 4.5)
        ticks = [t for t in hm_ax.get_yticks() if -0.5 <= t <= 4.5]
        assert all(float(t).is_integer() for t in ticks)

    @pytest.mark.parametrize(
        ("max_rows", "label"), [(None, "Molecule"), (3, "Row (5 molecules)")]
    )
    def test_ylabel(self, tmp_path, max_rows, label):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions="chr1", name="bm", max_rows=max_rows, show=False
        )
        assert plt.gcf().axes[0].get_ylabel() == label

    def test_link_mat_orders_rows(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm"
        )
        rows = _rows(dataset, configs)
        rows_float = np.where(rows < 0, 0, rows).astype(float)
        _, link_mat = utils.compute_linkage(rows_float)
        SimPlot().plot_config(
            dataset, regions="chr1", name="bm", link_mat=link_mat, show=False
        )
        order = sch.leaves_list(link_mat)
        np.testing.assert_array_equal(_image(), rows[order])
        assert plt.gcf().axes[0].get_ylabel() == "Row (leaf order)"
        assert len(plt.gcf().axes) == 3

    def test_link_mat_then_stride(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm"
        )
        rows = _rows(dataset, configs)
        _, link_mat = utils.compute_linkage(
            np.where(rows < 0, 0, rows).astype(float)
        )
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            link_mat=pd.DataFrame(link_mat),
            max_rows=3,
            show=False,
        )
        order = sch.leaves_list(link_mat)
        np.testing.assert_array_equal(_image(), rows[order[::2]])

    def test_link_mat_with_mols_raises(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        link_mat = utils.compute_linkage(np.random.rand(5, 3))[1]
        with pytest.raises(ValueError, match="mols cannot be combined"):
            SimPlot().plot_config(
                dataset,
                regions="chr1",
                name="bm",
                mols=[0, 1],
                link_mat=link_mat,
                show=False,
            )

    def test_link_mat_size_mismatch_raises(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        link_mat = utils.compute_linkage(np.random.rand(4, 3))[1]
        with pytest.raises(ValueError, match="4 leaves but the data has 5"):
            SimPlot().plot_config(
                dataset,
                regions="chr1",
                name="bm",
                link_mat=link_mat,
                show=False,
            )

    def test_missing_name_raises(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        with pytest.raises(KeyError, match="nope"):
            SimPlot().plot_config(
                dataset, regions="chr1", name="nope", show=False
            )

    def test_bad_downsample_how_raises(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        with pytest.raises(ValueError, match="downsample_how must be"):
            SimPlot().plot_config(
                dataset,
                regions="chr1",
                name="bm",
                max_rows=2,
                downsample_how="sum",
                show=False,
            )


def _expected_mean(rows):
    return np.nanmean(np.where(rows < 0, np.nan, rows.astype(float)), axis=0)


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotConfigMean:
    def _rows(self, dataset):
        configs = SimAnalysis().extract_config(
            dataset, region="chr1", name="bm"
        )
        return _rows(dataset, configs)

    def test_mean_over_molecules_with_data(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions="chr1", name="bm", plot_mean=True, show=False
        )
        rows = self._rows(dataset)
        assert (rows[1] == -1).all()
        np.testing.assert_allclose(
            _mean_line(plt.gcf()), _expected_mean(rows)
        )
        assert _mean_axes(plt.gcf())[0].get_ylabel() == "Mean"

    def test_stride_uses_shown_molecules(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            max_rows=3,
            plot_mean=True,
            show=False,
        )
        rows = self._rows(dataset)
        np.testing.assert_allclose(
            _mean_line(plt.gcf()), _expected_mean(rows[[0, 2, 4]])
        )

    def test_mean_merging_uses_all_molecules(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            max_rows=2,
            downsample_how="mean",
            plot_mean=True,
            show=False,
        )
        np.testing.assert_allclose(
            _mean_line(plt.gcf()), _expected_mean(self._rows(dataset))
        )

    def test_mean_above_eseq(self, tmp_path):
        dataset = _tm_calls_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions="chr1",
            name="bm",
            plot_mean=True,
            plot_eseq=True,
            show=False,
        )
        fig = plt.gcf()
        hm_ax = _heatmap_axes(fig)[0]
        mean_ax = _mean_axes(fig)[0]
        seq_ax = next(
            ax for ax in fig.axes if ax.lines and ax is not mean_ax
        )
        hm, mean, seq = _boxes([hm_ax, mean_ax, seq_ax])
        assert hm.y0 > mean.y1 and mean.y0 > seq.y1
        assert not hm_ax.get_xaxis().get_visible()
        assert not mean_ax.get_xaxis().get_visible()
        assert seq_ax.get_xlabel().startswith("Position")

    def test_several_regions(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions=["chr1", "chr2"],
            name="bm",
            xscale=1,
            plot_mean=True,
            sharex=True,
            show=False,
        )
        mean_axes = _mean_axes(plt.gcf())
        assert [ax.get_xlim() for ax in mean_axes] == [(0, 30), (0, 30)]
        assert [ax.get_ylabel() for ax in mean_axes] == ["Mean", ""]


def _tm_two_region_dataset(tmp_path):
    settings = SimSettings(
        nucbp=3, llink=2, elink=1.0, mu=0.5, end_temp=1.0, emax=5.0
    )
    rng = np.random.default_rng(9)
    chr1 = rng.random((5, 20))
    chr1[1] = np.nan
    dataset = SimManager(nworker=1, verbose=False).run(
        regions=["chr1", "chr2"],
        nsim=0,
        method="tm",
        settings=settings,
        meth_prob={"chr1": chr1, "chr2": rng.random((4, 30))},
        out_dir=tmp_path / "tm_two",
        seed=9,
    )
    SimAnalysis().compute_best_match_config(dataset=dataset, name="bm")
    return dataset


def _legend_labels(fig):
    return [t.get_text() for leg in fig.legends for t in leg.get_texts()]


def _config_rows(dataset, region, mols=None):
    configs = SimAnalysis().extract_config(
        dataset, region=region, name="bm", mols=mols
    )
    nuc_map = NucFiberMap(nbp=dataset.nbp[region], nucbp=3)
    return np.array(
        [
            np.full(dataset.nbp[region], -1, dtype=np.int8)
            if c is None
            else nuc_map.project(np.asarray(c))
            for c in configs
        ]
    )


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotConfigMultiRegion:
    def test_one_panel_per_region(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions=["chr1", "chr2"], name="bm", show=False
        )
        hm_axes = _heatmap_axes(plt.gcf())
        for region, ax in zip(["chr1", "chr2"], hm_axes):
            np.testing.assert_array_equal(
                np.ma.getdata(ax.images[0].get_array()),
                _config_rows(dataset, region),
            )
            assert ax.get_title() == region

    def test_binary_panels_share_one_legend(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions=["chr1", "chr2"], name="bm", show=False
        )
        fig = plt.gcf()
        legends = [ax.get_legend() for ax in fig.axes if ax.get_legend()]
        assert len(legends) == 1
        labels = [t.get_text() for t in legends[0].get_texts()]
        assert labels == ["Nucleosome", "Linker", "No data"]

    def test_merged_panel_gives_shared_colorbar(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions=["chr1", "chr2"],
            name="bm",
            max_rows=4,
            downsample_how="mean",
            show=False,
        )
        fig = plt.gcf()
        dtypes = [
            ax.images[0].get_array().dtype for ax in _heatmap_axes(fig)
        ]
        assert dtypes == [np.float64, np.int8]
        labels = [ax.get_ylabel() for ax in fig.axes]
        assert labels.count("Covered share") == 1
        assert _legend_labels(fig) == ["No data"]

    def test_panels_side_by_side(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        SimPlot().plot_config(
            dataset, regions=["chr1", "chr2"], name="bm", show=False
        )
        boxes = _boxes(_heatmap_axes(plt.gcf()))
        assert boxes[0].y0 == pytest.approx(boxes[1].y0)
        assert boxes[0].y1 == pytest.approx(boxes[1].y1)
        assert boxes[0].x1 < boxes[1].x0
        labels = [ax.get_ylabel() for ax in _heatmap_axes(plt.gcf())]
        assert labels == ["Molecule", ""]

    def test_sharex_uses_longest_region(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        SimPlot().plot_config(
            dataset,
            regions=["chr1", "chr2"],
            name="bm",
            xscale=1,
            sharex=True,
            show=False,
        )
        xlims = [ax.get_xlim() for ax in _heatmap_axes(plt.gcf())]
        assert xlims == [(0, 30), (0, 30)]

    def test_link_mat_dict_sorts_each_region(self, tmp_path):
        dataset = _tm_two_region_dataset(tmp_path)
        rng = np.random.default_rng(1)
        links = {
            "chr1": utils.compute_linkage(rng.random((5, 3)))[1],
            "chr2": utils.compute_linkage(rng.random((4, 3)))[1],
        }
        SimPlot().plot_config(
            dataset,
            regions=["chr1", "chr2"],
            name="bm",
            link_mat=links,
            show=False,
        )
        for region, ax in zip(["chr1", "chr2"], _heatmap_axes(plt.gcf())):
            order = sch.leaves_list(links[region])
            np.testing.assert_array_equal(
                np.ma.getdata(ax.images[0].get_array()),
                _config_rows(dataset, region)[order],
            )

    @pytest.mark.parametrize("name", ["link_mat", "mols"])
    def test_non_dict_with_several_regions_raises(self, tmp_path, name):
        dataset = _tm_two_region_dataset(tmp_path)
        value = np.zeros((3, 4)) if name == "link_mat" else [0, 1]
        with pytest.raises(ValueError, match=f"{name} must be a dict"):
            SimPlot().plot_config(
                dataset,
                regions=["chr1", "chr2"],
                name="bm",
                show=False,
                **{name: value},
            )

    def test_size_warning_counts_all_panels(self, tmp_path, monkeypatch):
        dataset = _tm_two_region_dataset(tmp_path)
        monkeypatch.setattr(
            "catella._plotting._RENDER_WARN_BYTES", 8 * 30 * 80
        )
        with pytest.warns(UserWarning, match="9 rows x 30 bp"):
            SimPlot().plot_config(
                dataset, regions=["chr1", "chr2"], name="bm", show=False
            )


_LINKER_ROWS = np.array(
    [
        [1.0, 2.0, 3.0, 0.5],
        [2.0, 2.0, 2.0, 1.0],
        [np.nan, np.nan, np.nan, np.nan],
        [0.0, 1.0, 1.0, 0.0],
    ]
)


def _put_linker_hist(dataset, region, values, name="linker_hist"):
    values = np.asarray(values, dtype=np.float64)
    columns = [*(str(i) for i in range(values.shape[1] - 1)), "tail"]
    arr = H5Array.create(values.shape, columns=columns)
    arr.write_batch(0, values.shape[0], values)
    dataset.analysis[region][name] = arr


def _linker_dataset(tmp_path, nregion, scales=None):
    dataset = _make_region_dataset(tmp_path, nregion)
    scales = scales or [1.0] * nregion
    for region, scale in zip(dataset.regions, scales):
        _put_linker_hist(dataset, region, scale * _LINKER_ROWS)
    return dataset


def _step_ydata(ax):
    return [np.asarray(line.get_ydata()) for line in ax.get_lines()]


class TestPlotLinkerHist:
    def test_overlay_draws_one_filled_step_per_region(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 2)
        SimPlot().plot_linker_hist(
            dataset, regions=["chr1", "chr2"], show=False
        )
        axes = plt.gcf().axes
        assert len(axes) == 1
        assert len(axes[0].get_lines()) == 2
        assert len(axes[0].collections) == 2

    def test_density_is_fraction_of_linkers(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        SimPlot().plot_linker_hist(dataset, regions="chr1", show=False)
        (ydata,) = _step_ydata(plt.gca())
        counts = np.nansum(_LINKER_ROWS[:, :-1], axis=0)
        np.testing.assert_allclose(ydata, counts / counts.sum())
        assert plt.gca().get_ylabel() == "Fraction of linkers"

    def test_counts_per_molecule(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        SimPlot().plot_linker_hist(
            dataset, regions="chr1", density=False, show=False
        )
        (ydata,) = _step_ydata(plt.gca())
        expected = np.nanmean(_LINKER_ROWS[:, :-1], axis=0)
        np.testing.assert_allclose(ydata, expected)
        assert plt.gca().get_ylabel() == "Linkers per molecule"

    def test_mols_subset(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        SimPlot().plot_linker_hist(
            dataset, regions="chr1", mols=[3], density=False, show=False
        )
        (ydata,) = _step_ydata(plt.gca())
        np.testing.assert_allclose(ydata, [0.0, 1.0, 1.0])

    def test_legend_shows_pooled_stats(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 2)
        SimPlot().plot_linker_hist(
            dataset, regions=["chr1", "chr2"], show=False
        )
        stats = SimAnalysis().extract_linker_stats(dataset, pooled=True)
        texts = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        row = stats.loc["chr1"]
        assert texts[0] == (
            f"chr1 (mean {row['mean']:.1f} bp, SD {row['sd']:.1f} bp)"
        )

    def test_legend_without_stats(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 2)
        SimPlot().plot_linker_hist(
            dataset, regions=["chr1", "chr2"], plot_stats=False, show=False
        )
        texts = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        assert texts == ["chr1", "chr2"]

    def test_panels_one_per_region(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 2)
        SimPlot().plot_linker_hist(
            dataset, regions=["chr1", "chr2"], panels=True, show=False
        )
        axes = [ax for ax in plt.gcf().axes if ax.get_visible()]
        assert [ax.get_title() for ax in axes] == ["chr1", "chr2"]
        for ax in axes:
            assert len(ax.get_lines()) == 1
            assert any("SD" in t.get_text() for t in ax.texts)

    def test_five_panels_use_two_rows(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 5)
        SimPlot().plot_linker_hist(
            dataset, regions=dataset.regions, panels=True, show=False
        )
        axes = plt.gcf().axes
        assert len(axes) == 8
        assert sum(ax.get_visible() for ax in axes) == 5

    @pytest.mark.parametrize("sharey", [True, False])
    def test_sharey(self, tmp_path, sharey):
        dataset = _linker_dataset(tmp_path, 2, scales=[1.0, 10.0])
        SimPlot().plot_linker_hist(
            dataset,
            regions=["chr1", "chr2"],
            panels=True,
            sharey=sharey,
            density=False,
            show=False,
        )
        ax1, ax2 = plt.gcf().axes
        assert (ax1.get_ylim() == ax2.get_ylim()) == sharey

    def test_logy(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        SimPlot().plot_linker_hist(
            dataset, regions="chr1", logy=True, show=False
        )
        assert plt.gca().get_yscale() == "log"

    def test_region_without_linkers_is_skipped(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 2)
        _put_linker_hist(dataset, "chr2", np.zeros((4, 4)))
        with pytest.warns(UserWarning, match="no linkers up to lmax"):
            SimPlot().plot_linker_hist(
                dataset, regions=["chr1", "chr2"], show=False
            )
        assert len(plt.gca().get_lines()) == 1

    def test_writes_out_file(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        out_file = tmp_path / "plots" / "linker.png"
        SimPlot().plot_linker_hist(
            dataset, regions="chr1", out_file=out_file, show=False
        )
        assert out_file.exists()

    def test_empty_regions_raises(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        with pytest.raises(ValueError, match="must not be empty"):
            SimPlot().plot_linker_hist(dataset, regions=[], show=False)

    def test_non_dict_mols_with_several_regions_raises(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 2)
        with pytest.raises(ValueError, match="dict keyed by region"):
            SimPlot().plot_linker_hist(
                dataset, regions=["chr1", "chr2"], mols=[0], show=False
            )

    def test_missing_name_raises(self, tmp_path):
        dataset = _linker_dataset(tmp_path, 1)
        with pytest.raises(KeyError, match="run compute_linker_hist first"):
            SimPlot().plot_linker_hist(
                dataset, regions="chr1", name="other", show=False
            )


_FOOTPRINTS = np.array(
    [[1, 0, -1], [0, 0, -1], [-1, -1, -1], [-1, 1, -1]], dtype=np.int8
)


class TestMergeRows:
    def test_max_ignores_no_data_rows(self):
        result = _merge_rows(_FOOTPRINTS, 2, "max")
        np.testing.assert_array_equal(result, [[1, 0, -1], [-1, 1, -1]])
        assert result.dtype == np.int8

    def test_mean_is_covered_share_among_rows_with_data(self):
        result = _merge_rows(_FOOTPRINTS, 2, "mean")
        expected = [[0.5, 0.0, np.nan], [np.nan, 1.0, np.nan]]
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64

    @pytest.mark.parametrize(
        ("max_rows", "how"), [(2, "stride"), (None, "mean"), (4, "max")]
    )
    def test_returns_matrix_as_is(self, max_rows, how):
        assert _merge_rows(_FOOTPRINTS, max_rows, how) is _FOOTPRINTS
