# test_plot.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.analysis import SimAnalysis
from catella.simulation.plot import SimPlot


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


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning")
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
        SimPlot().plot_occup(chrom="chr1", dataset=dataset, mols=mols,
                             show=False)

        img = plt.gcf().axes[0].images[0].get_array()
        assert img.shape[0] == len(mols)
        np.testing.assert_allclose(np.asarray(img), occup[mols])

    def test_mols_with_plot_eseq_does_not_raise(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(chrom="chr1", dataset=dataset, mols=[1, 3],
                             plot_eseq=True, show=False)

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
        dataset = manager.run(chroms="chr1", nsim=2, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "dataset",
                              seed=3)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        assert np.isnan(dataset.eseq["chr1"][:, 5]).all()

        SimPlot().plot_occup(chrom="chr1", dataset=dataset, plot_eseq=True,
                             show=False)

    def test_plot_eseq_skips_ylim_when_all_nan(self, tmp_path):
        # If every selected molecule has no coverage at all, the eseq
        # average is all-NaN; set_ylim must be skipped instead of
        # raising "Axis limits cannot be NaN or Inf".
        rng = np.random.default_rng(4)
        meth_prob = rng.random((6, 20))
        meth_prob[0, :] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=2, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "dataset",
                              seed=4)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)
        assert np.isnan(dataset.eseq["chr1"][0, :]).all()

        SimPlot().plot_occup(chrom="chr1", dataset=dataset, mols=[0],
                             plot_eseq=True, show=False)

        seq_ax = plt.gcf().axes[2]
        assert np.isfinite(seq_ax.get_ylim()).all()

    def test_plot_eseq_ylim_matches_full_data_range(self, tmp_path):
        # emin/emax must span the actual eseq range rather than a
        # narrower window, otherwise real energy values get clipped
        # off the visible axis.
        dataset = _make_dataset(tmp_path, nmol=6, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(chrom="chr1", dataset=dataset, plot_eseq=True,
                             show=False)

        seq_ax = plt.gcf().axes[2]
        eseq_mean = np.nanmean(dataset.eseq["chr1"], axis=0)
        np.testing.assert_allclose(seq_ax.get_ylim(),
                                   (np.nanmin(eseq_mean), np.nanmax(eseq_mean)))


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning")
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
        simplot.plot_occup(chrom="chr1", dataset=dataset, cmap="viridis",
                           show=False)
        assert simplot.cmap == "OrRd"


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning")
class TestPositionalDatasetArg:
    def test_plot_occup_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=20)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, batch_size=2)

        SimPlot().plot_occup(dataset, chrom="chr1", show=False)

        assert plt.gcf().axes[0].images

    def test_plot_energy_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_energy(dataset, chrom="chr1", mol=0, run=0,
                              show=False)

        assert plt.gcf().axes

    def test_plot_nuc_pos_accepts_dataset_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_nuc_pos(dataset, chrom="chr1", mol=0, run=0,
                               show=False)

        assert plt.gcf().axes


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning")
class TestPlotNucPosCmapOverride:
    def test_cmap_override_does_not_mutate_instance(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        simplot = SimPlot()
        simplot.plot_nuc_pos(dataset, chrom="chr1", mol=0, run=0,
                             cmap="viridis", show=False)
        assert simplot.cmap == "OrRd"


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning")
class TestPlotNucPosEseq:
    def test_plot_eseq_handles_nan_from_missing_coverage(self, tmp_path):
        # A genomic position with no coverage (NaN in meth_prob) propagates
        # NaN into eseq for that molecule; the y-limit calculation must
        # tolerate that instead of crashing.
        rng = np.random.default_rng(5)
        meth_prob = rng.random((2, 20))
        meth_prob[:, 5] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=2, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "dataset",
                              seed=5)
        assert np.isnan(dataset.eseq["chr1"][:, 5]).all()

        SimPlot().plot_nuc_pos(dataset, chrom="chr1", mol=0, run=0,
                               plot_eseq=True, show=False)

    def test_plot_eseq_skips_ylim_when_all_nan(self, tmp_path):
        rng = np.random.default_rng(6)
        meth_prob = rng.random((2, 20))
        meth_prob[0, :] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=2, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "dataset",
                              seed=6)
        assert np.isnan(dataset.eseq["chr1"][0, :]).all()

        SimPlot().plot_nuc_pos(dataset, chrom="chr1", mol=0, run=0,
                               plot_eseq=True, show=False)

        seq_ax = plt.gcf().axes[2]
        assert np.isfinite(seq_ax.get_ylim()).all()

    def test_plot_eseq_ylim_matches_full_data_range(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nbp=20)

        SimPlot().plot_nuc_pos(dataset, chrom="chr1", mol=0, run=0,
                               plot_eseq=True, show=False)

        seq_ax = plt.gcf().axes[2]
        eseq = dataset.eseq["chr1"][0, :]
        np.testing.assert_allclose(seq_ax.get_ylim(),
                                   (np.nanmin(eseq), np.nanmax(eseq)))
