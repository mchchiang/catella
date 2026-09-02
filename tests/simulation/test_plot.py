# test_plot.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
