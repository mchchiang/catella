# test_plot.py

import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

from catella.h5_array import H5Array
from catella.experiment.plot import MethPlot
from catella.experiment.methdata import MethPrintData, MethPrintExperiment
from catella.experiment.preprocessing import MethPrintAnalysis
from catella import utils


def _h5array(values):
    arr = H5Array.create(values.shape, dtype=values.dtype)
    arr.write_batch(0, values.shape[0], values)
    return arr


@pytest.fixture
def methplot():
    return MethPlot()


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_plot_methmap_smoke_across_input_types(methplot, tmp_path, kind):
    values = np.random.default_rng(0).random((5, 4))
    if kind == "h5array":
        data = _h5array(values)
    elif kind == "dataframe":
        data = pd.DataFrame(values)
    else:
        data = values

    out_file = tmp_path / "plot.png"
    methplot.plot_methmap(data, out_file=out_file, show=False)
    assert out_file.exists()


def test_plot_methmap_end_to_end_with_to_dense(tmp_path):
    nbp, nmol = 5, 30
    rng = np.random.default_rng(3)
    rows = [(m, p, float(rng.random())) for m in range(nmol)
           for p in rng.choice(nbp, size=3, replace=False)]
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "mod_qual"])
    df["strand"] = "+"
    df["mod_code"] = 0
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)

    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, test_mol_id=mol_id, test_data=df,
        meth_mol_id=None, meth_data=None,
        unmeth_mol_id=None, unmeth_data=None)
    exp = MethPrintExperiment._create(_raw_data={"chr1": raw})

    dense = exp.to_dense("chr1")
    out_file = tmp_path / "methmap.png"
    MethPlot().plot_methmap(dense, out_file=out_file, show=False)
    assert out_file.exists()


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_plot_methmap_with_link_mat_draws_dendrogram(
        methplot, tmp_path, kind):
    values = np.random.default_rng(4).random((8, 3))
    if kind == "h5array":
        data = _h5array(values)
    elif kind == "dataframe":
        data = pd.DataFrame(values)
    else:
        data = values

    order, link_mat = utils.compute_linkage(values)
    sorted_values = values[order]

    out_file = tmp_path / "dendro.png"
    methplot.plot_methmap(sorted_values, link_mat=link_mat,
                          out_file=out_file, show=False)
    assert out_file.exists()


def test_plot_methmap_end_to_end_with_sort_by_linkage(tmp_path):
    # size > nbp/2 guarantees any two molecules' covered positions
    # overlap (pigeonhole), avoiding the zero-overlap nan-distance
    # edge case for this smoke test.
    nbp, nmol = 10, 10
    rng = np.random.default_rng(5)
    rows = [(m, p, "+", float(rng.random()), 0) for m in range(nmol)
           for p in rng.choice(nbp, size=7, replace=False)]
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                     "mod_qual", "mod_code"])
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)

    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, test_mol_id=mol_id, test_data=df,
        meth_mol_id=None, meth_data=None,
        unmeth_mol_id=None, unmeth_data=None)
    exp = MethPrintExperiment._create(_raw_data={"chr1": raw})

    ana = MethPrintAnalysis()
    link_mats = ana.sort_by_linkage(exp=exp, raw_which="test",
                                    batch_size=4)
    sorted_arr = exp.analysis["chr1"]["test_sorted"]

    out_file = tmp_path / "sorted_methmap.png"
    MethPlot().plot_methmap(sorted_arr, link_mat=link_mats["chr1"],
                            out_file=out_file, show=False)
    assert out_file.exists()


def test_plot_methmap_warns_above_plot_warn_rows(methplot, monkeypatch):
    import catella.experiment.plot as plot_module
    monkeypatch.setattr(plot_module, "_PLOT_WARN_ROWS", 5)
    values = np.random.default_rng(6).random((8, 3))

    with pytest.warns(UserWarning):
        methplot.plot_methmap(values, show=False)


def test_plot_methmap_no_warning_below_plot_warn_rows(methplot,
                                                       monkeypatch):
    import catella.experiment.plot as plot_module
    monkeypatch.setattr(plot_module, "_PLOT_WARN_ROWS", 100)
    values = np.random.default_rng(7).random((8, 3))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        methplot.plot_methmap(values, show=False)
    assert not any(issubclass(w.category, UserWarning) for w in caught)
