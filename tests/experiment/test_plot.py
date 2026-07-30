# test_plot.py

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from nucmc.h5_array import H5Array
from nucmc.experiment.plot import MethPlot, _downsample_array


def _h5array(values):
    arr = H5Array.create(values.shape, dtype=values.dtype)
    arr.write_batch(0, values.shape[0], values)
    return arr


@pytest.fixture
def methplot():
    return MethPlot()


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
@pytest.mark.parametrize("nrow,max_rows", [(5, 2000), (50, 5)])
def test_plot_methmap_smoke_across_input_types(
        methplot, tmp_path, kind, nrow, max_rows):
    values = np.random.default_rng(0).random((nrow, 4))
    if kind == "h5array":
        data = _h5array(values)
    elif kind == "dataframe":
        data = pd.DataFrame(values)
    else:
        data = values

    out_file = tmp_path / "plot.png"
    methplot.plot_methmap(data, max_rows=max_rows, out_file=out_file,
                          show=False)
    assert out_file.exists()


@pytest.mark.parametrize("how", ["mean", "sum", "min", "max", "stride"])
def test_plot_methmap_h5array_and_array_paths_agree(how):
    values = np.random.default_rng(1).random((23, 3))
    max_rows = 4

    h5_matrix = _h5array(values).downsample(max_rows, how=how)
    arr_matrix = _downsample_array(values, max_rows, how)
    df_matrix = _downsample_array(pd.DataFrame(values).to_numpy(),
                                  max_rows, how)

    np.testing.assert_allclose(h5_matrix, arr_matrix)
    np.testing.assert_allclose(h5_matrix, df_matrix)


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_plot_methmap_invalid_how_raises(methplot, kind):
    values = np.random.default_rng(2).random((5, 3))
    data = {"h5array": lambda: _h5array(values),
           "dataframe": lambda: pd.DataFrame(values),
           "ndarray": lambda: values}[kind]()
    with pytest.raises(ValueError):
        methplot.plot_methmap(data, how="bogus", show=False)


def test_plot_methmap_end_to_end_with_to_dense(tmp_path):
    from nucmc.experiment.methdata import MethPrintData, MethPrintExperiment

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
    MethPlot().plot_methmap(dense, max_rows=10, out_file=out_file,
                            show=False)
    assert out_file.exists()
