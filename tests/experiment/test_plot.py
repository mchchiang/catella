# test_plot.py

import matplotlib

matplotlib.use("Agg")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
from catella import utils
from catella.experiment.methdata import MethPrintData, MethPrintExperiment
from catella.experiment.plot import MethPlot
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.h5_array import H5Array
from scipy.special import logit


def _h5array(values):
    arr = H5Array.create(values.shape, dtype=values.dtype)
    arr.write_batch(0, values.shape[0], values)
    return arr


@pytest.fixture
def methplot():
    return MethPlot()


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_plot_meth_prob_smoke_across_input_types(methplot, tmp_path, kind):
    values = np.random.default_rng(0).random((5, 4))
    if kind == "h5array":
        data = _h5array(values)
    elif kind == "dataframe":
        data = pd.DataFrame(values)
    else:
        data = values

    out_file = tmp_path / "plot.png"
    methplot.plot_meth_prob(data, out_file=out_file, show=False)
    assert out_file.exists()


def _raw_test_exp(nbp=5, nmol=30, seed=3):
    rng = np.random.default_rng(seed)
    rows = [
        (m, p, float(rng.random()))
        for m in range(nmol)
        for p in rng.choice(nbp, size=3, replace=False)
    ]
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "mod_qual"])
    df["strand"] = "+"
    df["mod_code"] = 0
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)

    raw = MethPrintData._create(
        chrom="chr1",
        nbp=nbp,
        test_mol_id=mol_id,
        test_data=df,
        meth_mol_id=None,
        meth_data=None,
        unmeth_mol_id=None,
        unmeth_data=None,
    )
    return MethPrintExperiment._create(_raw_data={"chr1": raw})


def test_plot_meth_prob_end_to_end_with_to_dense(tmp_path):
    exp = _raw_test_exp()
    dense = exp.to_dense("chr1")
    out_file = tmp_path / "methmap.png"
    MethPlot().plot_meth_prob(dense, out_file=out_file, show=False)
    assert out_file.exists()


def test_plot_meth_prob_with_raw_which(tmp_path):
    exp = _raw_test_exp()
    out_file = tmp_path / "methmap_raw_which.png"
    MethPlot().plot_meth_prob(
        exp=exp, chroms="chr1", raw_which="test", out_file=out_file, show=False
    )
    assert out_file.exists()


def test_plot_meth_prob_raises_without_data_or_raw_which(methplot):
    with pytest.raises(ValueError):
        methplot.plot_meth_prob(show=False)


def test_plot_meth_prob_raises_when_data_and_exp_combined(methplot):
    exp = _raw_test_exp()
    values = np.random.default_rng(1).random((5, 4))
    with pytest.raises(ValueError):
        methplot.plot_meth_prob(
            values, exp=exp, chroms="chr1", raw_which="test", show=False
        )


def test_plot_meth_prob_mask_name_forwarded(tmp_path):
    nbp, nmol = 5, 6
    exp = _raw_test_exp(nbp=nbp, nmol=nmol)
    keep = np.ones(nmol, dtype=bool)
    keep[0] = False
    exp.analysis["chr1"]["test_qc"] = pd.DataFrame({"keep": keep})

    out_file = tmp_path / "methmap_mask_name.png"
    MethPlot().plot_meth_prob(
        exp=exp,
        chroms="chr1",
        raw_which="test",
        mask_name="qc",
        out_file=out_file,
        show=False,
    )
    assert out_file.exists()


def test_plot_meth_prob_max_rows_downsamples(methplot, monkeypatch):
    monkeypatch.setattr("catella._plotting._RENDER_WARN_BYTES", 1000)
    values = np.random.default_rng(2).random((20, 3))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        methplot.plot_meth_prob(values, max_rows=4, show=False)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


def test_plot_meth_prob_raw_which_closes_scratch_file(tmp_path, monkeypatch):
    exp = _raw_test_exp()
    orig_to_dense = MethPrintExperiment.to_dense
    paths = []

    def spy_to_dense(self, *args, **kwargs):
        arr = orig_to_dense(self, *args, **kwargs)
        paths.append(arr.path)
        return arr

    monkeypatch.setattr(MethPrintExperiment, "to_dense", spy_to_dense)

    out_file = tmp_path / "methmap_scratch.png"
    MethPlot().plot_meth_prob(
        exp=exp, chroms="chr1", raw_which="test", out_file=out_file, show=False
    )
    assert out_file.exists()
    assert len(paths) == 1
    assert not Path(paths[0]).exists()


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_plot_meth_prob_with_link_mat_draws_dendrogram(
    methplot, tmp_path, kind
):
    values = np.random.default_rng(4).random((8, 3))
    order, link_mat = utils.compute_linkage(values)
    sorted_values = values[order]
    if kind == "h5array":
        data = _h5array(sorted_values)
    elif kind == "dataframe":
        data = pd.DataFrame(sorted_values)
    else:
        data = sorted_values

    out_file = tmp_path / "dendro.png"
    methplot.plot_meth_prob(
        data, link_mat=link_mat, out_file=out_file, show=False
    )
    assert out_file.exists()


def test_plot_meth_prob_end_to_end_with_sort_by_linkage(tmp_path):
    # size > nbp/2 guarantees any two molecules' covered positions
    # overlap (pigeonhole), avoiding the zero-overlap nan-distance
    # edge case for this smoke test.
    nbp, nmol = 10, 10
    rng = np.random.default_rng(5)
    rows = [
        (m, p, "+", float(rng.random()), 0)
        for m in range(nmol)
        for p in rng.choice(nbp, size=7, replace=False)
    ]
    df = pd.DataFrame(
        rows, columns=["mol_index", "pos", "strand", "mod_qual", "mod_code"]
    )
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)

    raw = MethPrintData._create(
        chrom="chr1",
        nbp=nbp,
        test_mol_id=mol_id,
        test_data=df,
        meth_mol_id=None,
        meth_data=None,
        unmeth_mol_id=None,
        unmeth_data=None,
    )
    exp = MethPrintExperiment._create(_raw_data={"chr1": raw})

    ana = MethPrintAnalysis()
    link_mats = ana.sort_by_linkage(exp=exp, raw_which="test", batch_size=4)

    out_file = tmp_path / "sorted_methmap.png"
    MethPlot().plot_meth_prob(
        exp=exp,
        chroms="chr1",
        raw_which="test",
        link_mat=link_mats["chr1"],
        out_file=out_file,
        show=False,
    )
    assert out_file.exists()
    dense = exp.to_dense("chr1", which="test").to_numpy()
    order = sch.leaves_list(link_mats["chr1"])
    image = plt.gcf().axes[0].images[0].get_array()
    np.testing.assert_allclose(
        np.ma.filled(image, np.nan), dense[order], equal_nan=True
    )


def test_plot_meth_prob_link_mat_orders_data_rows(methplot):
    values = np.random.default_rng(4).random((6, 5))
    _, link_mat = utils.compute_linkage(values)
    methplot.plot_meth_prob(values, link_mat=link_mat, show=False)
    image = plt.gcf().axes[0].images[0].get_array()
    order = sch.leaves_list(link_mat)
    np.testing.assert_allclose(np.ma.getdata(image), values[order])
    assert plt.gcf().axes[0].get_ylabel() == "Row (leaf order)"


def test_plot_meth_prob_link_mat_size_mismatch_raises(methplot):
    _, link_mat = utils.compute_linkage(np.random.rand(4, 3))
    with pytest.raises(ValueError, match="4 leaves but the data has 6"):
        methplot.plot_meth_prob(
            np.random.rand(6, 3), link_mat=link_mat, show=False
        )


def test_plot_meth_prob_with_nan_values(methplot, tmp_path):
    values = np.random.default_rng(8).random((5, 4))
    values[0, 0] = np.nan
    values[2, :] = np.nan

    out_file = tmp_path / "nan_methmap.png"
    methplot.plot_meth_prob(values, out_file=out_file, show=False)
    assert out_file.exists()


def test_plot_meth_prob_cmap_override_does_not_mutate_instance(
    methplot, tmp_path
):
    values = np.random.default_rng(9).random((5, 4))

    out_file = tmp_path / "override_methmap.png"
    methplot.plot_meth_prob(
        values, cmap="viridis", out_file=out_file, show=False
    )
    assert out_file.exists()
    assert methplot.cmap == "OrRd"


def test_plot_meth_prob_warns_above_render_limit(methplot, monkeypatch):
    monkeypatch.setattr("catella._plotting._RENDER_WARN_BYTES", 1000)
    values = np.random.default_rng(6).random((8, 3))

    with pytest.warns(UserWarning, match="8 rows x 3 bp"):
        methplot.plot_meth_prob(values, show=False)


def test_plot_meth_prob_no_warning_below_render_limit(methplot, monkeypatch):
    monkeypatch.setattr("catella._plotting._RENDER_WARN_BYTES", 10**4)
    values = np.random.default_rng(7).random((8, 3))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        methplot.plot_meth_prob(values, show=False)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_plot_meth_energy_smoke_across_input_types(methplot, tmp_path, kind):
    values = np.random.default_rng(10).random((5, 4))
    if kind == "h5array":
        data = _h5array(values)
    elif kind == "dataframe":
        data = pd.DataFrame(values)
    else:
        data = values

    out_file = tmp_path / "energymap.png"
    methplot.plot_meth_energy(data, out_file=out_file, show=False)
    assert out_file.exists()


def _drawn(methplot, values, **kwargs):
    methplot.plot_meth_energy(values, show=False, **kwargs)
    fig = plt.gcf()
    image = fig.axes[0].images[0]
    return {
        "data": np.ma.getdata(image.get_array()),
        "vmin": image.norm.vmin,
        "vmax": image.norm.vmax,
        "cmap": image.cmap.name,
        "cbar_label": fig.axes[-1].get_ylabel(),
    }


def test_plot_meth_energy_transforms_to_logit(methplot):
    values = np.random.default_rng(11).random((4, 3))
    drawn = _drawn(methplot, values)
    np.testing.assert_allclose(drawn["data"], logit(values))


def test_plot_meth_energy_default_vmin_vmax_symmetric(methplot):
    values = np.random.default_rng(12).random((4, 3))
    drawn = _drawn(methplot, values)
    assert drawn["vmin"] == -drawn["vmax"]
    assert drawn["vmax"] == pytest.approx(np.nanmax(np.abs(logit(values))))


def test_plot_meth_energy_explicit_vmin_vmax_passthrough(methplot):
    values = np.random.default_rng(13).random((4, 3))
    drawn = _drawn(methplot, values, vmin=-5, vmax=5)
    assert (drawn["vmin"], drawn["vmax"]) == (-5, 5)


def test_plot_meth_energy_asymmetric_vmin_vmax_centered_at_zero(methplot):
    values = np.random.default_rng(17).random((4, 3))
    drawn = _drawn(methplot, values, vmin=-3, vmax=10)
    assert (drawn["vmin"], drawn["vmax"]) == (-10, 10)

    drawn = _drawn(methplot, values, vmax=7)
    assert (drawn["vmin"], drawn["vmax"]) == (-7, 7)


def test_plot_meth_energy_emax_clamps_and_sets_default_range(methplot):
    values = np.random.default_rng(14).random((4, 3))
    drawn = _drawn(methplot, values, emax=1.0)
    assert (drawn["vmin"], drawn["vmax"]) == (-1.0, 1.0)
    assert np.all(drawn["data"] >= -1.0) and np.all(drawn["data"] <= 1.0)


def test_plot_meth_energy_extreme_probabilities_do_not_break_range(methplot):
    values = np.array([[0.0, 0.5], [1.0, 0.5]])
    drawn = _drawn(methplot, values)
    assert np.isfinite(drawn["vmin"]) and np.isfinite(drawn["vmax"])


def test_plot_meth_energy_nan_passthrough(methplot):
    values = np.random.default_rng(15).random((4, 3))
    values[0, 0] = np.nan
    drawn = _drawn(methplot, values)
    assert np.isnan(drawn["data"][0, 0])


def test_plot_meth_energy_default_cmap_and_label(methplot):
    values = np.random.default_rng(16).random((4, 3))
    drawn = _drawn(methplot, values)
    assert drawn["cmap"] == "RdBu_r"
    assert drawn["cbar_label"] == r"Energy [$k_BT$]"

    drawn = _drawn(methplot, values, cmap="viridis", cbar_label="custom")
    assert drawn["cmap"] == "viridis"
    assert drawn["cbar_label"] == "custom"


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write(
            "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"
        )
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        f.writelines(f"{chrom}\t{length}\n" for chrom, length in sizes.items())


def _write_fasta(path, records):
    with open(path, "w") as f:
        f.writelines(f">{chrom}\n{seq}\n" for chrom, seq in records.items())


def _dropout_fractions_exp(tmp_path):
    # refseq "AAAAAAAAAACG": 10 A-sites, 1 CG-site. Two molecules
    # with different coverage, so groups have >1 value to sort.
    chrom, refseq = "chr1", "AAAAAAAAAACG"
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {chrom: len(refseq)})
    fasta = tmp_path / "ref.fa"
    _write_fasta(fasta, {chrom: refseq})
    test_file = tmp_path / "test.tsv"
    rows = [("m0", i, chrom, "+", 0.5, "a") for i in range(9)] + [
        ("m1", i, chrom, "+", 0.5, "a") for i in range(5)
    ]
    _write_tsv(test_file, rows)
    return MethPrintExperiment.load_raw(
        chromsize=chromsize,
        test_file=test_file,
        fasta_file=fasta,
        mtase=["A", "CG"],
    )


def _two_chrom_exp(tmp_path):
    refseq = "AAAAAAAAAACG"
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {"chr1": len(refseq), "chr2": len(refseq)})
    fasta = tmp_path / "ref.fa"
    _write_fasta(fasta, {"chr1": refseq, "chr2": refseq})
    test_file = tmp_path / "test.tsv"
    rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(9)] + [
        ("m1", i, "chr2", "+", 0.5, "a") for i in range(5)
    ]
    _write_tsv(test_file, rows)
    return MethPrintExperiment.load_raw(
        chromsize=chromsize,
        test_file=test_file,
        fasta_file=fasta,
        mtase=["A", "CG"],
    )


def _legend_labels():
    return plt.gcf().axes[0].get_legend_handles_labels()[1]


class TestPlotDropoutEcdf:
    def test_end_to_end_smoke(self, tmp_path):
        exp = _dropout_fractions_exp(tmp_path)
        fracs = exp.dropout_fractions()

        out_file = tmp_path / "dropout_ecdf.png"
        MethPlot().plot_dropout_ecdf(
            fracs, "chr1", out_file=out_file, show=False
        )
        assert out_file.exists()
        exp.close()

    def test_which_filter_narrows_lines(self, tmp_path):
        exp = _dropout_fractions_exp(tmp_path)
        fracs = exp.dropout_fractions()

        out_file = tmp_path / "dropout_ecdf_which.png"
        MethPlot().plot_dropout_ecdf(
            fracs, "chr1", which="test", out_file=out_file, show=False
        )
        assert out_file.exists()
        exp.close()

    def test_unknown_chrom_raises(self, tmp_path):
        exp = _dropout_fractions_exp(tmp_path)
        fracs = exp.dropout_fractions()

        with pytest.raises(ValueError):
            MethPlot().plot_dropout_ecdf(fracs, "bogus", show=False)
        exp.close()

    def test_unknown_which_raises(self, tmp_path):
        exp = _dropout_fractions_exp(tmp_path)
        fracs = exp.dropout_fractions()

        with pytest.raises(ValueError):
            MethPlot().plot_dropout_ecdf(
                fracs, "chr1", which="meth", show=False
            )
        exp.close()

    def test_multi_chrom_legend_shows_varying_fields(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions(channels=["A"])

        MethPlot().plot_dropout_ecdf(fracs, ["chr1", "chr2"], show=False)
        assert _legend_labels() == ["chr1", "chr2"]
        assert plt.gcf().axes[0].get_title() == "test/A"
        plt.close("all")
        exp.close()

    def test_single_chrom_legend_unchanged_fields(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions(chroms=["chr1"])

        MethPlot().plot_dropout_ecdf(fracs, "chr1", show=False)
        assert _legend_labels() == ["A", "CG", "all"]
        assert plt.gcf().axes[0].get_title() == "chr1/test"
        plt.close("all")
        exp.close()

    def test_unknown_chrom_in_list_raises(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions()

        with pytest.raises(ValueError):
            MethPlot().plot_dropout_ecdf(
                fracs, ["chr1", "bogus"], show=False
            )
        exp.close()

    def test_empty_chroms_raises(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions()

        with pytest.raises(ValueError):
            MethPlot().plot_dropout_ecdf(fracs, [], show=False)
        exp.close()

    def test_pooled_false_omits_all(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions(chroms=["chr1"])

        MethPlot().plot_dropout_ecdf(fracs, "chr1", pooled=False, show=False)
        assert _legend_labels() == ["A", "CG"]
        plt.close("all")
        exp.close()

    def test_pooled_false_single_channel_still_plots(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions(chroms=["chr1"], channels=["A"])

        out_file = tmp_path / "single.png"
        MethPlot().plot_dropout_ecdf(
            fracs, "chr1", pooled=False, out_file=out_file, show=False
        )
        assert out_file.exists()
        plt.close("all")
        exp.close()

    def test_split_by_channel_one_panel_per_channel(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions()

        MethPlot().plot_dropout_ecdf(
            fracs, ["chr1", "chr2"], split_by="channel", show=False
        )
        visible = [ax for ax in plt.gcf().axes if ax.get_visible()]
        assert [ax.get_title() for ax in visible] == ["A", "CG", "all"]
        assert visible[0].get_legend_handles_labels()[1] == ["chr1", "chr2"]
        assert plt.gcf()._suptitle.get_text() == "test"
        plt.close("all")
        exp.close()

    def test_split_by_grid_hides_unused_panels(self, tmp_path):
        fracs = {
            (f"chr{i}", "test", "A"): np.array([0.1, 0.2])
            for i in range(5)
        }
        MethPlot().plot_dropout_ecdf(
            fracs, [k[0] for k in fracs], split_by="chrom", show=False
        )
        axes = plt.gcf().axes
        assert len([ax for ax in axes if ax.get_visible()]) == 5
        assert len([ax for ax in axes if not ax.get_visible()]) == 3
        plt.close("all")

    def test_more_than_10_lines_warns(self):
        fracs = {
            (f"chr{i}", "test", "A"): np.array([0.1, 0.2])
            for i in range(11)
        }
        with pytest.warns(UserWarning, match="colors repeat"):
            MethPlot().plot_dropout_ecdf(
                fracs, [k[0] for k in fracs], show=False
            )
        plt.close("all")

    def test_10_lines_does_not_warn(self):
        fracs = {
            (f"chr{i}", "test", "A"): np.array([0.1, 0.2])
            for i in range(10)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MethPlot().plot_dropout_ecdf(
                fracs, [k[0] for k in fracs], show=False
            )
        plt.close("all")

    def test_split_by_chrom_many_chroms_does_not_warn(self):
        fracs = {
            (f"chr{i}", "test", "A"): np.array([0.1, 0.2])
            for i in range(11)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MethPlot().plot_dropout_ecdf(
                fracs, [k[0] for k in fracs], split_by="chrom", show=False
            )
        plt.close("all")

    def test_invalid_split_by_raises(self, tmp_path):
        exp = _two_chrom_exp(tmp_path)
        fracs = exp.dropout_fractions()

        with pytest.raises(ValueError):
            MethPlot().plot_dropout_ecdf(
                fracs, "chr1", split_by="bogus", show=False
            )
        exp.close()

    def test_curve_is_monotonic_and_bounded(self, tmp_path):
        exp = _dropout_fractions_exp(tmp_path)
        fracs = exp.dropout_fractions()

        frac = np.sort(fracs[("chr1", "test", "A")])
        n = len(frac)
        ecdf_pct = 100 * np.arange(1, n + 1) / n
        assert np.all(np.diff(ecdf_pct) >= 0)
        assert ecdf_pct.min() >= 0
        assert ecdf_pct.max() <= 100
        exp.close()
