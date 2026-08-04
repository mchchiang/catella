# test_preprocessing.py

import numpy as np
import pandas as pd
import pytest

from nucmc.experiment.methdata import MethPrintData, MethPrintExperiment
from nucmc.experiment.preprocessing import MethPrintAnalysis
from nucmc.h5_array import H5Array
from nucmc import utils


def _make_mol_data(nmol, nbp, rng, unmapped_mol=None):
    rows = []
    for m in range(nmol):
        npos = rng.integers(max(1, nbp // 2), nbp + 1)
        positions = rng.choice(nbp, size=npos, replace=False)
        if unmapped_mol is not None and m == unmapped_mol:
            strand = "."
        else:
            strand = "+" if m % 2 == 0 else "-"
        for p in positions:
            rows.append((m, int(p), strand, float(rng.random()), 0))
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                     "mod_qual", "mod_code"])
    # object dtype matches what load_raw produces via Series.unique()
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)
    return mol_id, df


def _make_experiment(*, nmol=6, nbp=10, with_controls=True, seed=0,
                     unmapped_test_mol=None):
    rng = np.random.default_rng(seed)
    test_mol_id, test_data = _make_mol_data(
        nmol, nbp, rng, unmapped_mol=unmapped_test_mol)
    if with_controls:
        meth_mol_id, meth_data = _make_mol_data(nmol, nbp, rng)
        unmeth_mol_id, unmeth_data = _make_mol_data(nmol, nbp, rng)
    else:
        meth_mol_id = meth_data = unmeth_mol_id = unmeth_data = None

    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, test_mol_id=test_mol_id,
        test_data=test_data, meth_mol_id=meth_mol_id, meth_data=meth_data,
        unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_data)
    return MethPrintExperiment._create(_raw_data={"chr1": raw})


def _make_comparability_experiment(*, test_values, nbp=1):
    # Fixed (identical, non-random) meth/unmeth controls plus a given
    # test population, for testing that clip bounds derived from the
    # controls don't depend on what the test population looks like.
    cols = ["mol_index", "pos", "strand", "mod_qual", "mod_code"]
    meth_df = pd.DataFrame(
        [(m, 0, "+", v, 0)
         for m, v in enumerate([0.8, 0.9, 1.0, 0.95, 0.85])], columns=cols)
    unmeth_df = pd.DataFrame(
        [(m, 0, "+", v, 0)
         for m, v in enumerate([0.0, 0.05, 0.1, 0.02, 0.08])], columns=cols)
    test_df = pd.DataFrame(
        [(m, 0, "+", v, 0) for m, v in enumerate(test_values)], columns=cols)

    meth_mol_id = np.array([f"m{m}" for m in range(len(meth_df))],
                           dtype=object)
    unmeth_mol_id = np.array([f"u{m}" for m in range(len(unmeth_df))],
                             dtype=object)
    test_mol_id = np.array([f"t{m}" for m in range(len(test_values))],
                           dtype=object)

    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, test_mol_id=test_mol_id, test_data=test_df,
        meth_mol_id=meth_mol_id, meth_data=meth_df,
        unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_df)
    return MethPrintExperiment._create(_raw_data={"chr1": raw})


def _make_single_mol_experiment(*, positions, values, nbp):
    # One molecule with data at explicit positions/values, gaps elsewhere.
    # Deterministic layout for testing nan_method/fill_edge behavior.
    rows = [(0, p, "+", v, 0) for p, v in zip(positions, values)]
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                     "mod_qual", "mod_code"])
    mol_id = np.array(["mol0"], dtype=object)
    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, test_mol_id=mol_id, test_data=df,
        meth_mol_id=None, meth_data=None,
        unmeth_mol_id=None, unmeth_data=None)
    return MethPrintExperiment._create(_raw_data={"chr1": raw})


class TestSmooth:
    def test_output_shape(self):
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, batch_size=100)
        arr = exp.analysis["chr1"]["test_smoothed"]
        assert isinstance(arr, H5Array)
        assert arr.shape == (5, 8)

    def test_batching_matches_single_batch(self):
        exp_a = _make_experiment(nmol=7, nbp=12)
        exp_b = _make_experiment(nmol=7, nbp=12)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp_a, batch_size=1000)
        ana.smooth(binsize=3, exp=exp_b, batch_size=2)

        for key in ("test_smoothed", "meth_smoothed", "unmeth_smoothed"):
            a = exp_a.analysis["chr1"][key].to_numpy()
            b = exp_b.analysis["chr1"][key].to_numpy()
            np.testing.assert_array_equal(a, b)

    def test_nan_method_interior_only(self):
        # Valid data at pos 1, 2, 5; interior gap at 3, 4 (bounded by
        # valid data on both sides); edge gaps at 0 (leading) and 6
        # (trailing). binsize=1 keeps res identical to the raw pivot so
        # gap locations are exact.
        exp_mean = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        exp_interp = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=1, exp=exp_mean, nan_method="mean",
                  fill_edge=0.0, batch_size=100)
        ana.smooth(binsize=1, exp=exp_interp, nan_method="interpolate",
                  fill_edge=0.0, batch_size=100)
        row_mean = exp_mean.analysis["chr1"]["test_smoothed"].to_numpy()[0]
        row_interp = exp_interp.analysis["chr1"]["test_smoothed"].to_numpy()[0]

        # nan_method="mean" fills both interior gap positions identically
        col_mean = (0.0 + 0.4 + 1.0) / 3
        np.testing.assert_allclose(row_mean[3:5], [col_mean, col_mean])

        # nan_method="interpolate" fills them along the line from pos 2 to
        # pos 5
        np.testing.assert_allclose(row_interp[3:5], [0.6, 0.8])

        # fill_edge is identical across both nan_method choices
        np.testing.assert_allclose(row_mean[[0, 6]], [0.0, 0.0])
        np.testing.assert_allclose(row_interp[[0, 6]], [0.0, 0.0])

    def test_fill_edge_choices(self):
        make = lambda: _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        ana = MethPrintAnalysis()

        exp_nan = make()
        ana.smooth(binsize=1, exp=exp_nan, nan_method="interpolate",
                  fill_edge=np.nan, batch_size=100)
        row_nan = exp_nan.analysis["chr1"]["test_smoothed"].to_numpy()[0]
        assert np.isnan(row_nan[0]) and np.isnan(row_nan[6])

        exp_lit = make()
        ana.smooth(binsize=1, exp=exp_lit, nan_method="interpolate",
                  fill_edge=0.5, batch_size=100)
        row_lit = exp_lit.analysis["chr1"]["test_smoothed"].to_numpy()[0]
        np.testing.assert_allclose(row_lit[[0, 6]], [0.5, 0.5])

        exp_avg = make()
        ana.smooth(binsize=1, exp=exp_avg, nan_method="interpolate",
                  fill_edge="mean", batch_size=100)
        row_avg = exp_avg.analysis["chr1"]["test_smoothed"].to_numpy()[0]
        # Mean of all non-edge values after interior interpolation
        expected = np.mean([0.0, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(row_avg[[0, 6]], [expected, expected])

        # Interior values are unchanged by fill_edge
        for row in (row_nan, row_lit, row_avg):
            np.testing.assert_allclose(row[3:5], [0.6, 0.8])

    def test_nan_method_invalid_raises(self):
        exp = _make_experiment(nmol=3, nbp=6)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.smooth(binsize=2, exp=exp, nan_method="bogus",
                      batch_size=100)

    def test_fill_edge_out_of_range_raises(self):
        exp = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.smooth(binsize=1, exp=exp, fill_edge=1.5, batch_size=100)


class TestMethProb:
    def test_output_in_unit_range(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.meth_prob(exp=exp, binsize=3, batch_size=100,
                      percentile_sample_size=1000, fill_edge="mean")
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (6, 10)
        assert np.all(prob >= 0.0) and np.all(prob <= 1.0)

    @pytest.mark.parametrize("norm_by_strand", [False, True])
    def test_batching_matches_single_batch(self, norm_by_strand):
        exp_a = _make_experiment(nmol=8, nbp=10)
        exp_b = _make_experiment(nmol=8, nbp=10)
        ana = MethPrintAnalysis()
        ana.meth_prob(exp=exp_a, binsize=3, batch_size=1000,
                      percentile_sample_size=1000,
                      norm_by_strand=norm_by_strand, seed=0)
        ana.meth_prob(exp=exp_b, binsize=3, batch_size=2,
                      percentile_sample_size=1000,
                      norm_by_strand=norm_by_strand, seed=0)
        a = exp_a.analysis["chr1"]["meth_prob"].to_numpy()
        b = exp_b.analysis["chr1"]["meth_prob"].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_no_controls_skips_normalization(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        ana = MethPrintAnalysis()
        ana.meth_prob(exp=exp, binsize=2, batch_size=100,
                      percentile_sample_size=1000, fill_edge="mean")
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (4, 6)
        assert np.all(prob >= 0.0) and np.all(prob <= 1.0)

    def test_clip_bounds_independent_of_test_condition(self):
        # Same (fixed) meth/unmeth controls, but different test
        # populations -- one narrow, one wide -- sharing one common
        # "probe" value. Since clip bounds should come from the
        # controls (not the test signal's own range), the probe must
        # map to the same probability in both, regardless of what other
        # test molecules are present.
        probe = 0.5
        narrow_values = [0.48, probe, 0.52]
        wide_values = [probe, 0.6, 0.7, 0.8, 0.9]

        exp_narrow = _make_comparability_experiment(test_values=narrow_values)
        exp_wide = _make_comparability_experiment(test_values=wide_values)

        ana = MethPrintAnalysis()
        ana.meth_prob(exp=exp_narrow, binsize=1, batch_size=100,
                      percentile_sample_size=1000)
        ana.meth_prob(exp=exp_wide, binsize=1, batch_size=100,
                      percentile_sample_size=1000)

        prob_narrow = exp_narrow.analysis["chr1"]["meth_prob"].to_numpy()
        prob_wide = exp_wide.analysis["chr1"]["meth_prob"].to_numpy()

        idx_narrow = narrow_values.index(probe)
        idx_wide = wide_values.index(probe)
        np.testing.assert_allclose(prob_narrow[idx_narrow, 0],
                                   prob_wide[idx_wide, 0])

    def test_percentile_ignores_nan_in_control_channel(self):
        # unmeth mol 0 is missing pos 0, leaving it NaN after smoothing
        # (fill_edge defaults to np.nan). That NaN must not poison the
        # percentile-based vmin/vmax (and thus every probability).
        cols = ["mol_index", "pos", "strand", "mod_qual", "mod_code"]
        unmeth_df = pd.DataFrame([
            (0, 1, "+", 0.05, 0), (0, 2, "+", 0.06, 0),
            (1, 0, "+", 0.10, 0), (1, 1, "+", 0.11, 0),
            (1, 2, "+", 0.12, 0),
        ], columns=cols)
        meth_df = pd.DataFrame([
            (0, 0, "+", 0.90, 0), (0, 1, "+", 0.91, 0), (0, 2, "+", 0.92, 0),
            (1, 0, "+", 0.95, 0), (1, 1, "+", 0.94, 0), (1, 2, "+", 0.93, 0),
        ], columns=cols)
        test_df = pd.DataFrame([
            (0, 0, "+", 0.50, 0), (0, 1, "+", 0.50, 0), (0, 2, "+", 0.50, 0),
        ], columns=cols)

        unmeth_mol_id = np.array(["u0", "u1"], dtype=object)
        meth_mol_id = np.array(["m0", "m1"], dtype=object)
        test_mol_id = np.array(["t0"], dtype=object)

        raw = MethPrintData._create(
            chrom="chr1", nbp=3, test_mol_id=test_mol_id, test_data=test_df,
            meth_mol_id=meth_mol_id, meth_data=meth_df,
            unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_df)
        exp = MethPrintExperiment._create(_raw_data={"chr1": raw})

        ana = MethPrintAnalysis()
        ana.meth_prob(exp=exp, binsize=1, batch_size=100,
                      percentile_sample_size=1000)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        # Positions 1 and 2 are fully covered by both control molecules
        # and must not be NaN-poisoned by the gap at position 0.
        assert not np.any(np.isnan(prob[:, 1:]))

    def test_norm_by_strand_rejects_unmapped_strand(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=True,
                               unmapped_test_mol=0)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.meth_prob(exp=exp, binsize=2, batch_size=100,
                          norm_by_strand=True)

    def test_nan_method_forwarded_to_lazy_smooth(self):
        exp_lazy = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        exp_direct = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        ana = MethPrintAnalysis()

        ana.meth_prob(exp=exp_lazy, binsize=1, nan_method="interpolate",
                      fill_edge=0, batch_size=100,
                      percentile_sample_size=1000)
        ana.smooth(binsize=1, exp=exp_direct, nan_method="interpolate",
                  fill_edge=0, batch_size=100)

        lazy = exp_lazy.analysis["chr1"]["test_smoothed"].to_numpy()
        direct = exp_direct.analysis["chr1"]["test_smoothed"].to_numpy()
        np.testing.assert_allclose(lazy, direct)


class TestSaveLoadRoundTrip:
    def test_h5array_analysis_round_trips(self, tmp_path):
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2)
        ana.meth_prob(exp=exp, binsize=2, batch_size=2,
                      percentile_sample_size=1000)

        expected = exp.analysis["chr1"]["meth_prob"].to_numpy()
        out_file = tmp_path / "experiment.h5"
        exp.save(out_file)

        loaded = MethPrintExperiment.load(out_file)
        arr = loaded.analysis["chr1"]["meth_prob"]
        assert isinstance(arr, H5Array)
        np.testing.assert_array_equal(arr.to_numpy(), expected)

    def test_save_rejects_same_path_as_backing_h5array(self, tmp_path):
        exp = _make_experiment(nmol=3, nbp=4)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2)

        out_file = tmp_path / "experiment.h5"
        exp.save(out_file)
        loaded = MethPrintExperiment.load(out_file)
        with pytest.raises(ValueError):
            loaded.save(out_file)


class TestSortByLinkage:
    def test_default_falls_back_to_raw_test_before_smoothing(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()

        link_mats = ana.sort_by_linkage(exp=exp, batch_size=2)
        sorted_arr = exp.analysis["chr1"]["test_sorted"]
        assert isinstance(sorted_arr, H5Array)
        assert sorted_arr.shape == (6, 10)
        assert "chr1" in link_mats

    def test_default_prefers_smoothed_once_available(self):
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2, fill_edge="mean")

        ana.sort_by_linkage(exp=exp, batch_size=2)
        assert "test_smoothed_sorted" in exp.analysis["chr1"]
        assert "test_sorted" not in exp.analysis["chr1"]

    def test_default_handles_unfilled_smoothing_edge_nan(self):
        # smooth()'s default fill_edge=np.nan leaves trailing-edge nan;
        # sort_by_linkage must not choke on it.
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, batch_size=2)

        ana.sort_by_linkage(exp=exp, batch_size=2)
        assert "test_smoothed_sorted" in exp.analysis["chr1"]

    def test_meth_prob_data_name_works(self):
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2, fill_edge="mean")
        ana.meth_prob(exp=exp, binsize=2, batch_size=2,
                      percentile_sample_size=1000)

        ana.sort_by_linkage(exp=exp, data_name="meth_prob", batch_size=2)
        sorted_arr = exp.analysis["chr1"]["meth_prob_sorted"]
        assert isinstance(sorted_arr, H5Array)
        assert sorted_arr.shape == (5, 8)

    def test_raw_which_works_without_smoothing(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()

        ana.sort_by_linkage(exp=exp, raw_which="meth", batch_size=2)
        assert "meth_sorted" in exp.analysis["chr1"]

    def test_raw_which_takes_priority_over_data_name(self):
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()

        ana.sort_by_linkage(exp=exp, raw_which="test",
                            data_name="does_not_exist", batch_size=2)
        assert "test_sorted" in exp.analysis["chr1"]

    def test_missing_non_default_data_name_raises_key_error(self):
        exp = _make_experiment(nmol=4, nbp=6)
        ana = MethPrintAnalysis()
        with pytest.raises(KeyError):
            ana.sort_by_linkage(exp=exp, data_name="meth_prob")

    def test_custom_sorted_name(self):
        exp = _make_experiment(nmol=4, nbp=6)
        ana = MethPrintAnalysis()
        ana.sort_by_linkage(exp=exp, sorted_name="custom", batch_size=2)
        assert "custom" in exp.analysis["chr1"]

    def test_sorted_rows_are_a_permutation_of_original_rows(self):
        exp = _make_experiment(nmol=6, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2, fill_edge="mean")
        original = exp.analysis["chr1"]["test_smoothed"].to_numpy()

        ana.sort_by_linkage(exp=exp, batch_size=2)
        sorted_arr = exp.analysis["chr1"]["test_smoothed_sorted"].to_numpy()

        orig_rows = sorted(map(tuple, original.tolist()))
        got_rows = sorted(map(tuple, sorted_arr.tolist()))
        assert orig_rows == got_rows

    def test_works_with_legacy_dataframe_source(self):
        exp = _make_experiment(nmol=5, nbp=6)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2, fill_edge="mean")
        smoothed = exp.analysis["chr1"]["test_smoothed"].to_numpy()
        exp.analysis["chr1"]["test_df"] = pd.DataFrame(smoothed)

        ana.sort_by_linkage(exp=exp, data_name="test_df", batch_size=2)
        sorted_df = exp.analysis["chr1"]["test_df_sorted"]
        order, _ = utils.compute_linkage(smoothed)
        assert isinstance(sorted_df, pd.DataFrame)
        np.testing.assert_allclose(sorted_df.to_numpy(), smoothed[order])

    def test_fill_nan_avoids_crash_on_all_nan_row(self):
        exp = _make_experiment(nmol=5, nbp=6)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=2, fill_edge="mean")
        smoothed = exp.analysis["chr1"]["test_smoothed"].to_numpy()
        smoothed[0, :] = np.nan
        exp.analysis["chr1"]["test_nan"] = pd.DataFrame(smoothed)

        with pytest.raises(ValueError):
            ana.sort_by_linkage(exp=exp, data_name="test_nan", batch_size=2)

        link_mats = ana.sort_by_linkage(exp=exp, data_name="test_nan",
                                        batch_size=2, fill_nan="mean")
        assert np.isfinite(link_mats["chr1"]).all()
