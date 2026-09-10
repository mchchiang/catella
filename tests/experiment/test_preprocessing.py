# test_preprocessing.py

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catella.experiment.methdata import MethPrintData, MethPrintExperiment
from catella.experiment.preprocessing import (
    MethPrintAnalysis, _reference_contexts, NONE)
from catella.h5_array import H5Array
from catella import utils


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

    def test_nan_method_none_leaves_interior_gaps_unfilled(self):
        exp = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=1, exp=exp, nan_method="none", batch_size=100)
        row = exp.analysis["chr1"]["test_smoothed"].to_numpy()[0]
        assert np.isnan(row[[0, 3, 4, 6]]).all()
        np.testing.assert_allclose(row[[1, 2, 5]], [0.0, 0.4, 1.0])

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

    def test_exp_accepted_positionally(self):
        exp_kw = _make_experiment(nmol=5, nbp=8)
        exp_pos = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(exp_kw, binsize=3, batch_size=100)
        ana.smooth(exp_pos, binsize=3, batch_size=100)
        np.testing.assert_array_equal(
            exp_kw.analysis["chr1"]["test_smoothed"].to_numpy(),
            exp_pos.analysis["chr1"]["test_smoothed"].to_numpy())


class TestEmpiricalProb:
    def test_output_in_unit_range(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp, binsize=3, batch_size=100,
                      percentile_sample_size=1000, fill_edge=0.5)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (6, 10)
        assert np.all(prob >= 0.0) and np.all(prob <= 1.0)

    @pytest.mark.parametrize("norm_by_strand", [False, True])
    def test_batching_matches_single_batch(self, norm_by_strand):
        exp_a = _make_experiment(nmol=8, nbp=10)
        exp_b = _make_experiment(nmol=8, nbp=10)
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp_a, binsize=3, batch_size=1000,
                      percentile_sample_size=1000,
                      norm_by_strand=norm_by_strand, seed=0)
        ana.empirical_prob(exp=exp_b, binsize=3, batch_size=2,
                      percentile_sample_size=1000,
                      norm_by_strand=norm_by_strand, seed=0)
        a = exp_a.analysis["chr1"]["meth_prob"].to_numpy()
        b = exp_b.analysis["chr1"]["meth_prob"].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_no_controls_skips_normalization(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp, binsize=2, batch_size=100,
                      percentile_sample_size=1000, fill_edge=0.5)
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
        ana.empirical_prob(exp=exp_narrow, binsize=1, batch_size=100,
                      percentile_sample_size=1000)
        ana.empirical_prob(exp=exp_wide, binsize=1, batch_size=100,
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
        ana.empirical_prob(exp=exp, binsize=1, batch_size=100,
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
            ana.empirical_prob(exp=exp, binsize=2, batch_size=100,
                          norm_by_strand=True)

    def test_missing_positions_become_neutral_probability(self):
        # Positions 0, 3, 4, 6 have no raw data anywhere, so they can
        # never be covered by any window -- they must be filled with
        # the neutral, no-evidence probability 0.5, not imputed from
        # neighboring or molecule-mean values.
        exp = _make_single_mol_experiment(
            positions=[1, 2, 5], values=[0.0, 0.4, 1.0], nbp=7)
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp, binsize=1, batch_size=100,
                      percentile_sample_size=1000)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        missing = [0, 3, 4, 6]
        assert np.all(prob[0, missing] == 0.5)

    def test_binsize_none_uses_cached_smooth_call(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, nan_method="none", batch_size=100)
        ana.empirical_prob(exp=exp, binsize=None, batch_size=100,
                      percentile_sample_size=1000)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (6, 10)

    def test_binsize_none_raises_without_cache(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.empirical_prob(exp=exp, binsize=None, batch_size=100,
                          percentile_sample_size=1000)

    def test_exp_accepted_positionally(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp, binsize=3, batch_size=100,
                      percentile_sample_size=1000)
        assert "meth_prob" in exp.analysis["chr1"]


class TestEmpiricalProbResmooth:
    def test_fill_edge_applies_to_trailing_probability_columns(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp, binsize=3, fill_edge=0.5,
                      batch_size=100, percentile_sample_size=1000)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert np.all(prob[:, -2:] == 0.5)

    def test_mismatched_binsize_raises(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, nan_method="none", batch_size=100)
        with pytest.raises(ValueError):
            ana.empirical_prob(exp=exp, binsize=5, batch_size=100,
                          percentile_sample_size=1000)

    def test_reusing_smooth_not_run_with_nan_method_none_raises(self):
        # smooth() used the default "mean"; empirical_prob() requires
        # a prior smooth() to have used nan_method="none", so this is
        # caught as a mismatch.
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, batch_size=100)
        with pytest.raises(ValueError):
            ana.empirical_prob(exp=exp, binsize=3, batch_size=100,
                          percentile_sample_size=1000)

    def test_resmooth_recomputes_and_updates_params(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, nan_method="none", batch_size=100)
        ana.empirical_prob(exp=exp, binsize=5, resmooth=True,
                      batch_size=100, percentile_sample_size=1000)
        assert exp.global_analysis[
            "smoothed_params"]["binsize"].iloc[0] == 5

    def test_reuse_smooth_called_with_nan_method_none(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=exp, nan_method="none", batch_size=100)
        ana.empirical_prob(exp=exp, batch_size=100,
                      percentile_sample_size=1000)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (6, 10)

    def test_mask_name_not_required_to_match_smooth(self):
        # mask_name is reapplied fresh every call, independent of what
        # smooth() itself used -- not a "stale smoothing" mismatch.
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, True]})
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, nan_method="none",
                  batch_size=100)  # no mask_name
        ana.empirical_prob(exp=exp, binsize=2, batch_size=100,
                      percentile_sample_size=1000,
                      mask_name="dropout_mask")
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert np.isnan(prob[1]).all()


class TestMaskName:
    def test_smooth_nans_masked_rows(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]})
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=100,
                  mask_name="dropout_mask")
        arr = exp.analysis["chr1"]["test_smoothed"].to_numpy()
        assert np.isnan(arr[1]).all()
        assert np.isnan(arr[3]).all()
        assert not np.isnan(arr[0]).all()
        assert not np.isnan(arr[2]).all()

    def test_smooth_no_mask_name_unaffected(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]})
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, batch_size=100)
        arr = exp.analysis["chr1"]["test_smoothed"].to_numpy()
        assert not np.isnan(arr[1]).all()

    def test_smooth_missing_mask_raises(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        ana = MethPrintAnalysis()
        with pytest.raises(KeyError):
            ana.smooth(binsize=2, exp=exp, batch_size=100,
                      mask_name="nonexistent")

    def test_meth_prob_masks_test_output(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]})
        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp, binsize=2, batch_size=100,
                      percentile_sample_size=1000,
                      mask_name="dropout_mask")
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert np.isnan(prob[1]).all()
        assert np.isnan(prob[3]).all()

    def test_meth_prob_masks_even_when_smooth_ran_unmasked_first(self):
        # Regression test: meth_prob only triggers smooth() itself when
        # no smoothed array exists yet. If smooth() already ran without
        # a mask, meth_prob(mask_name=...) must still mask its output
        # by applying the mask directly, not just by forwarding.
        exp = _make_experiment(nmol=4, nbp=6, with_controls=False)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, True]})
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, nan_method="none",
                  batch_size=100)  # no mask_name
        smoothed_before = exp.analysis["chr1"]["test_smoothed"].to_numpy()
        assert not np.isnan(smoothed_before[1]).all()

        ana.empirical_prob(exp=exp, binsize=2, batch_size=100,
                      percentile_sample_size=1000,
                      mask_name="dropout_mask")
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert np.isnan(prob[1]).all()

    def test_meth_prob_excludes_masked_controls_from_pooling(self):
        cols = ["mol_index", "pos", "strand", "mod_qual", "mod_code"]
        # meth control mol0 -> low values that would corrupt the
        # control average/percentile if included; mol1 -> the "real"
        # signal. mol0 is masked out.
        meth_df = pd.DataFrame([
            (0, 0, "+", 0.1, 0), (0, 1, "+", 0.1, 0),
            (1, 0, "+", 0.9, 0), (1, 1, "+", 0.9, 0),
        ], columns=cols)
        unmeth_df = pd.DataFrame([
            (0, 0, "+", 0.0, 0), (0, 1, "+", 0.0, 0),
        ], columns=cols)
        test_df = pd.DataFrame([
            (0, 0, "+", 0.5, 0), (0, 1, "+", 0.5, 0),
        ], columns=cols)
        meth_mol_id = np.array(["m0", "m1"], dtype=object)
        unmeth_mol_id = np.array(["u0"], dtype=object)
        test_mol_id = np.array(["t0"], dtype=object)

        raw_masked = MethPrintData._create(
            chrom="chr1", nbp=2, test_mol_id=test_mol_id, test_data=test_df,
            meth_mol_id=meth_mol_id, meth_data=meth_df,
            unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_df)
        exp_masked = MethPrintExperiment._create(
            _raw_data={"chr1": raw_masked})
        exp_masked.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True]})
        exp_masked.analysis["chr1"]["meth_dropout_mask"] = pd.DataFrame(
            {"keep": [False, True]})
        exp_masked.analysis["chr1"]["unmeth_dropout_mask"] = pd.DataFrame(
            {"keep": [True]})

        # Reference: meth control physically only has the kept molecule.
        raw_ref = MethPrintData._create(
            chrom="chr1", nbp=2, test_mol_id=test_mol_id, test_data=test_df,
            meth_mol_id=np.array(["m1"], dtype=object),
            meth_data=pd.DataFrame([
                (0, 0, "+", 0.9, 0), (0, 1, "+", 0.9, 0),
            ], columns=cols),
            unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_df)
        exp_reference = MethPrintExperiment._create(
            _raw_data={"chr1": raw_ref})

        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=exp_masked, binsize=1, batch_size=100,
                      percentile_sample_size=1000,
                      mask_name="dropout_mask")
        ana.empirical_prob(exp=exp_reference, binsize=1, batch_size=100,
                      percentile_sample_size=1000)

        prob_masked = exp_masked.analysis["chr1"]["meth_prob"].to_numpy()
        prob_reference = exp_reference.analysis["chr1"]["meth_prob"].to_numpy()
        np.testing.assert_allclose(prob_masked[0], prob_reference[0])


class TestSaveLoadRoundTrip:
    def test_h5array_analysis_round_trips(self, tmp_path):
        exp = _make_experiment(nmol=5, nbp=8)
        ana = MethPrintAnalysis()
        ana.smooth(binsize=2, exp=exp, nan_method="none", batch_size=2)
        ana.empirical_prob(exp=exp, binsize=2, batch_size=2,
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

    def test_exp_accepted_positionally(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        link_mats = ana.sort_by_linkage(exp, batch_size=2)
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
        ana.smooth(binsize=2, exp=exp, nan_method="none", batch_size=2)
        ana.empirical_prob(exp=exp, binsize=2, batch_size=2,
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

    def test_mask_name_excludes_dropped_molecules(self):
        exp = _make_experiment(nmol=6, nbp=10)
        keep = np.array([i not in (1, 3) for i in range(6)])
        exp.analysis["chr1"]["meth_drop"] = pd.DataFrame({"keep": keep})
        ana = MethPrintAnalysis()

        ana.sort_by_linkage(exp=exp, raw_which="meth", mask_name="drop",
                            batch_size=2, fill_nan="mean")
        sorted_arr = exp.analysis["chr1"]["meth_sorted"].to_numpy()

        nan_rows = np.isnan(sorted_arr).all(axis=1)
        assert nan_rows.sum() == 2
        assert (~nan_rows).sum() == 4

    def test_mask_name_without_raw_which_raises(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.sort_by_linkage(exp=exp, mask_name="drop", batch_size=2)

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

    def test_chroms_restricts_processing_to_selected_chromosomes(self):
        rng = np.random.default_rng(1)
        mol_id_1, data_1 = _make_mol_data(6, 10, rng)
        mol_id_2, data_2 = _make_mol_data(6, 10, rng)
        raw_1 = MethPrintData._create(
            chrom="chr1", nbp=10, test_mol_id=mol_id_1, test_data=data_1,
            meth_mol_id=None, meth_data=None, unmeth_mol_id=None,
            unmeth_data=None)
        raw_2 = MethPrintData._create(
            chrom="chr2", nbp=10, test_mol_id=mol_id_2, test_data=data_2,
            meth_mol_id=None, meth_data=None, unmeth_mol_id=None,
            unmeth_data=None)
        exp = MethPrintExperiment._create(
            _raw_data={"chr1": raw_1, "chr2": raw_2})
        ana = MethPrintAnalysis()

        link_mats = ana.sort_by_linkage(exp=exp, chroms="chr1", batch_size=2)

        assert "test_sorted" in exp.analysis["chr1"]
        assert "test_sorted" not in exp.analysis["chr2"]
        assert list(link_mats) == ["chr1"]

    def test_raw_which_closes_transient_scratch_file(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        tmp_dir = Path(exp.resolve_tmp_dir())
        before = set(tmp_dir.glob("*.h5"))

        ana.sort_by_linkage(exp=exp, raw_which="meth", batch_size=2)

        # Only the persisted meth_sorted H5Array should remain; the
        # transient to_dense() array feeding compute_linkage/
        # reorder_rows must be closed.
        new_files = set(tmp_dir.glob("*.h5")) - before
        assert len(new_files) == 1

    def test_default_fallback_closes_transient_scratch_file(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        tmp_dir = Path(exp.resolve_tmp_dir())
        before = set(tmp_dir.glob("*.h5"))

        ana.sort_by_linkage(exp=exp, batch_size=2)

        new_files = set(tmp_dir.glob("*.h5")) - before
        assert len(new_files) == 1


def _make_footprint_experiment(*, nmol=20, meth_nmol=30, unmeth_nmol=30,
                               with_controls=True, planted_edges=(30,),
                               l_nuc=30, seed=0, soft_q=False,
                               corr_source=None, corr_channel=None,
                               corr_lag=1, corr_strength=0.95,
                               strand_of=None, strand_call_bias=None,
                               unmapped_test_mol=None, mtase=None):
    # Synthetic multi-channel footprinting experiment with a known
    # planted "protected" region, for testing model_prob end to end.
    # corr_source/corr_channel inject deterministic lag-k correlation
    # into one raw source's calls, for testing eta auto-estimation.
    # strand_of/strand_call_bias plant a strand-correlated call-rate
    # bias, for testing norm_by_strand. mtase sets exp.mtase, for
    # testing model_prob's channel restriction.
    from catella.experiment.preprocessing import (
        _reference_contexts, NONE, M6A, GCH, HCG, GCG)

    if strand_of is None:
        strand_of = lambda m: "+"

    rng = np.random.default_rng(seed)
    seq = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT" * 8
    ctx = _reference_contexts(seq)
    L = len(seq)
    site = ctx != NONE

    true_acc = np.zeros(L)
    true_fpr = np.zeros(L)
    rates = {M6A: (0.02, 0.60), GCH: (0.02, 0.75),
            HCG: (0.03, 0.50), GCG: (0.03, 0.80)}
    for code, (f, a) in rates.items():
        sel = ctx == code
        true_acc[sel] = np.clip(a + rng.normal(0, 0.10, sel.sum()),
                                0.05, 0.95)
        true_fpr[sel] = f
    rho_leak = 0.15

    occ = np.zeros(L, dtype=bool)
    for e in planted_edges:
        occ[e:e + l_nuc] = True
    test_prob = np.where(occ, true_fpr + rho_leak * (true_acc - true_fpr),
                         true_acc)

    def inject_corr(calls, channel, lag, strength):
        # Force calls[x+lag] to copy calls[x] (with probability
        # strength) for every x in the given channel that has a
        # same-channel partner exactly lag bp away.
        idx = np.where(ctx == channel)[0]
        idx_set = set(idx.tolist())
        for x in idx:
            y = x + lag
            if y in idx_set and rng.random() < strength:
                calls[y] = calls[x]
        return calls

    def make_df(n, prob, source_name):
        rows = []
        for m in range(n):
            strand = strand_of(m)
            if (source_name == "test" and unmapped_test_mol is not None
                    and m == unmapped_test_mol):
                strand = "."
            prob_eff = prob
            if strand_call_bias is not None and strand in strand_call_bias:
                prob_eff = np.clip(prob + strand_call_bias[strand],
                                   0.0, 1.0)
            calls = rng.random(L) < prob_eff
            if source_name == corr_source and corr_channel is not None:
                calls = inject_corr(calls, corr_channel, corr_lag,
                                    corr_strength)
            if soft_q:
                qual = np.where(calls, rng.uniform(0.6, 0.95, L),
                                rng.uniform(0.05, 0.4, L))
            else:
                qual = calls.astype(float)
            for pos in np.where(site)[0]:
                rows.append((m, int(pos), strand, float(qual[pos]), 0))
        return pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                           "mod_qual", "mod_code"])

    test_df = make_df(nmol, test_prob, "test")
    test_mol_id = np.array([f"t{m}" for m in range(nmol)], dtype=object)

    if with_controls:
        meth_df = make_df(meth_nmol, true_acc, "meth")
        unmeth_df = make_df(unmeth_nmol, true_fpr, "unmeth")
        meth_mol_id = np.array([f"m{m}" for m in range(meth_nmol)],
                               dtype=object)
        unmeth_mol_id = np.array([f"u{m}" for m in range(unmeth_nmol)],
                                 dtype=object)
    else:
        meth_df = unmeth_df = meth_mol_id = unmeth_mol_id = None

    raw = MethPrintData._create(
        chrom="chr1", nbp=L, refseq=seq, test_mol_id=test_mol_id,
        test_data=test_df, meth_mol_id=meth_mol_id, meth_data=meth_df,
        unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_df)
    mtase_tuple = tuple(mtase) if mtase is not None else None
    return MethPrintExperiment._create(_raw_data={"chr1": raw},
                                       _mtase=mtase_tuple)


class TestModelProb:
    def test_controls_path_favors_planted_region(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (20, len(exp.raw["chr1"].refseq))
        planted = np.nanmean(prob[:, 30])
        background = np.nanmean(prob[:, 0])
        assert planted < 0.1
        assert background > 0.5

    def test_exp_accepted_positionally(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp, l_nuc=30, batch_size=7)
        assert "meth_prob" in exp.analysis["chr1"]

    def test_continuous_confidence_favors_planted_region(self):
        # mod_qual values are graded confidence scores rather than hard
        # 0/1 calls; exercises the continuous-q_x likelihood path.
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30,
                                         soft_q=True)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        planted = np.nanmean(prob[:, 30])
        background = np.nanmean(prob[:, 0])
        assert planted < background

    def test_no_controls_em_path_favors_planted_region(self):
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        planted = np.nanmean(prob[:, 30])
        background = np.nanmean(prob[:, 0])
        assert planted < background

    def test_mask_name_excludes_dropped_molecules(self):
        exp = _make_footprint_experiment(with_controls=True, nmol=20,
                                         meth_nmol=30, unmeth_nmol=30,
                                         planted_edges=(30,), l_nuc=30)
        keep_test = np.array([i not in (1, 3) for i in range(20)])
        keep_ctrl = np.array([i not in (1, 3) for i in range(30)])
        exp.analysis["chr1"]["test_drop"] = pd.DataFrame({"keep": keep_test})
        exp.analysis["chr1"]["meth_drop"] = pd.DataFrame({"keep": keep_ctrl})
        exp.analysis["chr1"]["unmeth_drop"] = pd.DataFrame(
            {"keep": keep_ctrl})

        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, mask_name="drop")
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert np.isnan(prob[1]).all()
        assert np.isnan(prob[3]).all()
        assert not np.isnan(prob[0]).all()

    def test_missing_mask_raises(self):
        exp = _make_footprint_experiment(with_controls=True, nmol=10,
                                         meth_nmol=10, unmeth_nmol=10,
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        with pytest.raises(KeyError):
            ana.model_prob(exp=exp, l_nuc=30, mask_name="nonexistent")

    def test_closes_transient_scratch_files(self):
        # with_controls=False also exercises the no-control EM
        # calibration path (_streamed_call_count/_streamed_window_count),
        # and the default eta calibration exercises
        # _accumulate_eta_source -- both create transient to_dense()
        # scratch arrays that must be closed, same as the main
        # per-chromosome test_arr loop.
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        tmp_dir = Path(exp.resolve_tmp_dir())
        before = set(tmp_dir.glob("*.h5"))

        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17)

        # Only the persisted meth_prob output (one H5Array per
        # chromosome) should remain.
        new_files = set(tmp_dir.glob("*.h5")) - before
        assert len(new_files) == len(exp.chroms)

    def test_closes_transient_scratch_file_on_early_raise(self):
        # CPython GC already cleans up transient arrays on the success
        # path, so that path cannot distinguish explicit close() from
        # implicit refcounting. Keeping excinfo bound keeps its
        # traceback, and thus the raising frame's locals, alive
        # instead, so this path can.
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        tmp_dir = Path(exp.resolve_tmp_dir())
        before = set(tmp_dir.glob("*.h5"))

        with pytest.raises(ValueError) as excinfo:
            # n_min impossibly high -> _streamed_window_count raises
            # "too few windows" after its to_dense() array is created.
            ana.model_prob(exp=exp, l_nuc=30, n_min=10**9, batch_size=17)

        assert set(tmp_dir.glob("*.h5")) == before
        del excinfo

    def test_missing_refseq_raises(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.model_prob(exp=exp)

    def test_fill_edge_default_is_nan_on_trailing_positions(self):
        exp = _make_footprint_experiment(with_controls=True, nmol=10,
                                         meth_nmol=10, unmeth_nmol=10,
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=5)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        nbp = prob.shape[1]
        assert np.isnan(prob[:, nbp - 30 + 1:]).all()
        assert not np.isnan(prob[:, :nbp - 30 + 1]).any()

    def test_fill_edge_custom_value(self):
        exp = _make_footprint_experiment(with_controls=True, nmol=10,
                                         meth_nmol=10, unmeth_nmol=10,
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=5, fill_edge=0.25)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        nbp = prob.shape[1]
        np.testing.assert_allclose(prob[:, nbp - 30 + 1:], 0.25)

    def test_default_prob_name_matches_empirical_prob(self):
        exp = _make_footprint_experiment(with_controls=True, nmol=10,
                                         meth_nmol=10, unmeth_nmol=10,
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=5)
        assert "meth_prob" in exp.analysis["chr1"]


def _eta_from(exp, prob_name="meth_prob"):
    row = exp.global_analysis[f"{prob_name}_eta"].iloc[0]
    return {ch: row[f"eta_{ch}"] for ch in ("M6A", "GCH", "HCG", "GCG")}


class TestModelProbEta:
    def test_default_populates_channel_eta_for_present_channels(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        eta = _eta_from(exp)
        assert set(eta) == {"M6A", "GCH", "HCG", "GCG"}
        assert all(v > 0 for v in eta.values())

    def test_meth_control_correlation_lowers_that_channel_eta(self):
        from catella.experiment.preprocessing import M6A
        exp = _make_footprint_experiment(
            with_controls=True, planted_edges=(30,), l_nuc=30,
            corr_source="meth", corr_channel=M6A, corr_lag=1)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        eta = _eta_from(exp)
        assert eta["M6A"] < 0.8
        assert eta["GCH"] > 0.8

    def test_test_data_correlation_does_not_move_estimate_with_controls(
            self):
        from catella.experiment.preprocessing import M6A
        exp = _make_footprint_experiment(
            with_controls=True, planted_edges=(30,), l_nuc=30,
            corr_source="test", corr_channel=M6A, corr_lag=1)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        assert _eta_from(exp)["M6A"] > 0.8

    def test_no_controls_falls_back_to_test_data(self):
        from catella.experiment.preprocessing import M6A
        exp = _make_footprint_experiment(
            with_controls=False, nmol=120, planted_edges=(30, 120, 200),
            l_nuc=30, corr_source="test", corr_channel=M6A, corr_lag=1)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17)
        assert _eta_from(exp)["M6A"] < 0.8

    def test_float_override_applies_uniformly(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, eta=0.5)
        assert _eta_from(exp) == {
            "M6A": 0.5, "GCH": 0.5, "HCG": 0.5, "GCG": 0.5}

    def test_dict_override_partial_leaves_rest_auto(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, eta={"M6A": 0.5})
        eta = _eta_from(exp)
        assert eta["M6A"] == 0.5
        assert eta["GCH"] != 0.5

    def test_unknown_channel_name_raises(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.model_prob(exp=exp, l_nuc=30, batch_size=7,
                           eta={"bogus": 0.5})

    def test_eta_max_lag_changes_estimate(self):
        from catella.experiment.preprocessing import M6A
        exp_short = _make_footprint_experiment(
            with_controls=True, planted_edges=(30,), l_nuc=30,
            corr_source="meth", corr_channel=M6A, corr_lag=5)
        exp_long = _make_footprint_experiment(
            with_controls=True, planted_edges=(30,), l_nuc=30,
            corr_source="meth", corr_channel=M6A, corr_lag=5)
        ana_short = MethPrintAnalysis()
        ana_short.model_prob(exp=exp_short, l_nuc=30, batch_size=7,
                             eta_max_lag=1)
        ana_long = MethPrintAnalysis()
        ana_long.model_prob(exp=exp_long, l_nuc=30, batch_size=7,
                            eta_max_lag=10)
        assert _eta_from(exp_short)["M6A"] != pytest.approx(
            _eta_from(exp_long)["M6A"], rel=1e-6)


class TestModelProbStoreRho:
    def test_default_does_not_store_rho(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        assert "meth_prob_rho" not in exp.global_analysis

    def test_stores_rho_for_auto_estimated_channels(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, eta_max_lag=5,
                       store_rho=True)
        rho = exp.global_analysis["meth_prob_rho"]
        assert list(rho["lag"]) == [1, 2, 3, 4, 5]
        assert {"rho_M6A", "rho_GCH", "rho_HCG", "rho_GCG"} <= set(
            rho.columns)

    def test_pinned_channel_excluded_from_rho(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7,
                       eta={"M6A": 0.5}, store_rho=True)
        rho = exp.global_analysis["meth_prob_rho"]
        assert "rho_M6A" not in rho.columns
        assert "rho_GCH" in rho.columns

    def test_all_channels_pinned_skips_rho_entry(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, eta=0.5,
                       store_rho=True)
        assert "meth_prob_rho" not in exp.global_analysis


def _revcomp(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


class TestModelProbWrap:
    def test_folds_ctx_and_runs(self):
        # A reverse-complement palindrome: S + revcomp(S) is always
        # symmetric around its center, matching what wrap requires.
        half = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT"
        seq = half + _revcomp(half)
        nbp = len(half)
        rng = np.random.default_rng(0)

        from catella.experiment.preprocessing import (
            _reference_contexts, NONE)
        site = np.where(_reference_contexts(seq)[:nbp] != NONE)[0]

        def make_df(n, prob):
            rows = [(m, int(pos), "+",
                    float(rng.uniform(0.6, 0.9) if rng.random() < prob
                          else rng.uniform(0.05, 0.3)), 0)
                   for m in range(n) for pos in site]
            return pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                               "mod_qual", "mod_code"])

        nmol = 8
        test_mol_id = np.array([f"t{m}" for m in range(nmol)], dtype=object)
        meth_mol_id = np.array([f"m{m}" for m in range(10)], dtype=object)
        unmeth_mol_id = np.array([f"u{m}" for m in range(10)], dtype=object)
        raw = MethPrintData._create(
            chrom="chr1", nbp=nbp, refseq=seq, test_mol_id=test_mol_id,
            test_data=make_df(nmol, 0.5), meth_mol_id=meth_mol_id,
            meth_data=make_df(10, 0.8), unmeth_mol_id=unmeth_mol_id,
            unmeth_data=make_df(10, 0.05))
        exp = MethPrintExperiment._create(_raw_data={"chr1": raw},
                                          _wrap=True)

        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=10, batch_size=4)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (nmol, nbp)

    def test_no_warning_when_fully_symmetric(self, recwarn):
        half = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT"
        seq = half + _revcomp(half)
        nbp = len(half)
        rng = np.random.default_rng(2)
        site = np.where(_reference_contexts(seq)[:nbp] != NONE)[0]

        def make_df(n, prob):
            rows = [(m, int(pos), "+",
                    float(rng.uniform(0.6, 0.9) if rng.random() < prob
                          else rng.uniform(0.05, 0.3)), 0)
                   for m in range(n) for pos in site]
            return pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                               "mod_qual", "mod_code"])

        raw = MethPrintData._create(
            chrom="chr1", nbp=nbp, refseq=seq,
            test_mol_id=np.array(["t0", "t1"], dtype=object),
            test_data=make_df(2, 0.5),
            meth_mol_id=np.array([f"m{i}" for i in range(6)], dtype=object),
            meth_data=make_df(6, 0.8),
            unmeth_mol_id=np.array([f"u{i}" for i in range(6)], dtype=object),
            unmeth_data=make_df(6, 0.05))
        exp = MethPrintExperiment._create(_raw_data={"chr1": raw},
                                          _wrap=True)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=10)
        assert not any(issubclass(w.category, UserWarning) for w in recwarn)

    def test_small_disagreement_ignored_in_output(self):
        # A single mismatched base near the center, mirroring the real
        # loop/junction region that motivated this behavior: it should
        # fold to no context and have zero effect on the output.
        half = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT"
        nbp = len(half)
        disagree_pos = 6
        seq = list(half + _revcomp(half))
        seq[disagree_pos] = "C"
        seq = "".join(seq)

        full_ctx = _reference_contexts(seq)
        lower = full_ctx[:nbp]
        upper = full_ctx[len(seq) - nbp:][::-1]
        assert lower[disagree_pos] != upper[disagree_pos]

        # Informative on at least one side pre-fold, so a call there
        # would matter if not properly excluded after folding.
        site = np.where((lower != NONE) | (upper != NONE))[0]
        rng = np.random.default_rng(1)

        def fixed_df(n, prob):
            rows = [(m, int(pos), "+",
                    float(rng.uniform(0.6, 0.9) if rng.random() < prob
                          else rng.uniform(0.05, 0.3)), 0)
                   for m in range(n) for pos in site]
            return pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                               "mod_qual", "mod_code"])

        meth_df = fixed_df(6, 0.8)
        unmeth_df = fixed_df(6, 0.05)
        meth_mol_id = np.array([f"m{i}" for i in range(6)], dtype=object)
        unmeth_mol_id = np.array([f"u{i}" for i in range(6)], dtype=object)

        def make_test_df(qual_at_disagree):
            rows = [(0, int(pos), "+", 0.5, 0) for pos in site
                    if pos != disagree_pos]
            rows.append((0, disagree_pos, "+", qual_at_disagree, 0))
            return pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                               "mod_qual", "mod_code"])

        def make_exp(qual_at_disagree):
            raw = MethPrintData._create(
                chrom="chr1", nbp=nbp, refseq=seq,
                test_mol_id=np.array(["t0"], dtype=object),
                test_data=make_test_df(qual_at_disagree),
                meth_mol_id=meth_mol_id, meth_data=meth_df,
                unmeth_mol_id=unmeth_mol_id, unmeth_data=unmeth_df)
            return MethPrintExperiment._create(_raw_data={"chr1": raw},
                                               _wrap=True)

        ana = MethPrintAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp_low = make_exp(0.01)
            ana.model_prob(exp=exp_low, l_nuc=10)
            exp_high = make_exp(0.99)
            ana.model_prob(exp=exp_high, l_nuc=10)
        prob_low = exp_low.analysis["chr1"]["meth_prob"].to_numpy()
        prob_high = exp_high.analysis["chr1"]["meth_prob"].to_numpy()
        np.testing.assert_allclose(prob_low, prob_high)

    def test_large_disagreement_warns(self):
        # No reverse-complement relationship at all -- almost every
        # wrap-mirrored pair disagrees on context.
        half = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT"
        seq = half + half
        nbp = len(half)
        rng = np.random.default_rng(3)
        site = np.where(_reference_contexts(seq)[:nbp] != NONE)[0]

        def make_df(n, prob):
            rows = [(m, int(pos), "+",
                    float(rng.uniform(0.6, 0.9) if rng.random() < prob
                          else rng.uniform(0.05, 0.3)), 0)
                   for m in range(n) for pos in site]
            return pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                               "mod_qual", "mod_code"])

        raw = MethPrintData._create(
            chrom="chr1", nbp=nbp, refseq=seq,
            test_mol_id=np.array(["t0", "t1"], dtype=object),
            test_data=make_df(2, 0.5),
            meth_mol_id=np.array([f"m{i}" for i in range(6)], dtype=object),
            meth_data=make_df(6, 0.8),
            unmeth_mol_id=np.array([f"u{i}" for i in range(6)], dtype=object),
            unmeth_data=make_df(6, 0.05))
        exp = MethPrintExperiment._create(_raw_data={"chr1": raw},
                                          _wrap=True)
        ana = MethPrintAnalysis()
        with pytest.warns(UserWarning):
            ana.model_prob(exp=exp, l_nuc=10)


class TestModelProbStrand:
    _strand_of = staticmethod(lambda m: "+" if m % 2 == 0 else "-")

    def test_rejects_unmapped_strand_with_controls(self):
        exp = _make_footprint_experiment(with_controls=True, nmol=10,
                                         meth_nmol=10, unmeth_nmol=10,
                                         l_nuc=30, unmapped_test_mol=0)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.model_prob(exp=exp, l_nuc=30, norm_by_strand=True)

    def test_rejects_unmapped_strand_no_controls(self):
        exp = _make_footprint_experiment(with_controls=False, nmol=10,
                                         l_nuc=30, unmapped_test_mol=0)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.model_prob(exp=exp, l_nuc=30, norm_by_strand=True)

    @pytest.mark.parametrize("norm_by_strand", [False, True])
    def test_batching_matches_single_batch(self, norm_by_strand):
        exp_a = _make_footprint_experiment(with_controls=True, nmol=40,
                                           l_nuc=30, strand_of=self._strand_of)
        exp_b = _make_footprint_experiment(with_controls=True, nmol=40,
                                           l_nuc=30, strand_of=self._strand_of)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp_a, l_nuc=30, batch_size=1000,
                       norm_by_strand=norm_by_strand)
        ana.model_prob(exp=exp_b, l_nuc=30, batch_size=3,
                       norm_by_strand=norm_by_strand)
        a = exp_a.analysis["chr1"]["meth_prob"].to_numpy()
        b = exp_b.analysis["chr1"]["meth_prob"].to_numpy()
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_pooled_calibration_is_biased_by_strand_but_split_recovers_it(
            self):
        bias = {"+": 0.0, "-": 0.35}
        exp_pooled = _make_footprint_experiment(
            with_controls=True, nmol=40, planted_edges=(30,), l_nuc=30,
            strand_of=self._strand_of, strand_call_bias=bias)
        exp_split = _make_footprint_experiment(
            with_controls=True, nmol=40, planted_edges=(30,), l_nuc=30,
            strand_of=self._strand_of, strand_call_bias=bias)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp_pooled, l_nuc=30, batch_size=7,
                       norm_by_strand=False)
        ana.model_prob(exp=exp_split, l_nuc=30, batch_size=7,
                       norm_by_strand=True)

        neg = np.array([self._strand_of(m) for m in range(40)]) == "-"
        prob_pooled = exp_pooled.analysis["chr1"]["meth_prob"].to_numpy()
        prob_split = exp_split.analysis["chr1"]["meth_prob"].to_numpy()

        planted_pooled = np.nanmean(prob_pooled[neg, 30])
        planted_split = np.nanmean(prob_split[neg, 30])
        assert planted_split < planted_pooled
        assert planted_split < 0.1

    def test_no_controls_path_norm_by_strand_favors_planted_region(self):
        exp = _make_footprint_experiment(
            with_controls=False, nmol=200, planted_edges=(30, 120, 200),
            l_nuc=30, strand_of=self._strand_of)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17,
                       norm_by_strand=True)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        planted = np.nanmean(prob[:, 30])
        background = np.nanmean(prob[:, 0])
        assert planted < background

    def test_eta_pooled_regardless_of_norm_by_strand(self):
        exp_a = _make_footprint_experiment(
            with_controls=True, nmol=40, strand_of=self._strand_of, seed=1)
        exp_b = _make_footprint_experiment(
            with_controls=True, nmol=40, strand_of=self._strand_of, seed=1)
        ana_a = MethPrintAnalysis()
        ana_a.model_prob(exp=exp_a, l_nuc=30, norm_by_strand=False)
        ana_b = MethPrintAnalysis()
        ana_b.model_prob(exp=exp_b, l_nuc=30, norm_by_strand=True)
        eta_a, eta_b = _eta_from(exp_a), _eta_from(exp_b)
        for channel in ("M6A", "GCH", "HCG", "GCG"):
            assert eta_b[channel] == pytest.approx(
                eta_a[channel], rel=0.1)


class TestModelProbEtaTable:
    def test_eta_table_always_created_and_separate_from_params(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        assert "meth_prob_eta" in exp.global_analysis
        eta_row = exp.global_analysis["meth_prob_eta"].iloc[0]
        for ch in ("M6A", "GCH", "HCG", "GCG"):
            assert f"eta_{ch}" in eta_row
        params = exp.global_analysis["meth_prob_params"]
        assert not any(c.startswith("eta_") and c != "eta_max_lag"
                      for c in params.columns)


class TestModelProbCalib:
    def test_no_controls_path_populates_calib_table(self):
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17)
        calib = exp.global_analysis["meth_prob_calib"]
        assert len(calib) == 1
        row = calib.iloc[0]
        assert row["chrom"] == "chr1"
        assert row["has_controls"] == False
        frac_cols = [c for c in calib.columns
                    if c.startswith("frac_informative_")]
        assert frac_cols
        for c in frac_cols:
            assert 0 <= row[c] <= 1
        assert 1 <= row["iters"] <= 200
        assert np.isfinite(row["log_likelihood"])
        assert 0 <= row["frac_protected"] <= 1
        for ch in ("M6A", "GCH", "HCG", "GCG"):
            assert f"theta_prot_{ch}" in calib.columns
            assert f"theta_acc_{ch}" in calib.columns

    def test_controls_path_row_has_no_em_diagnostics(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        calib = exp.global_analysis["meth_prob_calib"]
        row = calib.iloc[0]
        assert row["has_controls"] == True
        frac_cols = [c for c in calib.columns
                    if c.startswith("frac_informative_")]
        assert frac_cols
        for c in frac_cols:
            assert 0 <= row[c] <= 1
        for col in ("iters", "log_likelihood", "frac_protected"):
            assert col not in calib.columns or pd.isna(row[col])

    def test_max_iters_caps_actual_iterations(self):
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17,
                       max_iters=1)
        calib = exp.global_analysis["meth_prob_calib"]
        assert calib.iloc[0]["iters"] <= 1

    def test_norm_by_strand_gives_two_rows_per_chrom(self):
        strand_of = lambda m: "+" if m % 2 == 0 else "-"
        exp = _make_footprint_experiment(
            with_controls=False, nmol=200, planted_edges=(30, 120, 200),
            l_nuc=30, strand_of=strand_of)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17,
                       norm_by_strand=True)
        calib = exp.global_analysis["meth_prob_calib"]
        assert len(calib) == 2
        assert set(calib["strand"]) == {"+", "-"}
        for _, row in calib.iterrows():
            assert 1 <= row["iters"] <= 200

    def test_params_table_has_max_iters_not_iters(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, max_iters=50)
        params = exp.global_analysis["meth_prob_params"]
        assert params.iloc[0]["max_iters"] == 50
        assert "iters" not in params.columns


class TestModelProbMtase:
    def test_mtase_a_restricts_calib_and_eta_to_m6a(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30,
                                         mtase=["A"])
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        calib = exp.global_analysis["meth_prob_calib"]
        frac_cols = [c for c in calib.columns
                    if c.startswith("frac_informative_")]
        assert frac_cols == ["frac_informative_M6A"]
        eta_row = exp.global_analysis["meth_prob_eta"].iloc[0]
        assert list(eta_row.index) == ["eta_M6A"]

    def test_mtase_cg_gives_hcg_and_gcg_only(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30,
                                         mtase=["CG"])
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        eta_row = exp.global_analysis["meth_prob_eta"].iloc[0]
        assert set(eta_row.index) == {"eta_HCG", "eta_GCG"}
        calib = exp.global_analysis["meth_prob_calib"]
        frac_cols = {c for c in calib.columns
                    if c.startswith("frac_informative_")}
        assert frac_cols == {"frac_informative_HCG", "frac_informative_GCG"}

    def test_mtase_a_restricts_no_controls_theta_columns(self):
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30, mtase=["A"])
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17)
        calib = exp.global_analysis["meth_prob_calib"]
        assert "theta_prot_M6A" in calib.columns
        assert "theta_acc_M6A" in calib.columns
        for ch in ("GCH", "HCG", "GCG"):
            assert f"theta_prot_{ch}" not in calib.columns
            assert f"theta_acc_{ch}" not in calib.columns


class TestModelProbTheta:
    def test_theta_table_present_for_no_controls_path(self):
        exp = _make_footprint_experiment(with_controls=False, nmol=120,
                                         planted_edges=(30, 120, 200),
                                         l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, n_min=3, batch_size=17)
        theta = exp.analysis["chr1"]["meth_prob_theta"]
        nbp = len(exp.raw["chr1"].refseq)
        assert len(theta) == nbp
        assert set(theta.columns) == {"theta_prot", "theta_acc",
                                      "informative"}

    def test_theta_table_present_for_controls_path(self):
        exp = _make_footprint_experiment(with_controls=True,
                                         planted_edges=(30,), l_nuc=30)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7)
        theta = exp.analysis["chr1"]["meth_prob_theta"]
        nbp = len(exp.raw["chr1"].refseq)
        assert len(theta) == nbp
        assert set(theta.columns) == {"theta_prot", "theta_acc",
                                      "informative"}
        informative = theta["informative"].to_numpy()
        assert informative.any()
        assert (theta["theta_acc"][informative]
               > theta["theta_prot"][informative]).all()

    def test_norm_by_strand_gives_suffixed_columns(self):
        strand_of = lambda m: "+" if m % 2 == 0 else "-"
        exp = _make_footprint_experiment(
            with_controls=True, nmol=40, planted_edges=(30,), l_nuc=30,
            strand_of=strand_of)
        ana = MethPrintAnalysis()
        ana.model_prob(exp=exp, l_nuc=30, batch_size=7, norm_by_strand=True)
        theta = exp.analysis["chr1"]["meth_prob_theta"]
        expected = {"theta_prot_pos", "theta_acc_pos", "informative_pos",
                   "theta_prot_neg", "theta_acc_neg", "informative_neg"}
        assert set(theta.columns) == expected
