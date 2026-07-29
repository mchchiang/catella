# test_preprocessing.py

import numpy as np
import pandas as pd
import pytest

from nucmc.experiment.methdata import MethPrintData, MethPrintExperiment
from nucmc.experiment.preprocessing import MethPrintAnalysis
from nucmc.h5_array import H5Array


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


class TestMethProb:
    def test_output_in_unit_range(self):
        exp = _make_experiment(nmol=6, nbp=10)
        ana = MethPrintAnalysis()
        ana.meth_prob(exp=exp, binsize=3, batch_size=100,
                      percentile_sample_size=1000)
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
                      percentile_sample_size=1000)
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        assert prob.shape == (4, 6)

    def test_norm_by_strand_rejects_unmapped_strand(self):
        exp = _make_experiment(nmol=4, nbp=6, with_controls=True,
                               unmapped_test_mol=0)
        ana = MethPrintAnalysis()
        with pytest.raises(ValueError):
            ana.meth_prob(exp=exp, binsize=2, batch_size=100,
                          norm_by_strand=True)


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
