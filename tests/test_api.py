# test_api.py

import numpy as np
import pandas as pd
import pytest

import catella
from catella import utils
from catella.experiment.methdata import MethPrintData, MethPrintExperiment
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager


def _settings():
    return SimSettings(nucbp=5, llink=2, mu=-1.0, start_temp=1.0,
                       end_temp=0.1, cool_option="linear", nsweep=2,
                       print_freq=1, emax=5.0)


def _make_dataset(tmp_path, nmol=6, nbp=10, seed=1):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(chroms="chr1", nsim=2, settings=_settings(),
                       meth_prob=meth_prob, out_dir=tmp_path / "sim",
                       seed=seed)


def _make_experiment(nmol=6, nbp=10, seed=3, refseq=None, mtase=None):
    rng = np.random.default_rng(seed)
    rows = [(m, p, "+", float(rng.random()), 0) for m in range(nmol)
           for p in rng.choice(nbp, size=min(6, nbp), replace=False)]
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                     "mod_qual", "mod_code"])
    mol_id = np.array([f"m{m}" for m in range(nmol)], dtype=object)
    raw = MethPrintData._create(
        chrom="chr1", nbp=nbp, refseq=refseq, test_mol_id=mol_id,
        test_data=df, meth_mol_id=None, meth_data=None, unmeth_mol_id=None,
        unmeth_data=None)
    kwargs = {"_raw_data": {"chr1": raw}}
    if mtase is not None:
        kwargs["_mtase"] = mtase
    return MethPrintExperiment._create(**kwargs)


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write("read_id\tref_position\tchrom\tref_strand\tmod_qual\t"
                "mod_code\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        for chrom, length in sizes.items():
            f.write(f"{chrom}\t{length}\n")


def _write_fasta(path, records):
    with open(path, "w") as f:
        for chrom, seq in records.items():
            f.write(f">{chrom}\n{seq}\n")


_MODEL_SEQ = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT" * 8


class TestDownsample:
    def test_matches_utils_downsample(self):
        data = np.random.default_rng(0).random((10, 4))
        expected = utils.downsample(data, 3, how="mean")
        got = catella.downsample(data=data, max_rows=3, how="mean")
        np.testing.assert_allclose(got, expected)


class TestSortByLinkage:
    def test_raises_if_neither_given(self):
        with pytest.raises(ValueError):
            catella.sort_by_linkage()

    def test_raises_if_both_given(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        exp = _make_experiment()
        with pytest.raises(ValueError):
            catella.sort_by_linkage(dataset=dataset, exp=exp)

    def test_dataset_dispatch_sorts_and_stores_link_mat(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        order, ref_link = utils.compute_linkage(occup)

        catella.sort_by_linkage(dataset=dataset)

        sorted_arr = dataset.analysis["chr1"]["occup_sorted"].to_numpy()
        link_mat = dataset.analysis["chr1"]["occup_linkage"].to_numpy()
        np.testing.assert_allclose(sorted_arr, occup[order])
        np.testing.assert_allclose(link_mat, ref_link)

    def test_experiment_dispatch_with_raw_which(self):
        exp = _make_experiment()

        catella.sort_by_linkage(exp=exp, raw_which="test")

        assert "test_sorted" in exp.analysis["chr1"]
        assert "test_linkage" in exp.analysis["chr1"]

    def test_experiment_dispatch_with_mask_name(self):
        exp = _make_experiment()
        keep = np.zeros(6, dtype=bool)
        keep[0] = True
        exp.analysis["chr1"]["test_drop"] = pd.DataFrame({"keep": keep})

        catella.sort_by_linkage(exp=exp, raw_which="test", mask_name="drop",
                              fill_nan="mean")

        sorted_arr = exp.analysis["chr1"]["test_sorted"].to_numpy()
        nan_rows = np.isnan(sorted_arr).all(axis=1)
        assert nan_rows.sum() == 5

    def test_mask_name_with_dataset_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError):
            catella.sort_by_linkage(dataset=dataset, mask_name="drop")

    def test_store_link_mat_false_skips_persistence(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)

        catella.sort_by_linkage(dataset=dataset, store_link_mat=False)

        assert "occup_sorted" in dataset.analysis["chr1"]
        assert "occup_linkage" not in dataset.analysis["chr1"]

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)

        with pytest.raises(ValueError):
            catella.sort_by_linkage(dataset=dataset, data_name="occup_nan")

        catella.sort_by_linkage(dataset=dataset, data_name="occup_nan",
                              fill_nan="mean")
        link_mat = dataset.analysis["chr1"]["occup_nan_linkage"].to_numpy()
        assert np.isfinite(link_mat).all()


class TestLoadRaw:
    def test_returns_experiment_with_raw_data(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, [("m0", 0, "chr1", "+", 0.5, "a"),
                               ("m1", 1, "chr1", "+", 0.7, "a")])
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 5})

        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)

        assert exp.chroms == ("chr1",)
        assert sorted(exp.raw["chr1"].test_mol_id) == ["m0", "m1"]

    def test_chroms_forwarded(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, [("m0", 0, "chr1", "+", 0.5, "a"),
                               ("m1", 0, "chr2", "+", 0.5, "a")])
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 5, "chr2": 5})

        exp = catella.load_raw(chromsize=chromsize, test_file=test_file,
                               chroms=["chr1"])

        assert exp.chroms == ("chr1",)

        with pytest.raises(ValueError):
            catella.load_raw(chromsize=chromsize, test_file=test_file,
                             chroms=["bogus"])


class TestFilterDropout:
    def test_matches_direct_method_call(self):
        nbp = 10
        exp = _make_experiment(nmol=6, nbp=nbp, seed=3, refseq="A" * nbp,
                               mtase=("A",))
        expected = _make_experiment(nmol=6, nbp=nbp, seed=3,
                                    refseq="A" * nbp, mtase=("A",))

        catella.filter_dropout(exp=exp, threshold=0.5)
        expected.filter_dropout(threshold=0.5)

        np.testing.assert_array_equal(
            exp.analysis["chr1"]["test_dropout_mask"]["keep"].to_numpy(),
            expected.analysis["chr1"]["test_dropout_mask"]["keep"]
            .to_numpy())

    def test_chroms_forwarded(self):
        nbp = 10
        exp = _make_experiment(nmol=6, nbp=nbp, seed=3, refseq="A" * nbp,
                               mtase=("A",))
        with pytest.raises(ValueError):
            catella.filter_dropout(exp=exp, chroms=["bogus"])


class TestSummarizeDropout:
    def test_matches_direct_method_call(self):
        nbp = 10
        exp = _make_experiment(nmol=6, nbp=nbp, seed=3, refseq="A" * nbp,
                               mtase=("A",))

        got = catella.summarize_dropout(exp=exp)
        expected = exp.summarize_dropout()

        pd.testing.assert_frame_equal(got, expected)

    def test_chroms_forwarded(self):
        nbp = 10
        exp = _make_experiment(nmol=6, nbp=nbp, seed=3, refseq="A" * nbp,
                               mtase=("A",))
        with pytest.raises(ValueError):
            catella.summarize_dropout(exp=exp, chroms=["bogus"])


class TestComputeEmpiricalProb:
    def test_matches_manual_smooth_and_empirical_prob(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        expected = _make_experiment(nmol=6, nbp=10, seed=3)

        catella.compute_empirical_prob(exp=exp, binsize=3)

        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=expected)
        ana.empirical_prob(exp=expected)

        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy())

    def test_mask_name_excludes_dropped_molecules(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        keep = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": keep})

        catella.compute_empirical_prob(exp=exp, binsize=3,
                                       mask_name="dropout_mask")

        # Last binsize-1 columns have no full window to summarize and
        # are NaN regardless of masking; only check the filled columns.
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()[:, :10 - 3 + 1]
        assert np.isnan(prob[keep == 0]).all()
        assert not np.isnan(prob[keep == 1]).any()

    def test_nan_method_fill_edge_seed_propagate(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        expected = _make_experiment(nmol=6, nbp=10, seed=3)

        catella.compute_empirical_prob(exp=exp, binsize=3,
                                       nan_method="interpolate",
                                       fill_edge="mean", seed=7)

        ana = MethPrintAnalysis()
        ana.smooth(binsize=3, exp=expected, nan_method="interpolate",
                  fill_edge="mean")
        ana.empirical_prob(exp=expected, seed=7)

        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True)


class TestComputeModelProb:
    def test_matches_direct_api_call(self, tmp_path):
        rows = [(f"m{m}", p, "chr1", "+", 0.5, "a") for m in range(5)
               for p in range(len(_MODEL_SEQ))]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        exp = catella.load_raw(chromsize=chromsize, test_file=test_file,
                               fasta_file=fasta_file)
        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    fasta_file=fasta_file)

        catella.compute_model_prob(exp=exp, l_nuc=30, n_min=3)

        ana = MethPrintAnalysis()
        ana.model_prob(exp=expected, l_nuc=30, n_min=3)

        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True)

    def test_missing_refseq_raises(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        with pytest.raises(ValueError):
            catella.compute_model_prob(exp=exp, l_nuc=10)

    def test_store_rho_propagates(self, tmp_path):
        rng = np.random.default_rng(11)
        rows = [(f"m{m}", p, "chr1", "+", float(rng.random()), "a")
               for m in range(5) for p in range(len(_MODEL_SEQ))]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        exp = catella.load_raw(chromsize=chromsize, test_file=test_file,
                               fasta_file=fasta_file)
        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    fasta_file=fasta_file)

        catella.compute_model_prob(exp=exp, l_nuc=30, n_min=3,
                                   eta_max_lag=4, store_rho=True)

        ana = MethPrintAnalysis()
        ana.model_prob(exp=expected, l_nuc=30, n_min=3, eta_max_lag=4,
                       store_rho=True)

        assert ("meth_prob_rho" in exp.global_analysis) == (
            "meth_prob_rho" in expected.global_analysis)
        if "meth_prob_rho" in exp.global_analysis:
            pd.testing.assert_frame_equal(
                exp.global_analysis["meth_prob_rho"],
                expected.global_analysis["meth_prob_rho"])
