# test_api.py

from dataclasses import replace

import catella
import numpy as np
import pandas as pd
import pytest
from catella import utils
from catella.experiment.methdata import MethPrintData, MethPrintExperiment
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from scipy.special import logit


def _settings():
    return SimSettings(
        nucbp=5,
        llink=2,
        elink=1.0,
        mu=-1.0,
        start_temp=1.0,
        end_temp=0.1,
        cool_option="linear",
        nsweep=2,
        print_freq=1,
        emax=5.0,
    )


def _make_dataset(tmp_path, nmol=6, nbp=10, seed=1):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        chroms="chr1",
        nsim=2,
        settings=_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "sim",
        seed=seed,
    )


def _make_experiment(nmol=6, nbp=10, seed=3, refseq=None, mtase=None):
    rng = np.random.default_rng(seed)
    rows = [
        (m, p, "+", float(rng.random()), 0)
        for m in range(nmol)
        for p in rng.choice(nbp, size=min(6, nbp), replace=False)
    ]
    df = pd.DataFrame(
        rows, columns=["mol_index", "pos", "strand", "mod_qual", "mod_code"]
    )
    mol_id = np.array([f"m{m}" for m in range(nmol)], dtype=object)
    raw = MethPrintData._create(
        chrom="chr1",
        nbp=nbp,
        refseq=refseq,
        test_mol_id=mol_id,
        test_data=df,
        meth_mol_id=None,
        meth_data=None,
        unmeth_mol_id=None,
        unmeth_data=None,
    )
    kwargs = {"_raw_data": {"chr1": raw}}
    if mtase is not None:
        kwargs["_mtase"] = mtase
    return MethPrintExperiment._create(**kwargs)


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

        catella.sort_by_linkage(
            exp=exp, raw_which="test", mask_name="drop", fill_nan="mean"
        )

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

        catella.sort_by_linkage(
            dataset=dataset, data_name="occup_nan", fill_nan="mean"
        )
        link_mat = dataset.analysis["chr1"]["occup_nan_linkage"].to_numpy()
        assert np.isfinite(link_mat).all()


class TestLoadRaw:
    def test_returns_experiment_with_raw_data(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(
            test_file,
            [
                ("m0", 0, "chr1", "+", 0.5, "a"),
                ("m1", 1, "chr1", "+", 0.7, "a"),
            ],
        )
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 5})

        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)

        assert exp.chroms == ("chr1",)
        assert sorted(exp.raw["chr1"].test_mol_id) == ["m0", "m1"]

    def test_chroms_forwarded(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(
            test_file,
            [
                ("m0", 0, "chr1", "+", 0.5, "a"),
                ("m1", 0, "chr2", "+", 0.5, "a"),
            ],
        )
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 5, "chr2": 5})

        exp = catella.load_raw(
            chromsize=chromsize, test_file=test_file, chroms=["chr1"]
        )

        assert exp.chroms == ("chr1",)

        with pytest.raises(ValueError):
            catella.load_raw(
                chromsize=chromsize, test_file=test_file, chroms=["bogus"]
            )


class TestFilterDropout:
    def test_matches_direct_method_call(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        expected = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )

        catella.filter_dropout(exp=exp, thres_max=0.5)
        expected.filter_dropout(thres_max=0.5)

        np.testing.assert_array_equal(
            exp.analysis["chr1"]["test_dropout_mask"]["keep"].to_numpy(),
            expected.analysis["chr1"]["test_dropout_mask"]["keep"].to_numpy(),
        )

    def test_chroms_forwarded(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        with pytest.raises(ValueError):
            catella.filter_dropout(exp=exp, chroms=["bogus"])

    def test_exp_accepted_positionally(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        catella.filter_dropout(exp, thres_max=0.5)
        assert "test_dropout_mask" in exp.analysis["chr1"]


class TestSummarizeDropout:
    def test_matches_direct_method_call(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )

        got = catella.summarize_dropout(exp=exp)
        expected = exp.summarize_dropout()

        pd.testing.assert_frame_equal(got, expected)

    def test_chroms_forwarded(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        with pytest.raises(ValueError):
            catella.summarize_dropout(exp=exp, chroms=["bogus"])


class TestComputeEmpiricalProb:
    def test_matches_manual_smooth_and_empirical_prob(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        expected = _make_experiment(nmol=6, nbp=10, seed=3)

        catella.compute_empirical_prob(exp=exp, lnuc=3)

        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=expected, lnuc=3)

        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
        )

    def test_exp_accepted_positionally(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        catella.compute_empirical_prob(exp, lnuc=3)
        assert "meth_prob" in exp.analysis["chr1"]

    def test_mask_name_excludes_dropped_molecules(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        keep = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": keep}
        )

        catella.compute_empirical_prob(
            exp=exp, lnuc=3, mask_name="dropout_mask"
        )

        # Last lnuc-1 columns have no full window to summarize and
        # are NaN regardless of masking; only check the filled columns.
        prob = exp.analysis["chr1"]["meth_prob"].to_numpy()[:, : 10 - 3 + 1]
        assert np.isnan(prob[keep == 0]).all()
        assert not np.isnan(prob[keep == 1]).any()

    def test_fill_edge_seed_propagate(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        expected = _make_experiment(nmol=6, nbp=10, seed=3)

        catella.compute_empirical_prob(exp=exp, lnuc=3, fill_edge=0.5, seed=7)

        ana = MethPrintAnalysis()
        ana.empirical_prob(exp=expected, lnuc=3, fill_edge=0.5, seed=7)

        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True,
        )


class TestComputeModelProb:
    def test_matches_direct_api_call(self, tmp_path):
        rows = [
            (f"m{m}", p, "chr1", "+", 0.5, "a")
            for m in range(5)
            for p in range(len(_MODEL_SEQ))
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        exp = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )

        catella.compute_model_prob(exp=exp, lnuc=30)

        ana = MethPrintAnalysis()
        ana.model_prob(exp=expected, lnuc=30)

        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True,
        )

    def test_missing_refseq_raises(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        with pytest.raises(ValueError):
            catella.compute_model_prob(exp=exp, lnuc=10)

    def test_store_rho_propagates(self, tmp_path):
        rng = np.random.default_rng(11)
        rows = [
            (f"m{m}", p, "chr1", "+", float(rng.random()), "a")
            for m in range(5)
            for p in range(len(_MODEL_SEQ))
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        exp = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )

        catella.compute_model_prob(
            exp=exp, lnuc=30, eta_max_lag=4, store_rho=True
        )

        ana = MethPrintAnalysis()
        ana.model_prob(exp=expected, lnuc=30, eta_max_lag=4, store_rho=True)

        assert ("meth_prob_rho" in exp.global_analysis) == (
            "meth_prob_rho" in expected.global_analysis
        )
        if "meth_prob_rho" in exp.global_analysis:
            pd.testing.assert_frame_equal(
                exp.global_analysis["meth_prob_rho"],
                expected.global_analysis["meth_prob_rho"],
            )

    def test_span_propagates(self, tmp_path):
        rows = [
            (f"m{m}", p, "chr1", "+", 0.5, "a")
            for m in range(5)
            for p in range(5, len(_MODEL_SEQ) - 5)
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        exp = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )

        catella.compute_model_prob(exp=exp, lnuc=30, span="calls")
        MethPrintAnalysis().model_prob(exp=expected, lnuc=30, span="calls")

        assert exp.global_analysis["meth_prob_params"]["span"][0] == "calls"
        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True,
        )

    def test_channels_propagate(self, tmp_path):
        rows = [
            (f"m{m}", p, "chr1", "+", 0.5, "a")
            for m in range(5)
            for p in range(5, len(_MODEL_SEQ) - 5)
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        exp = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )

        catella.compute_model_prob(exp=exp, lnuc=30, channels="GCH")
        MethPrintAnalysis().model_prob(exp=expected, lnuc=30, channels="GCH")

        params = exp.global_analysis["meth_prob_params"]
        assert params["channels"][0] == "GCH"
        eta_row = exp.global_analysis["meth_prob_eta"].iloc[0]
        assert list(eta_row.index) == ["eta_GCH"]
        np.testing.assert_allclose(
            exp.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True,
        )


class TestEstimateStartTemp:
    def test_returns_percentile_of_abs_logit(self):
        exp = _make_experiment(nmol=20, nbp=4, seed=5)
        rng = np.random.default_rng(9)
        known = rng.uniform(0.01, 0.99, size=(20, 4))
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(known)

        got = catella.estimate_start_temp(
            exp, percentile=90.0, percentile_sample_size=100, seed=1
        )

        expected = np.nanpercentile(np.abs(logit(known)), 90.0)
        assert got == pytest.approx(expected)

    def test_batching_matches_single_batch(self):
        exp_a = _make_experiment(nmol=20, nbp=4, seed=5)
        exp_b = _make_experiment(nmol=20, nbp=4, seed=5)
        rng = np.random.default_rng(2)
        known = rng.uniform(0.01, 0.99, size=(20, 4))
        exp_a.analysis["chr1"]["meth_prob"] = pd.DataFrame(known)
        exp_b.analysis["chr1"]["meth_prob"] = pd.DataFrame(known)

        a = catella.estimate_start_temp(
            exp_a, batch_size=1000, percentile_sample_size=100, seed=0
        )
        b = catella.estimate_start_temp(
            exp_b, batch_size=3, percentile_sample_size=100, seed=0
        )
        assert a == pytest.approx(b)

    def test_mask_name_excludes_dropped_molecules(self):
        exp = _make_experiment(nmol=6, nbp=4, seed=5)
        rng = np.random.default_rng(3)
        known = rng.uniform(0.4, 0.6, size=(6, 4))
        known[0] = 0.001  # extreme outlier molecule
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(known)
        keep = np.array([0, 1, 1, 1, 1, 1], dtype=np.int8)
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": keep}
        )

        masked = catella.estimate_start_temp(
            exp,
            percentile=90.0,
            percentile_sample_size=100,
            seed=0,
            mask_name="dropout_mask",
        )
        unmasked = catella.estimate_start_temp(
            exp, percentile=90.0, percentile_sample_size=100, seed=0
        )

        expected = np.nanpercentile(np.abs(logit(known[keep == 1])), 90.0)
        assert masked == pytest.approx(expected)
        assert masked < unmasked

    def test_missing_mask_raises_key_error(self):
        exp = _make_experiment(nmol=6, nbp=4, seed=5)
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(np.full((6, 4), 0.5))
        with pytest.raises(KeyError):
            catella.estimate_start_temp(exp, mask_name="missing")

    def test_nan_and_inf_logit_values_do_not_raise(self):
        exp = _make_experiment(nmol=4, nbp=3, seed=5)
        known = np.array(
            [
                [0.0, 0.5, 1.0],
                [np.nan, 0.5, 0.5],
                [0.2, 0.3, 0.4],
                [0.6, 0.7, 0.8],
            ]
        )
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(known)

        got = catella.estimate_start_temp(
            exp, percentile=50.0, percentile_sample_size=100, seed=0
        )
        assert np.isfinite(got)

    @pytest.mark.parametrize(
        "saturated, bound",
        [
            (1.0, 1 - np.finfo(np.float64).epsneg),
            (0.0, np.finfo(np.float64).tiny),
        ],
    )
    def test_saturated_probabilities_give_finite_estimate(
        self, saturated, bound
    ):
        exp = _make_experiment(nmol=10, nbp=4, seed=5)
        known = np.full((10, 4), 0.5)
        known[:5] = saturated
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(known)

        got = catella.estimate_start_temp(
            exp, percentile=99.0, percentile_sample_size=100, seed=0
        )
        assert got == pytest.approx(abs(logit(bound)))

    def test_chroms_restricts_to_selected_chromosomes(self):
        raw1 = MethPrintData._create(
            chrom="chr1",
            nbp=3,
            test_mol_id=np.array(["m0"], dtype=object),
            test_data=pd.DataFrame(
                [(0, 0, "+", 0.5, 0)],
                columns=["mol_index", "pos", "strand", "mod_qual", "mod_code"],
            ),
            meth_mol_id=None,
            meth_data=None,
            unmeth_mol_id=None,
            unmeth_data=None,
        )
        raw2 = MethPrintData._create(
            chrom="chr2",
            nbp=3,
            test_mol_id=np.array(["m0"], dtype=object),
            test_data=pd.DataFrame(
                [(0, 0, "+", 0.5, 0)],
                columns=["mol_index", "pos", "strand", "mod_qual", "mod_code"],
            ),
            meth_mol_id=None,
            meth_data=None,
            unmeth_mol_id=None,
            unmeth_data=None,
        )
        exp = MethPrintExperiment._create(
            _raw_data={"chr1": raw1, "chr2": raw2}
        )
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(np.full((5, 3), 0.5))
        exp.analysis["chr2"]["meth_prob"] = pd.DataFrame(np.full((5, 3), 0.99))

        got = catella.estimate_start_temp(
            exp, chroms="chr1", percentile_sample_size=100, seed=0
        )
        expected = np.nanpercentile(np.abs(logit(np.full((5, 3), 0.5))), 90.0)
        assert got == pytest.approx(expected)

    def test_exp_accepted_positionally(self):
        exp = _make_experiment(nmol=6, nbp=4, seed=5)
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(np.full((6, 4), 0.5))
        got = catella.estimate_start_temp(exp)
        assert np.isfinite(got)


def _make_tm_dataset(tmp_path, nmol=4, nbp=15, seed=2):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        chroms="chr1",
        nsim=0,
        method="tm",
        settings=_settings(),
        meth_prob=meth_prob,
        out_dir=tmp_path / "tm_sim",
        seed=seed,
    )


class TestEstimateLlink:
    def _experiment(self, linker=10, nmol=50, nbp=200):
        exp = _make_experiment(nmol=nmol, nbp=nbp)
        energy = np.full((nmol, nbp), 5.0)
        energy[:, 0 : nbp - 4 : 5 + linker] = -20.0
        exp.analysis["chr1"]["meth_prob"] = pd.DataFrame(
            1.0 / (1.0 + np.exp(-energy))
        )
        return exp

    def _settings(self):
        return SimSettings(
            nucbp=5, llink=99, elink=25.0, mu=0.0, end_temp=1.0, emax=100.0
        )

    def test_is_exported(self):
        assert "estimate_llink" in catella.__all__

    def test_returns_value_per_chromosome(self):
        best = catella.estimate_llink(
            self._experiment(), self._settings(), llinks=[0, 10, 30]
        )
        assert best == {"chr1": 10}

    def test_forwards_keywords(self):
        exp = self._experiment()
        with pytest.warns(UserWarning, match="from 5 molecules"):
            catella.estimate_llink(
                exp,
                self._settings(),
                llinks=[0, 10, 30],
                chroms="chr1",
                max_nmol=5,
                seed=1,
            )

    def test_parallel_matches_serial(self):
        exp = self._experiment()
        serial = catella.estimate_llink(
            exp, self._settings(), llinks=[0, 10, 30]
        )
        parallel = catella.estimate_llink(
            exp, self._settings(), llinks=[0, 10, 30], nworker=2
        )
        assert parallel == serial == {"chr1": 10}

    def test_invalid_nworker_raises(self):
        with pytest.raises(ValueError, match="nworker must be a positive"):
            catella.estimate_llink(
                self._experiment(), self._settings(), nworker=0
            )

    def test_does_not_modify_experiment(self):
        exp = self._experiment()
        keys = set(exp.analysis["chr1"])
        catella.estimate_llink(exp, self._settings(), llinks=[0, 10, 30])
        assert set(exp.analysis["chr1"]) == keys

    def test_estimate_reaches_simulation_through_settings(self, tmp_path):
        exp = self._experiment()
        settings = self._settings()
        best = catella.estimate_llink(exp, settings, llinks=[0, 10, 30])
        dataset = catella.run(
            chroms="chr1",
            nsim=1,
            method="tm",
            settings=replace(settings, llink=best["chr1"]),
            meth_prob={"chr1": exp.analysis["chr1"]["meth_prob"]},
            out_dir=tmp_path / "sim",
            seed=1,
            verbose=False,
        )
        assert dataset.settings["llink"] == 10
        assert dataset.raw["chr1", 0, 0].llink == 10

    def test_estimate_dict_runs_directly(self, tmp_path):
        exp = self._experiment()
        settings = self._settings()
        best = catella.estimate_llink(exp, settings, llinks=[0, 10, 30])
        dataset = catella.run(
            chroms="chr1",
            nsim=0,
            method="tm",
            settings=replace(settings, llink=best),
            meth_prob={"chr1": exp.analysis["chr1"]["meth_prob"]},
            out_dir=tmp_path / "sim",
            seed=1,
            verbose=False,
        )
        catella.analyze(dataset)
        direct = catella.run(
            chroms="chr1",
            nsim=0,
            method="tm",
            settings=replace(settings, llink=10),
            meth_prob={"chr1": exp.analysis["chr1"]["meth_prob"]},
            out_dir=tmp_path / "direct",
            seed=1,
            verbose=False,
        )
        catella.analyze(direct)
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["occup"][:],
            direct.analysis["chr1"]["occup"][:],
        )


class TestRun:
    def test_method_tm_creates_tm_dataset(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        assert dataset.method == "tm"
        assert dataset.nsim == 0


class TestAnalyze:
    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        catella.analyze(dataset)
        assert "occup" in dataset.analysis["chr1"]
        assert "mean_nnuc" in dataset.analysis["chr1"]

    def test_tm_dataset_computes_occup_and_mean_nnuc(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.analyze(dataset)
        assert "occup" in dataset.analysis["chr1"]
        assert "mean_nnuc" in dataset.analysis["chr1"]

    def test_compute_var_on_tm_dataset(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.analyze(dataset, compute_var=True)
        assert "var_nnuc" in dataset.analysis["chr1"]
        assert "var_occup" in dataset.analysis["chr1"]

    def test_compute_var_requires_tm_method(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError):
            catella.analyze(dataset, compute_var=True)
