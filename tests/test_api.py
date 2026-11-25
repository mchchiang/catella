# test_api.py

from dataclasses import replace

import catella
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
from catella import utils
from catella.experiment.methdata import MethPrintData, MethPrintExperiment
from catella.experiment.plot import MethPlot
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from scipy.special import logit


def _settings():
    return SimSettings(
        lnuc=5,
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
        regions="chr1",
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
        region="chr1",
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
        f.writelines(f">{region}\n{seq}\n" for region, seq in records.items())


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

    def test_dataset_dispatch_stores_link_mat(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        _, ref_link = utils.compute_linkage(occup)

        link_mats = catella.sort_by_linkage(dataset=dataset)

        link_mat = dataset.analysis["chr1"]["occup_linkage"].to_numpy()
        np.testing.assert_allclose(link_mat, ref_link)
        np.testing.assert_allclose(link_mats["chr1"], ref_link)
        assert "occup_sorted" not in dataset.analysis["chr1"]

    def test_experiment_dispatch_with_raw_which(self):
        exp = _make_experiment()

        catella.sort_by_linkage(exp=exp, raw_which="test")

        assert "test_linkage" in exp.analysis["chr1"]

    def test_experiment_dispatch_with_mask_name(self):
        exp = _make_experiment()
        keep = np.zeros(6, dtype=bool)
        keep[0] = True
        exp.analysis["chr1"]["test_drop"] = pd.DataFrame({"keep": keep})

        link_mats = catella.sort_by_linkage(
            exp=exp, raw_which="test", mask_name="drop", fill_nan="mean"
        )

        assert len(link_mats["chr1"]) == 5

    def test_dataset_dispatch_with_config_name(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_config(dataset)
        link_mats = catella.sort_by_linkage(
            dataset=dataset, config_name="best_match"
        )
        assert len(link_mats["chr1"]) == 3
        assert "best_match_linkage" in dataset.analysis["chr1"]

    def test_config_name_with_exp_raises(self):
        with pytest.raises(ValueError, match="only used with 'dataset'"):
            catella.sort_by_linkage(
                exp=_make_experiment(), config_name="best_match"
            )

    def test_mask_name_with_dataset_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError):
            catella.sort_by_linkage(dataset=dataset, mask_name="drop")

    def test_store_link_mat_false_skips_persistence(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)

        link_mats = catella.sort_by_linkage(
            dataset=dataset, store_link_mat=False
        )

        assert "chr1" in link_mats
        assert "occup_linkage" not in dataset.analysis["chr1"]

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)

        catella.sort_by_linkage(dataset=dataset, data_name="occup_nan")
        link_mat = dataset.analysis["chr1"]["occup_nan_linkage"].to_numpy()
        assert np.isfinite(link_mat).all()

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

        assert exp.regions == ("chr1",)
        assert sorted(exp.raw["chr1"].test.mol_id) == ["m0", "m1"]

    def test_count_5hmc_forwarded(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(
            test_file,
            [
                ("m0", 1, "chr1", "+", 0.3, "m"),
                ("m0", 1, "chr1", "+", 0.2, "h"),
            ],
        )
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 5})

        exp = catella.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            mtase="CG",
            count_5hmc=True,
        )

        qual = exp.raw["chr1"].test.data["mod_qual"].tolist()
        assert qual == pytest.approx([0.5])

    def test_regions_forwarded(self, tmp_path):
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
            chromsize=chromsize, test_file=test_file, regions=["chr1"]
        )

        assert exp.regions == ("chr1",)

        with pytest.raises(ValueError):
            catella.load_raw(
                chromsize=chromsize, test_file=test_file, regions=["bogus"]
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

    def test_regions_forwarded(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        with pytest.raises(ValueError):
            catella.filter_dropout(exp=exp, regions=["bogus"])

    def test_channels_forwarded(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        with pytest.raises(ValueError):
            catella.filter_dropout(exp=exp, channels=["GC"])

    def test_exp_accepted_positionally(self):
        nbp = 10
        exp = _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )
        catella.filter_dropout(exp, thres_max=0.5)
        assert "test_dropout_mask" in exp.analysis["chr1"]


class TestPlotDropoutEcdf:
    def _make(self):
        nbp = 10
        return _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )

    def test_which_writes_figure(self, tmp_path):
        out_file = tmp_path / "ecdf.png"
        catella.plot_dropout_ecdf(
            self._make(), regions="chr1", which="test", out_file=out_file,
            show=False,
        )
        assert out_file.exists()

    def test_missing_which_source_raises(self):
        with pytest.raises(ValueError):
            catella.plot_dropout_ecdf(
                self._make(), regions="chr1", which="meth", show=False
            )


class TestQC:
    def _make(self):
        nbp = 10
        return _make_experiment(
            nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
        )

    def test_matches_direct_method_call(self):
        got = catella.qc(exp=self._make(), percentiles=[25, 75])
        expected = self._make().qc(percentiles=[25, 75])
        pd.testing.assert_frame_equal(got, expected)

    def test_stores_table(self):
        exp = self._make()
        df = catella.qc(exp=exp)
        pd.testing.assert_frame_equal(exp.global_analysis["qc"], df)

    def test_overwrite_forwarded(self):
        exp = self._make()
        catella.qc(exp=exp)
        with pytest.raises(ValueError):
            catella.qc(exp=exp)
        catella.qc(exp=exp, overwrite=True)

    def test_regions_forwarded(self):
        with pytest.raises(ValueError):
            catella.qc(exp=self._make(), regions=["bogus"])


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

    def test_regions_restricts_to_selected_regions(self):
        raw1 = MethPrintData._create(
            region="chr1",
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
            region="chr2",
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
            exp, regions="chr1", percentile_sample_size=100, seed=0
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
        regions="chr1",
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
            lnuc=5, llink=99, elink=25.0, mu=0.0, end_temp=1.0, emax=100.0
        )

    def test_is_exported(self):
        assert "estimate_llink" in catella.__all__

    def test_returns_value_per_region(self):
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
                regions="chr1",
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
        with pytest.raises(ValueError, match="nworker must be"):
            catella.estimate_llink(
                self._experiment(), self._settings(), nworker=0
            )

    def test_does_not_modify_experiment(self):
        exp = self._experiment()
        keys = set(exp.analysis["chr1"])
        catella.estimate_llink(exp, self._settings(), llinks=[0, 10, 30])
        assert set(exp.analysis["chr1"]) == keys

    def test_llink_evid_name_stores_table(self):
        exp = self._experiment()
        catella.estimate_llink(
            exp,
            self._settings(),
            llinks=[30, 0, 10],
            llink_evid_name="llink_evid",
        )
        table = exp.analysis["chr1"]["llink_evid"]
        assert list(table.columns) == ["llink", "evidence", "nmol"]
        assert table["llink"].tolist() == [0, 10, 30]
        assert (table["nmol"] == 50).all()

    def test_llink_evid_name_matches_estimate(self):
        exp = self._experiment()
        best = catella.estimate_llink(
            exp,
            self._settings(),
            llinks=[0, 10, 30],
            llink_evid_name="llink_evid",
        )
        table = exp.analysis["chr1"]["llink_evid"]
        top = table.loc[table["evidence"].idxmax(), "llink"]
        assert best == {"chr1": int(top)}

    def test_llink_evid_name_survives_save(self, tmp_path):
        exp = self._experiment()
        catella.estimate_llink(
            exp,
            self._settings(),
            llinks=[0, 10, 30],
            llink_evid_name="llink_evid",
        )
        exp.save(tmp_path / "exp.h5")
        loaded = MethPrintExperiment.load(tmp_path / "exp.h5")
        pd.testing.assert_frame_equal(
            loaded.analysis["chr1"]["llink_evid"],
            exp.analysis["chr1"]["llink_evid"],
        )

    def test_estimate_reaches_simulation_through_settings(self, tmp_path):
        exp = self._experiment()
        settings = self._settings()
        best = catella.estimate_llink(exp, settings, llinks=[0, 10, 30])
        dataset = catella.run(
            regions="chr1",
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
            regions="chr1",
            nsim=0,
            method="tm",
            settings=replace(settings, llink=best),
            meth_prob={"chr1": exp.analysis["chr1"]["meth_prob"]},
            out_dir=tmp_path / "sim",
            seed=1,
            verbose=False,
        )
        catella.compute_occup(dataset)
        direct = catella.run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=replace(settings, llink=10),
            meth_prob={"chr1": exp.analysis["chr1"]["meth_prob"]},
            out_dir=tmp_path / "direct",
            seed=1,
            verbose=False,
        )
        catella.compute_occup(direct)
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["occup"][:],
            direct.analysis["chr1"]["occup"][:],
        )


class TestComputeConfig:
    def test_default_is_best_match(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_config(dataset)
        ref = _make_tm_dataset(tmp_path / "ref")
        SimAnalysis().compute_best_match_config(dataset=ref, name="ref")
        got = catella.extract_config(dataset, region="chr1", name="best_match")
        want = catella.extract_config(ref, region="chr1", name="ref")
        for a, b in zip(got, want, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_most_likely_default_name(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_config(dataset, method="most_likely")
        assert "most_likely_energy" in dataset.analysis["chr1"]

    def test_two_gammas_coexist(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_config(dataset, gamma=0.5, name="g05")
        catella.compute_config(dataset, gamma=4.0, name="g4")
        low = catella.extract_config(dataset, region="chr1", name="g05")
        high = catella.extract_config(dataset, region="chr1", name="g4")
        assert sum(len(c) for c in high) >= sum(len(c) for c in low)

    def test_unknown_method_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="method must be"):
            catella.compute_config(dataset, method="map")

    def test_gamma_with_most_likely_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="gamma is only used"):
            catella.compute_config(dataset, method="most_likely", gamma=2.0)

    def test_mcmc_dataset_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError, match="requires a method='tm'"):
            catella.compute_config(dataset)


class TestComputeLinkerHist:
    def test_is_exported(self):
        assert "compute_linker_hist" in catella.__all__

    def test_matches_analysis_method(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_linker_hist(
            dataset, lmax=3, trim=1, regions="chr1", name="lh", nworker=1
        )
        ref = _make_tm_dataset(tmp_path / "ref")
        SimAnalysis().compute_linker_hist(ref, lmax=3, trim=1, name="ref")
        np.testing.assert_allclose(
            dataset.analysis["chr1"]["lh"].to_numpy(),
            ref.analysis["chr1"]["ref"].to_numpy(),
        )
        pd.testing.assert_frame_equal(
            dataset.analysis["chr1"]["lh_settings"],
            ref.analysis["chr1"]["ref_settings"],
        )

    def test_window_forwarded(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        nbp = dataset.nbp["chr1"]
        catella.compute_linker_hist(dataset, lmax=2, window=(1, nbp - 1))
        settings = dataset.analysis["chr1"]["linker_hist_settings"]
        assert settings["xlo"].iloc[0] == 1
        assert settings["xhi"].iloc[0] == nbp - 1

    def test_mcmc_dataset_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError, match="requires a method='tm'"):
            catella.compute_linker_hist(dataset)


class TestExtractLinkerStats:
    def test_is_exported(self):
        assert "extract_linker_stats" in catella.__all__

    @pytest.mark.parametrize("pooled", [False, True])
    def test_matches_analysis_method(self, tmp_path, pooled):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_linker_hist(dataset, lmax=4)
        got = catella.extract_linker_stats(
            dataset, mols=[0, 1], pooled=pooled
        )
        want = SimAnalysis().extract_linker_stats(
            dataset, mols=[0, 1], pooled=pooled
        )
        pd.testing.assert_frame_equal(got, want)


class TestPlotLinkerHist:
    def test_is_exported(self):
        assert "plot_linker_hist" in catella.__all__

    def test_writes_figure_file(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_linker_hist(dataset, lmax=4)
        out_file = tmp_path / "linker.png"
        catella.plot_linker_hist(
            dataset,
            regions="chr1",
            panels=True,
            density=False,
            out_file=out_file,
            show=False,
        )
        assert out_file.exists()
        assert plt.gcf().axes[0].get_title() == "chr1"


class TestPlotConfig:
    def test_writes_figure_file(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_config(dataset)
        out_file = tmp_path / "configs.png"
        catella.plot_config(
            dataset,
            regions="chr1",
            name="best_match",
            max_rows=2,
            out_file=out_file,
            show=False,
        )
        assert out_file.exists()


class TestPlotOccup:
    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_several_regions_with_linkage_dict(self, tmp_path):
        rng = np.random.default_rng(4)
        dataset = SimManager(nworker=1, verbose=False).run(
            regions=["chr1", "chr2"],
            nsim=2,
            settings=_settings(),
            meth_prob={
                "chr1": rng.random((4, 10)),
                "chr2": rng.random((3, 12)),
            },
            out_dir=tmp_path / "sim",
            seed=4,
        )
        for region, shape in (("chr1", (4, 10)), ("chr2", (3, 12))):
            dataset.analysis[region]["occup"] = pd.DataFrame(rng.random(shape))
        link = catella.sort_by_linkage(dataset=dataset, data_name="occup")
        out_file = tmp_path / "occup.png"
        catella.plot_occup(
            dataset,
            regions=["chr1", "chr2"],
            link_mat=link,
            sharex=True,
            out_file=out_file,
            show=False,
        )
        assert out_file.exists()

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_plot_mean(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        catella.compute_occup(dataset)
        catella.plot_occup(dataset, regions="chr1", plot_mean=True, show=False)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        mean_ax = next(
            ax for ax in plt.gcf().axes if ax.get_ylabel() == "Mean"
        )
        np.testing.assert_allclose(
            mean_ax.lines[0].get_ydata(), occup.mean(axis=0)
        )


class TestPlotMeth:
    def _data(self):
        return np.random.default_rng(0).uniform(0.05, 0.95, (6, 10))

    def _record(self, monkeypatch):
        calls = {}

        def fake(name):
            def method(self, *args, **kwargs):
                calls[name] = (args, kwargs)

            return method

        monkeypatch.setattr(MethPlot, "plot_meth_prob", fake("prob"))
        monkeypatch.setattr(MethPlot, "plot_meth_energy", fake("energy"))
        return calls

    @pytest.mark.parametrize("value", ["prob", "energy"])
    def test_writes_figure_file(self, tmp_path, value):
        out_file = tmp_path / f"{value}.png"
        catella.plot_meth(
            self._data(), value=value, out_file=out_file, show=False
        )
        assert out_file.exists()

    def test_raw_writes_figure_file(self, tmp_path):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        out_file = tmp_path / "raw.png"
        catella.plot_meth(
            exp=exp, regions="chr1", raw_which="test", out_file=out_file,
            show=False,
        )
        assert out_file.exists()

    def _two_region_exp(self):
        rng = np.random.default_rng(5)
        raws = {}
        for region, nmol, nbp in (("chr1", 6, 10), ("chr2", 4, 14)):
            rows = [
                (m, p, "+", float(rng.random()), 0)
                for m in range(nmol)
                for p in rng.choice(nbp, size=6, replace=False)
            ]
            raws[region] = MethPrintData._create(
                region=region,
                nbp=nbp,
                test_mol_id=np.array(
                    [f"m{m}" for m in range(nmol)], dtype=object
                ),
                test_data=pd.DataFrame(
                    rows,
                    columns=["mol_index", "pos", "strand", "mod_qual",
                             "mod_code"],
                ),
                meth_mol_id=None,
                meth_data=None,
                unmeth_mol_id=None,
                unmeth_data=None,
            )
        exp = MethPrintExperiment._create(_raw_data=raws)
        exp.analysis["chr1"]["prob"] = pd.DataFrame(
            rng.uniform(0.1, 0.5, (6, 10))
        )
        exp.analysis["chr2"]["prob"] = pd.DataFrame(
            rng.uniform(0.4, 0.9, (4, 14))
        )
        return exp

    def _images(self):
        return [ax.images[0] for ax in plt.gcf().axes if ax.images]

    @pytest.mark.parametrize("value", ["prob", "energy"])
    def test_data_name_several_regions(self, value):
        exp = self._two_region_exp()
        catella.plot_meth(
            exp=exp,
            regions=["chr1", "chr2"],
            data_name="prob",
            value=value,
            show=False,
        )
        images = self._images()
        assert [im.get_array().shape for im in images] == [(6, 10), (4, 14)]
        assert len(plt.gcf().axes) == 3

    def test_raw_several_regions_with_mols_dict(self):
        exp = self._two_region_exp()
        catella.plot_meth(
            exp=exp,
            regions=["chr1", "chr2"],
            raw_which="test",
            mols={"chr2": [0, 1]},
            show=False,
        )
        shapes = [im.get_array().shape for im in self._images()]
        assert shapes == [(6, 10), (2, 14)]

    def test_common_color_scale(self):
        exp = self._two_region_exp()
        catella.plot_meth(
            exp=exp, regions=["chr1", "chr2"], data_name="prob", show=False
        )
        data = [exp.analysis[r]["prob"].to_numpy() for r in ("chr1", "chr2")]
        for im in self._images():
            assert im.norm.vmin == pytest.approx(min(d.min() for d in data))
            assert im.norm.vmax == pytest.approx(max(d.max() for d in data))

    def test_energy_common_symmetric_scale(self):
        exp = self._two_region_exp()
        catella.plot_meth(
            exp=exp,
            regions=["chr1", "chr2"],
            data_name="prob",
            value="energy",
            show=False,
        )
        halfrange = max(
            np.abs(logit(exp.analysis[r]["prob"].to_numpy())).max()
            for r in ("chr1", "chr2")
        )
        for im in self._images():
            assert im.norm.vmin == pytest.approx(-halfrange)
            assert im.norm.vmax == pytest.approx(halfrange)

    def test_link_mat_dict_sorts_each_region(self):
        exp = self._two_region_exp()
        link = catella.sort_by_linkage(exp=exp, data_name="prob")
        catella.plot_meth(
            exp=exp,
            regions=["chr1", "chr2"],
            data_name="prob",
            link_mat=link,
            show=False,
        )
        for region, im in zip(("chr1", "chr2"), self._images()):
            order = sch.leaves_list(link[region])
            np.testing.assert_allclose(
                np.ma.getdata(im.get_array()),
                exp.analysis[region]["prob"].to_numpy()[order],
            )

    def test_sharex_uses_widest_region(self):
        exp = self._two_region_exp()
        catella.plot_meth(
            exp=exp,
            regions=["chr1", "chr2"],
            data_name="prob",
            sharex=True,
            show=False,
        )
        xlims = [ax.get_xlim() for ax in plt.gcf().axes if ax.images]
        assert xlims == [(0, 14), (0, 14)]

    def test_positions_drawn_in_bp(self):
        catella.plot_meth(self._data(), show=False)
        fig = plt.gcf()
        fig.canvas.draw()
        hm_ax = fig.axes[0]
        assert hm_ax.get_xlim() == pytest.approx((0, 10))
        assert hm_ax.get_xlabel() == "Position $x$ [bp]"

    @pytest.mark.parametrize("value", ["prob", "energy"])
    def test_xscale_removed(self, value):
        with pytest.raises(TypeError, match="xscale"):
            catella.plot_meth(
                self._data(), value=value, xscale=1000, show=False
            )

    def test_data_with_regions_raises(self):
        exp = self._two_region_exp()
        with pytest.raises(ValueError, match="cannot be combined"):
            catella.plot_meth(
                self._data(), exp=exp, regions="chr1", show=False
            )

    def test_raw_which_and_data_name_raises(self):
        exp = self._two_region_exp()
        with pytest.raises(ValueError, match="exactly one of"):
            catella.plot_meth(
                exp=exp,
                regions="chr1",
                raw_which="test",
                data_name="prob",
                show=False,
            )

    def test_mols_with_data_name_raises(self):
        exp = self._two_region_exp()
        with pytest.raises(ValueError, match="require 'raw_which'"):
            catella.plot_meth(
                exp=exp, regions="chr1", data_name="prob", mols=[0],
                show=False,
            )

    def test_prob_dispatches_to_plot_meth_prob(self, monkeypatch):
        calls = self._record(monkeypatch)
        catella.plot_meth(self._data(), show=False)
        assert list(calls) == ["prob"]
        assert "cbar_label" not in calls["prob"][1]

    def test_energy_dispatches_to_plot_meth_energy(self, monkeypatch):
        calls = self._record(monkeypatch)
        catella.plot_meth(self._data(), value="energy", emax=3.0, show=False)
        assert list(calls) == ["energy"]
        assert calls["energy"][1]["emax"] == 3.0
        assert "cbar_label" not in calls["energy"][1]

    def test_cbar_label_forwarded(self, monkeypatch):
        calls = self._record(monkeypatch)
        catella.plot_meth(
            self._data(), value="energy", cbar_label="E", show=False
        )
        assert calls["energy"][1]["cbar_label"] == "E"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="value must be"):
            catella.plot_meth(self._data(), value="logit", show=False)

    def test_emax_with_prob_raises(self):
        with pytest.raises(ValueError, match="emax"):
            catella.plot_meth(self._data(), emax=3.0, show=False)

    def test_energy_with_raw_which_raises(self):
        exp = _make_experiment(nmol=6, nbp=10, seed=3)
        with pytest.raises(ValueError, match="does not support raw_which"):
            catella.plot_meth(
                exp=exp, regions="chr1", raw_which="test", value="energy",
                show=False,
            )

    def test_energy_without_data_raises(self):
        with pytest.raises(ValueError, match="provide either 'data'"):
            catella.plot_meth(value="energy", show=False)


def _interval_exp(tmp_path, start, length=10000, wrap=False):
    rows = [
        (f"m{m}", start + p, "chr1", "+", 0.5, "a")
        for m in range(3)
        for p in range(0, length, 500)
    ]
    test_file = tmp_path / "test.tsv"
    _write_tsv(test_file, rows)
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {"chr1": 4000000})
    if wrap:
        _write_chromsize(chromsize, {"chr1": length})
        return catella.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True
        )
    return catella.load_raw(
        chromsize=chromsize,
        test_file=test_file,
        regions=[("chr1", start, start + length, "reg")],
    )


def _tick_texts(ax):
    return [t.get_text() for t in ax.get_xticklabels() if t.get_text()]


class TestPlotMethCoords:
    def _plot(self, exp, **kwargs):
        catella.plot_meth(
            exp=exp, regions="reg", raw_which="test", show=False, **kwargs
        )
        fig = plt.gcf()
        fig.canvas.draw()
        return fig.axes[0]

    def test_genomic_labels(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        ax = self._plot(exp, coords=True)
        assert ax.get_xlabel() == "Position on chr1 [kb]"
        assert "3,170" in _tick_texts(ax)
        assert ax.get_xlim() == (0, 10000)

    def test_odd_start_gives_round_ticks(self, tmp_path):
        exp = _interval_exp(tmp_path, 3171234)
        ax = self._plot(exp, coords=True)
        values = [float(t.replace(",", "")) for t in _tick_texts(ax)]
        assert values
        assert all(round(v * 1000) % 500 == 0 for v in values)

    def test_table_equals_true(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        from_true = _tick_texts(self._plot(exp, coords=True))
        plt.close("all")
        from_table = _tick_texts(self._plot(exp, coords=exp.region_coords))
        assert from_true == from_table

    def test_default_is_zero_based(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        ax = self._plot(exp)
        assert ax.get_xlabel() == "Position $x$ [kb]"
        assert "0" in _tick_texts(ax)

    def test_narrower_data_passes(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        exp.analysis["reg"]["narrow"] = pd.DataFrame(np.full((3, 9854), 0.5))
        catella.plot_meth(
            exp=exp, regions="reg", data_name="narrow", coords=True,
            show=False,
        )
        assert plt.gcf().axes[0].get_xlabel() == "Position on chr1 [kb]"

    def test_data_with_coords_raises(self):
        data = np.full((2, 10), 0.5)
        with pytest.raises(ValueError, match="coords requires 'exp'"):
            catella.plot_meth(data, coords=True, show=False)

    def test_missing_region_raises(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        table = exp.region_coords.rename(index={"reg": "other"})
        with pytest.raises(ValueError, match="coords has no region 'reg'"):
            self._plot(exp, coords=table)

    def test_nbp_mismatch_raises(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        table = exp.region_coords
        table.loc["reg", "nbp"] = 9000
        with pytest.raises(ValueError, match="coords has nbp 9000"):
            self._plot(exp, coords=table)

    def test_no_coordinates_raises(self, tmp_path):
        exp = _interval_exp(tmp_path, 3170000)
        table = exp.region_coords
        table.loc["reg", "ref_start"] = np.nan
        with pytest.raises(ValueError, match="has no genomic coordinates"):
            self._plot(exp, coords=table)

    def test_wrap_raises(self, tmp_path):
        exp = _interval_exp(tmp_path, 0, length=1000, wrap=True)
        with pytest.raises(ValueError, match="is folded"):
            catella.plot_meth(
                exp=exp, regions="chr1", raw_which="test", coords=True,
                show=False,
            )


class TestExtractConfig:
    def test_matches_simanalysis(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_config(dataset)
        got = catella.extract_config(dataset, region="chr1", name="best_match")
        want = SimAnalysis().extract_config(
            dataset, region="chr1", name="best_match"
        )
        assert len(got) == 4
        for a, b in zip(got, want, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_mcmc_raw_run(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        got = catella.extract_config(dataset, region="chr1", run=1, mols=[3])
        np.testing.assert_array_equal(
            got[0], dataset.raw["chr1", 3, 1].position[-1]
        )


class TestRun:
    def test_method_tm_creates_tm_dataset(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        assert dataset.method == "tm"
        assert dataset.nsim == 0


class TestComputeOccup:
    def test_dataset_accepted_positionally(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        catella.compute_occup(dataset)
        assert "occup" in dataset.analysis["chr1"]
        assert "mean_nnuc" in dataset.analysis["chr1"]

    def test_tm_dataset_computes_occup_and_mean_nnuc(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_occup(dataset)
        assert "occup" in dataset.analysis["chr1"]
        assert "mean_nnuc" in dataset.analysis["chr1"]

    def test_compute_var_on_tm_dataset(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_occup(dataset, compute_var=True)
        assert "var_nnuc" in dataset.analysis["chr1"]
        assert "var_occup" in dataset.analysis["chr1"]

    def test_compute_var_requires_tm_method(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError):
            catella.compute_occup(dataset, compute_var=True)

    def test_tm_matches_separate_calls(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_occup(dataset, compute_var=True)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset, name="ref_occup")
        ana.compute_mean_nnuc(dataset=dataset, name="ref_mean_nnuc")
        ana.compute_var_nnuc(dataset=dataset, name="ref_var_nnuc")
        ana.compute_var_occup(dataset=dataset, name="ref_var_occup")
        analysis = dataset.analysis["chr1"]
        for name in ["occup", "var_occup"]:
            np.testing.assert_array_equal(
                analysis[name][:], analysis[f"ref_{name}"][:]
            )
        for name in ["mean_nnuc", "var_nnuc"]:
            pd.testing.assert_frame_equal(
                analysis[name], analysis[f"ref_{name}"]
            )

    def test_tm_custom_names(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_occup(
            dataset,
            compute_var=True,
            occup_name="o",
            mean_nnuc_name="m",
            var_nnuc_name="vn",
            var_occup_name="vo",
        )
        assert {"o", "m", "vn", "vo"} <= set(dataset.analysis["chr1"])

    def test_tm_without_var_skips_var(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        catella.compute_occup(dataset)
        assert "var_nnuc" not in dataset.analysis["chr1"]
        assert "var_occup" not in dataset.analysis["chr1"]

    def test_tm_parallel_matches_serial(self, tmp_path):
        serial = _make_tm_dataset(tmp_path / "a")
        par = _make_tm_dataset(tmp_path / "b")
        catella.compute_occup(serial, compute_var=True)
        catella.compute_occup(par, compute_var=True, nworker=2)
        for name in ["occup", "var_occup"]:
            np.testing.assert_array_equal(
                par.analysis["chr1"][name][:],
                serial.analysis["chr1"][name][:],
            )
        for name in ["mean_nnuc", "var_nnuc"]:
            pd.testing.assert_frame_equal(
                par.analysis["chr1"][name], serial.analysis["chr1"][name]
            )

    def test_tm_rejects_time(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError):
            catella.compute_occup(dataset, time=0)

    def test_nonpositive_nworker_raises(self, tmp_path):
        dataset = _make_tm_dataset(tmp_path)
        with pytest.raises(ValueError, match="nworker must be positive"):
            catella.compute_occup(dataset, nworker=0)

    def test_parallel_mcmc_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        with pytest.raises(ValueError, match="nworker > 1 requires"):
            catella.compute_occup(dataset, nworker=2)


def _plot_calls(tmp_path):
    nbp = 10
    exp = _make_experiment(
        nmol=6, nbp=nbp, seed=3, refseq="A" * nbp, mtase=("A",)
    )
    dataset = _make_tm_dataset(tmp_path)
    catella.compute_linker_hist(dataset, lmax=3, regions="chr1")
    SimAnalysis().compute_best_match_config(dataset=dataset, name="bm")
    return {
        "plot_meth": lambda **kw: catella.plot_meth(
            np.full((2, 10), 0.5), **kw
        ),
        "plot_dropout_ecdf": lambda **kw: catella.plot_dropout_ecdf(
            exp, regions="chr1", which="test", **kw
        ),
        "plot_config": lambda **kw: catella.plot_config(
            dataset, regions="chr1", name="bm", **kw
        ),
        "plot_occup": lambda **kw: catella.plot_occup(
            dataset, regions="chr1", **kw
        ),
        "plot_linker_hist": lambda **kw: catella.plot_linker_hist(
            dataset, regions="chr1", **kw
        ),
    }


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword"
    ":DeprecationWarning"
)
class TestPlotReturnsFigure:
    @pytest.mark.parametrize(
        "name",
        [
            "plot_meth",
            "plot_dropout_ecdf",
            "plot_config",
            "plot_occup",
            "plot_linker_hist",
        ],
    )
    def test_figure_without_show(self, tmp_path, name):
        call = _plot_calls(tmp_path)[name]
        fig = call(show=False)
        assert isinstance(fig, plt.Figure)
        assert fig is plt.gcf()
        assert call(show=True) is None

    def test_saving_still_works(self, tmp_path):
        out_file = tmp_path / "meth.png"
        fig = catella.plot_meth(
            np.full((2, 10), 0.5), out_file=out_file, show=False
        )
        assert out_file.exists()
        assert isinstance(fig, plt.Figure)
