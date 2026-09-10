# test_cli.py

import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import catella
from catella import utils
from catella.cli import app
from catella.experiment.methdata import MethPrintExperiment
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset

runner = CliRunner()

_TSV_HEADER = "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write(_TSV_HEADER)
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


def _settings_dict(**overrides):
    defaults = dict(nucbp=5, llink=2, mu=-1.0, start_temp=1.0,
                    end_temp=0.1, cool_option="linear", nsweep=2,
                    print_freq=1, emax=5.0)
    defaults.update(overrides)
    return defaults


def _write_settings_json(path, **overrides):
    with open(path, "w") as f:
        json.dump(_settings_dict(**overrides), f)


def _make_test_rows(nmol, nbp, seed=0, step=5):
    rng = np.random.default_rng(seed)
    return [(f"m{m}", pos, "chr1", "+", round(float(rng.random()), 3), "a")
           for m in range(nmol) for pos in range(0, nbp, step)]


def _make_dataset(tmp_path, *, nmol=4, nbp=20, nsim=2, seed=1,
                  name="results"):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(chroms="chr1", nsim=nsim,
                       settings=SimSettings(**_settings_dict()),
                       meth_prob=meth_prob, out_dir=tmp_path / "sim_out",
                       dataset_name=name, seed=seed)


class TestLoadRaw:
    def test_fasta_file_populates_refseq(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "A" * 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["load_raw", str(chromsize),
                                     str(test_file), str(cli_out),
                                     "--fasta-file", str(fasta_file)])
        assert result.exit_code == 0, result.output

        got = MethPrintExperiment.load(cli_out)
        assert got.raw["chr1"].refseq == "A" * 30

    def test_mtase_populates_experiment(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["load_raw", str(chromsize),
                                     str(test_file), str(cli_out),
                                     "--mtase", "CG,GC"])
        assert result.exit_code == 0, result.output

        got = MethPrintExperiment.load(cli_out)
        assert got.mtase == ("CG", "GC")

    def test_max_nmol_and_seed_subset_molecules(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["load_raw", str(chromsize),
                                     str(test_file), str(cli_out),
                                     "--max-nmol", "3", "--seed", "7"])
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    max_nmol=3, seed=7)
        got = MethPrintExperiment.load(cli_out)
        assert len(got.raw["chr1"].test_mol_id) == 3
        assert (sorted(got.raw["chr1"].test_mol_id)
               == sorted(expected.raw["chr1"].test_mol_id))

    def test_chroms_restricts_loaded_chromosomes(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30, "chr2": 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["load_raw", str(chromsize),
                                     str(test_file), str(cli_out),
                                     "--chroms", "chr1"])
        assert result.exit_code == 0, result.output

        got = MethPrintExperiment.load(cli_out)
        assert got.chroms == ("chr1",)


def _load_raw_a_mtase(tmp_path, *, nmol=5, nbp=30, step=1):
    rows = _make_test_rows(nmol=nmol, nbp=nbp, step=step)
    test_file = tmp_path / "test.tsv"
    _write_tsv(test_file, rows)
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {"chr1": nbp})
    fasta_file = tmp_path / "ref.fa"
    _write_fasta(fasta_file, {"chr1": "A" * nbp})

    raw_file = tmp_path / "raw_exp.h5"
    result = runner.invoke(app, ["load_raw", str(chromsize), str(test_file),
                                 str(raw_file), "--fasta-file",
                                 str(fasta_file), "--mtase", "A"])
    assert result.exit_code == 0, result.output
    return raw_file, chromsize, test_file, fasta_file


class TestFilterDropout:
    def test_matches_direct_api_call_and_persists_mask(self, tmp_path):
        raw_file, chromsize, test_file, fasta_file = _load_raw_a_mtase(
            tmp_path)

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["filter_dropout", str(raw_file),
                                     "--thres-max", "0.5",
                                     "--out-file", str(cli_out)])
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    fasta_file=fasta_file, mtase="A")
        catella.filter_dropout(exp=expected, thres_max=0.5)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_array_equal(
            got.analysis["chr1"]["test_dropout_mask"]["keep"].to_numpy(),
            expected.analysis["chr1"]["test_dropout_mask"]["keep"]
            .to_numpy())

    def test_mask_name_option(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        result = runner.invoke(app, ["filter_dropout", str(raw_file),
                                     "--thres-max", "0.5",
                                     "--mask-name", "qc_mask"])
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(raw_file)
        assert "test_qc_mask" in reloaded.analysis["chr1"]


class TestSummarizeDropout:
    def test_prints_table_and_writes_csv(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        expected = catella.summarize_dropout(
            exp=MethPrintExperiment.load(raw_file))

        csv_out = tmp_path / "summary.csv"
        result = runner.invoke(app, ["summarize_dropout", str(raw_file),
                                     "--out-file", str(csv_out)])
        assert result.exit_code == 0, result.output
        for col in expected.columns:
            assert col in result.output

        got = pd.read_csv(csv_out)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)


class TestPlotDropoutEcdf:
    def test_writes_figure_file(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        out_file = tmp_path / "dropout.png"
        result = runner.invoke(app, ["plot_dropout_ecdf", str(raw_file),
                                     "chr1", "--out-file", str(out_file),
                                     "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_skips_chromosomes_missing_refseq(self, tmp_path):
        # chr2 has no refseq; plotting chr1 must not touch it (before
        # the chroms= restriction, dropout_fractions() always looped
        # over every chromosome and would raise here).
        rows = ([("m0", i, "chr1", "+", 0.5, "a") for i in range(4)]
               + [("m0", 0, "chr2", "+", 0.5, "a")])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 4, "chr2": 4})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "AAAA"})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file),
                                          "--fasta-file", str(fasta_file),
                                          "--mtase", "A"])
        assert load_result.exit_code == 0, load_result.output

        out_file = tmp_path / "dropout.png"
        result = runner.invoke(app, ["plot_dropout_ecdf", str(raw_file),
                                     "chr1", "--out-file", str(out_file),
                                     "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()


class TestComputeEmpiricalProb:
    def test_matches_direct_api_call(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file),
                                          "--max-nmol", "3", "--seed", "7"])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_empirical_prob",
                                     str(raw_file), str(cli_out),
                                     "--binsize", "5"])
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    max_nmol=3, seed=7)
        catella.compute_empirical_prob(exp=expected, binsize=5)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy())

    def test_prob_name_renames_stored_probabilities(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file)])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_empirical_prob",
                                     str(raw_file), str(cli_out),
                                     "--binsize", "5",
                                     "--prob-name", "custom_prob"])
        assert result.exit_code == 0, result.output

        got = MethPrintExperiment.load(cli_out)
        assert "custom_prob" in got.analysis["chr1"]
        assert "meth_prob" not in got.analysis["chr1"]

    def test_fill_edge_seed_propagate(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file),
                                          "--max-nmol", "3", "--seed", "7"])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_empirical_prob",
                                     str(raw_file), str(cli_out),
                                     "--binsize", "5",
                                     "--fill-edge", "0.5",
                                     "--seed", "9"])
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    max_nmol=3, seed=7)
        catella.compute_empirical_prob(exp=expected, binsize=5,
                                       fill_edge=0.5, seed=9)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True)


_MODEL_SEQ = "AATTGCGTTAAGCTTTAACGTTAAGCGCAATT" * 8


class TestComputeModelProb:
    def test_matches_direct_api_call(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file),
                                          "--fasta-file", str(fasta_file),
                                          "--max-nmol", "3", "--seed", "7"])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_model_prob", str(raw_file),
                                     str(cli_out), "--l-nuc", "30",
                                     "--n-min", "3"])
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    fasta_file=fasta_file, max_nmol=3,
                                    seed=7)
        catella.compute_model_prob(exp=expected, l_nuc=30, n_min=3)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy())

    def test_store_rho_propagates(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1, seed=11)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file),
                                          "--fasta-file", str(fasta_file)])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_model_prob", str(raw_file),
                                     str(cli_out), "--l-nuc", "30",
                                     "--n-min", "3", "--eta-max-lag", "4",
                                     "--store-rho"])
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(chromsize=chromsize, test_file=test_file,
                                    fasta_file=fasta_file)
        catella.compute_model_prob(exp=expected, l_nuc=30, n_min=3,
                                   eta_max_lag=4, store_rho=True)

        got = MethPrintExperiment.load(cli_out)
        assert ("meth_prob_rho" in got.global_analysis) == (
            "meth_prob_rho" in expected.global_analysis)
        if "meth_prob_rho" in expected.global_analysis:
            pd.testing.assert_frame_equal(
                got.global_analysis["meth_prob_rho"],
                expected.global_analysis["meth_prob_rho"],
                check_dtype=False)

    def test_prob_name_renames_stored_probabilities(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file),
                                          "--fasta-file", str(fasta_file)])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_model_prob", str(raw_file),
                                     str(cli_out), "--l-nuc", "30",
                                     "--n-min", "3",
                                     "--prob-name", "custom_prob"])
        assert result.exit_code == 0, result.output

        got = MethPrintExperiment.load(cli_out)
        assert "custom_prob" in got.analysis["chr1"]
        assert "meth_prob" not in got.analysis["chr1"]

    def test_cli_requires_fasta_file(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(app, ["load_raw", str(chromsize),
                                          str(test_file), str(raw_file)])
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["compute_model_prob", str(raw_file),
                                     str(cli_out)])
        assert result.exit_code != 0

    def test_api_requires_fasta_file(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})

        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        with pytest.raises(ValueError):
            catella.compute_model_prob(exp=exp)


class TestRun:
    def test_matches_direct_api_call_with_mols_subset(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)

        settings_file = tmp_path / "settings.json"
        _write_settings_json(settings_file)
        out_dir = tmp_path / "sim_out"

        result = runner.invoke(app, [
            "run", "--chroms", "chr1", "--nsim", "2",
            "--settings", str(settings_file), "--exp-file", str(exp_file),
            "--out-dir", str(out_dir), "--dataset-name", "myrun",
            "--seed", "42", "--mols", "0,1", "--nworker", "1",
            "--no-verbose"])
        assert result.exit_code == 0, result.output
        cli_dataset = SimDataset.load(out_dir / "myrun.h5")
        assert list(cli_dataset.chroms) == ["chr1"]
        # nmol is the full molecule-index space, not the simulated
        # count; confirm --mols restricted simulation to mols 0 and 1.
        with pytest.raises(ValueError):
            cli_dataset.raw["chr1", 2, 0]

        exp_data = MethPrintExperiment.load(exp_file)
        meth_prob = exp_data.analysis["chr1"]["meth_prob"].to_numpy()
        expected = catella.run(chroms="chr1", nsim=2,
                             settings=SimSettings(**_settings_dict()),
                             meth_prob={"chr1": meth_prob},
                             out_dir=tmp_path / "direct_out",
                             dataset_name="direct", seed=42, mols=[0, 1],
                             nworker=1, verbose=False)
        for mol in range(2):
            for run_idx in range(2):
                np.testing.assert_allclose(
                    cli_dataset.raw["chr1", mol, run_idx].energy,
                    expected.raw["chr1", mol, run_idx].energy)
                np.testing.assert_array_equal(
                    cli_dataset.raw["chr1", mol, run_idx].position[-1],
                    expected.raw["chr1", mol, run_idx].position[-1])


class TestAnalyze:
    def test_matches_direct_api_call(self, tmp_path):
        cli_dataset = _make_dataset(tmp_path, nmol=6, nbp=15, name="cli")
        direct_dataset = SimDataset.load(cli_dataset._dataset_file)

        result = runner.invoke(app, ["analyze",
                                     str(cli_dataset._dataset_file),
                                     "--occup-name", "my_occup",
                                     "--mean-nnuc-name", "my_nnuc"])
        assert result.exit_code == 0, result.output

        catella.analyze(dataset=direct_dataset, occup_name="my_occup",
                     mean_nnuc_name="my_nnuc")

        reloaded = SimDataset.load(cli_dataset._dataset_file)
        np.testing.assert_allclose(
            reloaded.analysis["chr1"]["my_occup"].to_numpy(),
            direct_dataset.analysis["chr1"]["my_occup"].to_numpy())
        np.testing.assert_allclose(
            reloaded.analysis["chr1"]["my_nnuc"].to_numpy(),
            direct_dataset.analysis["chr1"]["my_nnuc"].to_numpy())


class TestDownsample:
    def test_dataset_downsample_matches_utils(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=8, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        expected = utils.downsample(occup, 3, how="mean")

        result = runner.invoke(app, [
            "downsample", str(dataset._dataset_file), "chr1", "occup", "3",
            "--kind", "dataset", "--how", "mean",
            "--out-key", "occup_small"])
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        got = reloaded.analysis["chr1"]["occup_small"]
        assert isinstance(got, pd.DataFrame)
        np.testing.assert_allclose(got.to_numpy(), expected)

    def test_experiment_downsample_matches_utils(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)
        exp = MethPrintExperiment.load(exp_file)
        expected = utils.downsample(
            exp.analysis["chr1"]["meth_prob"], 2, how="mean")

        result = runner.invoke(app, [
            "downsample", str(exp_file), "chr1", "meth_prob", "2",
            "--kind", "experiment"])
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(exp_file)
        got = reloaded.analysis["chr1"]["meth_prob_downsampled"]
        np.testing.assert_allclose(got.to_numpy(), expected)


class TestSortByLinkage:
    def test_dataset_sort_matches_oracle_and_persists_link_mat(
            self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        order, ref_link = utils.compute_linkage(occup)

        result = runner.invoke(app, ["sort_by_linkage",
                                     str(dataset._dataset_file),
                                     "--kind", "dataset"])
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        sorted_arr = reloaded.analysis["chr1"]["occup_sorted"].to_numpy()
        link_mat = reloaded.analysis["chr1"]["occup_linkage"].to_numpy()
        np.testing.assert_allclose(sorted_arr, occup[order])
        np.testing.assert_allclose(link_mat, ref_link)

    def test_experiment_store_link_mat_false_skips_persistence(
            self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)
        exp = MethPrintExperiment.load(exp_file)
        meth_prob = exp.analysis["chr1"]["meth_prob"].to_numpy()
        order, _ = utils.compute_linkage(meth_prob)

        result = runner.invoke(app, [
            "sort_by_linkage", str(exp_file), "--kind", "experiment",
            "--data-name", "meth_prob", "--no-store-link-mat"])
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(exp_file)
        sorted_arr = reloaded.analysis["chr1"]["meth_prob_sorted"]
        np.testing.assert_allclose(sorted_arr.to_numpy(), meth_prob[order])
        assert "meth_prob_linkage" not in reloaded.analysis["chr1"]

    def test_dataset_chroms_restricts_processing(self, tmp_path):
        rng = np.random.default_rng(3)
        nmol, nbp = 4, 20
        meth_prob = {chrom: rng.random((nmol, nbp))
                    for chrom in ("chr1", "chr2")}
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms=["chr1", "chr2"], nsim=2,
                              settings=SimSettings(**_settings_dict()),
                              meth_prob=meth_prob,
                              out_dir=tmp_path / "sim_out",
                              dataset_name="results", seed=3)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()

        result = runner.invoke(app, ["sort_by_linkage",
                                     str(dataset._dataset_file),
                                     "--kind", "dataset",
                                     "--chroms", "chr1"])
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        assert "occup_sorted" in reloaded.analysis["chr1"]
        assert "occup_sorted" not in reloaded.analysis["chr2"]

    def test_experiment_chroms_restricts_processing(self, tmp_path):
        rng = np.random.default_rng(2)
        rows = [(f"m{m}", pos, chrom, "+", round(float(rng.random()), 3),
                 "a")
               for chrom in ("chr1", "chr2")
               for m in range(6) for pos in range(0, 20, 5)]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20, "chr2": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)

        result = runner.invoke(app, [
            "sort_by_linkage", str(exp_file), "--kind", "experiment",
            "--data-name", "meth_prob", "--chroms", "chr1"])
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(exp_file)
        assert "meth_prob_sorted" in reloaded.analysis["chr1"]
        assert "meth_prob_sorted" not in reloaded.analysis["chr2"]

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)
        dataset.save()

        no_fill = runner.invoke(app, ["sort_by_linkage",
                                      str(dataset._dataset_file),
                                      "--kind", "dataset",
                                      "--data-name", "occup_nan"])
        assert no_fill.exit_code != 0

        result = runner.invoke(app, ["sort_by_linkage",
                                     str(dataset._dataset_file),
                                     "--kind", "dataset",
                                     "--data-name", "occup_nan",
                                     "--fill-nan", "mean"])
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        link_mat = reloaded.analysis["chr1"]["occup_nan_linkage"].to_numpy()
        assert np.isfinite(link_mat).all()

    def test_fill_nan_invalid_value_rejected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()

        result = runner.invoke(app, ["sort_by_linkage",
                                     str(dataset._dataset_file),
                                     "--kind", "dataset",
                                     "--fill-nan", "bogus"])
        assert result.exit_code != 0


class TestPlotOccup:
    # Plot content is not asserted (matches the repo's existing plotting
    # test convention in tests/experiment/test_plot.py); this only checks
    # the CLI wires arguments through to a successful render.
    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning")
    def test_writes_figure_file(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        SimAnalysis().sort_by_linkage(dataset=dataset)
        dataset.save()
        out_file = tmp_path / "occup_sorted.png"

        result = runner.invoke(app, [
            "plot_occup", "chr1", str(dataset._dataset_file),
            "--occup-name", "occup_sorted",
            "--link-mat-name", "occup_linkage",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning")
    def test_with_mols_xscale_cmap(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()
        out_file = tmp_path / "occup_subset.png"

        result = runner.invoke(app, [
            "plot_occup", "chr1", str(dataset._dataset_file),
            "--mols", "0", "--mols", "2",
            "--xscale", "1", "--cmap", "viridis",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning")
    def test_with_time_computes_occup(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        out_file = tmp_path / "occup_time.png"

        result = runner.invoke(app, [
            "plot_occup", "chr1", str(dataset._dataset_file),
            "--time", "0",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()


class TestPlotNucPos:
    def test_writes_figure_file(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        out_file = tmp_path / "nucpos.png"

        result = runner.invoke(app, ["plot_nuc_pos", "chr1", "0", "0",
                                     str(dataset._dataset_file),
                                     "--out-file", str(out_file),
                                     "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_time_scale_cmap(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        out_file = tmp_path / "nucpos_scaled.png"

        result = runner.invoke(app, [
            "plot_nuc_pos", "chr1", "0", "0", str(dataset._dataset_file),
            "--tstart", "0", "--tend", "1",
            "--tscale", "1", "--xscale", "1", "--cmap", "viridis",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()


class TestPlotEnergy:
    def test_writes_figure_file(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        out_file = tmp_path / "energy.png"

        result = runner.invoke(app, ["plot_energy", "chr1", "0", "0",
                                     str(dataset._dataset_file),
                                     "--out-file", str(out_file),
                                     "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_time_scale(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nbp=10)
        out_file = tmp_path / "energy_scaled.png"

        result = runner.invoke(app, [
            "plot_energy", "chr1", "0", "0", str(dataset._dataset_file),
            "--tstart", "0", "--tend", "1", "--tscale", "1",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()


class TestPlotMethmap:
    def test_with_link_mat_name(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)
        exp = MethPrintExperiment.load(exp_file)
        MethPrintAnalysis().sort_by_linkage(exp=exp, data_name="meth_prob")
        exp.save(overwrite=True)
        out_file = tmp_path / "methmap_sorted.png"

        result = runner.invoke(app, [
            "plot_methmap", str(exp_file), "chr1",
            "--key", "meth_prob_sorted",
            "--link-mat-name", "meth_prob_linkage",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_cmap(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)
        out_file = tmp_path / "methmap_cmap.png"

        result = runner.invoke(app, [
            "plot_methmap", str(exp_file), "chr1",
            "--cmap", "viridis",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_cbar_label(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, binsize=5)
        out_file = tmp_path / "methmap_cbar_label.png"

        result = runner.invoke(app, [
            "plot_methmap", str(exp_file), "chr1",
            "--cbar-label", "Custom label",
            "--out-file", str(out_file), "--no-show"])
        assert result.exit_code == 0, result.output
        assert out_file.exists()
