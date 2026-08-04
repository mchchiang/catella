# test_cli.py

import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import nucmc
from nucmc import utils
from nucmc.cli import app
from nucmc.experiment.methdata import MethPrintExperiment
from nucmc.experiment.preprocessing import MethPrintAnalysis
from nucmc.simulation.analysis import SimAnalysis
from nucmc.simulation.config import SimSettings
from nucmc.simulation.engine import SimManager
from nucmc.simulation.results import SimDataset

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


class TestPreprocess:
    def test_matches_direct_api_call(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(app, ["preprocess", str(chromsize),
                                     str(test_file), str(cli_out),
                                     "--binsize", "5", "--max-nmol", "3",
                                     "--seed", "7"])
        assert result.exit_code == 0, result.output

        expected = nucmc.preprocess(chromsize=chromsize, test_file=test_file,
                                    binsize=5, max_nmol=3, seed=7)

        got = MethPrintExperiment.load(cli_out)
        assert len(got.raw["chr1"].test_mol_id) == 3
        assert (sorted(got.raw["chr1"].test_mol_id)
               == sorted(expected.raw["chr1"].test_mol_id))
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy())


class TestRun:
    def test_matches_direct_api_call_with_mols_subset(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        exp_file = tmp_path / "exp.h5"
        nucmc.preprocess(chromsize=chromsize, test_file=test_file,
                         out_file=exp_file, binsize=5)

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
        expected = nucmc.run(chroms="chr1", nsim=2,
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

        nucmc.analyze(dataset=direct_dataset, occup_name="my_occup",
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
        nucmc.preprocess(chromsize=chromsize, test_file=test_file,
                         out_file=exp_file, binsize=5)
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
        nucmc.preprocess(chromsize=chromsize, test_file=test_file,
                         out_file=exp_file, binsize=5)
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


class TestPlotOccup:
    # Plot content is not asserted (matches the repo's existing plotting
    # test convention in tests/experiment/test_plot.py); this only checks
    # the CLI wires arguments through to a successful render.
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


class TestPlotMethmap:
    def test_with_link_mat_name(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        nucmc.preprocess(chromsize=chromsize, test_file=test_file,
                         out_file=exp_file, binsize=5)
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
