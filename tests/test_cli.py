# test_cli.py

import gc
import io
import json

import matplotlib

matplotlib.use("Agg")

import catella
import numpy as np
import pandas as pd
import pytest
from catella import utils
from catella.cli import app
from catella.experiment.methdata import MethPrintExperiment
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.h5_array import H5Array
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset
from typer.testing import CliRunner

runner = CliRunner()

_TSV_HEADER = "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"


def _close_all_h5arrays(obj):
    # Loading eagerly opens a handle per analysis entry, not just the
    # one a test reads; leaving any open blocks a Windows rename.
    for data in list(obj.analysis.values()) + [obj.global_analysis]:
        for entry in data.values():
            if isinstance(entry, H5Array):
                entry.close()


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write(_TSV_HEADER)
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        f.writelines(f"{chrom}\t{length}\n" for chrom, length in sizes.items())


def _write_fasta(path, records):
    with open(path, "w") as f:
        f.writelines(f">{region}\n{seq}\n" for region, seq in records.items())


def _settings_dict(**overrides):
    defaults = {
        "nucbp": 5,
        "llink": 2,
        "elink": 1.0,
        "mu": -1.0,
        "start_temp": 1.0,
        "end_temp": 0.1,
        "cool_option": "linear",
        "nsweep": 2,
        "print_freq": 1,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return defaults


def _write_settings_json(path, **overrides):
    with open(path, "w") as f:
        json.dump(_settings_dict(**overrides), f)


def _make_test_rows(nmol, nbp, seed=0, step=5):
    rng = np.random.default_rng(seed)
    return [
        (f"m{m}", pos, "chr1", "+", round(float(rng.random()), 3), "a")
        for m in range(nmol)
        for pos in range(0, nbp, step)
    ]


def _make_dataset(tmp_path, *, nmol=4, nbp=20, nsim=2, seed=1, name="results"):
    rng = np.random.default_rng(seed)
    meth_prob = rng.random((nmol, nbp))
    manager = SimManager(nworker=1, verbose=False)
    return manager.run(
        regions="chr1",
        nsim=nsim,
        settings=SimSettings(**_settings_dict()),
        meth_prob=meth_prob,
        out_dir=tmp_path / "sim_out",
        dataset_name=name,
        seed=seed,
    )


class TestLoadRaw:
    def test_regions_bed_file(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, _make_test_rows(nmol=3, nbp=30))
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        bed = tmp_path / "regions.bed"
        bed.write_text("chr1\t5\t25\tmid\n")
        out = tmp_path / "exp.h5"
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(out),
                "--regions",
                str(bed),
            ],
        )
        assert result.exit_code == 0, result.output
        got = MethPrintExperiment.load(out)
        assert got.regions == ("mid",)
        assert got.raw["mid"].nbp == 20
        got.close()

    def test_min_overlap_forwarded(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, _make_test_rows(nmol=3, nbp=30))
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        bed = tmp_path / "regions.bed"
        bed.write_text("chr1\t5\t25\tmid\n")
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(tmp_path / "exp.h5"),
                "--regions",
                str(bed),
                "--min-overlap",
                "21",
            ],
        )
        assert result.exit_code != 0
        assert "no region has molecules" in str(result.exception)

    def test_fasta_file_populates_refseq(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "A" * 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(cli_out),
                "--fasta-file",
                str(fasta_file),
            ],
        )
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
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(cli_out),
                "--mtase",
                "CG,GC",
            ],
        )
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
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(cli_out),
                "--max-nmol",
                "3",
                "--seed",
                "7",
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, max_nmol=3, seed=7
        )
        got = MethPrintExperiment.load(cli_out)
        assert len(got.raw["chr1"].test.mol_id) == 3
        assert sorted(got.raw["chr1"].test.mol_id) == sorted(
            expected.raw["chr1"].test.mol_id
        )

    def test_balance_strand_matches_api(self, tmp_path):
        rows = [
            (m, p, c, "-" if int(m[1:]) % 2 else "+", q, a)
            for m, p, c, _, q, a in _make_test_rows(nmol=10, nbp=30)
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(cli_out),
                "--max-nmol",
                "4",
                "--balance-strand",
                "--seed",
                "7",
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            max_nmol=4,
            balance_strand=True,
            seed=7,
        )
        got = MethPrintExperiment.load(cli_out)
        mol_ids = got.raw["chr1"].test.mol_id
        assert sum(int(m[1:]) % 2 for m in mol_ids) == 2
        assert sorted(mol_ids) == sorted(expected.raw["chr1"].test.mol_id)

    def test_regions_restricts_loaded_regions(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30, "chr2": 30})

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(cli_out),
                "--regions",
                "chr1",
            ],
        )
        assert result.exit_code == 0, result.output

        got = MethPrintExperiment.load(cli_out)
        assert got.regions == ("chr1",)


def _load_raw_a_mtase(tmp_path, *, nmol=5, nbp=30, step=1):
    rows = _make_test_rows(nmol=nmol, nbp=nbp, step=step)
    test_file = tmp_path / "test.tsv"
    _write_tsv(test_file, rows)
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {"chr1": nbp})
    fasta_file = tmp_path / "ref.fa"
    _write_fasta(fasta_file, {"chr1": "A" * nbp})

    raw_file = tmp_path / "raw_exp.h5"
    result = runner.invoke(
        app,
        [
            "load_raw",
            str(chromsize),
            str(test_file),
            str(raw_file),
            "--fasta-file",
            str(fasta_file),
            "--mtase",
            "A",
        ],
    )
    assert result.exit_code == 0, result.output
    return raw_file, chromsize, test_file, fasta_file


class TestFilterDropout:
    def test_matches_direct_api_call_and_persists_mask(self, tmp_path):
        raw_file, chromsize, test_file, fasta_file = _load_raw_a_mtase(
            tmp_path
        )

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "filter_dropout",
                str(raw_file),
                "--thres-max",
                "0.5",
                "--out-file",
                str(cli_out),
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
        )
        catella.filter_dropout(exp=expected, thres_max=0.5)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_array_equal(
            got.analysis["chr1"]["test_dropout_mask"]["keep"].to_numpy(),
            expected.analysis["chr1"]["test_dropout_mask"]["keep"].to_numpy(),
        )

    def test_mask_name_option(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        result = runner.invoke(
            app,
            [
                "filter_dropout",
                str(raw_file),
                "--thres-max",
                "0.5",
                "--mask-name",
                "qc_mask",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(raw_file)
        assert "test_qc_mask" in reloaded.analysis["chr1"]

    def test_channels_option(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        ok = runner.invoke(
            app, ["filter_dropout", str(raw_file), "--channels", "A"]
        )
        assert ok.exit_code == 0, ok.output

        bad = runner.invoke(
            app, ["filter_dropout", str(raw_file), "--channels", "GC"]
        )
        assert bad.exit_code != 0

    def test_span_option(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        ok = runner.invoke(
            app, ["filter_dropout", str(raw_file), "--span", "read"]
        )
        assert ok.exit_code == 0, ok.output

        bad = runner.invoke(
            app, ["filter_dropout", str(raw_file), "--span", "bogus"]
        )
        assert bad.exit_code != 0
        assert "span must be one of" in str(bad.exception)


class TestQC:
    def test_prints_tsv_writes_tsv_and_saves(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)
        expected = catella.qc(
            exp=MethPrintExperiment.load(raw_file), percentiles=[25, 75]
        )

        tsv_out = tmp_path / "qc.tsv"
        result = runner.invoke(
            app,
            [
                "qc",
                str(raw_file),
                "--percentiles",
                "25,75",
                "--tsv-file",
                str(tsv_out),
            ],
        )
        assert result.exit_code == 0, result.output
        printed = pd.read_csv(io.StringIO(result.stdout), sep="\t")
        pd.testing.assert_frame_equal(printed, expected, check_dtype=False)

        got = pd.read_csv(tsv_out, sep="\t")
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)

        reloaded = MethPrintExperiment.load(raw_file)
        assert "qc" in reloaded.global_analysis

    def test_rerun_needs_overwrite(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)
        assert runner.invoke(app, ["qc", str(raw_file)]).exit_code == 0

        result = runner.invoke(app, ["qc", str(raw_file)])
        assert result.exit_code != 0

        result = runner.invoke(app, ["qc", str(raw_file), "--overwrite"])
        assert result.exit_code == 0, result.output

    def test_span_option(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)
        ok = runner.invoke(app, ["qc", str(raw_file), "--span", "read"])
        assert ok.exit_code == 0, ok.output

        bad = runner.invoke(
            app, ["qc", str(raw_file), "--span", "bogus", "--overwrite"]
        )
        assert bad.exit_code != 0
        assert "span must be one of" in str(bad.exception)


class TestPlotDropoutEcdf:
    def test_writes_figure_file(self, tmp_path):
        raw_file, *_ = _load_raw_a_mtase(tmp_path)

        out_file = tmp_path / "dropout.png"
        result = runner.invoke(
            app,
            [
                "plot_dropout_ecdf",
                str(raw_file),
                "chr1",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_skips_regions_missing_refseq(self, tmp_path):
        # chr2 has no refseq; plotting chr1 must not touch it (before
        # the regions= restriction, dropout_fractions() always looped
        # over every region and would raise here).
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(4)] + [
            ("m0", 0, "chr2", "+", 0.5, "a")
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 4, "chr2": 4})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "AAAA"})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
                "--mtase",
                "A",
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        out_file = tmp_path / "dropout.png"
        result = runner.invoke(
            app,
            [
                "plot_dropout_ecdf",
                str(raw_file),
                "chr1",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_multi_regions_no_pooled_and_split_by(self, tmp_path):
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(4)] + [
            ("m1", i, "chr2", "+", 0.5, "a") for i in range(2)
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 4, "chr2": 4})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "AACG", "chr2": "AACG"})
        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
                "--mtase",
                "A,CG",
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        out_file = tmp_path / "dropout.png"
        result = runner.invoke(
            app,
            [
                "plot_dropout_ecdf",
                str(raw_file),
                "chr1,chr2",
                "--no-pooled",
                "--split-by",
                "channel",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
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
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--max-nmol",
                "3",
                "--seed",
                "7",
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_empirical_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, max_nmol=3, seed=7
        )
        catella.compute_empirical_prob(exp=expected, lnuc=5)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
        )

    def test_no_out_file_overwrites_source_file(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app, ["load_raw", str(chromsize), str(test_file), str(raw_file)]
        )
        assert load_result.exit_code == 0, load_result.output

        result = runner.invoke(
            app, ["compute_empirical_prob", str(raw_file), "--lnuc", "5"]
        )
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(raw_file)
        assert "meth_prob" in reloaded.analysis["chr1"]

    def test_span_option(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, _make_test_rows(nmol=5, nbp=30))
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})
        raw_file = tmp_path / "raw_exp.h5"
        runner.invoke(
            app, ["load_raw", str(chromsize), str(test_file), str(raw_file)]
        )
        args = ["compute_empirical_prob", str(raw_file), "--lnuc", "5"]

        ok = runner.invoke(app, [*args, "--span", "read"])
        assert ok.exit_code == 0, ok.output

        bad = runner.invoke(app, [*args, "--span", "bogus"])
        assert bad.exit_code != 0
        assert "span must be one of" in str(bad.exception)

    def test_prob_name_renames_stored_probabilities(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=30)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 30})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app, ["load_raw", str(chromsize), str(test_file), str(raw_file)]
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_empirical_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "5",
                "--prob-name",
                "custom_prob",
            ],
        )
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
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--max-nmol",
                "3",
                "--seed",
                "7",
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_empirical_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "5",
                "--fill-edge",
                "0.5",
                "--seed",
                "9",
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, max_nmol=3, seed=7
        )
        catella.compute_empirical_prob(
            exp=expected, lnuc=5, fill_edge=0.5, seed=9
        )

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
            equal_nan=True,
        )


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
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
                "--max-nmol",
                "3",
                "--seed",
                "7",
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            max_nmol=3,
            seed=7,
        )
        catella.compute_model_prob(exp=expected, lnuc=30)

        got = MethPrintExperiment.load(cli_out)
        np.testing.assert_allclose(
            got.analysis["chr1"]["meth_prob"].to_numpy(),
            expected.analysis["chr1"]["meth_prob"].to_numpy(),
        )

    def test_span_option_is_forwarded(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})
        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--span",
                "calls",
            ],
        )
        assert result.exit_code == 0, result.output
        got = MethPrintExperiment.load(cli_out)
        assert got.global_analysis["meth_prob_params"]["span"][0] == "calls"

    def _load_model_raw(self, tmp_path, extra=()):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})
        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
                *extra,
            ],
        )
        assert load_result.exit_code == 0, load_result.output
        return raw_file

    def test_channels_option_is_forwarded(self, tmp_path):
        raw_file = self._load_model_raw(tmp_path)
        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--channels",
                "GCG,GCH",
            ],
        )
        assert result.exit_code == 0, result.output
        got = MethPrintExperiment.load(cli_out)
        params = got.global_analysis["meth_prob_params"]
        assert params["channels"][0] == "GCH,GCG"
        eta_row = got.global_analysis["meth_prob_eta"].iloc[0]
        assert set(eta_row.index) == {"eta_GCH", "eta_GCG"}

    def test_merged_channel_is_forwarded(self, tmp_path):
        raw_file = self._load_model_raw(tmp_path, ("--mtase", "GC"))
        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--channels",
                "GC",
                "--eta",
                "0.8",
            ],
        )
        assert result.exit_code == 0, result.output
        got = MethPrintExperiment.load(cli_out)
        params = got.global_analysis["meth_prob_params"]
        assert params["channels"][0] == "GC"
        eta_row = got.global_analysis["meth_prob_eta"].iloc[0]
        assert eta_row.to_dict() == {"eta_GC": 0.8}

    def test_merged_channel_without_mtase_fails(self, tmp_path):
        raw_file = self._load_model_raw(tmp_path)
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(tmp_path / "cli_exp.h5"),
                "--lnuc",
                "30",
                "--channels",
                "GC",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "requires mtase" in str(result.exception)

    def test_unknown_channel_fails(self, tmp_path):
        raw_file = self._load_model_raw(tmp_path)
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(tmp_path / "cli_exp.h5"),
                "--lnuc",
                "30",
                "--channels",
                "GpC",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_q_thres_option_is_forwarded(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})
        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--q-thres",
                "0.7",
            ],
        )
        assert result.exit_code == 0, result.output
        got = MethPrintExperiment.load(cli_out)
        assert got.global_analysis["meth_prob_params"]["q_thres"][0] == 0.7

    def test_lbin_option_is_forwarded(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})
        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--lbin",
                "15",
            ],
        )
        assert result.exit_code == 0, result.output
        got = MethPrintExperiment.load(cli_out)
        assert got.global_analysis["meth_prob_params"]["lbin"][0] == 15

    def test_no_out_file_overwrites_source_file(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
                "--max-nmol",
                "3",
                "--seed",
                "7",
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--lnuc",
                "30",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(raw_file)
        assert "meth_prob" in reloaded.analysis["chr1"]

    def test_store_rho_propagates(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1, seed=11)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--eta-max-lag",
                "4",
                "--store-rho",
            ],
        )
        assert result.exit_code == 0, result.output

        expected = catella.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
        catella.compute_model_prob(
            exp=expected, lnuc=30, eta_max_lag=4, store_rho=True
        )

        got = MethPrintExperiment.load(cli_out)
        assert ("meth_prob_rho" in got.global_analysis) == (
            "meth_prob_rho" in expected.global_analysis
        )
        if "meth_prob_rho" in expected.global_analysis:
            pd.testing.assert_frame_equal(
                got.global_analysis["meth_prob_rho"],
                expected.global_analysis["meth_prob_rho"],
                check_dtype=False,
            )

    def test_prob_name_renames_stored_probabilities(self, tmp_path):
        rows = _make_test_rows(nmol=5, nbp=len(_MODEL_SEQ), step=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(_MODEL_SEQ)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": _MODEL_SEQ})

        raw_file = tmp_path / "raw_exp.h5"
        load_result = runner.invoke(
            app,
            [
                "load_raw",
                str(chromsize),
                str(test_file),
                str(raw_file),
                "--fasta-file",
                str(fasta_file),
            ],
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            [
                "compute_model_prob",
                str(raw_file),
                "--out-file",
                str(cli_out),
                "--lnuc",
                "30",
                "--prob-name",
                "custom_prob",
            ],
        )
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
        load_result = runner.invoke(
            app, ["load_raw", str(chromsize), str(test_file), str(raw_file)]
        )
        assert load_result.exit_code == 0, load_result.output

        cli_out = tmp_path / "cli_exp.h5"
        result = runner.invoke(
            app,
            ["compute_model_prob", str(raw_file), "--out-file", str(cli_out)],
        )
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
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)

        settings_file = tmp_path / "settings.json"
        _write_settings_json(settings_file)
        out_dir = tmp_path / "sim_out"

        result = runner.invoke(
            app,
            [
                "run",
                "--regions",
                "chr1",
                "--nsim",
                "2",
                "--settings",
                str(settings_file),
                "--exp-file",
                str(exp_file),
                "--out-dir",
                str(out_dir),
                "--dataset-name",
                "myrun",
                "--seed",
                "42",
                "--mols",
                "0,1",
                "--nworker",
                "1",
                "--no-verbose",
            ],
        )
        assert result.exit_code == 0, result.output
        cli_dataset = SimDataset.load(out_dir / "myrun.h5")
        assert list(cli_dataset.regions) == ["chr1"]
        # nmol is the full molecule-index space, not the simulated
        # count; confirm --mols restricted simulation to mols 0 and 1.
        with pytest.raises(ValueError):
            cli_dataset.raw["chr1", 2, 0]

        exp_data = MethPrintExperiment.load(exp_file)
        meth_prob = exp_data.analysis["chr1"]["meth_prob"].to_numpy()
        expected = catella.run(
            regions="chr1",
            nsim=2,
            settings=SimSettings(**_settings_dict()),
            meth_prob={"chr1": meth_prob},
            out_dir=tmp_path / "direct_out",
            dataset_name="direct",
            seed=42,
            mols=[0, 1],
            nworker=1,
            verbose=False,
        )
        for mol in range(2):
            for run_idx in range(2):
                np.testing.assert_allclose(
                    cli_dataset.raw["chr1", mol, run_idx].energy,
                    expected.raw["chr1", mol, run_idx].energy,
                )
                np.testing.assert_array_equal(
                    cli_dataset.raw["chr1", mol, run_idx].position[-1],
                    expected.raw["chr1", mol, run_idx].position[-1],
                )

    def test_method_tm_creates_tm_dataset(self, tmp_path):
        rows = _make_test_rows(nmol=3, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)

        settings_file = tmp_path / "settings.json"
        _write_settings_json(settings_file)
        out_dir = tmp_path / "tm_out"

        result = runner.invoke(
            app,
            [
                "run",
                "--regions",
                "chr1",
                "--nsim",
                "0",
                "--settings",
                str(settings_file),
                "--exp-file",
                str(exp_file),
                "--out-dir",
                str(out_dir),
                "--dataset-name",
                "tmrun",
                "--method",
                "tm",
                "--seed",
                "42",
                "--nworker",
                "1",
                "--no-verbose",
            ],
        )
        assert result.exit_code == 0, result.output
        cli_dataset = SimDataset.load(out_dir / "tmrun.h5")
        assert cli_dataset.method == "tm"
        assert cli_dataset.nsim == 0

    @pytest.mark.parametrize(
        ("value", "expected"), [("0.5", 0.5), ("mean", "mean")]
    )
    def test_fill_nan_is_stored(self, tmp_path, value, expected):
        rows = _make_test_rows(nmol=3, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        settings_file = tmp_path / "settings.json"
        _write_settings_json(settings_file)
        out_dir = tmp_path / "fill_out"

        result = runner.invoke(
            app,
            [
                "run",
                "--regions",
                "chr1",
                "--nsim",
                "0",
                "--settings",
                str(settings_file),
                "--exp-file",
                str(exp_file),
                "--out-dir",
                str(out_dir),
                "--method",
                "tm",
                "--fill-nan",
                value,
                "--no-verbose",
            ],
        )
        assert result.exit_code == 0, result.output
        assert SimDataset.load(out_dir / "results.h5").nan_fill == expected


class TestComputeOccup:
    def test_matches_direct_api_call(self, tmp_path):
        cli_dataset = _make_dataset(tmp_path, nmol=6, nbp=15, name="cli")
        direct_dataset = SimDataset.load(cli_dataset._dataset_file)

        result = runner.invoke(
            app,
            [
                "compute_occup",
                str(cli_dataset._dataset_file),
                "--occup-name",
                "my_occup",
                "--mean-nnuc-name",
                "my_nnuc",
            ],
        )
        assert result.exit_code == 0, result.output

        catella.compute_occup(
            dataset=direct_dataset,
            occup_name="my_occup",
            mean_nnuc_name="my_nnuc",
        )

        reloaded = SimDataset.load(cli_dataset._dataset_file)
        np.testing.assert_allclose(
            reloaded.analysis["chr1"]["my_occup"].to_numpy(),
            direct_dataset.analysis["chr1"]["my_occup"].to_numpy(),
        )
        np.testing.assert_allclose(
            reloaded.analysis["chr1"]["my_nnuc"].to_numpy(),
            direct_dataset.analysis["chr1"]["my_nnuc"].to_numpy(),
        )

    def test_compute_var_flag_on_tm_dataset(self, tmp_path):
        rng = np.random.default_rng(3)
        meth_prob = rng.random((3, 15))
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=SimSettings(**_settings_dict()),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_sim",
            dataset_name="tm_results",
            seed=1,
        )

        result = runner.invoke(
            app,
            [
                "compute_occup",
                str(dataset._dataset_file),
                "--compute-var",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        assert "var_nnuc" in reloaded.analysis["chr1"]
        assert "var_occup" in reloaded.analysis["chr1"]

    def test_regions_and_nworker_on_tm_dataset(self, tmp_path):
        rng = np.random.default_rng(4)
        dataset = SimManager(nworker=1, verbose=False).run(
            regions=["chr1", "chr2"],
            nsim=0,
            method="tm",
            settings=SimSettings(**_settings_dict()),
            meth_prob={
                "chr1": rng.random((3, 15)),
                "chr2": rng.random((2, 15)),
            },
            out_dir=tmp_path / "tm_sim",
            seed=1,
        )
        direct = SimDataset.load(dataset._dataset_file)

        result = runner.invoke(
            app,
            [
                "compute_occup",
                str(dataset._dataset_file),
                "--regions",
                "chr2",
                "--nworker",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output

        catella.compute_occup(direct, regions="chr2")
        reloaded = SimDataset.load(dataset._dataset_file)
        assert "occup" not in reloaded.analysis["chr1"]
        np.testing.assert_allclose(
            reloaded.analysis["chr2"]["occup"].to_numpy(),
            direct.analysis["chr2"]["occup"].to_numpy(),
        )


class TestComputeConfig:
    def _dataset(self, tmp_path):
        meth_prob = np.random.default_rng(5).random((3, 15))
        meth_prob[1] = np.nan
        return SimManager(nworker=1, verbose=False).run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=SimSettings(**_settings_dict()),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_sim",
            seed=1,
        )

    def test_matches_api_and_writes_tsv(self, tmp_path):
        dataset = self._dataset(tmp_path)
        direct = SimDataset.load(dataset._dataset_file)
        tsv_file = tmp_path / "calls.tsv"

        result = runner.invoke(
            app,
            [
                "compute_config",
                str(dataset._dataset_file),
                "--gamma",
                "2",
                "--name",
                "bm2",
                "--tsv-file",
                str(tsv_file),
            ],
        )
        assert result.exit_code == 0, result.output

        catella.compute_config(direct, gamma=2.0, name="bm2")
        want = catella.extract_config(direct, region="chr1", name="bm2")
        reloaded = SimDataset.load(dataset._dataset_file)
        got = catella.extract_config(reloaded, region="chr1", name="bm2")
        assert got[1] is None
        for mol in (0, 2):
            np.testing.assert_array_equal(got[mol], want[mol])

        table = pd.read_csv(
            tsv_file, sep="\t", keep_default_na=False, dtype=str
        )
        assert list(table.columns) == ["region", "mol", "starts"]
        assert list(table["mol"]) == ["0", "1", "2"]
        assert table["starts"].iloc[1] == "NA"
        assert table["starts"].iloc[0] == ",".join(str(s) for s in want[0])

    def test_most_likely_default_name(self, tmp_path):
        dataset = self._dataset(tmp_path)
        result = runner.invoke(
            app,
            [
                "compute_config",
                str(dataset._dataset_file),
                "--method",
                "most_likely",
            ],
        )
        assert result.exit_code == 0, result.output
        reloaded = SimDataset.load(dataset._dataset_file)
        assert "most_likely_energy" in reloaded.analysis["chr1"]


class TestDownsample:
    @pytest.mark.parametrize("batch_size", ["0", "-5"])
    def test_nonpositive_batch_size_rejected(self, tmp_path, batch_size):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        dataset.save()
        result = runner.invoke(
            app,
            [
                "downsample",
                str(dataset._dataset_file),
                "chr1",
                "occup",
                "3",
                "--kind",
                "dataset",
                "--batch-size",
                batch_size,
            ],
        )
        assert result.exit_code == 2
        assert "x>=1" in result.output

    def test_dataset_downsample_matches_utils(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=8, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        expected = utils.downsample(occup, 3, how="mean")

        result = runner.invoke(
            app,
            [
                "downsample",
                str(dataset._dataset_file),
                "chr1",
                "occup",
                "3",
                "--kind",
                "dataset",
                "--how",
                "mean",
                "--out-key",
                "occup_small",
            ],
        )
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
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        exp = MethPrintExperiment.load(exp_file)
        expected = utils.downsample(
            exp.analysis["chr1"]["meth_prob"], 2, how="mean"
        )
        _close_all_h5arrays(exp)

        result = runner.invoke(
            app,
            [
                "downsample",
                str(exp_file),
                "chr1",
                "meth_prob",
                "2",
                "--kind",
                "experiment",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(exp_file)
        got = reloaded.analysis["chr1"]["meth_prob_downsampled"]
        np.testing.assert_allclose(got.to_numpy(), expected)


class TestSortByLinkage:
    def test_dataset_linkage_matches_oracle_and_is_stored(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        _, ref_link = utils.compute_linkage(occup)

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(dataset._dataset_file),
                "--kind",
                "dataset",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        link_mat = reloaded.analysis["chr1"]["occup_linkage"].to_numpy()
        np.testing.assert_allclose(link_mat, ref_link)
        assert "occup_sorted" not in reloaded.analysis["chr1"]

    def test_experiment_store_link_mat_false_skips_persistence(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        exp = MethPrintExperiment.load(exp_file)
        before = set(exp.analysis["chr1"])
        _close_all_h5arrays(exp)

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(exp_file),
                "--kind",
                "experiment",
                "--data-name",
                "meth_prob",
                "--no-store-link-mat",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(exp_file)
        assert set(reloaded.analysis["chr1"]) == before

    def test_dataset_regions_restricts_processing(self, tmp_path):
        rng = np.random.default_rng(3)
        nmol, nbp = 4, 20
        meth_prob = {
            region: rng.random((nmol, nbp)) for region in ("chr1", "chr2")
        }
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            regions=["chr1", "chr2"],
            nsim=2,
            settings=SimSettings(**_settings_dict()),
            meth_prob=meth_prob,
            out_dir=tmp_path / "sim_out",
            dataset_name="results",
            seed=3,
        )
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(dataset._dataset_file),
                "--kind",
                "dataset",
                "--regions",
                "chr1",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        assert "occup_linkage" in reloaded.analysis["chr1"]
        assert "occup_linkage" not in reloaded.analysis["chr2"]

    def test_experiment_regions_restricts_processing(self, tmp_path):
        rng = np.random.default_rng(2)
        rows = [
            (f"m{m}", pos, region, "+", round(float(rng.random()), 3), "a")
            for region in ("chr1", "chr2")
            for m in range(6)
            for pos in range(0, 20, 5)
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20, "chr2": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(exp_file),
                "--kind",
                "experiment",
                "--data-name",
                "meth_prob",
                "--regions",
                "chr1",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = MethPrintExperiment.load(exp_file)
        assert "meth_prob_linkage" in reloaded.analysis["chr1"]
        assert "meth_prob_linkage" not in reloaded.analysis["chr2"]

    def test_dataset_config_name(self, tmp_path):
        dataset = SimManager(nworker=1, verbose=False).run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=SimSettings(**_settings_dict()),
            meth_prob=np.random.default_rng(5).random((4, 15)),
            out_dir=tmp_path / "tm_sim",
            seed=1,
        )
        catella.compute_config(dataset)
        dataset.save()

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(dataset._dataset_file),
                "--kind",
                "dataset",
                "--config-name",
                "best_match",
            ],
        )
        assert result.exit_code == 0, result.output
        reloaded = SimDataset.load(dataset._dataset_file)
        assert "best_match_linkage" in reloaded.analysis["chr1"]

    def test_fill_nan_avoids_crash_on_all_nan_row(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = dataset.analysis["chr1"]["occup"].to_numpy()
        occup[0, :] = np.nan
        dataset.analysis["chr1"]["occup_nan"] = pd.DataFrame(occup)
        dataset.save()

        no_fill = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(dataset._dataset_file),
                "--kind",
                "dataset",
                "--data-name",
                "occup_nan",
            ],
        )
        assert no_fill.exit_code != 0
        # Drop the retained traceback, which keeps the failed
        # invocation's own file handle open otherwise.
        del no_fill
        gc.collect()

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(dataset._dataset_file),
                "--kind",
                "dataset",
                "--data-name",
                "occup_nan",
                "--fill-nan",
                "mean",
            ],
        )
        assert result.exit_code == 0, result.output

        reloaded = SimDataset.load(dataset._dataset_file)
        link_mat = reloaded.analysis["chr1"]["occup_nan_linkage"].to_numpy()
        assert np.isfinite(link_mat).all()

    def test_fill_nan_invalid_value_rejected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()

        result = runner.invoke(
            app,
            [
                "sort_by_linkage",
                str(dataset._dataset_file),
                "--kind",
                "dataset",
                "--fill-nan",
                "bogus",
            ],
        )
        assert result.exit_code != 0


class TestPlotConfig:
    def test_writes_figure_file(self, tmp_path):
        dataset = SimManager(nworker=1, verbose=False).run(
            regions="chr1",
            nsim=0,
            method="tm",
            settings=SimSettings(**_settings_dict()),
            meth_prob=np.random.default_rng(2).random((4, 15)),
            out_dir=tmp_path / "tm_sim",
            seed=1,
        )
        catella.compute_config(dataset)
        dataset.save()
        out_file = tmp_path / "configs.png"

        result = runner.invoke(
            app,
            [
                "plot_config",
                "chr1",
                str(dataset._dataset_file),
                "--name",
                "best_match",
                "--mols",
                "2",
                "--mols",
                "0",
                "--max-rows",
                "1",
                "--downsample-how",
                "mean",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def _two_region_file(self, tmp_path):
        rng = np.random.default_rng(2)
        dataset = SimManager(nworker=1, verbose=False).run(
            regions=["chr1", "chr2"],
            nsim=0,
            method="tm",
            settings=SimSettings(**_settings_dict()),
            meth_prob={
                "chr1": rng.random((4, 15)),
                "chr2": rng.random((3, 20)),
            },
            out_dir=tmp_path / "tm_sim",
            seed=1,
        )
        catella.compute_config(dataset)
        dataset.save()
        return dataset._dataset_file

    def test_several_regions(self, tmp_path):
        dataset_file = self._two_region_file(tmp_path)
        out_file = tmp_path / "configs_two.png"

        result = runner.invoke(
            app,
            [
                "plot_config",
                "chr1,chr2",
                str(dataset_file),
                "--name",
                "best_match",
                "--sharex",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_plot_mean_and_eseq(self, tmp_path):
        dataset_file = self._two_region_file(tmp_path)
        out_file = tmp_path / "configs_mean.png"

        result = runner.invoke(
            app,
            [
                "plot_config",
                "chr1,chr2",
                str(dataset_file),
                "--name",
                "best_match",
                "--plot-mean",
                "--plot-eseq",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_mols_with_several_regions_fails(self, tmp_path):
        dataset_file = self._two_region_file(tmp_path)

        result = runner.invoke(
            app,
            [
                "plot_config",
                "chr1,chr2",
                str(dataset_file),
                "--mols",
                "0",
                "--no-show",
            ],
        )
        assert result.exit_code != 0
        assert "single region" in result.output


class TestPlotOccup:
    # Plot content is not asserted (matches the repo's existing plotting
    # test convention in tests/experiment/test_plot.py); this only checks
    # the CLI wires arguments through to a successful render.
    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_writes_figure_file(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        SimAnalysis().sort_by_linkage(dataset=dataset)
        dataset.save()
        out_file = tmp_path / "occup_sorted.png"

        result = runner.invoke(
            app,
            [
                "plot_occup",
                "chr1",
                str(dataset._dataset_file),
                "--link-mat-name",
                "occup_linkage",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_with_mols_xscale_cmap(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=6, nbp=10)
        SimAnalysis().compute_occup(dataset=dataset)
        dataset.save()
        out_file = tmp_path / "occup_subset.png"

        result = runner.invoke(
            app,
            [
                "plot_occup",
                "chr1",
                str(dataset._dataset_file),
                "--mols",
                "0",
                "--mols",
                "2",
                "--xscale",
                "1",
                "--cmap",
                "viridis",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_with_time_computes_occup(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        out_file = tmp_path / "occup_time.png"

        result = runner.invoke(
            app,
            [
                "plot_occup",
                "chr1",
                str(dataset._dataset_file),
                "--time",
                "0",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_plot_mean_and_eseq(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nbp=10)
        out_file = tmp_path / "occup_mean.png"

        result = runner.invoke(
            app,
            [
                "plot_occup",
                "chr1",
                str(dataset._dataset_file),
                "--time",
                "0",
                "--plot-mean",
                "--plot-eseq",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def _two_region_file(self, tmp_path):
        rng = np.random.default_rng(2)
        dataset = SimManager(nworker=1, verbose=False).run(
            regions=["chr1", "chr2"],
            nsim=2,
            settings=SimSettings(**_settings_dict()),
            meth_prob={
                "chr1": rng.random((4, 20)),
                "chr2": rng.random((3, 30)),
            },
            out_dir=tmp_path / "sim_out",
            seed=2,
        )
        for region, shape in (("chr1", (4, 20)), ("chr2", (3, 30))):
            dataset.analysis[region]["occup"] = pd.DataFrame(rng.random(shape))
        SimAnalysis().sort_by_linkage(dataset=dataset)
        dataset.save()
        return dataset._dataset_file

    @pytest.mark.filterwarnings(
        "ignore:__array__ implementation doesn't accept a copy keyword"
        ":DeprecationWarning"
    )
    def test_several_regions(self, tmp_path):
        dataset_file = self._two_region_file(tmp_path)
        out_file = tmp_path / "occup_two.png"

        result = runner.invoke(
            app,
            [
                "plot_occup",
                "chr1,chr2",
                str(dataset_file),
                "--link-mat-name",
                "occup_linkage",
                "--sharex",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_mols_with_several_regions_fails(self, tmp_path):
        dataset_file = self._two_region_file(tmp_path)

        result = runner.invoke(
            app,
            [
                "plot_occup",
                "chr1,chr2",
                str(dataset_file),
                "--mols",
                "0",
                "--no-show",
            ],
        )
        assert result.exit_code != 0
        assert "single region" in result.output


class TestPlotMeth:
    def test_with_link_mat_name(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        exp = MethPrintExperiment.load(exp_file)
        MethPrintAnalysis().sort_by_linkage(exp=exp, data_name="meth_prob")
        exp.save()
        out_file = tmp_path / "methmap_sorted.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--key",
                "meth_prob",
                "--link-mat-name",
                "meth_prob_linkage",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
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
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        out_file = tmp_path / "methmap_cmap.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--key",
                "meth_prob",
                "--cmap",
                "viridis",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
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
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        out_file = tmp_path / "methmap_cbar_label.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--key",
                "meth_prob",
                "--cbar-label",
                "Custom label",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_raw_which(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        exp.save(exp_file)
        out_file = tmp_path / "methmap_raw_which.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--raw-which",
                "test",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_key_and_raw_which_are_mutually_exclusive(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        exp.save(exp_file)

        result = runner.invoke(
            app, ["plot_meth", str(exp_file), "chr1", "--no-show"]
        )
        assert result.exit_code != 0

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--key",
                "meth_prob",
                "--raw-which",
                "test",
                "--no-show",
            ],
        )
        assert result.exit_code != 0

    def _two_region_exp_file(self, tmp_path):
        rng = np.random.default_rng(7)
        rows = _make_test_rows(nmol=6, nbp=20) + [
            (f"n{m}", pos, "chr2", "+", round(float(rng.random()), 3), "a")
            for m in range(4)
            for pos in range(0, 30, 5)
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20, "chr2": 30})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        exp = MethPrintExperiment.load(exp_file)
        MethPrintAnalysis().sort_by_linkage(
            exp=exp, data_name="meth_prob", fill_nan=0.0
        )
        exp.save()
        return exp_file

    @pytest.mark.parametrize("value", ["prob", "energy"])
    def test_several_regions_with_key(self, tmp_path, value):
        exp_file = self._two_region_exp_file(tmp_path)
        out_file = tmp_path / f"methmap_two_{value}.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1,chr2",
                "--key",
                "meth_prob",
                "--value",
                value,
                "--link-mat-name",
                "meth_prob_linkage",
                "--sharex",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_several_regions_with_raw_which(self, tmp_path):
        exp_file = self._two_region_exp_file(tmp_path)
        out_file = tmp_path / "methmap_two_raw.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1,chr2",
                "--raw-which",
                "test",
                "--xscale",
                "1",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()


class TestPlotMethEnergy:
    def _make_exp_file(self, tmp_path):
        rows = _make_test_rows(nmol=6, nbp=20)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})
        exp_file = tmp_path / "exp.h5"
        exp = catella.load_raw(chromsize=chromsize, test_file=test_file)
        catella.compute_empirical_prob(exp=exp, out_file=exp_file, lnuc=5)
        return exp_file

    def test_writes_figure_file(self, tmp_path):
        exp_file = self._make_exp_file(tmp_path)
        out_file = tmp_path / "energymap.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--value",
                "energy",
                "--key",
                "meth_prob",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_emax(self, tmp_path):
        exp_file = self._make_exp_file(tmp_path)
        out_file = tmp_path / "energymap_emax.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--value",
                "energy",
                "--key",
                "meth_prob",
                "--emax",
                "30",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_with_cmap_and_cbar_label(self, tmp_path):
        exp_file = self._make_exp_file(tmp_path)
        out_file = tmp_path / "energymap_cmap.png"

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--value",
                "energy",
                "--key",
                "meth_prob",
                "--cmap",
                "viridis",
                "--cbar-label",
                "Custom label",
                "--out-file",
                str(out_file),
                "--no-show",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_energy_with_raw_which_fails(self, tmp_path):
        exp_file = self._make_exp_file(tmp_path)

        result = runner.invoke(
            app,
            [
                "plot_meth",
                str(exp_file),
                "chr1",
                "--value",
                "energy",
                "--raw-which",
                "test",
                "--no-show",
            ],
        )
        assert result.exit_code != 0
