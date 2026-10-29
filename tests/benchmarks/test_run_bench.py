# test_run_bench.py

import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import run_bench
from typer.testing import CliRunner

runner = CliRunner()


def _has_env(env):
    try:
        run_bench.run_in_env(env, ["samtools", "--version"])
    except (RuntimeError, FileNotFoundError):
        return False
    return True


needs_fiberseq = pytest.mark.skipif(
    not _has_env("fiberseq"), reason="conda env fiberseq not available"
)


class TestDatasetSpecs:
    def test_full_count(self):
        specs = run_bench.dataset_specs()
        n_test = len(run_bench.CONDITIONS) * len(run_bench.TEST_SEEDS)
        assert len(specs) == 1 + n_test

    def test_tuning_dataset_first(self):
        spec = run_bench.dataset_specs()[0]
        assert spec["cond"] == "tune"
        assert spec["seed"] == run_bench.TUNE_SEED
        assert spec["seed"] not in run_bench.TEST_SEEDS

    def test_overrides_applied(self):
        specs = run_bench.dataset_specs()
        for spec in specs[1:]:
            override = run_bench.CONDITIONS[spec["cond"]]
            for key, value in override.items():
                assert spec["params"][key] == value
            for key, value in run_bench.BASE_PARAMS.items():
                if key not in override:
                    assert spec["params"][key] == value

    def test_common_params(self):
        for spec in run_bench.dataset_specs():
            params = spec["params"]
            assert params["controls"] is True
            assert params["double_strand"] is True
            assert params["nmol"] == run_bench.NMOL
            assert params["seed"] == spec["seed"]

    def test_quick(self):
        specs = run_bench.dataset_specs(quick=True)
        assert [s["cond"] for s in specs] == ["tune", "default"]
        assert all(s["params"]["nmol"] == run_bench.QUICK_NMOL for s in specs)


class TestRunTasks:
    def test_serial_and_pool_match(self, tmp_path):
        args = [(tmp_path, "c", seed) for seed in range(5)]
        serial = run_bench._run_tasks(run_bench.dataset_dir, args, 1)
        pooled = run_bench._run_tasks(run_bench.dataset_dir, args, 2)
        assert serial == pooled
        assert serial[3] == tmp_path / "data" / "c" / "s3"


class TestDatasetDir:
    def test_layout(self, tmp_path):
        path = run_bench.dataset_dir(tmp_path, "default", 2)
        assert path == tmp_path / "data" / "default" / "s2"


class TestMakeData:
    def test_writes_datasets_without_conversion(self, tmp_path):
        specs = run_bench.make_data(tmp_path, quick=True, convert=False)
        assert len(specs) == 2
        for spec in specs:
            ddir = run_bench.dataset_dir(tmp_path, spec["cond"], spec["seed"])
            assert spec["dir"] == str(ddir)
            for name in (
                "synth.tsv",
                "synth.sam",
                "synth.meth.tsv",
                "synth.unmeth.sam",
                "synth.truth.bed",
            ):
                assert (ddir / name).exists()
        record = json.loads((tmp_path / "datasets.json").read_text())
        assert record["quick"] is True
        assert len(record["datasets"]) == 2
        assert record["grids"] == json.loads(json.dumps(run_bench.GRIDS))

    def test_parallel_matches_serial(self, tmp_path):
        run_bench.make_data(tmp_path / "a", quick=True, convert=False)
        run_bench.make_data(
            tmp_path / "b", quick=True, convert=False, nworker=2
        )
        files_a = sorted(
            p.relative_to(tmp_path / "a")
            for p in (tmp_path / "a" / "data").rglob("*")
            if p.is_file()
        )
        assert files_a
        for rel in files_a:
            a = (tmp_path / "a" / rel).read_bytes()
            b = (tmp_path / "b" / rel).read_bytes()
            assert a == b

    @needs_fiberseq
    def test_writes_indexed_bams(self, tmp_path):
        run_bench.make_data(tmp_path, quick=True)
        ddir = run_bench.dataset_dir(tmp_path, "default", 1)
        for name in ("synth", "synth.meth", "synth.unmeth"):
            assert (ddir / f"{name}.bam").exists()
            assert (ddir / f"{name}.bam.bai").exists()


class TestRunInEnv:
    @pytest.mark.parametrize(
        "conda_exe, mamba_exe, expected",
        [
            ("/x/conda", "/y/mamba", "/x/conda"),
            (None, "/y/mamba", "/y/mamba"),
            (None, None, "conda"),
        ],
    )
    def test_executable_choice(
        self, monkeypatch, conda_exe, mamba_exe, expected
    ):
        env_vars = (("CONDA_EXE", conda_exe), ("MAMBA_EXE", mamba_exe))
        for name, value in env_vars:
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        seen = []

        def fake_run(args, **kwargs):
            seen.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(run_bench.subprocess, "run", fake_run)
        run_bench.run_in_env("e", ["samtools", "--version"])
        assert seen == [[expected, "run", "-n", "e", "samtools", "--version"]]

    def test_failure_raises(self):
        with pytest.raises((RuntimeError, FileNotFoundError)):
            run_bench.run_in_env("no_such_env_xyz", ["true"])

    @needs_fiberseq
    def test_success_returns_process(self):
        proc = run_bench.run_in_env("fiberseq", ["samtools", "--version"])
        assert isinstance(proc, subprocess.CompletedProcess)
        assert "samtools" in proc.stdout


class TestIntervalsToCalls:
    def test_occupancy_and_count(self):
        intervals = pd.DataFrame(
            {"mol_id": ["a", "a", "b"], "start": [2, 6, 0], "end": [4, 8, 3]}
        )
        calls = run_bench.intervals_to_calls(intervals, ["a", "b", "c"], 10)
        assert calls.mol_ids.tolist() == ["a", "b", "c"]
        assert calls.occ[0].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        assert calls.occ[1].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        assert calls.occ[2].sum() == 0
        assert calls.count.tolist() == [2, 1, 0]

    def test_clipping_and_unknown_molecules(self):
        intervals = pd.DataFrame(
            {"mol_id": ["a", "z"], "start": [-3, 0], "end": [12, 5]}
        )
        calls = run_bench.intervals_to_calls(intervals, ["a"], 10)
        assert calls.occ.shape == (1, 10)
        assert calls.occ[0].sum() == 10
        assert calls.count.tolist() == [1]


class TestParseFtExtract:
    def test_parse(self, tmp_path):
        path = tmp_path / "x.tsv"
        pd.DataFrame(
            {
                "fiber": ["m1", "m2", "m3"],
                "ref_nuc_starts": ["10,200,", ".", "5"],
                "ref_nuc_lengths": ["147,150,", ".", "100"],
            }
        ).to_csv(path, sep="\t", index=False)
        intervals = run_bench.parse_ft_extract(path)
        assert intervals.values.tolist() == [
            ["m1", 10, 157],
            ["m1", 200, 350],
            ["m3", 5, 105],
        ]


class TestSaveLoadCalls:
    def test_roundtrip(self, tmp_path):
        calls = run_bench.Calls(
            np.array(["a", "b"]),
            np.array([[0.0, 0.25, 1.0], [0.5, 0.75, 0.0]], dtype=np.float32),
            np.array([1.5, 2.0]),
        )
        path = run_bench.save_calls(calls, tmp_path / "sub" / "c.h5")
        back = run_bench.load_calls(path)
        assert back.mol_ids.tolist() == ["a", "b"]
        np.testing.assert_allclose(back.occ, calls.occ, atol=1e-3)
        np.testing.assert_array_equal(back.count, calls.count)


class TestMcc:
    def test_perfect(self):
        truth = np.array([1, 1, 0, 0, 1], dtype=bool)
        assert run_bench.mcc(truth, truth) == pytest.approx(1.0)

    def test_inverted(self):
        truth = np.array([1, 1, 0, 0, 1], dtype=bool)
        assert run_bench.mcc(truth, ~truth) == pytest.approx(-1.0)

    def test_constant_prediction(self):
        truth = np.array([1, 1, 0, 0, 1], dtype=bool)
        assert run_bench.mcc(truth, np.ones(5, dtype=bool)) == 0.0

    def test_known_value(self):
        truth = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        pred = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
        assert run_bench.mcc(truth, pred) == pytest.approx(1 / 3)


class TestScoreCalls:
    def _truth(self):
        return pd.DataFrame(
            {
                "mol_id": ["a", "a", "b"],
                "start": [10, 40, 20],
                "end": [30, 60, 50],
            }
        )

    def test_perfect_calls(self):
        truth = self._truth()
        calls = run_bench.intervals_to_calls(truth, ["a", "b"], 80)
        scores = run_bench.score_calls(calls, truth, nbp=80, trim=5)
        assert scores["mol_id"].tolist() == ["a", "b"]
        assert scores["mcc"].tolist() == pytest.approx([1.0, 1.0])
        assert scores["count_err"].tolist() == [0, 0]

    def test_merged_calls(self):
        truth = self._truth()
        merged = pd.DataFrame({"mol_id": ["a"], "start": [10], "end": [60]})
        calls = run_bench.intervals_to_calls(merged, ["a"], 80)
        scores = run_bench.score_calls(calls, truth, nbp=80, trim=5)
        assert scores["count_err"].tolist() == [-1]
        assert 0 < scores["mcc"].iloc[0] < 1

    def test_probability_occupancy(self):
        truth = self._truth()
        calls = run_bench.intervals_to_calls(truth, ["a", "b"], 80)
        calls.occ = 0.4 + 0.2 * calls.occ
        calls.count = np.array([2.4, 0.8])
        scores = run_bench.score_calls(calls, truth, nbp=80, trim=5)
        assert scores["mcc"].tolist() == pytest.approx([1.0, 1.0])
        assert scores["count_err"].tolist() == pytest.approx([0.4, -0.2])

    def test_trim_excludes_ends(self):
        truth = pd.DataFrame({"mol_id": ["a"], "start": [0], "end": [10]})
        empty = pd.DataFrame(columns=["mol_id", "start", "end"])
        calls = run_bench.intervals_to_calls(empty, ["a"], 80)
        calls.occ[0, 40:50] = 1
        truth_mid = pd.concat(
            [
                truth,
                pd.DataFrame({"mol_id": ["a"], "start": [40], "end": [50]}),
            ]
        )
        scores = run_bench.score_calls(calls, truth_mid, nbp=80, trim=20)
        assert scores["mcc"].iloc[0] == pytest.approx(1.0)


class TestGridCombos:
    def test_product(self):
        combos = run_bench.grid_combos({"a": [1, 2], "b": ["x", "y"]})
        assert combos == [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "x"},
            {"a": 2, "b": "y"},
        ]

    def test_all_grids_have_four_combos(self):
        for tool in run_bench.TOOLS:
            assert len(run_bench.grid_combos(run_bench.GRIDS[tool])) == 4

    def test_ft_grid_combined_not_shorter(self):
        for params in run_bench.grid_combos(run_bench.GRIDS["ft"]):
            assert params["combined_length"] >= params["nucleosome_length"]


class TestSummarize:
    def _results(self):
        rows = []
        for seed, (cat, ft) in enumerate([(0.9, 0.7), (0.8, 0.85)], 1):
            for mol, shift in (("m0", 0.05), ("m1", -0.05)):
                rows.append(("catella", "tuned", "c", seed, mol, cat + shift))
                rows.append(("ft", "tuned", "c", seed, mol, ft + shift))
        results = pd.DataFrame(
            rows, columns=["tool", "variant", "cond", "seed", "mol_id", "mcc"]
        )
        results["count_err"] = 1.0
        return results

    def test_per_tool(self):
        per_tool, _ = run_bench.summarize(self._results())
        cat = per_tool[per_tool["tool"] == "catella"].iloc[0]
        assert cat["mcc_mean"] == pytest.approx(0.85)
        assert cat["mcc_sd"] == pytest.approx(np.std([0.9, 0.8], ddof=1))
        assert cat["count_err_mean"] == pytest.approx(1.0)
        assert cat["n"] == 2

    def test_paired(self):
        _, paired = run_bench.summarize(self._results())
        assert len(paired) == 1
        row = paired.iloc[0]
        assert row["other"] == "ft"
        assert row["diff_mean"] == pytest.approx((0.2 - 0.05) / 2)
        assert row["wins"] == 1
        assert row["n"] == 2


@pytest.fixture
def fake_tools(monkeypatch):
    runs = []

    def fake_run_tool(tool, ddir, workdir, params, env="fiberseq"):
        runs.append(tool)
        truth = run_bench.read_truth(ddir)
        mol_ids = list(dict.fromkeys(truth["mol_id"]))
        perfect = tool == "catella" or 85 in params.values()
        if not perfect:
            truth = truth.iloc[::2]
        return run_bench.intervals_to_calls(truth, mol_ids, run_bench.NBP)

    monkeypatch.setattr(run_bench, "run_tool", fake_run_tool)
    return runs


@pytest.fixture
def tuned_out(tmp_path, fake_tools):
    run_bench.make_data(tmp_path, quick=True, convert=False)
    run_bench.tune(tmp_path)
    fake_tools.clear()
    return tmp_path


class TestEvaluateUnits:
    def test_one_result_file_per_unit(self, tuned_out):
        run_bench.evaluate(tuned_out)
        for tool in run_bench.TOOLS:
            path = run_bench.unit_result_path(tuned_out, tool, "default", 1)
            table = pd.read_csv(path)
            assert set(table["tool"]) == {tool}
            assert set(table["variant"]) == {"tuned", "default"}
            assert len(table) == 2 * run_bench.QUICK_NMOL

    def test_existing_units_skipped(self, tuned_out, fake_tools):
        run_bench.evaluate(tuned_out)
        assert len(fake_tools) == 2 * len(run_bench.TOOLS)
        fake_tools.clear()
        run_bench.evaluate(tuned_out)
        assert fake_tools == []
        run_bench.evaluate(tuned_out, tools=["ft"], overwrite=True)
        assert fake_tools == ["ft", "ft"]

    def test_filters(self, tuned_out, fake_tools):
        run_bench.evaluate(tuned_out, tools=["ft"], conds=["default"])
        assert set(fake_tools) == {"ft"}
        results = pd.read_csv(tuned_out / "results.csv")
        assert set(results["tool"]) == {"ft"}

    @pytest.mark.parametrize(
        "kwargs",
        [{"tools": ["nope"]}, {"conds": ["nope"]}, {"seeds": [99]}],
    )
    def test_unknown_filter_values_raise(self, tuned_out, kwargs):
        with pytest.raises(ValueError):
            run_bench.evaluate(tuned_out, **kwargs)

    def test_partial_summary(self, tuned_out):
        run_bench.evaluate(tuned_out, tools=["catella", "ft"])
        paired = pd.read_csv(tuned_out / "summary_paired.csv")
        assert set(paired["other"]) == {"ft"}
        assert (tuned_out / "summary.md").exists()

    def test_moved_output_directory(self, tuned_out, tmp_path_factory):
        moved = tmp_path_factory.mktemp("moved") / "out"
        shutil.move(str(tuned_out), str(moved))
        results = run_bench.evaluate(moved)
        assert len(results) == len(run_bench.TOOLS) * 2 * run_bench.QUICK_NMOL

    def test_summary_without_results_raises(self, tmp_path):
        with pytest.raises(ValueError):
            run_bench.write_summary(tmp_path)


def _pdf_pages(path):
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", path.read_bytes()))


@pytest.fixture
def evaluated_out(tuned_out):
    run_bench.evaluate(tuned_out)
    return tuned_out


class TestPlotOccupancy:
    def test_default_path_and_pages(self, evaluated_out):
        path = run_bench.plot_occupancy(evaluated_out, "default", 1)
        expected = (
            evaluated_out / "plots" / "occupancy_default_s1_tuned.pdf"
        )
        assert path == expected
        assert path.read_bytes().startswith(b"%PDF")
        assert _pdf_pages(path) == -(-run_bench.QUICK_NMOL // 4)

    def test_nmol(self, evaluated_out, tmp_path):
        path = run_bench.plot_occupancy(
            evaluated_out, "default", 1, nmol=5, path=tmp_path / "a.pdf"
        )
        assert _pdf_pages(path) == 2

    def test_nmol_larger_than_dataset(self, evaluated_out, tmp_path):
        path = run_bench.plot_occupancy(
            evaluated_out, "default", 1, nmol=999, path=tmp_path / "a.pdf"
        )
        assert _pdf_pages(path) == -(-run_bench.QUICK_NMOL // 4)

    @pytest.mark.parametrize(
        "kwargs", [{"nmol": 0}, {"per_page": 0}, {"tools": ["nope"]}]
    )
    def test_invalid_args(self, evaluated_out, kwargs):
        with pytest.raises(ValueError):
            run_bench.plot_occupancy(evaluated_out, "default", 1, **kwargs)

    def test_tool_subset_and_default_variant(self, evaluated_out, tmp_path):
        path = run_bench.plot_occupancy(
            evaluated_out,
            "default",
            1,
            variant="default",
            tools=["ft", "catella"],
            path=tmp_path / "a.pdf",
        )
        assert path.exists()

    def test_missing_tool_calls_skipped(self, evaluated_out, tmp_path):
        shutil.rmtree(evaluated_out / "calls" / "ft_tuned")
        path = run_bench.plot_occupancy(
            evaluated_out, "default", 1, path=tmp_path / "a.pdf"
        )
        assert path.exists()

    def test_no_calls_raises(self, tuned_out):
        with pytest.raises(ValueError):
            run_bench.plot_occupancy(tuned_out, "default", 1)

    def test_cli(self, evaluated_out):
        result = runner.invoke(
            run_bench.app,
            [
                "plot_occupancy",
                str(evaluated_out),
                "--cond",
                "default",
                "--seed",
                "1",
                "--nmol",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        path = evaluated_out / "plots" / "occupancy_default_s1_tuned.pdf"
        assert _pdf_pages(path) == 1


class TestPlotSummary:
    def test_all_pages(self, evaluated_out):
        path = run_bench.plot_summary(evaluated_out)
        assert path == evaluated_out / "plots" / "summary.pdf"
        assert _pdf_pages(path) == 5

    def test_partial_results(self, tuned_out):
        run_bench.evaluate(tuned_out, tools=["catella", "ft"])
        path = run_bench.plot_summary(tuned_out)
        assert _pdf_pages(path) == 5

    def test_without_catella_skips_paired_page(self, tuned_out):
        run_bench.evaluate(tuned_out, tools=["ft", "fiberhmm"])
        path = run_bench.plot_summary(tuned_out)
        assert _pdf_pages(path) == 4

    def test_without_tune_scores(self, evaluated_out, tmp_path):
        (evaluated_out / "tune_scores.csv").unlink()
        path = run_bench.plot_summary(evaluated_out, path=tmp_path / "s.pdf")
        assert path == tmp_path / "s.pdf"
        assert _pdf_pages(path) == 4

    def test_without_results_raises(self, tmp_path):
        with pytest.raises(ValueError):
            run_bench.plot_summary(tmp_path)

    @pytest.mark.parametrize("command", ["summarize", "plot_summary"])
    def test_cli(self, evaluated_out, command):
        (evaluated_out / "plots" / "summary.pdf").unlink(missing_ok=True)
        result = runner.invoke(run_bench.app, [command, str(evaluated_out)])
        assert result.exit_code == 0, result.output
        assert _pdf_pages(evaluated_out / "plots" / "summary.pdf") == 5


class TestTuneEvaluate:
    def test_pipeline(self, tmp_path, fake_tools):
        run_bench.make_data(tmp_path, quick=True, convert=False)
        tuned = run_bench.tune(tmp_path)
        assert set(tuned) == set(run_bench.TOOLS)
        assert tuned["ft"]["nucleosome_length"] == 85
        scores = pd.read_csv(tmp_path / "tune_scores.csv")
        assert len(scores) == 4 * len(run_bench.TOOLS)
        saved = json.loads((tmp_path / "tuned_params.json").read_text())
        assert saved == tuned

        results = run_bench.evaluate(tmp_path)
        n_rows = len(run_bench.TOOLS) * 2 * run_bench.QUICK_NMOL
        assert len(results) == n_rows
        assert set(results["variant"]) == {"tuned", "default"}
        calls = run_bench.load_calls(
            tmp_path / "calls" / "ft_tuned" / "default" / "s1.h5"
        )
        assert calls.occ.shape == (run_bench.QUICK_NMOL, run_bench.NBP)
        for name in (
            "results.csv",
            "summary_per_tool.csv",
            "summary_paired.csv",
            "summary.md",
        ):
            assert (tmp_path / name).exists()
        paired = pd.read_csv(tmp_path / "summary_paired.csv")
        tuned_ft = paired[
            (paired["variant"] == "tuned") & (paired["other"] == "ft")
        ]
        assert tuned_ft["diff_mean"].iloc[0] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def quick_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    specs = run_bench.make_data(
        out, quick=True, convert=_has_env("fiberseq")
    )
    return out, specs[1]["dir"]


def _check_calls(calls, ddir):
    truth = run_bench.read_truth(ddir)
    nmol = run_bench.QUICK_NMOL
    assert calls.occ.shape == (nmol, run_bench.NBP)
    assert set(calls.mol_ids) == set(truth["mol_id"])
    assert ((calls.occ >= 0) & (calls.occ <= 1)).all()
    assert (calls.count > 0).all()


class TestRunners:
    def test_read_truth(self, quick_data):
        _, ddir = quick_data
        truth = run_bench.read_truth(ddir)
        assert list(truth.columns) == ["mol_id", "start", "end"]
        assert truth["mol_id"].nunique() == run_bench.QUICK_NMOL

    def test_catella(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_catella(ddir, tmp_path, llink=10, mu=0.0)
        _check_calls(calls, ddir)
        scores = run_bench.score_calls(calls, run_bench.read_truth(ddir))
        assert scores["mcc"].mean() > 0.8

    def test_catella_explicit_emax(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_catella(ddir, tmp_path, llink=10, emax=25.0)
        _check_calls(calls, ddir)

    @needs_fiberseq
    def test_ft(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_ft(ddir, tmp_path, nucleosome_length=75)
        _check_calls(calls, ddir)
        assert set(np.unique(calls.occ)) <= {0.0, 1.0}

    @needs_fiberseq
    def test_fiberhmm_pretrained(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_fiberhmm(ddir, tmp_path, retrain=False)
        _check_calls(calls, ddir)

    @needs_fiberseq
    def test_fiberhmm_retrained(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_fiberhmm(
            ddir, tmp_path, retrain=True, iterations=2, nuc_min_size=85
        )
        _check_calls(calls, ddir)
        model = run_bench.train_fiberhmm(ddir, iterations=2)
        assert model.exists()
