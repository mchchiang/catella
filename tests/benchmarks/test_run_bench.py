# test_run_bench.py

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

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


needs_env = pytest.mark.skipif(
    not _has_env(run_bench.ENV),
    reason=f"conda env {run_bench.ENV} not available",
)
needs_hiddenfoot = pytest.mark.skipif(
    run_bench._hiddenfoot_binary() is None,
    reason="HiddenFoot executable not available",
)

FAKE_HF = """#!{python}
import os, sys
nrow = sum(1 for _ in open("met.txt")) - 1
width = len(open("seq.txt").read().strip())
open("args.txt", "w").write(" ".join(sys.argv[1:]))
ll = os.environ.get("FAKE_HF_LL", "-100")
open("hf.params_SGD.txt", "w").write(f"99 {{ll}} 1 1 1 0.3 0.96\\n")
for name, val in (("hfprofile", 0.5), ("hfbinding", 0.001)):
    with open(f"hf.{{name}}.nucleosome.txt", "w") as fh:
        for _ in range(nrow):
            fh.write(" ".join([str(val)] * width) + "\\n")
"""


def _fake_hiddenfoot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / "hiddenfoot_omp"
    exe.write_text(FAKE_HF.format(python=sys.executable))
    exe.chmod(0o755)
    return exe


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
            fai = ddir / "synth.fa.fai"
            assert fai.read_text().split("\t")[:2] == [
                "chr1",
                str(run_bench.NBP),
            ]
            assert fai.stat().st_mtime >= (ddir / "synth.fa").stat().st_mtime
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

    @needs_env
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

    @needs_env
    def test_success_returns_process(self):
        proc = run_bench.run_in_env(run_bench.ENV, ["samtools", "--version"])
        assert isinstance(proc, subprocess.CompletedProcess)
        assert "samtools" in proc.stdout


class TestToolVersions:
    def _fake_run(self, monkeypatch, outputs):
        def fake_run(args, **kwargs):
            out = outputs.get(args[4])
            if out is None:
                return subprocess.CompletedProcess(args, 1, "", "missing")
            return subprocess.CompletedProcess(args, 0, out, "")

        monkeypatch.setenv("HIDDENFOOT", "/no/such/hiddenfoot")
        monkeypatch.setattr(run_bench.subprocess, "run", fake_run)

    def test_parsed_versions(self, monkeypatch):
        self._fake_run(
            monkeypatch,
            {
                "ft": "fibertools-rs v0.13.1\tgit-commit VERGEN_\n",
                "fiberhmm-call": "fiberhmm 3.0.2\n",
                "samtools": "samtools 1.25\nUsing htslib 1.22\n",
            },
        )
        versions = run_bench.tool_versions("e")
        assert versions == {
            "catella": run_bench.catella.__version__,
            "fibertools": "0.13.1",
            "fiberhmm": "3.0.2",
            "samtools": "1.25",
            "hiddenfoot": "unknown",
        }

    def test_unknown_versions(self, monkeypatch):
        self._fake_run(monkeypatch, {"samtools": "samtools\n"})
        versions = run_bench.tool_versions("e")
        assert versions["fibertools"] == "unknown"
        assert versions["fiberhmm"] == "unknown"
        assert versions["samtools"] == "unknown"

    def test_hiddenfoot_git_commit(self, monkeypatch, tmp_path):
        repo = tmp_path / "HiddenFoot"
        exe = _fake_hiddenfoot(repo / "build")
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q", str(repo)], check=True)
        subprocess.run([*git, "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [*git, "-C", str(repo), "commit", "-qm", "init"], check=True
        )
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        def no_env(env, args, cwd=None):
            raise RuntimeError("no env")

        monkeypatch.setattr(run_bench, "run_in_env", no_env)
        monkeypatch.setenv("HIDDENFOOT", str(exe))
        assert run_bench.tool_versions("e")["hiddenfoot"] == head

    @needs_env
    def test_real_env(self):
        versions = run_bench.tool_versions(run_bench.ENV)
        assert set(versions) == {
            "catella", "hiddenfoot", *run_bench.VERSION_CMDS
        }
        for tool in run_bench.VERSION_CMDS:
            assert versions[tool] != "unknown"


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


FAKE_VERSIONS = {
    "catella": "0.1.0",
    "fibertools": "0.13.1",
    "fiberhmm": "3.0.2",
    "samtools": "1.25",
}


@pytest.fixture
def fake_tools(monkeypatch):
    runs = []

    def fake_run_tool(tool, ddir, workdir, params, env=run_bench.ENV):
        runs.append(tool)
        truth = run_bench.read_truth(ddir)
        mol_ids = list(dict.fromkeys(truth["mol_id"]))
        perfect = tool == "catella" or 85 in params.values()
        if not perfect:
            truth = truth.iloc[::2]
        return run_bench.intervals_to_calls(truth, mol_ids, run_bench.NBP)

    monkeypatch.setattr(run_bench, "run_tool", fake_run_tool)
    monkeypatch.setattr(
        run_bench, "tool_versions", lambda env=run_bench.ENV: FAKE_VERSIONS
    )
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
            assert path.suffix == ".tsv"
            table = pd.read_csv(path, sep="\t")
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
        results = pd.read_csv(tuned_out / "results.tsv", sep="\t")
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
        paired = pd.read_csv(tuned_out / "summary_paired.tsv", sep="\t")
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

    def test_versions_recorded(self, tuned_out):
        run_bench.evaluate(tuned_out)
        saved = json.loads((tuned_out / "versions.json").read_text())
        assert saved == FAKE_VERSIONS
        text = (tuned_out / "summary.md").read_text()
        assert "## Tool versions" in text
        assert "- fibertools: 0.13.1" in text

    def test_summary_without_versions(self, tuned_out):
        run_bench.evaluate(tuned_out)
        (tuned_out / "versions.json").unlink()
        run_bench.write_summary(tuned_out)
        text = (tuned_out / "summary.md").read_text()
        assert "## Tool versions" not in text


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
        (evaluated_out / "tune_scores.tsv").unlink()
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
        scores = pd.read_csv(tmp_path / "tune_scores.tsv", sep="\t")
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
            "results.tsv",
            "summary_per_tool.tsv",
            "summary_paired.tsv",
            "summary.md",
        ):
            assert (tmp_path / name).exists()
        paired = pd.read_csv(tmp_path / "summary_paired.tsv", sep="\t")
        tuned_ft = paired[
            (paired["variant"] == "tuned") & (paired["other"] == "ft")
        ]
        assert tuned_ft["diff_mean"].iloc[0] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def quick_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    specs = run_bench.make_data(
        out, quick=True, convert=_has_env(run_bench.ENV)
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

    @needs_env
    def test_ft(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_ft(ddir, tmp_path, nucleosome_length=75)
        _check_calls(calls, ddir)
        assert set(np.unique(calls.occ)) <= {0.0, 1.0}

    @needs_env
    def test_fiberhmm_pretrained(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_fiberhmm(ddir, tmp_path, retrain=False)
        _check_calls(calls, ddir)

    @needs_env
    def test_fiberhmm_retrained(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = run_bench.run_fiberhmm(
            ddir, tmp_path, retrain=True, iterations=2, nuc_min_size=85
        )
        _check_calls(calls, ddir)
        model = run_bench.train_fiberhmm(ddir, iterations=2)
        assert model.exists()


class TestHiddenFootWindows:
    def test_starts(self):
        assert run_bench.hiddenfoot_windows(5000, 1000, 500) == list(
            range(0, 4001, 500)
        )
        assert run_bench.hiddenfoot_windows(5000, 1000, 700) == [
            0, 700, 1400, 2100, 2800, 3500, 4000
        ]
        assert run_bench.hiddenfoot_windows(5000, 5000, 100) == [0]

    @pytest.mark.parametrize(
        "window, step", [(0, 1), (6000, 500), (1000, 0), (1000, 1500)]
    )
    def test_invalid(self, window, step):
        with pytest.raises(ValueError):
            run_bench.hiddenfoot_windows(5000, window, step)


class TestStitchWindows:
    def _keys(self, mol_ids):
        starts = run_bench.hiddenfoot_windows(5000, 1000, 500)
        return starts, [(m, s) for s in starts for m in mol_ids]

    def test_nearest_centre(self):
        starts, keys = self._keys(["a", "b"])
        index = {s: k for k, s in enumerate(starts)}
        values = np.array([np.full(1000, index[s]) for _, s in keys])
        out = run_bench.stitch_windows(values, keys, ["a", "b"], 5000,
                                       1000, 500)
        assert out.shape == (2, 5000)
        assert out[0, 0] == 0
        assert out[0, 4999] == 8
        assert out[1, 760] == 1
        assert out[1, 740] == 0

    def test_reconstructs_position(self):
        _, keys = self._keys(["a", "b"])
        values = np.array([np.arange(s, s + 1000) for _, s in keys])
        out = run_bench.stitch_windows(values, keys, ["b", "a"], 5000,
                                       1000, 500)
        assert (out == np.arange(5000)).all()


class TestHiddenFoot:
    def test_input_files(self, quick_data, tmp_path):
        _, ddir = quick_data
        keys = run_bench.write_hiddenfoot_input(ddir, tmp_path, 1000, 500)
        mol_ids = run_bench._test_mol_ids(ddir)
        assert len(keys) == len(mol_ids) * 9
        assert (tmp_path / "seq.txt").read_text() == "A" * 1000 + "\n"
        assert (tmp_path / "wms.txt").read_text() == ""
        lines = (tmp_path / "met.txt").read_text().splitlines()
        sites = np.array(lines[0].split(), dtype=int)
        mat = np.array([line.split() for line in lines[1:]], dtype=int)
        assert mat.shape == (len(keys), len(sites))
        assert ((sites >= 0) & (sites < 1000)).all()
        assert set(np.unique(mat)) <= {0, 1, 2}
        k = keys.index((mol_ids[3], 500))
        tsv = pd.read_csv(Path(ddir) / "synth.tsv", sep="\t")
        mol = tsv[(tsv["read_id"] == mol_ids[3])]
        mol = mol[(mol["ref_position"] >= 500) & (mol["ref_position"] < 1500)]
        expected = dict.fromkeys(sites, 2)
        expected.update(
            zip(mol["ref_position"] - 500, (mol["mod_qual"] >= 0.5) * 1)
        )
        assert mat[k].tolist() == [expected[s] for s in sites]

    def test_fake_binary(self, quick_data, tmp_path):
        _, ddir = quick_data
        exe = _fake_hiddenfoot(tmp_path / "bin")
        work = tmp_path / "work"
        work.mkdir()
        calls = run_bench.run_hiddenfoot(ddir, work, binary=str(exe))
        nmol = run_bench.QUICK_NMOL
        assert calls.occ.shape == (nmol, run_bench.NBP)
        assert np.allclose(calls.occ, 0.5)
        assert np.allclose(calls.count, 0.001 * run_bench.NBP)
        args = (work / "args.txt").read_text().split()
        assert args[args.index("-nuclen") + 1] == "147"
        assert args[args.index("-numthreads") + 1] == "2"

    def test_underflow_raises(self, quick_data, tmp_path, monkeypatch):
        _, ddir = quick_data
        exe = _fake_hiddenfoot(tmp_path / "bin")
        monkeypatch.setenv("FAKE_HF_LL", "-inf")
        with pytest.raises(RuntimeError, match="underflow"):
            run_bench.run_hiddenfoot(ddir, tmp_path, binary=str(exe))

    def test_window_too_large(self, quick_data, tmp_path):
        _, ddir = quick_data
        with pytest.raises(ValueError):
            run_bench.run_hiddenfoot(ddir, tmp_path, window=2500)

    def test_missing_binary(self, quick_data, tmp_path):
        _, ddir = quick_data
        with pytest.raises(FileNotFoundError):
            run_bench.run_hiddenfoot(ddir, tmp_path, binary="/no/such/hf")

    def test_binary_resolution(self, tmp_path, monkeypatch):
        arg = _fake_hiddenfoot(tmp_path / "arg")
        env = _fake_hiddenfoot(tmp_path / "env")
        path = _fake_hiddenfoot(tmp_path / "path")
        monkeypatch.setenv("PATH", str(path.parent), prepend=":")
        monkeypatch.setenv("HIDDENFOOT", str(env))
        assert run_bench._hiddenfoot_binary(str(arg)) == str(arg)
        assert run_bench._hiddenfoot_binary() == str(env)
        monkeypatch.delenv("HIDDENFOOT")
        assert run_bench._hiddenfoot_binary() == str(path)
        monkeypatch.chdir(tmp_path)
        resolved = run_bench._hiddenfoot_binary("arg/hiddenfoot_omp")
        assert resolved == os.path.abspath("arg/hiddenfoot_omp")

    @pytest.mark.parametrize("command", ["tune", "evaluate"])
    def test_cli_option(self, tuned_out, monkeypatch, command):
        monkeypatch.setenv("HIDDENFOOT", "unset")
        monkeypatch.chdir(tuned_out)
        result = runner.invoke(
            run_bench.app, [command, str(tuned_out), "--hiddenfoot", "hf"]
        )
        assert result.exit_code == 0, result.output
        assert os.environ["HIDDENFOOT"] == os.path.abspath("hf")

    @needs_hiddenfoot
    def test_real_binary(self, quick_data, tmp_path):
        _, ddir = quick_data
        truth = run_bench.read_truth(ddir)
        mccs = []
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            calls = run_bench.run_hiddenfoot(ddir, tmp_path / name)
            _check_calls(calls, ddir)
            mccs.append(run_bench.score_calls(calls, truth)["mcc"].mean())
        assert min(mccs) > 0.8
        assert abs(mccs[0] - mccs[1]) < 0.01
