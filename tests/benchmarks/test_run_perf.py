# test_run_perf.py

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

if sys.platform == "win32":
    pytest.skip(
        "perf benchmarks need the Unix-only resource module",
        allow_module_level=True,
    )
psutil = pytest.importorskip("psutil")

import callers
import perf_worker
import run_perf
import run_synth

HOG_MB = 200
HOG = (
    "import sys, time; b = b'x' * ({mb} * 1024 * 1024); "
    "sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep({sleep})"
)


def _start_hog(sleep=2.0):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOG.format(mb=HOG_MB, sleep=sleep)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    return proc


@pytest.fixture(scope="module")
def quick_ddir(tmp_path_factory):
    out = tmp_path_factory.mktemp("perf")
    specs = run_synth.make_data(out, quick=True, convert=False)
    return specs[1]["dir"]


class TestPerfSpecs:
    def test_full_axes(self):
        specs = run_perf.perf_specs()
        nmol = [s["value"] for s in specs if s["axis"] == "nmol"]
        nbp = [s["value"] for s in specs if s["axis"] == "nbp"]
        assert nmol == list(run_perf.NMOL_AXIS)
        assert nbp == list(run_perf.NBP_AXIS)

    def test_quick_axes(self):
        specs = run_perf.perf_specs(quick=True)
        assert len(specs) == len(run_perf.QUICK_NMOL_AXIS) + len(
            run_perf.QUICK_NBP_AXIS
        )
        for spec in specs:
            if spec["axis"] == "nbp":
                assert spec["params"]["nmol"] == run_perf.QUICK_BASE_NMOL

    def test_axis_sizes(self):
        for spec in run_perf.perf_specs():
            params = spec["params"]
            if spec["axis"] == "nmol":
                assert params["nmol"] == spec["value"]
                assert params["nbp"] == run_perf.BASE_NBP
            else:
                assert params["nbp"] == spec["value"]
                assert params["nmol"] == run_perf.BASE_NMOL

    def test_default_condition(self):
        params = run_perf.perf_specs()[0]["params"]
        for key, value in run_synth.BASE_PARAMS.items():
            assert params[key] == value
        assert params["controls"] and params["double_strand"]
        assert params["seed"] == run_perf.SEED

    def test_shared_dataset_has_one_name(self):
        specs = run_perf.perf_specs()
        name = run_perf.dataset_name(run_perf.BASE_NMOL, run_perf.BASE_NBP)
        assert sum(s["cond"] == name for s in specs) == 2


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("perf_data")
    return out, run_perf.make_data(out, quick=True, convert=False)


class TestMakeData:
    def test_one_directory_per_dataset(self, made):
        _, specs = made
        dirs = {s["dir"] for s in specs}
        assert len(dirs) == len({s["cond"] for s in specs}) == 3

    def test_datasets_written(self, made):
        _, specs = made
        for spec in specs:
            assert (Path(spec["dir"]) / "synth.tsv").exists()
            nmol = len(callers._test_mol_ids(spec["dir"]))
            assert nmol == spec["params"]["nmol"]

    def test_record_written(self, made):
        out, specs = made
        record = json.loads((out / "perf_datasets.json").read_text())
        assert record["quick"] is True
        assert record["datasets"] == specs


class TestThreadParams:
    @pytest.mark.parametrize(
        ("tool", "expected"),
        [
            ("catella", {"nworker": 3}),
            ("ft", {"threads": 3}),
            ("hiddenfoot", {"threads": 3}),
            ("fiberhmm", {"threads": 3, "io_threads": 3}),
            ("fiberhmm_pre", {"threads": 3, "io_threads": 3}),
        ],
    )
    def test_tools(self, tool, expected):
        assert run_perf.thread_params(tool, 3) == expected

    def test_every_tool_covered(self):
        for tool in callers.TOOLS:
            run_perf.thread_params(tool, 1)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown tool 'x'"):
            run_perf.thread_params("x", 1)


class TestResultPath:
    def test_layout(self, tmp_path):
        path = run_perf.result_path(tmp_path, "ft", "nmol", 25, 2)
        assert path == tmp_path / "results" / "ft" / "nmol_25_r2.json"


class TestMachineInfo:
    def test_keys(self):
        info = run_perf.machine_info()
        assert set(info) == {
            "platform",
            "machine",
            "cpu",
            "physical_cores",
            "logical_cores",
            "memory_gb",
            "python",
        }
        assert info["logical_cores"] >= 1
        assert info["memory_gb"] > 0


class TestRun:
    @pytest.fixture
    def fake(self, made, monkeypatch, tmp_path):
        src, _ = made
        out = tmp_path / "out"
        out.mkdir()
        shutil.copy(src / "perf_datasets.json", out)
        calls = []

        def fake_measure(tool, ddir, params, **kwargs):
            calls.append({"tool": tool, "ddir": ddir, "params": params})
            if tool == "fiberhmm":
                assert not list(Path(ddir).glob("fiberhmm_model*"))
            if tool == "hiddenfoot":
                raise RuntimeError("measuring hiddenfoot failed; boom")
            return {"wall": 1.0, "peak_mem": 2.0, "fit_time": None}

        monkeypatch.setattr(run_perf, "measure_tool", fake_measure)
        monkeypatch.setattr(run_perf, "tool_versions", lambda env: {})
        monkeypatch.setattr(
            run_perf, "env_bin_dir", lambda env: tmp_path / "bin"
        )
        monkeypatch.setenv("BENCH_TOOL_BIN", "unset")
        monkeypatch.delenv("BENCH_TOOL_BIN")
        return out, calls

    def test_every_run_written(self, fake):
        out, calls = fake
        record = json.loads((out / "perf_datasets.json").read_text())
        paths = run_perf.run(out, nrep=2)
        expected = 2 * len(record["datasets"]) * len(callers.TOOLS)
        assert len(paths) == len(calls) == expected
        assert all(p.exists() for p in paths)

    def test_quick_defaults_to_one_repeat(self, fake):
        out, _ = fake
        paths = run_perf.run(out, tools=["catella"])
        assert {json.loads(p.read_text())["rep"] for p in paths} == {0}

    def test_result_metadata(self, fake):
        out, _ = fake
        run_perf.run(out, tools=["catella"], nthread=3)
        path = run_perf.result_path(out, "catella", "nbp", 2000, 0)
        result = json.loads(path.read_text())
        assert result["tool"] == "catella"
        assert result["axis"] == "nbp"
        assert result["value"] == 2000
        assert result["nbp"] == 2000
        assert result["nmol"] == run_perf.QUICK_BASE_NMOL
        assert result["nthread"] == 3
        assert result["params"] == {"nworker": 3}
        assert result["wall"] == 1.0

    def test_resume_skips_finished(self, fake):
        out, calls = fake
        run_perf.run(out, tools=["catella"])
        n = len(calls)
        assert run_perf.run(out, tools=["catella"]) == []
        assert len(calls) == n

    def test_overwrite_reruns(self, fake):
        out, _ = fake
        first = run_perf.run(out, tools=["catella"])
        second = run_perf.run(out, tools=["catella"], overwrite=True)
        assert second == first

    def test_failure_recorded_and_retried(self, fake):
        out, _ = fake
        paths = run_perf.run(out, tools=["hiddenfoot"])
        for path in paths:
            assert "boom" in json.loads(path.read_text())["error"]
        assert run_perf.run(out, tools=["hiddenfoot"]) == paths

    def test_tuned_params_merged(self, fake, tmp_path):
        out, calls = fake
        tuned = tmp_path / "tuned_params.json"
        tuned.write_text(json.dumps({"ft": {"nucleosome_length": 75}}))
        run_perf.run(out, tools=["ft"], tuned=tuned, nthread=4)
        assert calls[0]["params"] == {"nucleosome_length": 75, "threads": 4}

    def test_fiberhmm_cache_removed(self, fake):
        out, calls = fake
        record = json.loads((out / "perf_datasets.json").read_text())
        for spec in record["datasets"]:
            cache = Path(spec["dir"]) / "fiberhmm_model"
            cache.mkdir(exist_ok=True)
            (cache / "best-model.json").write_text("{}")
        run_perf.run(out, tools=["fiberhmm"])
        assert calls

    def test_bin_dir_set_for_external_tools(self, fake, tmp_path):
        out, _ = fake
        run_perf.run(out, tools=["ft"])
        assert os.environ["BENCH_TOOL_BIN"] == str(tmp_path / "bin")

    def test_bin_dir_not_needed_for_catella(self, fake, monkeypatch):
        out, _ = fake

        def fail(env):
            raise AssertionError("env_bin_dir called")

        monkeypatch.setattr(run_perf, "env_bin_dir", fail)
        run_perf.run(out, tools=["catella"])
        assert "BENCH_TOOL_BIN" not in os.environ

    def test_metadata_files_written(self, fake):
        out, _ = fake
        run_perf.run(out, tools=["catella"])
        assert json.loads((out / "versions.json").read_text()) == {}
        machine = json.loads((out / "machine.json").read_text())
        assert "cpu" in machine

    def test_unknown_tool_raises(self, fake):
        out, _ = fake
        with pytest.raises(ValueError, match="unknown tools"):
            run_perf.run(out, tools=["nosuch"])

    def test_real_catella_run(self, made, tmp_path, monkeypatch):
        src, _ = made
        out = tmp_path / "out"
        out.mkdir()
        shutil.copy(src / "perf_datasets.json", out)
        monkeypatch.setattr(run_perf, "tool_versions", lambda env: {})
        paths = run_perf.run(out, tools=["catella"], nthread=1)
        for path in paths:
            result = json.loads(path.read_text())
            assert "error" not in result
            assert result["wall"] > 0
            assert result["peak_mem"] > result["base_mem"]


def _write_result(out, tool, axis, value, rep, wall, fit=None, error=None):
    result = {
        "tool": tool,
        "axis": axis,
        "value": value,
        "rep": rep,
        "nmol": value if axis == "nmol" else 10,
        "nbp": value if axis == "nbp" else 5000,
        "nthread": 2,
        "params": {"threads": 2},
    }
    if error is None:
        result.update(
            {
                "wall": wall,
                "cpu": 2 * wall,
                "peak_mem": 300.0,
                "base_mem": 100.0,
                "mem_metric": "footprint",
                "fit_time": fit,
            }
        )
    else:
        result["error"] = error
    path = run_perf.result_path(out, tool, axis, value, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))


@pytest.fixture
def results_dir(tmp_path):
    for rep, wall in enumerate((1.0, 3.0, 2.0)):
        _write_result(tmp_path, "ft", "nmol", 10, rep, wall)
        _write_result(tmp_path, "catella", "nmol", 10, rep, wall, fit=0.5)
        _write_result(tmp_path, "catella", "nbp", 2000, rep, wall, fit=0.5)
    _write_result(
        tmp_path, "hiddenfoot", "nmol", 10, 0, None,
        error="measuring hiddenfoot failed; Traceback\nOSError: no binary",
    )
    return tmp_path


class TestCollect:
    def test_rows_and_errors(self, results_dir):
        results, errors = run_perf.collect(results_dir)
        assert len(results) == 9
        assert errors["tool"].tolist() == ["hiddenfoot"]

    def test_derived_measures(self, results_dir):
        results, _ = run_perf.collect(results_dir)
        catella = results[results["tool"] == "catella"]
        assert (catella["call_time"] == catella["wall"] - 0.5).all()
        assert (results["tool_mem"] == 200.0).all()

    def test_old_measure_rejected(self, results_dir):
        path = run_perf.result_path(results_dir, "ft", "nmol", 10, 0)
        result = json.loads(path.read_text())
        result["peak_rss"] = result.pop("peak_mem")
        path.write_text(json.dumps(result))
        with pytest.raises(ValueError, match="rerun with --overwrite"):
            run_perf.collect(results_dir)

    def test_no_fit_gives_nan(self, results_dir):
        results, _ = run_perf.collect(results_dir)
        ft = results[results["tool"] == "ft"]
        assert ft["fit_time"].isna().all()
        assert ft["call_time"].isna().all()

    def test_params_dropped(self, results_dir):
        results, _ = run_perf.collect(results_dir)
        assert "params" not in results

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no run results found"):
            run_perf.collect(tmp_path)


class TestSummarize:
    def test_median_over_repeats(self, results_dir):
        results, _ = run_perf.collect(results_dir)
        table = run_perf.summarize(results)
        row = table[(table["tool"] == "ft") & (table["axis"] == "nmol")]
        assert row["wall"].item() == 2.0
        assert row["cpu"].item() == 4.0
        assert row["nrep"].item() == 3

    def test_one_row_per_point_and_tool(self, results_dir):
        results, _ = run_perf.collect(results_dir)
        table = run_perf.summarize(results)
        assert len(table) == 3
        for metric in run_perf.METRICS:
            assert metric in table


class TestWriteSummary:
    def test_files_written(self, results_dir):
        (results_dir / "machine.json").write_text(json.dumps({"cpu": "X1"}))
        (results_dir / "versions.json").write_text(
            json.dumps({"catella": "9.9"})
        )
        table = run_perf.write_summary(results_dir)
        assert len(table) == 3
        tsv = pd.read_csv(results_dir / "perf.tsv", sep="\t")
        assert len(tsv) == 9
        text = (results_dir / "perf.md").read_text()
        assert "## Scaling with molecule count" in text
        assert "## Scaling with length" in text
        assert "hiddenfoot nmol=10 rep 0: OSError: no binary" in text
        assert "- cpu: X1" in text
        assert "- catella: 9.9" in text
        assert "## Notes" in text
        assert "Memory measure: footprint" in text

    def test_all_failed_raises(self, tmp_path):
        _write_result(tmp_path, "ft", "nmol", 10, 0, None, error="boom")
        with pytest.raises(ValueError, match="every run"):
            run_perf.write_summary(tmp_path)


def _pdf_pages(path):
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", path.read_bytes()))


class TestPlotPerf:
    def test_default_path_two_pages(self, results_dir):
        run_perf.write_summary(results_dir)
        path = run_perf.plot_perf(results_dir)
        assert path == results_dir / "plots" / "perf.pdf"
        assert _pdf_pages(path) == 2

    def test_no_fit_step_one_page(self, tmp_path):
        for rep in range(2):
            _write_result(tmp_path, "ft", "nmol", 10, rep, 1.0)
            _write_result(tmp_path, "ft", "nbp", 2000, rep, 1.0)
        run_perf.write_summary(tmp_path)
        path = run_perf.plot_perf(tmp_path, tmp_path / "a.pdf")
        assert path == tmp_path / "a.pdf"
        assert _pdf_pages(path) == 1

    def test_missing_summary_raises(self, tmp_path):
        with pytest.raises(ValueError, match="run summarize first"):
            run_perf.plot_perf(tmp_path)


needs_macos = pytest.mark.skipif(
    sys.platform != "darwin", reason="physical footprint is macOS only"
)


class TestProcessMemory:
    def test_metric_matches_platform(self):
        expected = "footprint" if sys.platform == "darwin" else "rss"
        assert perf_worker.MEMORY_METRIC == expected

    def test_positive_for_current_process(self):
        assert perf_worker.process_memory(psutil.Process()) > 0

    def test_rss_fallback(self, monkeypatch):
        monkeypatch.setattr(perf_worker, "MEMORY_METRIC", "rss")
        me = psutil.Process()
        rss = me.memory_info().rss
        assert abs(perf_worker.process_memory(me) - rss) < 50 * perf_worker.MB
        assert perf_worker.lifetime_peak_memory() == 0

    def test_exited_process_raises(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        child = psutil.Process(proc.pid)
        proc.wait()
        with pytest.raises(psutil.Error):
            perf_worker.process_memory(child)

    @needs_macos
    def test_footprint_drops_after_free(self):
        me = psutil.Process()
        before = perf_worker.process_memory(me)
        blocks = [bytes(range(256)) * 400 for _ in range(2000)]
        during = perf_worker.process_memory(me)
        del blocks
        after = perf_worker.process_memory(me)
        assert during - before > 150 * perf_worker.MB
        assert after - before < 50 * perf_worker.MB

    @needs_macos
    def test_lifetime_peak_at_least_current(self):
        current = perf_worker.process_memory(psutil.Process())
        assert perf_worker.lifetime_peak_memory() >= current


class TestTreeMemory:
    def test_positive_for_current_process(self):
        assert perf_worker.tree_memory(psutil.Process()) > 0

    def test_includes_children(self):
        me = psutil.Process()
        before = perf_worker.tree_memory(me)
        hog = _start_hog()
        try:
            during = perf_worker.tree_memory(me)
        finally:
            hog.kill()
            hog.wait()
        assert during - before > 0.8 * HOG_MB * perf_worker.MB


class TestPeakMemory:
    def test_peak_at_least_start(self):
        with perf_worker.PeakMemory() as mem:
            start = mem.peak
        assert mem.peak >= start > 0

    def test_catches_finished_child(self):
        me = psutil.Process()
        before = perf_worker.tree_memory(me)
        with perf_worker.PeakMemory(interval=0.02) as mem:
            hog = _start_hog(sleep=0.5)
            hog.wait()
        assert mem.peak - before > 0.8 * HOG_MB * perf_worker.MB


class TestMeasureTool:
    def test_catella(self, quick_ddir):
        result = run_perf.measure_tool(
            "catella", quick_ddir, {"llink": 10}, nthread=1
        )
        assert set(result) == {
            "wall",
            "cpu",
            "peak_mem",
            "base_mem",
            "mem_metric",
            "fit_time",
        }
        assert result["wall"] > 0
        assert result["cpu"] > 0
        assert result["peak_mem"] >= result["base_mem"] > 0
        assert result["mem_metric"] == perf_worker.MEMORY_METRIC
        assert result["fit_time"] == 0.0

    def test_fit_time_reported(self, quick_ddir):
        result = run_perf.measure_tool("catella", quick_ddir, {}, nthread=1)
        assert 0 < result["fit_time"] < result["wall"]

    def test_thread_variables_set(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(args, **kwargs):
            seen.update(kwargs["env"])
            result = {"wall": 1.0}
            with open(args[-1], "w") as fh:
                json.dump(result, fh)
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(run_perf.subprocess, "run", fake_run)
        result = run_perf.measure_tool("catella", tmp_path, {}, nthread=3)
        assert result == {"wall": 1.0}
        for name in run_perf.THREAD_ENV_VARS:
            assert seen[name] == "3"

    def test_unknown_tool_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="measuring nosuch failed"):
            run_perf.measure_tool("nosuch", tmp_path, {})

    def test_nonpositive_nthread_raises(self, tmp_path):
        with pytest.raises(ValueError, match="nthread must be positive"):
            run_perf.measure_tool("catella", tmp_path, {}, nthread=0)

    def test_workdir_left_clean(self, quick_ddir, tmp_path):
        run_perf.measure_tool(
            "catella", quick_ddir, {"llink": 10}, nthread=1, workdir=tmp_path
        )
        assert list(tmp_path.iterdir()) == []
