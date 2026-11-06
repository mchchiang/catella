# test_run_perf.py

import json
import subprocess
import sys
from pathlib import Path

import callers
import perf_worker
import psutil
import pytest
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


class TestTreeRss:
    def test_positive_for_current_process(self):
        assert perf_worker.tree_rss(psutil.Process()) > 0

    def test_includes_children(self):
        me = psutil.Process()
        before = perf_worker.tree_rss(me)
        hog = _start_hog()
        try:
            during = perf_worker.tree_rss(me)
        finally:
            hog.kill()
            hog.wait()
        assert during - before > 0.8 * HOG_MB * perf_worker.MB


class TestPeakRSS:
    def test_peak_at_least_start(self):
        with perf_worker.PeakRSS() as rss:
            start = rss.peak
        assert rss.peak >= start > 0

    def test_catches_finished_child(self):
        me = psutil.Process()
        before = perf_worker.tree_rss(me)
        with perf_worker.PeakRSS(interval=0.02) as rss:
            hog = _start_hog(sleep=0.5)
            hog.wait()
        assert rss.peak - before > 0.8 * HOG_MB * perf_worker.MB


class TestMeasureTool:
    def test_catella(self, quick_ddir):
        result = run_perf.measure_tool(
            "catella", quick_ddir, {"llink": 10}, nthread=1
        )
        assert set(result) == {
            "wall",
            "cpu",
            "peak_rss",
            "base_rss",
            "fit_time",
        }
        assert result["wall"] > 0
        assert result["cpu"] > 0
        assert result["peak_rss"] >= result["base_rss"] > 0
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
