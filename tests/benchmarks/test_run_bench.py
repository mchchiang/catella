# test_run_bench.py

import json
import subprocess

import pytest
import run_bench


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

    @needs_fiberseq
    def test_writes_indexed_bams(self, tmp_path):
        run_bench.make_data(tmp_path, quick=True)
        ddir = run_bench.dataset_dir(tmp_path, "default", 1)
        for name in ("synth", "synth.meth", "synth.unmeth"):
            assert (ddir / f"{name}.bam").exists()
            assert (ddir / f"{name}.bam.bai").exists()


class TestRunInEnv:
    def test_failure_raises(self):
        with pytest.raises((RuntimeError, FileNotFoundError)):
            run_bench.run_in_env("no_such_env_xyz", ["true"])

    @needs_fiberseq
    def test_success_returns_process(self):
        proc = run_bench.run_in_env("fiberseq", ["samtools", "--version"])
        assert isinstance(proc, subprocess.CompletedProcess)
        assert "samtools" in proc.stdout
