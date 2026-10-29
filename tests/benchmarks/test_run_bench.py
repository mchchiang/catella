# test_run_bench.py

import json
import subprocess

import numpy as np
import pandas as pd
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
