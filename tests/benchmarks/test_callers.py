# test_callers.py

import os
import subprocess
import sys
from pathlib import Path

import callers
import numpy as np
import pandas as pd
import pytest
import run_synth
import synthgen


def _has_env(env):
    try:
        callers.run_in_env(env, ["samtools", "--version"])
    except (RuntimeError, FileNotFoundError):
        return False
    return True


needs_env = pytest.mark.skipif(
    not _has_env(callers.ENV),
    reason=f"conda env {callers.ENV} not available",
)
needs_hiddenfoot = pytest.mark.skipif(
    callers._hiddenfoot_binary() is None,
    reason="HiddenFoot executable not available",
)

FAKE_HF = """import os, sys
nrow = sum(1 for _ in open("met.txt")) - 1
width = len(open("seq.txt").read().strip())
open("args.txt", "w").write(" ".join(sys.argv[1:]))
ll = os.environ.get("FAKE_HF_LL", "-100")
open("hf.params_SGD.txt", "w").write(f"99 {ll} 1 1 1 0.3 0.96\\n")
for name, val in (("hfprofile", 0.5), ("hfbinding", 0.001)):
    with open(f"hf.{name}.nucleosome.txt", "w") as fh:
        for _ in range(nrow):
            fh.write(" ".join([str(val)] * width) + "\\n")
"""


def _fake_hiddenfoot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_hiddenfoot.py"
    script.write_text(FAKE_HF)
    if os.name == "nt":
        exe = directory / "hiddenfoot_omp.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        exe = directory / "hiddenfoot_omp"
        exe.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n'
        )
        exe.chmod(0o755)
    return exe


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

        monkeypatch.setattr(callers.subprocess, "run", fake_run)
        callers.run_in_env("e", ["samtools", "--version"])
        assert seen == [[expected, "run", "-n", "e", "samtools", "--version"]]

    def test_failure_raises(self):
        with pytest.raises((RuntimeError, FileNotFoundError)):
            callers.run_in_env("no_such_env_xyz", ["true"])

    @needs_env
    def test_success_returns_process(self):
        proc = callers.run_in_env(callers.ENV, ["samtools", "--version"])
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
        monkeypatch.setattr(callers.subprocess, "run", fake_run)

    def test_parsed_versions(self, monkeypatch):
        self._fake_run(
            monkeypatch,
            {
                "ft": "fibertools-rs v0.13.1\tgit-commit VERGEN_\n",
                "fiberhmm-call": "fiberhmm 3.0.2\n",
                "samtools": "samtools 1.25\nUsing htslib 1.22\n",
            },
        )
        versions = callers.tool_versions("e")
        assert versions == {
            "catella": callers.catella.__version__,
            "fibertools": "0.13.1",
            "fiberhmm": "3.0.2",
            "samtools": "1.25",
            "hiddenfoot": "unknown",
        }

    def test_unknown_versions(self, monkeypatch):
        self._fake_run(monkeypatch, {"samtools": "samtools\n"})
        versions = callers.tool_versions("e")
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

        monkeypatch.setattr(callers, "run_in_env", no_env)
        monkeypatch.setenv("HIDDENFOOT", str(exe))
        assert callers.tool_versions("e")["hiddenfoot"] == head

    @needs_env
    def test_real_env(self):
        versions = callers.tool_versions(callers.ENV)
        assert set(versions) == {
            "catella", "hiddenfoot", *callers.VERSION_CMDS
        }
        for tool in callers.VERSION_CMDS:
            assert versions[tool] != "unknown"


class TestIntervalsToCalls:
    def test_occupancy_and_count(self):
        intervals = pd.DataFrame(
            {"mol_id": ["a", "a", "b"], "start": [2, 6, 0], "end": [4, 8, 3]}
        )
        calls = callers.intervals_to_calls(intervals, ["a", "b", "c"], 10)
        assert calls.mol_ids.tolist() == ["a", "b", "c"]
        assert calls.occ[0].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        assert calls.occ[1].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        assert calls.occ[2].sum() == 0
        assert calls.count.tolist() == [2, 1, 0]

    def test_clipping_and_unknown_molecules(self):
        intervals = pd.DataFrame(
            {"mol_id": ["a", "z"], "start": [-3, 0], "end": [12, 5]}
        )
        calls = callers.intervals_to_calls(intervals, ["a"], 10)
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
        intervals = callers.parse_ft_extract(path)
        assert intervals.values.tolist() == [
            ["m1", 10, 157],
            ["m1", 200, 350],
            ["m3", 5, 105],
        ]


class TestSaveLoadCalls:
    def test_roundtrip(self, tmp_path):
        calls = callers.Calls(
            np.array(["a", "b"]),
            np.array([[0.0, 0.25, 1.0], [0.5, 0.75, 0.0]], dtype=np.float32),
            np.array([1.5, 2.0]),
        )
        path = callers.save_calls(calls, tmp_path / "sub" / "c.h5")
        back = callers.load_calls(path)
        assert back.mol_ids.tolist() == ["a", "b"]
        np.testing.assert_allclose(back.occ, calls.occ, atol=1e-3)
        np.testing.assert_array_equal(back.count, calls.count)
        assert back.llink is None

    def test_llink_round_trip(self, tmp_path):
        calls = callers.Calls(
            np.array(["a"]),
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([1.0]),
            llink=0,
        )
        path = callers.save_calls(calls, tmp_path / "c.h5")
        assert callers.load_calls(path).llink == 0


@pytest.fixture(scope="module")
def quick_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    specs = run_synth.make_data(
        out, quick=True, convert=_has_env(callers.ENV)
    )
    return out, specs[1]["dir"]


def _check_calls(calls, ddir):
    truth = run_synth.read_truth(ddir)
    nmol = run_synth.QUICK_NMOL
    assert calls.occ.shape == (nmol, run_synth.NBP)
    assert set(calls.mol_ids) == set(truth["mol_id"])
    assert ((calls.occ >= 0) & (calls.occ <= 1)).all()
    assert (calls.count > 0).all()


class TestRunners:
    def test_catella(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = callers.run_catella(ddir, tmp_path, llink=10, mu=0.0)
        _check_calls(calls, ddir)
        scores = run_synth.score_calls(calls, run_synth.read_truth(ddir))
        assert scores["mcc"].mean() > 0.8

    def test_catella_explicit_emax(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = callers.run_catella(ddir, tmp_path, llink=10, emax=25.0)
        _check_calls(calls, ddir)

    def test_catella_records_fixed_llink(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = callers.run_catella(ddir, tmp_path, llink=10)
        assert calls.llink == 10

    def test_catella_ml_uses_estimate(self, quick_data, tmp_path, monkeypatch):
        _, ddir = quick_data
        seen = {}
        original = callers.catella.estimate_llink

        def spy(exp, settings, **kwargs):
            best = original(exp, settings, **kwargs)
            seen.update(best)
            return best

        monkeypatch.setattr(callers.catella, "estimate_llink", spy)
        calls = callers.run_catella(ddir, tmp_path, llink="ml")
        _check_calls(calls, ddir)
        assert len(seen) == 1
        assert calls.llink == next(iter(seen.values()))
        scores = run_synth.score_calls(calls, run_synth.read_truth(ddir))
        assert scores["mcc"].mean() > 0.8

    def test_catella_rejects_unknown_llink_text(self, quick_data, tmp_path):
        _, ddir = quick_data
        with pytest.raises(ValueError, match="llink must be an integer"):
            callers.run_catella(ddir, tmp_path, llink="auto")

    @pytest.mark.parametrize(
        ("tool", "fixed"),
        [
            ("catella_ml", {"llink": "ml"}),
            ("catella_ml_q05", {"llink": "ml", "q_thres": 0.5}),
        ],
    )
    def test_ml_tools_fix_arguments(self, tmp_path, monkeypatch, tool, fixed):
        seen = {}

        def fake_runner(ddir, workdir, **kwargs):
            seen.update(kwargs)

        monkeypatch.setitem(
            callers.TOOLS, tool, (fake_runner, callers.TOOLS[tool][1])
        )
        callers.run_tool(tool, tmp_path, tmp_path, {})
        assert seen == fixed

    @needs_env
    def test_ft(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = callers.run_ft(ddir, tmp_path, nucleosome_length=75)
        _check_calls(calls, ddir)
        assert set(np.unique(calls.occ)) <= {0.0, 1.0}

    @needs_env
    def test_fiberhmm_pretrained(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = callers.run_fiberhmm(ddir, tmp_path, retrain=False)
        _check_calls(calls, ddir)

    @needs_env
    def test_fiberhmm_retrained(self, quick_data, tmp_path):
        _, ddir = quick_data
        calls = callers.run_fiberhmm(
            ddir, tmp_path, retrain=True, iterations=2, nuc_min_size=85
        )
        _check_calls(calls, ddir)
        model = callers.train_fiberhmm(ddir, iterations=2)
        assert model.exists()


class TestChannels:
    def _record(self, monkeypatch):
        seen = []

        def fake_run_in_env(env, args, cwd=None):
            seen.append(args)
            if args[0] == "fiberhmm-probs":
                tables = Path(args[args.index("-o") + 1]) / "tables"
                tables.mkdir(parents=True)
                for state in ("accessible", "inaccessible"):
                    (tables / f"x_{state}_C_k3.tsv").write_text("")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(callers, "run_in_env", fake_run_in_env)
        return seen

    def test_fiberhmm_gpc_training(self, tmp_path, monkeypatch):
        seen = self._record(monkeypatch)
        model = callers.train_fiberhmm(tmp_path, channel="m")
        assert model == tmp_path / "fiberhmm_model.gpc" / "best-model.json"
        probs, train = seen
        assert str(tmp_path / "synth.meth.gpc.bam") in probs
        assert str(tmp_path / "synth.unmeth.gpc.bam") in probs
        assert probs[probs.index("--mode") + 1] == "gpc"
        assert str(tmp_path / "synth.gpc.bam") in train
        assert train[train.index("--mode") + 1] == "gpc"

    def test_fiberhmm_gpc_calling(self, tmp_path, monkeypatch):
        seen = self._record(monkeypatch)
        monkeypatch.setattr(callers, "_ft_calls", lambda *a: "calls")
        out = callers.run_fiberhmm(tmp_path, tmp_path, channel="m")
        assert out == "calls"
        call = seen[-1]
        assert call[call.index("-i") + 1] == str(tmp_path / "synth.gpc.bam")

    def test_fiberhmm_invalid_channel(self, tmp_path):
        with pytest.raises(ValueError):
            callers.train_fiberhmm(tmp_path, channel="x")
        with pytest.raises(ValueError):
            callers.run_fiberhmm(tmp_path, tmp_path, retrain=False,
                                 channel="m")

    def test_catella_mtase(self, tmp_path, monkeypatch):
        seen = {}

        def fake_load_raw(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop")

        monkeypatch.setattr(callers.catella, "load_raw", fake_load_raw)
        with pytest.raises(RuntimeError, match="stop"):
            callers.run_catella(tmp_path, tmp_path, mtase=["A", "GC"])
        assert seen["mtase"] == ["A", "GC"]

    @pytest.mark.parametrize("q_thres", [None, 0.5])
    def test_catella_q_thres(self, tmp_path, monkeypatch, q_thres):
        seen = {}

        def fake_model_prob(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop")

        monkeypatch.setattr(callers.catella, "load_raw", lambda **_: None)
        monkeypatch.setattr(
            callers.catella, "compute_model_prob", fake_model_prob
        )
        with pytest.raises(RuntimeError, match="stop"):
            callers.run_catella(tmp_path, tmp_path, q_thres=q_thres)
        assert seen["q_thres"] == q_thres

    def test_binarized_tool_fixes_threshold(self, tmp_path, monkeypatch):
        seen = {}

        def fake_runner(ddir, workdir, **kwargs):
            seen.update(kwargs)

        monkeypatch.setitem(
            callers.TOOLS,
            "catella_q05",
            (fake_runner, callers.TOOLS["catella_q05"][1]),
        )
        callers.run_tool("catella_q05", tmp_path, tmp_path, {"llink": 10})
        assert seen == {"q_thres": 0.5, "llink": 10}


class TestHiddenFootWindows:
    def test_starts(self):
        assert callers.hiddenfoot_windows(5000, 1000, 500) == list(
            range(0, 4001, 500)
        )
        assert callers.hiddenfoot_windows(5000, 1000, 700) == [
            0, 700, 1400, 2100, 2800, 3500, 4000
        ]
        assert callers.hiddenfoot_windows(5000, 5000, 100) == [0]

    @pytest.mark.parametrize(
        "window, step", [(0, 1), (6000, 500), (1000, 0), (1000, 1500)]
    )
    def test_invalid(self, window, step):
        with pytest.raises(ValueError):
            callers.hiddenfoot_windows(5000, window, step)


class TestStitchWindows:
    def _keys(self, mol_ids):
        starts = callers.hiddenfoot_windows(5000, 1000, 500)
        return starts, [(m, s) for s in starts for m in mol_ids]

    def test_nearest_centre(self):
        starts, keys = self._keys(["a", "b"])
        index = {s: k for k, s in enumerate(starts)}
        values = np.array([np.full(1000, index[s]) for _, s in keys])
        out = callers.stitch_windows(values, keys, ["a", "b"], 5000,
                                     1000, 500)
        assert out.shape == (2, 5000)
        assert out[0, 0] == 0
        assert out[0, 4999] == 8
        assert out[1, 760] == 1
        assert out[1, 740] == 0

    def test_reconstructs_position(self):
        _, keys = self._keys(["a", "b"])
        values = np.array([np.arange(s, s + 1000) for _, s in keys])
        out = callers.stitch_windows(values, keys, ["b", "a"], 5000,
                                     1000, 500)
        assert (out == np.arange(5000)).all()


class TestHiddenFoot:
    def test_input_files(self, quick_data, tmp_path):
        _, ddir = quick_data
        keys = callers.write_hiddenfoot_input(ddir, tmp_path, 1000, 500)
        mol_ids = callers._test_mol_ids(ddir)
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
        calls = callers.run_hiddenfoot(ddir, work, binary=str(exe))
        nmol = run_synth.QUICK_NMOL
        assert calls.occ.shape == (nmol, run_synth.NBP)
        assert np.allclose(calls.occ, 0.5)
        assert np.allclose(calls.count, 0.001 * run_synth.NBP)
        args = (work / "args.txt").read_text().split()
        assert args[args.index("-nuclen") + 1] == "147"
        assert args[args.index("-numthreads") + 1] == "2"

    @pytest.mark.parametrize(
        "mod_code, empty",
        [("a", lambda s: s >= 2500), ("m", lambda s: s < 2000)],
    )
    def test_input_mod_code(self, tmp_path, mod_code, empty):
        ddir = tmp_path / "data"
        synthgen.make_dataset(ddir, nbp=5000, nmol=3, seed=0)
        tsv = pd.read_csv(ddir / "synth.tsv", sep="\t")
        tsv.loc[tsv["ref_position"] >= 2500, "mod_code"] = "m"
        tsv.to_csv(ddir / "synth.tsv", sep="\t", index=False)
        keys = callers.write_hiddenfoot_input(
            ddir, tmp_path, 1000, 500, mod_code
        )
        lines = (tmp_path / "met.txt").read_text().splitlines()[1:]
        rows = np.array([line.split() for line in lines], dtype=int)
        for row, (_, start) in zip(rows, keys):
            assert (row == 2).all() == empty(start)

    def test_dataset_length(self, tmp_path):
        ddir = tmp_path / "data"
        synthgen.make_dataset(ddir, nbp=3000, nmol=4, seed=0)
        assert callers._dataset_length(ddir) == 3000
        exe = _fake_hiddenfoot(tmp_path / "bin")
        work = tmp_path / "work"
        work.mkdir()
        calls = callers.run_hiddenfoot(ddir, work, binary=str(exe))
        assert calls.occ.shape == (4, 3000)

    def test_underflow_raises(self, quick_data, tmp_path, monkeypatch):
        _, ddir = quick_data
        exe = _fake_hiddenfoot(tmp_path / "bin")
        monkeypatch.setenv("FAKE_HF_LL", "-inf")
        with pytest.raises(RuntimeError, match="underflow"):
            callers.run_hiddenfoot(ddir, tmp_path, binary=str(exe))

    def test_window_too_large(self, quick_data, tmp_path):
        _, ddir = quick_data
        with pytest.raises(ValueError):
            callers.run_hiddenfoot(ddir, tmp_path, window=2500)

    def test_missing_binary(self, quick_data, tmp_path):
        _, ddir = quick_data
        with pytest.raises(FileNotFoundError):
            callers.run_hiddenfoot(ddir, tmp_path, binary="/no/such/hf")

    def test_binary_resolution(self, tmp_path, monkeypatch):
        arg = _fake_hiddenfoot(tmp_path / "arg")
        env = _fake_hiddenfoot(tmp_path / "env")
        path = _fake_hiddenfoot(tmp_path / "path")

        def same(found, exe):
            return os.path.normcase(found) == os.path.normcase(str(exe))

        monkeypatch.setenv("PATH", str(path.parent), prepend=os.pathsep)
        monkeypatch.setenv("HIDDENFOOT", str(env))
        assert same(callers._hiddenfoot_binary(str(arg)), arg)
        assert same(callers._hiddenfoot_binary(), env)
        monkeypatch.delenv("HIDDENFOOT")
        assert same(callers._hiddenfoot_binary(), path)
        monkeypatch.chdir(tmp_path)
        relative = os.path.join("arg", arg.name)
        assert same(callers._hiddenfoot_binary(relative), arg)

    @needs_hiddenfoot
    def test_real_binary(self, quick_data, tmp_path):
        _, ddir = quick_data
        truth = run_synth.read_truth(ddir)
        mccs = []
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            calls = callers.run_hiddenfoot(ddir, tmp_path / name)
            _check_calls(calls, ddir)
            mccs.append(run_synth.score_calls(calls, truth)["mcc"].mean())
        assert min(mccs) > 0.8
        assert abs(mccs[0] - mccs[1]) < 0.01
