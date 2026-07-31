# test_engine.py

from pathlib import Path

import numpy as np
import pytest
from nucmc_cpp import Dump

from nucmc.simulation.config import SimSettings
from nucmc.simulation.engine import SimManager, SimRun


def _make_settings(**overrides):
    defaults = dict(nucbp=147, llink=20, mu=-1.0, start_temp=1.0,
                     end_temp=0.1, cool_option="linear", nsweep=5,
                     print_freq=1, emax=5.0)
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_meth_prob(nmol=3, nbp=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((nmol, nbp))


def _make_sim_run(out_file, **overrides):
    seq_prob = np.asarray(1.0 - _make_meth_prob(nmol=1)[0],
                          dtype=np.float64).tobytes()
    defaults = dict(chrom="chr1", mol=0, run=0, seed=1, seq_prob=seq_prob,
                    out_type=Dump.OutputType.All, out_file=out_file,
                    settings=_make_settings(), override={})
    defaults.update(overrides)
    return SimRun(**defaults)


class TestSimManagerInit:
    def test_invalid_nworker_raises(self):
        with pytest.raises(ValueError):
            SimManager(nworker=0)

    def test_default_construction(self):
        manager = SimManager()
        assert manager.nworker == 1
        assert manager.verbose is True

    def test_no_longer_accepts_mp_context(self):
        with pytest.raises(TypeError):
            SimManager(mp_context="spawn")


class TestRunJob:
    def test_success_returns_true(self, tmp_path):
        p = _make_sim_run(tmp_path / "out.h5")
        chrom, mol, run, success, err_msg = SimManager._run_job(p)
        assert (chrom, mol, run) == ("chr1", 0, 0)
        assert success is True
        assert err_msg is None

    def test_failure_returns_false_with_message(self, tmp_path):
        # An embedded null byte in the path makes mkdir fail, so the
        # job should report failure instead of raising.
        bad_path = Path(str(tmp_path) + "/bad\x00dir/out.h5")
        p = _make_sim_run(bad_path)
        chrom, mol, run, success, err_msg = SimManager._run_job(p)
        assert (chrom, mol, run) == ("chr1", 0, 0)
        assert success is False
        assert err_msg


class TestSimManagerRun:
    def test_serial_run_produces_dataset(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=3, nbp=200)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "serial",
                              seed=1)
        assert dataset.nmol["chr1"] == 3

    def test_parallel_run_produces_same_shape_dataset(self, tmp_path):
        # Regression test: parallel dispatch previously hardcoded a
        # 'fork' multiprocessing context, which raised NameError on
        # macOS (missing 'os' import) and duplicated the parent
        # process's memory into every worker. It now always uses
        # 'spawn' via the stdlib multiprocessing module.
        meth_prob = _make_meth_prob(nmol=3, nbp=200)
        manager = SimManager(nworker=2, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob,
                              out_dir=tmp_path / "parallel", seed=1)
        assert dataset.nmol["chr1"] == 3

    def test_failed_job_does_not_abort_the_batch(self, tmp_path, monkeypatch,
                                                 capsys):
        # Regression test: `if not result:` on the always-truthy result
        # tuple used to make failed jobs silently disappear. A failure
        # should now (a) be reported to the console and (b) not stop the
        # rest of the batch from completing. `_run_job` is monkeypatched
        # (rather than relying on a genuine backend failure) so this test
        # exercises the batch's error-handling regardless of what the C++
        # backend happens to reject.
        calls = []

        def fake_run_job(p):
            calls.append(p)
            if len(calls) == 1:
                return (p.chrom, p.mol, p.run, False, "synthetic failure")
            return (p.chrom, p.mol, p.run, True, None)

        monkeypatch.setattr(SimManager, "_run_job", staticmethod(fake_run_job))

        meth_prob = _make_meth_prob(nmol=2, nbp=200)
        manager = SimManager(nworker=1, verbose=True)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "fail",
                              seed=1)

        assert len(calls) == 2
        assert dataset.nmol["chr1"] == 2
        captured = capsys.readouterr()
        assert "Simulation failed" in captured.out

    def test_broken_pool_is_reported_and_reraised(self, tmp_path, capsys):
        # Regression test for the sliding-window dispatch: if a future's
        # result() raises for a reason _run_job's own try/except can't
        # catch (e.g. the worker process itself died), that must be
        # reported distinctly and re-raised rather than silently hanging
        # or being swallowed. A genuine worker crash is hard to trigger
        # portably, so an unpicklable settings field is used instead: it
        # makes ProcessPoolExecutor.submit()'s payload fail to serialize,
        # which surfaces via future.result() the same way a broken pool
        # would, without needing to actually kill a subprocess.
        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle this")

        settings = _make_settings(mu=Unpicklable())
        meth_prob = _make_meth_prob(nmol=2, nbp=100)
        manager = SimManager(nworker=2, verbose=False)

        with pytest.raises(TypeError, match="cannot pickle"):
            manager.run(chroms="chr1", nsim=1, settings=settings,
                       meth_prob=meth_prob, out_dir=tmp_path / "broken",
                       seed=1, store_eseq=False)

        captured = capsys.readouterr()
        assert "Worker pool crashed" in captured.out
