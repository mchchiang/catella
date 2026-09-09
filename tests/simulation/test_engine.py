# test_engine.py

from pathlib import Path

import h5py
import numpy as np
import pytest
from catella_cpp import Dump

from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager, SimRun
from catella.simulation.results import SimDataset


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
    seq_energy = np.asarray(_make_meth_prob(nmol=1)[0],
                            dtype=np.float64).tobytes()
    defaults = dict(chrom="chr1", mol=0, run=0, seed=1,
                    seq_energy=seq_energy,
                    out_type=Dump.OutputType.All, out_file=out_file,
                    settings=_make_settings())
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

    def test_multiple_out_types_combine_correctly(self, tmp_path):
        # Regression test: combining more than one out_type previously
        # always raised ValueError (the validation checked the resolved
        # enum value against a dict of string keys, which never matches),
        # and even past that, Dump.OutputType is a scoped C++ enum class
        # that pybind11 doesn't bind '|' for directly.
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "multi",
                              seed=1, out_types=["energy", "position"])
        sim_file = dataset.sim_file("chr1", 0, 0)
        with h5py.File(sim_file, "r") as f:
            keys = set(f["data"].keys())
        assert "energy" in keys
        assert {"position_flat", "position_offset"} <= keys

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

    def test_persists_out_type_seed_and_seed_table(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=2, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "persist",
                              seed=7)

        assert dataset.out_type == Dump.OutputType.All
        assert dataset.seed == 7
        assert dataset.seed_table["chr1"].shape == (2, 2)

        sim_file = dataset.sim_file("chr1", 1, 0)
        with h5py.File(sim_file, "r") as f:
            recorded_seed = int(f["params"].attrs["seed"])
        assert int(dataset.seed_table["chr1"][1, 0]) == recorded_seed

    def test_median_eseq_mu_reflected_in_persisted_settings(self, tmp_path):
        # Regression test: use_median_eseq_mu used to adjust mu only via a
        # per-job SimRun.override dict, so dataset.settings.mu silently
        # kept showing the pre-adjustment value. It's now applied via
        # dataclasses.replace() before the dataset is created, so the
        # persisted settings reflect the mu actually used.
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob,
                              out_dir=tmp_path / "medianmu", seed=1,
                              use_median_eseq_mu=True)
        assert dataset.settings["mu"] == np.median(dataset.eseq["chr1"])

    def test_same_seed_gives_same_seed_table_regardless_of_mols(self, tmp_path):
        # The seed table is order-independent: a triplet's seed doesn't
        # depend on which other molecules were selected via 'mols'.
        meth_prob = _make_meth_prob(nmol=3, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        full = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                           meth_prob=meth_prob, out_dir=tmp_path / "full",
                           seed=3)
        subset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                             meth_prob=meth_prob, out_dir=tmp_path / "subset",
                             seed=3, mols=slice(2, 3))
        assert (full.seed_table["chr1"] == subset.seed_table["chr1"]).all()

    def test_broken_pool_is_reported_and_reraised(self, tmp_path, capsys):
        # Regression test for the sliding-window dispatch: if a future's
        # result() raises for a reason _run_job's own try/except can't
        # catch (e.g. the worker process itself died), that must be
        # reported distinctly (including the status of every run that was
        # still in flight) and re-raised rather than silently hanging or
        # being swallowed. A genuine worker crash is hard to trigger
        # portably, so an unpicklable settings field is used instead: it
        # makes ProcessPoolExecutor.submit()'s payload fail to serialize,
        # which surfaces via future.result() the same way a broken pool
        # would, without needing to actually kill a subprocess. Exercised
        # directly against _dispatch() (rather than the full run()) so it
        # doesn't also need a dataset.save()-writable settings object.
        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle this")

        dataset = SimDataset.create(
            chroms=["chr1"], nmol={"chr1": 1}, nsim=1, nbp={"chr1": 10},
            settings=_make_settings(), out_dir=tmp_path / "broken",
            out_type=Dump.OutputType.All)

        bad_run = _make_sim_run(dataset.sim_file("chr1", 0, 0),
                                settings=Unpicklable())
        manager = SimManager(nworker=2, verbose=False)

        with pytest.raises(TypeError, match="cannot pickle"):
            manager._dispatch(dataset, iter([bad_run]), 1)

        captured = capsys.readouterr()
        assert "Worker pool crashed" in captured.out
        assert "chr1" in captured.out
        assert "missing" in captured.out


class TestSimManagerRerun:
    def test_only_none_fixes_missing_run(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "rr1",
                              seed=5)

        target_file = dataset.sim_file("chr1", 1, 0)
        original_seed = int(dataset.seed_table["chr1"][1, 0])
        target_file.unlink()
        assert dataset.find_incomplete_runs()["missing"] == [("chr1", 1, 0)]

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        incomplete = dataset.find_incomplete_runs()
        assert incomplete == {"missing": [], "corrupted": [], "truncated": []}
        with h5py.File(target_file, "r") as f:
            assert int(f["params"].attrs["seed"]) == original_seed

    def test_dataset_accepted_positionally(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "rr_pos",
                              seed=5)
        dataset.sim_file("chr1", 1, 0).unlink()

        manager.rerun(dataset, meth_prob=meth_prob)

        incomplete = dataset.find_incomplete_runs()
        assert incomplete == {"missing": [], "corrupted": [], "truncated": []}

    def test_explicit_only_reruns_exactly_that_target(self, tmp_path,
                                                       monkeypatch):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "rr2",
                              seed=5)

        calls = []
        real_run_job = SimManager._run_job

        def spy_run_job(p):
            calls.append((p.chrom, p.mol, p.run))
            return real_run_job(p)

        monkeypatch.setattr(SimManager, "_run_job", staticmethod(spy_run_job))

        manager.rerun(dataset=dataset, meth_prob=meth_prob,
                     only=[("chr1", 0, 0)])
        assert calls == [("chr1", 0, 0)]

    def test_legacy_dataset_requires_explicit_out_type_and_seed(self,
                                                                 tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"], nmol={"chr1": 1}, nsim=1, nbp={"chr1": 50},
            settings=_make_settings(), out_dir=tmp_path / "legacy")
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)

        with pytest.raises(ValueError, match="out_type"):
            manager.rerun(dataset=dataset, meth_prob=meth_prob,
                         only=[("chr1", 0, 0)])

        with pytest.raises(ValueError, match="seed"):
            manager.rerun(dataset=dataset, meth_prob=meth_prob,
                         only=[("chr1", 0, 0)], out_type=Dump.OutputType.All)

    def test_explicit_seed_override_differs_from_reproduction(self,
                                                               tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1, settings=_make_settings(),
                              meth_prob=meth_prob, out_dir=tmp_path / "rr3",
                              seed=5)
        original_seed = int(dataset.seed_table["chr1"][0, 0])
        target_file = dataset.sim_file("chr1", 0, 0)

        manager.rerun(dataset=dataset, meth_prob=meth_prob,
                     only=[("chr1", 0, 0)], seed=999)

        with h5py.File(target_file, "r") as f:
            reran_seed = int(f["params"].attrs["seed"])
        assert reran_seed != original_seed

    def test_rerun_overwrites_corrupted_file(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(chroms="chr1", nsim=1,
                              settings=_make_settings(nsweep=10),
                              meth_prob=meth_prob, out_dir=tmp_path / "rr4",
                              seed=5)
        target_file = dataset.sim_file("chr1", 0, 0)
        target_file.write_bytes(b"not a valid hdf5 file")
        assert dataset.find_incomplete_runs()["corrupted"] == [("chr1", 0, 0)]

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        assert dataset.find_incomplete_runs() == \
            {"missing": [], "corrupted": [], "truncated": []}
