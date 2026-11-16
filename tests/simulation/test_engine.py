# test_engine.py

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import MCMCJob, SimManager, TMJob
from catella.simulation.results import SimDataset, _read_tm_sample
from catella_cpp import Dump, NucPosTM
from scipy.special import logit


def _make_settings(**overrides):
    defaults = {
        "nucbp": 147,
        "llink": 20,
        "elink": 1.0,
        "mu": -1.0,
        "start_temp": 1.0,
        "end_temp": 0.1,
        "cool_option": "linear",
        "nsweep": 5,
        "print_freq": 1,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_meth_prob(nmol=3, nbp=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((nmol, nbp))


def _make_sim_run(out_file, **overrides):
    seq_energy = np.asarray(
        _make_meth_prob(nmol=1)[0], dtype=np.float64
    ).tobytes()
    defaults = {
        "chrom": "chr1",
        "mol": 0,
        "run": 0,
        "seed": 1,
        "seq_energy": seq_energy,
        "out_type": Dump.OutputType.All,
        "out_file": out_file,
        "settings": _make_settings(),
    }
    defaults.update(overrides)
    return MCMCJob(**defaults)


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
        chrom, mol, run, success, err_msg = SimManager._run_mcmc_job(p)
        assert (chrom, mol, run) == ("chr1", 0, 0)
        assert success is True
        assert err_msg is None

    def test_failure_returns_false_with_message(self, tmp_path):
        # An embedded null byte in the path makes mkdir fail, so the
        # job should report failure instead of raising.
        bad_path = Path(str(tmp_path) + "/bad\x00dir/out.h5")
        p = _make_sim_run(bad_path)
        chrom, mol, run, success, err_msg = SimManager._run_mcmc_job(p)
        assert (chrom, mol, run) == ("chr1", 0, 0)
        assert success is False
        assert err_msg

    def test_unwritable_out_file_returns_false(self, tmp_path):
        out_dir = tmp_path / "out.h5"
        out_dir.mkdir()
        p = _make_sim_run(out_dir)
        _, _, _, success, err_msg = SimManager._run_mcmc_job(p)
        assert success is False
        assert err_msg
        assert out_dir.is_dir()

    def test_writes_selected_observables(self, tmp_path):
        p = _make_sim_run(tmp_path / "out.h5", out_type=Dump.OutputType.Energy)
        assert SimManager._run_mcmc_job(p)[3] is True
        with h5py.File(tmp_path / "out.h5", "r") as f:
            assert set(f["data"]) == {"time", "energy"}
            assert set(f["params"].attrs) == {
                "nucbp",
                "nbp",
                "llink",
                "elink",
                "mu",
                "seed",
            }


class TestPerChromLlink:
    CHROMS = ("a", "b")

    def _probs(self):
        return {
            "a": _make_meth_prob(nmol=2, nbp=400, seed=1),
            "b": _make_meth_prob(nmol=2, nbp=400, seed=2),
        }

    def _run(self, out_dir, llink, method, nsim=2):
        return SimManager(nworker=1, verbose=False).run(
            chroms=self.CHROMS,
            nsim=nsim,
            method=method,
            settings=_make_settings(nucbp=20, llink=llink),
            meth_prob=self._probs(),
            out_dir=out_dir,
            seed=1,
        )

    @staticmethod
    def _position(sample):
        pos = sample.position
        if isinstance(pos, tuple):
            pos = np.concatenate(pos)
        return np.asarray(pos)

    @pytest.mark.parametrize("method", ["mcmc", "tm"])
    def test_matches_uniform_runs(self, tmp_path, method):
        mixed = self._run(tmp_path / "mix", {"a": 0, "b": 30}, method)
        for chrom, llink in (("a", 0), ("b", 30)):
            uniform = self._run(tmp_path / f"u{llink}", llink, method)
            for mol in range(2):
                for run in range(2):
                    got = mixed.raw[chrom, mol, run]
                    want = uniform.raw[chrom, mol, run]
                    assert got.llink == llink
                    np.testing.assert_array_equal(
                        self._position(got), self._position(want)
                    )

    def test_tm_logz_uses_chromosome_value(self, tmp_path):
        mixed = self._run(tmp_path / "mix", {"a": 0, "b": 30}, "tm")
        for chrom, llink in (("a", 0), ("b", 30)):
            uniform = self._run(tmp_path / f"u{llink}", llink, "tm")
            assert mixed.raw[chrom, 0, 0].logz == pytest.approx(
                uniform.raw[chrom, 0, 0].logz
            )

    def test_occupancy_uses_chromosome_value(self, tmp_path):
        mixed = self._run(tmp_path / "mix", {"a": 0, "b": 30}, "tm", nsim=0)
        SimAnalysis().compute_occup(dataset=mixed, batch_size=2)
        for chrom, llink in (("a", 0), ("b", 30)):
            uniform = self._run(tmp_path / f"u{llink}", llink, "tm", nsim=0)
            SimAnalysis().compute_occup(dataset=uniform, batch_size=2)
            np.testing.assert_allclose(
                mixed.analysis[chrom]["occup"][:],
                uniform.analysis[chrom]["occup"][:],
            )
        other = self._run(tmp_path / "o", 0, "tm", nsim=0)
        SimAnalysis().compute_occup(dataset=other, batch_size=2)
        assert not np.allclose(
            mixed.analysis["b"]["occup"][:], other.analysis["b"]["occup"][:]
        )

    def test_values_change_the_result(self, tmp_path):
        a = self._run(tmp_path / "a", {"a": 0, "b": 30}, "tm")
        b = self._run(tmp_path / "b", {"a": 0, "b": 0}, "tm")
        assert a.raw["b", 0, 0].logz != pytest.approx(b.raw["b", 0, 0].logz)

    def test_extra_keys_are_allowed(self, tmp_path):
        dataset = self._run(
            tmp_path / "x", {"a": 0, "b": 30, "other": 5}, "tm", nsim=0
        )
        assert dataset.settings["llink"]["b"] == 30

    def test_mapping_is_saved_and_loaded(self, tmp_path):
        dataset = self._run(tmp_path / "s", {"a": 0, "b": 30}, "tm", nsim=0)
        back = SimDataset.load(tmp_path / "s" / "results.h5")
        assert dict(back.settings["llink"]) == {"a": 0, "b": 30}
        assert dict(dataset.settings["llink"]) == {"a": 0, "b": 30}

    def test_integer_is_saved_and_loaded(self, tmp_path):
        self._run(tmp_path / "s", 30, "tm", nsim=0)
        back = SimDataset.load(tmp_path / "s" / "results.h5")
        assert back.settings["llink"] == 30

    def test_mcmc_rerun_keeps_chromosome_value(self, tmp_path):
        manager = SimManager(nworker=1, verbose=False)
        dataset = self._run(tmp_path / "r", {"a": 0, "b": 30}, "mcmc", nsim=1)
        before = self._position(dataset.raw["b", 1, 0])
        dataset.mcmc_file("b", 1, 0).unlink()
        manager.rerun(dataset, meth_prob=self._probs())
        fresh = SimDataset.load(tmp_path / "r" / "results.h5")
        assert fresh.raw["b", 1, 0].llink == 30
        np.testing.assert_array_equal(
            self._position(fresh.raw["b", 1, 0]), before
        )

    def test_missing_chromosome_raises_before_work(self, tmp_path):
        with pytest.raises(KeyError, match="no value for chromosome 'b'"):
            self._run(tmp_path / "m", {"a": 0}, "tm")
        assert not (tmp_path / "m").exists()

    @pytest.mark.parametrize("value", [-1, 1.5, True, "5"])
    def test_invalid_value_raises_before_work(self, tmp_path, value):
        with pytest.raises(ValueError, match="llink must be a non-negative"):
            self._run(tmp_path / "v", {"a": 0, "b": value}, "tm")
        assert not (tmp_path / "v").exists()


class TestSimManagerRun:
    @pytest.mark.parametrize("method", ["mcmc", "tm"])
    def test_samples_at_zero_llink_read_back(self, tmp_path, method):
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            method=method,
            settings=_make_settings(llink=0),
            meth_prob=_make_meth_prob(nmol=2, nbp=400),
            out_dir=tmp_path / method,
            seed=1,
        )
        sample = dataset.raw["chr1", 0, 0]
        assert sample.llink == 0
        assert len(sample.position) > 0

    def test_serial_run_produces_dataset(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=3, nbp=200)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "serial",
            seed=1,
        )
        assert dataset.nmol["chr1"] == 3

    def test_multiple_out_types_combine_correctly(self, tmp_path):
        # Regression test: combining more than one out_type previously
        # always raised ValueError (the validation checked the resolved
        # enum value against a dict of string keys, which never matches),
        # and even past that, Dump.OutputType is a scoped C++ enum class
        # that pybind11 doesn't bind '|' for directly.
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "multi",
            seed=1,
            out_types=["energy", "position"],
        )
        sim_file = dataset.mcmc_file("chr1", 0, 0)
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
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "parallel",
            seed=1,
        )
        assert dataset.nmol["chr1"] == 3

    def test_failed_job_does_not_abort_the_batch(
        self, tmp_path, monkeypatch, capsys
    ):
        # Regression test: `if not result:` on the always-truthy result
        # tuple used to make failed jobs silently disappear. A failure
        # should now (a) be reported to the console and (b) not stop the
        # rest of the batch from completing. `_run_mcmc_job` is monkeypatched
        # (rather than relying on a genuine backend failure) so this test
        # exercises the batch's error-handling regardless of what the C++
        # backend happens to reject.
        calls = []

        def fake_run_mcmc_job(p):
            calls.append(p)
            if len(calls) == 1:
                return (p.chrom, p.mol, p.run, False, "synthetic failure")
            return (p.chrom, p.mol, p.run, True, None)

        monkeypatch.setattr(
            SimManager, "_run_mcmc_job", staticmethod(fake_run_mcmc_job)
        )

        meth_prob = _make_meth_prob(nmol=2, nbp=200)
        manager = SimManager(nworker=1, verbose=True)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "fail",
            seed=1,
        )

        assert len(calls) == 2
        assert dataset.nmol["chr1"] == 2
        captured = capsys.readouterr()
        assert "Simulation failed" in captured.out

    def test_persists_out_type_seed_and_seed_table(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "persist",
            seed=7,
        )

        assert dataset.out_type == Dump.OutputType.All
        assert dataset.seed == 7
        assert dataset.seed_table["chr1"].shape == (2, 2)

        sim_file = dataset.mcmc_file("chr1", 1, 0)
        with h5py.File(sim_file, "r") as f:
            recorded_seed = int(f["params"].attrs["seed"])
        assert int(dataset.seed_table["chr1"][1, 0]) == recorded_seed

    @pytest.mark.parametrize("method", ["mcmc", "tm"])
    def test_none_seed_is_drawn_and_persisted(self, tmp_path, method):
        settings = (
            _make_settings() if method == "mcmc" else _make_tm_settings()
        )
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            method=method,
            settings=settings,
            meth_prob=_make_meth_prob(nmol=2, nbp=30),
            out_dir=tmp_path / "noseed",
        )
        assert isinstance(dataset.seed, int)
        assert 0 <= dataset.seed < 2**63
        loaded = SimDataset.load(dataset._dataset_file)
        assert loaded.seed == dataset.seed

    def test_median_eseq_mu_reflected_in_persisted_settings(self, tmp_path):
        # Regression test: use_median_eseq_mu used to adjust mu only via a
        # per-job MCMCJob.override dict, so dataset.settings.mu silently
        # kept showing the pre-adjustment value. It's now applied via
        # dataclasses.replace() before the dataset is created, so the
        # persisted settings reflect the mu actually used.
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "medianmu",
            seed=1,
            use_median_eseq_mu=True,
        )
        assert dataset.settings["mu"] == np.median(dataset.eseq["chr1"])

    def test_median_eseq_mu_ignores_nan(self, tmp_path):
        # meth_prob commonly has trailing NaN columns (left-aligned
        # window edge from compute_model_prob/compute_empirical_prob)
        # or all-NaN dropout rows. Those must not turn mu (and hence
        # the whole simulation's energy) into NaN.
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        meth_prob[:, -5:] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "mediannan",
            seed=1,
            use_median_eseq_mu=True,
        )
        assert not np.isnan(dataset.settings["mu"])
        assert dataset.settings["mu"] == np.nanmedian(dataset.eseq["chr1"])

    def test_median_eseq_mu_without_store_eseq(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        stored = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "stored",
            seed=1,
            use_median_eseq_mu=True,
        )
        unstored = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "unstored",
            seed=1,
            store_eseq=False,
            use_median_eseq_mu=True,
        )
        assert unstored.settings["mu"] == stored.settings["mu"]
        assert not unstored.has_eseq

    def test_same_seed_gives_same_seed_table_regardless_of_mols(
        self, tmp_path
    ):
        # The seed table is order-independent: a triplet's seed doesn't
        # depend on which other molecules were selected via 'mols'.
        meth_prob = _make_meth_prob(nmol=3, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        full = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "full",
            seed=3,
        )
        subset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "subset",
            seed=3,
            mols=slice(2, 3),
        )
        assert (full.seed_table["chr1"] == subset.seed_table["chr1"]).all()

    def test_single_int_mols_runs_one_molecule(self, tmp_path):
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=_make_meth_prob(nmol=3, nbp=50),
            out_dir=tmp_path / "ds",
            seed=3,
            mols=1,
        )
        assert sorted(dataset.iter_runs()) == [("chr1", 1, 0), ("chr1", 1, 1)]

    def test_broken_pool_is_reported_and_reraised(self, tmp_path, capsys):
        # Regression test for the sliding-window dispatch: if a future's
        # result() raises for a reason _run_mcmc_job's own try/except can't
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
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 10},
            settings=_make_settings(),
            out_dir=tmp_path / "broken",
            out_type=Dump.OutputType.All,
        )

        bad_run = _make_sim_run(
            dataset.mcmc_file("chr1", 0, 0), settings=Unpicklable()
        )
        manager = SimManager(nworker=2, verbose=False)

        with pytest.raises(TypeError, match="cannot pickle"):
            manager._dispatch(
                dataset, iter([bad_run]), 1, SimManager._run_mcmc_job
            )

        captured = capsys.readouterr()
        assert "Worker pool crashed" in captured.out
        assert "chr1" in captured.out
        assert "missing" in captured.out


class TestSimManagerRerun:
    def test_only_none_fixes_missing_run(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr1",
            seed=5,
        )

        target_file = dataset.mcmc_file("chr1", 1, 0)
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
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr_pos",
            seed=5,
        )
        dataset.mcmc_file("chr1", 1, 0).unlink()

        manager.rerun(dataset, meth_prob=meth_prob)

        incomplete = dataset.find_incomplete_runs()
        assert incomplete == {"missing": [], "corrupted": [], "truncated": []}

    def test_explicit_only_reruns_exactly_that_target(
        self, tmp_path, monkeypatch
    ):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr2",
            seed=5,
        )

        calls = []
        real_run_mcmc_job = SimManager._run_mcmc_job

        def spy_run_mcmc_job(p):
            calls.append((p.chrom, p.mol, p.run))
            return real_run_mcmc_job(p)

        monkeypatch.setattr(
            SimManager, "_run_mcmc_job", staticmethod(spy_run_mcmc_job)
        )

        manager.rerun(
            dataset=dataset, meth_prob=meth_prob, only=[("chr1", 0, 0)]
        )
        assert calls == [("chr1", 0, 0)]

    def test_legacy_dataset_requires_explicit_out_type_and_seed(
        self, tmp_path
    ):
        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "legacy",
        )
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)

        with pytest.raises(ValueError, match="out_type"):
            manager.rerun(
                dataset=dataset, meth_prob=meth_prob, only=[("chr1", 0, 0)]
            )

        with pytest.raises(ValueError, match="seed"):
            manager.rerun(
                dataset=dataset,
                meth_prob=meth_prob,
                only=[("chr1", 0, 0)],
                out_type=Dump.OutputType.All,
            )

    def test_explicit_seed_override_differs_from_reproduction(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr3",
            seed=5,
        )
        original_seed = int(dataset.seed_table["chr1"][0, 0])
        target_file = dataset.mcmc_file("chr1", 0, 0)

        manager.rerun(
            dataset=dataset,
            meth_prob=meth_prob,
            only=[("chr1", 0, 0)],
            seed=999,
        )

        with h5py.File(target_file, "r") as f:
            reran_seed = int(f["params"].attrs["seed"])
        assert reran_seed != original_seed

    def test_rerun_overwrites_corrupted_file(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(nsweep=10),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr4",
            seed=5,
        )
        target_file = dataset.mcmc_file("chr1", 0, 0)
        target_file.write_bytes(b"not a valid hdf5 file")
        assert dataset.find_incomplete_runs()["corrupted"] == [("chr1", 0, 0)]

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }


def _make_tm_settings(**overrides):
    defaults = {
        "nucbp": 3,
        "llink": 2,
        "elink": 1.0,
        "mu": 0.5,
        "end_temp": 1.0,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_tm_job(out_file, seeds=(1, 2, 3), nbp=20, **overrides):
    seq_energy = np.asarray(
        logit(_make_meth_prob(nmol=1, nbp=nbp)[0]), dtype=np.float64
    ).tobytes()
    defaults = {
        "chrom": "chr1",
        "mol": 0,
        "seeds": seeds,
        "seq_energy": seq_energy,
        "out_file": out_file,
        "settings": _make_tm_settings(),
    }
    defaults.update(overrides)
    return TMJob(**defaults)


class TestRunTMJob:
    def test_success_writes_all_samples(self, tmp_path):
        out_file = tmp_path / "sub" / "mol.h5"
        p = _make_tm_job(out_file, seeds=(4, 5, 6))
        result = SimManager._run_tm_job(p)
        assert result == ("chr1", 0, None, True, None)
        for run, s in enumerate((4, 5, 6)):
            sample = _read_tm_sample(out_file, run)
            assert sample.seed == s
            assert sample.nucbp == 3
            assert sample.nbp == 20

    def test_samples_match_fresh_solve_per_seed(self, tmp_path):
        out_file = tmp_path / "mol.h5"
        p = _make_tm_job(out_file, seeds=(11, 12, 13))
        SimManager._run_tm_job(p)
        settings = p.settings
        eseq = np.frombuffer(p.seq_energy, dtype=np.float64).tolist()
        for run, s in enumerate(p.seeds):
            tm = NucPosTM(
                settings.nucbp,
                len(eseq),
                settings.llink,
                settings.elink,
                settings.mu,
                s,
            )
            tm.setEnergy(eseq, settings.emax)
            tm.solve(settings.end_temp)
            position, energy = tm.sampleConfig()
            sample = _read_tm_sample(out_file, run)
            np.testing.assert_array_equal(sample.position, position)
            assert sample.energy == energy
            assert sample.logz == tm.getLogZ()

    def test_failure_reports_error_and_writes_nothing(self, tmp_path):
        out_file = tmp_path / "mol.h5"
        p = _make_tm_job(out_file, settings=_make_tm_settings(end_temp=0.0))
        chrom, mol, run, success, err_msg = SimManager._run_tm_job(p)
        assert (chrom, mol, run) == ("chr1", 0, None)
        assert success is False
        assert err_msg
        assert not out_file.exists()


class TestSimManagerRunTM:
    def test_zero_nsim_creates_dataset_with_no_samples(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=0,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm0",
            seed=1,
        )
        assert dataset.method == "tm"
        assert dataset.nsim == 0
        with pytest.raises(ValueError):
            dataset.raw["chr1", 0, 0]
        assert not dataset.tm_file("chr1", 0).exists()

    def test_samples_use_seed_table(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=3,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_seeds",
            seed=1,
        )
        for mol in range(2):
            for run in range(3):
                sample = dataset.raw["chr1", mol, run]
                assert sample.seed == dataset.seed_table["chr1"][mol, run]

    def test_parallel_matches_serial(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=3, nbp=30)
        datasets = [
            SimManager(nworker=nworker, verbose=False).run(
                chroms="chr1",
                nsim=2,
                method="tm",
                settings=_make_tm_settings(),
                meth_prob=meth_prob,
                out_dir=tmp_path / f"tm_par{nworker}",
                seed=1,
            )
            for nworker in (1, 2)
        ]
        for mol in range(3):
            for run in range(2):
                a = datasets[0].raw["chr1", mol, run]
                b = datasets[1].raw["chr1", mol, run]
                np.testing.assert_array_equal(a.position, b.position)
                assert a.energy == b.energy

    def test_generates_and_consolidates_samples(self, tmp_path):
        from catella.simulation.results import SimTMData

        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=3,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm1",
            seed=1,
        )
        assert dataset.method == "tm"
        assert dataset.nsim == 3

        sample = dataset.raw["chr1", 0, 1]
        assert isinstance(sample, SimTMData)
        assert sample.nucbp == 3
        assert sample.nbp == 30

        # Temp per-run files should be cleaned up after consolidation.
        for run in range(3):
            assert not dataset.mcmc_file("chr1", 0, run).exists()
        assert dataset.tm_file("chr1", 0).exists()

        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }

    def test_reload_preserves_method_and_samples(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm2",
            seed=1,
        )
        loaded = SimDataset.load(dataset._dataset_file)
        assert loaded.method == "tm"
        assert loaded.nsim == 2
        assert loaded.raw["chr1", 0, 0].nucbp == 3


class TestRerunTM:
    def test_rejects_triplets_in_only(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_rerun1",
            seed=1,
        )
        with pytest.raises(ValueError):
            manager.rerun(
                dataset=dataset,
                meth_prob=meth_prob,
                only=[("chr1", 0, 0)],
            )

    def test_regenerates_missing_molecule(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_rerun2",
            seed=1,
        )
        dataset.tm_file("chr1", 1).unlink()
        assert dataset.find_incomplete_runs()["missing"] != []

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }
        assert dataset.raw["chr1", 1, 0] is not None

    def test_only_pairs_regenerate_specific_molecule(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_rerun3",
            seed=1,
        )
        old_energy = dataset.raw["chr1", 0, 0].energy
        dataset.tm_file("chr1", 0).unlink()

        manager.rerun(
            dataset=dataset, meth_prob=meth_prob, only=[("chr1", 0)]
        )

        # Reproduced with the same seed_table, so the exact sample
        # should match what was there before.
        assert dataset.raw["chr1", 0, 0].energy == pytest.approx(
            old_energy
        )


def _no_data_meth_prob(nmol=3, nbp=400, nucbp=147):
    meth_prob = _make_meth_prob(nmol=nmol, nbp=nbp)
    meth_prob[1] = np.nan
    meth_prob[:, nbp - nucbp + 1 :] = np.nan
    return meth_prob


def _run_no_data(tmp_path, method="mcmc", nsim=2, name="ds"):
    if method == "mcmc":
        settings = _make_settings()
        meth_prob = _no_data_meth_prob()
    else:
        settings = _make_tm_settings()
        meth_prob = _no_data_meth_prob(nbp=30, nucbp=3)
    dataset = SimManager(nworker=1, verbose=False).run(
        chroms="chr1",
        nsim=nsim,
        method=method,
        settings=settings,
        meth_prob=meth_prob,
        out_dir=tmp_path / name,
        seed=1,
    )
    return dataset, meth_prob


class TestNoData:
    def test_mcmc_skips_molecule(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path)
        np.testing.assert_array_equal(
            dataset.no_data["chr1"], [False, True, False]
        )
        assert not dataset.mcmc_file("chr1", 1, 0).exists()
        assert dataset.mcmc_file("chr1", 0, 0).exists()
        assert {mol for _, mol, _ in dataset.iter_runs()} == {0, 2}

    def test_find_incomplete_runs_ignores_molecule(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path)
        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }

    def test_mcmc_occupancy_is_nan(self, tmp_path, capsys):
        dataset, _ = _run_no_data(tmp_path)
        SimAnalysis().compute_occup(dataset=dataset)
        occup = np.asarray(dataset.analysis["chr1"]["occup"])
        assert np.isnan(occup[1]).all()
        assert not np.isnan(occup[[0, 2]]).any()
        assert "no simulation data" not in capsys.readouterr().out

    def test_tm_skips_molecule(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path, method="tm")
        assert dataset.no_data["chr1"][1]
        assert not dataset.tm_file("chr1", 1).exists()
        assert dataset.tm_file("chr1", 0).exists()
        assert dataset.find_incomplete_runs()["missing"] == []

    def test_tm_exact_stats_are_nan(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path, method="tm", nsim=0)
        ana = SimAnalysis()
        ana.compute_occup(dataset=dataset)
        ana.compute_mean_nnuc(dataset=dataset)
        occup = np.asarray(dataset.analysis["chr1"]["occup"])
        mean_n = dataset.analysis["chr1"]["mean_nnuc"].to_numpy().ravel()
        assert np.isnan(occup[1]).all()
        assert not np.isnan(occup[[0, 2]]).any()
        assert np.isnan(mean_n[1])
        assert not np.isnan(mean_n[[0, 2]]).any()

    def test_tm_config_is_empty_with_nan_energy(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path, method="tm", nsim=0)
        ana = SimAnalysis()
        ana.compute_best_match_config(dataset=dataset)
        energy = dataset.analysis["chr1"]["best_match_energy"]
        assert np.isnan(energy["energy"][1])
        assert np.isnan(energy["log_prob"][1])
        assert not np.isnan(energy["energy"][[0, 2]]).any()
        configs = ana.extract_config(dataset, "chr1", name="best_match")
        assert configs[1] is None
        assert configs[0] is not None

    def test_partial_nan_raises_before_creating_dataset(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=400)
        meth_prob[0, 10] = np.nan
        with pytest.raises(ValueError, match="nan at nucleosome start"):
            SimManager(nworker=1, verbose=False).run(
                chroms="chr1",
                nsim=1,
                settings=_make_settings(),
                meth_prob=meth_prob,
                out_dir=tmp_path / "ds",
                seed=1,
            )
        assert not (tmp_path / "ds").exists()

    def test_no_data_survives_reload(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path)
        loaded = SimDataset.load(dataset._dataset_file)
        np.testing.assert_array_equal(
            loaded.no_data["chr1"], [False, True, False]
        )

    def test_no_data_is_read_only(self, tmp_path):
        dataset, _ = _run_no_data(tmp_path)
        with pytest.raises(ValueError):
            dataset.no_data["chr1"][0] = True

    def test_rerun_fixes_missing_run(self, tmp_path):
        dataset, meth_prob = _run_no_data(tmp_path)
        dataset.mcmc_file("chr1", 2, 0).unlink()
        manager = SimManager(nworker=1, verbose=False)
        manager.rerun(dataset=dataset, meth_prob=meth_prob)
        assert dataset.mcmc_file("chr1", 2, 0).exists()
        assert not dataset.mcmc_file("chr1", 1, 0).exists()

    def test_rerun_with_different_no_data_raises(self, tmp_path):
        dataset, meth_prob = _run_no_data(tmp_path)
        other = meth_prob.copy()
        other[0, : 400 - 147 + 1] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        with pytest.raises(ValueError, match="molecules with no data differ"):
            manager.rerun(dataset=dataset, meth_prob=other)

    @pytest.mark.parametrize(
        ("method", "target"),
        [("mcmc", ("chr1", 1, 0)), ("tm", ("chr1", 1))],
    )
    def test_rerun_of_no_data_molecule_raises(self, tmp_path, method, target):
        dataset, meth_prob = _run_no_data(tmp_path, method=method)
        manager = SimManager(nworker=1, verbose=False)
        with pytest.raises(ValueError, match="has no data"):
            manager.rerun(dataset=dataset, meth_prob=meth_prob, only=[target])


def _partial_nan_meth_prob(nmol=3, nbp=400):
    meth_prob = _make_meth_prob(nmol=nmol, nbp=nbp)
    meth_prob[0, 10] = np.nan
    return meth_prob


class TestFillNan:
    def _run(self, tmp_path, meth_prob, fill_nan, name="ds"):
        return SimManager(nworker=1, verbose=False).run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / name,
            seed=1,
            fill_nan=fill_nan,
        )

    @pytest.mark.parametrize("fill_nan", [0.5, "mean"])
    def test_partial_molecule_is_simulated(self, tmp_path, fill_nan):
        dataset = self._run(tmp_path, _partial_nan_meth_prob(), fill_nan)
        assert dataset.mcmc_file("chr1", 0, 0).exists()
        assert not dataset.no_data["chr1"].any()

    def test_fill_sets_energy(self, tmp_path):
        meth_prob = _partial_nan_meth_prob()
        dataset = self._run(tmp_path, meth_prob, 0.5)
        assert np.isfinite(dataset.eseq["chr1"][0, 10])
        assert np.isnan(meth_prob[0, 10])

    @pytest.mark.parametrize("fill_nan", [0.5, "mean", None])
    def test_nan_fill_survives_reload(self, tmp_path, fill_nan):
        dataset = self._run(tmp_path, _make_meth_prob(), fill_nan)
        assert dataset.nan_fill == fill_nan
        loaded = SimDataset.load(dataset._dataset_file)
        assert loaded.nan_fill == fill_nan

    def test_invalid_fill_nan_raises(self, tmp_path):
        with pytest.raises(ValueError, match="fill_nan must be"):
            self._run(tmp_path, _make_meth_prob(), 2.0)

    def test_rerun_reuses_nan_fill(self, tmp_path):
        meth_prob = _partial_nan_meth_prob()
        dataset = self._run(tmp_path, meth_prob, 0.5)
        target = dataset.mcmc_file("chr1", 0, 0)
        with h5py.File(target, "r") as f:
            before = f["data"]["energy"][()]
        target.unlink()
        SimManager(nworker=1, verbose=False).rerun(
            dataset=dataset, meth_prob=meth_prob
        )
        with h5py.File(target, "r") as f:
            np.testing.assert_array_equal(f["data"]["energy"][()], before)


def _llink_exp(linker=10, nmol=50, nbp=200):
    energy = np.full((nmol, nbp), 5.0)
    energy[:, 0 : nbp - 4 : 5 + linker] = -20.0
    prob = 1.0 / (1.0 + np.exp(-energy))
    return SimpleNamespace(
        chroms=("chr1",), analysis={"chr1": {"meth_prob": prob}}
    )


def _llink_settings():
    return SimSettings(
        nucbp=5, llink=99, elink=25.0, mu=0.0, end_temp=1.0, emax=100.0
    )


class TestSimManagerEstimateLlink:
    def test_returns_value_per_chromosome(self):
        best = SimManager().estimate_llink(
            _llink_exp(), _llink_settings(), llinks=[0, 10, 30]
        )
        assert best == {"chr1": 10}

    def test_parallel_matches_serial(self):
        exp = _llink_exp()
        serial = SimManager().estimate_llink(
            exp, _llink_settings(), llinks=[0, 10, 30]
        )
        parallel = SimManager(nworker=2).estimate_llink(
            exp, _llink_settings(), llinks=[0, 10, 30]
        )
        assert parallel == serial

    def test_does_not_modify_experiment(self):
        exp = _llink_exp()
        SimManager().estimate_llink(
            exp, _llink_settings(), llinks=[0, 10, 30]
        )
        assert set(exp.analysis["chr1"]) == {"meth_prob"}

    def test_llink_evid_name_stores_table(self):
        exp = _llink_exp()
        best = SimManager().estimate_llink(
            exp,
            _llink_settings(),
            llinks=[0, 10, 30],
            llink_evid_name="llink_evid",
        )
        table = exp.analysis["chr1"]["llink_evid"]
        assert table["llink"].tolist() == [0, 10, 30]
        top = table.loc[table["evidence"].idxmax(), "llink"]
        assert best == {"chr1": int(top)}

    def test_empty_llinks_raises(self):
        with pytest.raises(ValueError, match="llinks must not be empty"):
            SimManager().estimate_llink(
                _llink_exp(), _llink_settings(), llinks=[]
            )


def _prob_exp(arrays):
    return SimpleNamespace(
        chroms=tuple(arrays),
        analysis={c: {"meth_prob": a} for c, a in arrays.items()},
    )


class TestSimManagerEstimateStartTemp:
    def test_constant_prob_gives_its_log_odds(self):
        exp = _prob_exp({"chr1": np.full((20, 30), 0.9)})
        got = SimManager().estimate_start_temp(exp, seed=0)
        assert got == pytest.approx(abs(logit(0.9)))

    def test_same_seed_gives_same_value(self):
        rng = np.random.default_rng(0)
        exp = _prob_exp({"chr1": rng.uniform(0.01, 0.99, (500, 40))})
        kwargs = {"seed": 7, "percentile_sample_size": 50}
        a = SimManager().estimate_start_temp(exp, **kwargs)
        b = SimManager().estimate_start_temp(exp, **kwargs)
        assert a == b

    def test_percentile_orders_values(self):
        rng = np.random.default_rng(1)
        exp = _prob_exp({"chr1": rng.uniform(0.01, 0.99, (50, 40))})
        low = SimManager().estimate_start_temp(exp, percentile=10.0)
        high = SimManager().estimate_start_temp(exp, percentile=90.0)
        assert low < high

    def test_chroms_selects_chromosomes(self):
        exp = _prob_exp(
            {
                "chr1": np.full((10, 20), 0.9),
                "chr2": np.full((10, 20), 0.6),
            }
        )
        got = SimManager().estimate_start_temp(exp, chroms="chr2")
        assert got == pytest.approx(abs(logit(0.6)))

    def test_saturated_prob_is_finite(self):
        exp = _prob_exp({"chr1": np.ones((5, 10))})
        got = SimManager().estimate_start_temp(exp)
        assert np.isfinite(got)
