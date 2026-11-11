# test_io.py

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from catella.h5_array import H5Array
from catella.simulation._io import (
    SimFileMapper,
    _pack_jagged,
    _read_attrs,
    _read_entries,
    _read_mcmc_file,
    _read_metadata,
    _read_settings,
    _read_tm_file,
    _replace_group,
    _tm_sample_fields,
    _unpack_jagged,
    _write_entries,
    _write_mcmc_data,
    _write_mcmc_params,
    _write_metadata,
    _write_settings,
    _write_tm_samples,
)
from catella.simulation.config import SimSettings
from catella_cpp import Dump


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


def _metadata_kwargs(tmp_path, **overrides):
    kwargs = {
        "raw_dir": tmp_path / "raw",
        "nsim": 2,
        "method": "mcmc",
        "chroms": ["c1", "c2"],
        "nmol": {"c1": 3, "c2": 1},
        "nbp": {"c1": 40, "c2": 50},
        "settings": _make_settings(),
    }
    kwargs.update(overrides)
    return kwargs


class TestReadAttrs:
    def test_reads_named_attrs(self, tmp_path):
        with h5py.File(tmp_path / "f.h5", "w") as f:
            g = f.create_group("g")
            g.attrs["a"] = 1
            g.attrs["b"] = 2.5
            g.attrs["c"] = 3
            assert _read_attrs(g, ["a", "b"]) == {"a": 1, "b": 2.5}

    def test_missing_attr_raises(self, tmp_path):
        with (
            h5py.File(tmp_path / "f.h5", "w") as f,
            pytest.raises(KeyError),
        ):
            _read_attrs(f.create_group("g"), ["a"])


class TestJagged:
    def test_round_trip(self):
        arrays = [np.array([1, 5]), np.array([], dtype=int), np.array([7])]
        flat, offset = _pack_jagged(arrays)
        assert flat.dtype == np.int32
        assert offset.dtype == np.uint64
        np.testing.assert_array_equal(flat, [1, 5, 7])
        np.testing.assert_array_equal(offset, [0, 2, 2, 3])
        out = _unpack_jagged(flat, offset)
        assert len(out) == 3
        for got, expected in zip(out, arrays, strict=True):
            np.testing.assert_array_equal(got, expected)
            assert got.dtype == np.int32

    def test_empty(self):
        flat, offset = _pack_jagged([])
        assert flat.size == 0
        np.testing.assert_array_equal(offset, [0])
        assert _unpack_jagged(flat, offset) == ()

    def test_dtype(self):
        flat, _ = _pack_jagged([np.array([1.5])], dtype=np.float64)
        assert flat.dtype == np.float64


class TestReadMCMCFile:
    def _write(self, path, values):
        with h5py.File(path, "w") as f:
            _write_mcmc_params(
                f, nucbp=147, nbp=40, llink=20, elink=1.0, mu=-1.0, seed=9
            )
            _write_mcmc_data(f, values)

    def test_reads_params_and_observables(self, tmp_path):
        path = tmp_path / "run.h5"
        flat, offset = _pack_jagged([np.array([3]), np.array([1, 200])])
        self._write(
            path,
            {
                "time": np.array([0, 1]),
                "energy": np.array([-1.0, -2.0]),
                "temp": np.array([1.0, 0.5]),
                "position_flat": flat,
                "position_offset": offset,
            },
        )
        out = _read_mcmc_file(path)
        assert out["nucbp"] == 147
        assert out["seed"] == 9
        np.testing.assert_array_equal(out["time"], [0, 1])
        np.testing.assert_array_equal(out["temp"], [1.0, 0.5])
        assert len(out["position"]) == 2
        np.testing.assert_array_equal(out["position"][1], [1, 200])

    def test_missing_observable_is_nan(self, tmp_path):
        path = tmp_path / "run.h5"
        flat, offset = _pack_jagged([np.array([3])])
        self._write(
            path,
            {
                "time": np.array([0]),
                "position_flat": flat,
                "position_offset": offset,
            },
        )
        out = _read_mcmc_file(path)
        assert np.isnan(out["energy"])
        assert np.isnan(out["temp"])


class TestReadTMFile:
    def test_round_trip_and_sample_fields(self, tmp_path):
        path = tmp_path / "tm.h5"
        _write_tm_samples(
            path,
            nucbp=147,
            nbp=300,
            llink=10,
            elink=1.0,
            mu=-2.0,
            logz=12.5,
            positions=[np.array([0, 160]), np.array([5])],
            energies=[-3.0, -1.5],
            seeds=[11, 22],
        )
        arrays = _read_tm_file(path)
        assert arrays["params"]["llink"] == 10
        assert arrays["logz"] == pytest.approx(12.5)

        sample = _tm_sample_fields(arrays, 1)
        assert sample["nucbp"] == 147
        assert sample["mu"] == pytest.approx(-2.0)
        assert sample["seed"] == 22
        assert sample["energy"] == pytest.approx(-1.5)
        assert sample["logz"] == pytest.approx(12.5)
        np.testing.assert_array_equal(sample["position"], [5])


class TestSimFileMapper:
    def test_mcmc_file_layout(self, tmp_path):
        mapper = SimFileMapper(tmp_path, total_mols=250)
        path = mapper.params_to_mcmc_file("chr1", 123, 4)
        expected = (
            tmp_path / "chr1" / "mol_100-199" / "mol_123"
            / "chr1_mol_123_run_04.h5"
        )
        assert path == expected

    def test_tm_file_layout(self, tmp_path):
        mapper = SimFileMapper(tmp_path, total_mols=250)
        path = mapper.params_to_tm_file("chr1", 7)
        assert path == tmp_path / "chr1" / "mol_000-099" / "chr1_mol_007_tm.h5"

    def test_file_to_params_round_trip(self, tmp_path):
        mapper = SimFileMapper(tmp_path, total_mols=50)
        path = mapper.params_to_mcmc_file("chrX", 12, 3)
        assert mapper.file_to_params(path) == ("chrX", 12, 3)

    def test_file_to_params_bad_name_raises(self, tmp_path):
        mapper = SimFileMapper(tmp_path, total_mols=50)
        with pytest.raises(ValueError):
            mapper.file_to_params(tmp_path / "a" / "b" / "c" / "x.h5")


class TestSettings:
    def test_round_trip(self, tmp_path):
        settings = _make_settings(llink={"c1": 10, "c2": 15})
        with h5py.File(tmp_path / "f.h5", "w") as f:
            _write_settings(f.create_group("s"), settings)
            assert _read_settings(f["s"]) == settings

    def test_none_fields_skipped(self, tmp_path):
        settings = _make_settings(start_temp=None)
        with h5py.File(tmp_path / "f.h5", "w") as f:
            g = f.create_group("s")
            _write_settings(g, settings)
            assert "start_temp" not in g.attrs
            assert _read_settings(g) == settings


class TestMetadata:
    def test_round_trip(self, tmp_path):
        eseq = {"c1": np.ones((3, 40)), "c2": np.zeros((1, 50))}
        seed_table = {
            "c1": np.arange(6, dtype=np.uint32).reshape(3, 2),
            "c2": np.array([[7, 8]], dtype=np.uint32),
        }
        kwargs = _metadata_kwargs(
            tmp_path,
            eseq=eseq,
            out_type=Dump.OutputType.All,
            seed=42,
            seed_table=seed_table,
        )
        with h5py.File(tmp_path / "d.h5", "w") as f:
            _write_metadata(f, **kwargs)
        with h5py.File(tmp_path / "d.h5", "r") as f:
            meta = _read_metadata(f)
        assert meta["raw_dir"] == tmp_path / "raw"
        assert meta["nsim"] == 2
        assert meta["method"] == "mcmc"
        assert meta["chroms"] == ["c1", "c2"]
        assert meta["nmol"] == {"c1": 3, "c2": 1}
        assert meta["nbp"] == {"c1": 40, "c2": 50}
        assert meta["settings"] == kwargs["settings"]
        np.testing.assert_array_equal(meta["eseq"]["c1"], eseq["c1"])
        assert meta["out_type"] == Dump.OutputType.All
        assert meta["seed"] == 42
        np.testing.assert_array_equal(meta["seed_table"]["c2"], [[7, 8]])

    def test_optional_fields_default_to_none(self, tmp_path):
        with h5py.File(tmp_path / "d.h5", "w") as f:
            _write_metadata(f, **_metadata_kwargs(tmp_path))
        with h5py.File(tmp_path / "d.h5", "r") as f:
            meta = _read_metadata(f)
        for name in ("eseq", "out_type", "seed", "seed_table"):
            assert meta[name] is None

    def test_existing_metadata_only_updates_raw_dir_and_nsim(self, tmp_path):
        with h5py.File(tmp_path / "d.h5", "w") as f:
            _write_metadata(f, **_metadata_kwargs(tmp_path, nsim=0))
            _write_metadata(
                f,
                **_metadata_kwargs(
                    tmp_path,
                    raw_dir=tmp_path / "moved",
                    nsim=5,
                    method="tm",
                    seed=1,
                ),
            )
        with h5py.File(tmp_path / "d.h5", "r") as f:
            meta = _read_metadata(f)
        assert meta["raw_dir"] == tmp_path / "moved"
        assert meta["nsim"] == 5
        assert meta["method"] == "mcmc"
        assert meta["seed"] is None

    def test_legacy_raw_path_and_method(self, tmp_path):
        with h5py.File(tmp_path / "d.h5", "w") as f:
            _write_metadata(f, **_metadata_kwargs(tmp_path))
            gmeta = f["metadata"]
            del gmeta.attrs["raw_dir"]
            del gmeta.attrs["method"]
            gmeta.attrs["raw_path"] = str(tmp_path / "old")
        with h5py.File(tmp_path / "d.h5", "r") as f:
            meta = _read_metadata(f)
        assert meta["raw_dir"] == Path(tmp_path / "old")
        assert meta["method"] == "mcmc"


class TestReplaceGroup:
    def test_creates_empty_group(self, tmp_path):
        with h5py.File(tmp_path / "f.h5", "w") as f:
            g = f.create_group("a")
            g.create_dataset("x", data=[1])
            new = _replace_group(f, "a")
            assert len(new) == 0
            assert _replace_group(f, "b").name == "/b"


class TestEntries:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "d.h5"
        arr = H5Array.create((2, 3), dtype=np.float64, dir=tmp_path)
        arr.write_batch(0, 2, np.arange(6.0).reshape(2, 3))
        df = pd.DataFrame({"v": [1.0, 2.0]})
        with h5py.File(path, "w") as f:
            parent = f.create_group("analysis")
            _write_entries(parent.create_group("c1"), {"arr": arr, "df": df})
        with h5py.File(path, "r") as f:
            entries = _read_entries(f["analysis/c1"], path)
        assert set(entries) == {"arr", "df"}
        assert isinstance(entries["arr"], H5Array)
        np.testing.assert_array_equal(
            entries["arr"].to_numpy(), np.arange(6.0).reshape(2, 3)
        )
        pd.testing.assert_frame_equal(entries["df"], df)
