import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import pytest
from catella.experiment._ingest import (
    _ensure_fasta_index,
    _load_refseqs,
    _read_chromsize,
)


def _write(path, text):
    path.write_text(text)
    return path


class TestReadChromsize:
    def test_with_header(self, tmp_path):
        f = _write(tmp_path / "s.tsv", "chrom\tlength\nc1\t100\nc2\t51\n")
        full, sizes = _read_chromsize(f, False, None)
        assert full == {"c1": 100, "c2": 51}
        assert sizes == full

    def test_without_header(self, tmp_path):
        f = _write(tmp_path / "s.tsv", "c1\t100\nc2\t51\n")
        full, _ = _read_chromsize(f, False, None)
        assert full == {"c1": 100, "c2": 51}

    def test_wrap_halves_storage_length_only(self, tmp_path):
        f = _write(tmp_path / "s.tsv", "c1\t100\nc2\t51\n")
        full, sizes = _read_chromsize(f, True, None)
        assert full == {"c1": 100, "c2": 51}
        assert sizes == {"c1": 50, "c2": 25}

    def test_chroms_restricts(self, tmp_path):
        f = _write(tmp_path / "s.tsv", "c1\t100\nc2\t51\n")
        full, sizes = _read_chromsize(f, False, ["c2"])
        assert full == {"c2": 51}
        assert sizes == {"c2": 51}

    def test_unknown_chrom_raises(self, tmp_path):
        f = _write(tmp_path / "s.tsv", "c1\t100\n")
        with pytest.raises(ValueError, match="not found in chromsize"):
            _read_chromsize(f, False, ["c9"])

    def test_duplicate_chrom_raises(self, tmp_path):
        f = _write(tmp_path / "s.tsv", "c1\t100\nc1\t50\n")
        with pytest.raises(ValueError, match="unique"):
            _read_chromsize(f, False, None)


class TestLoadRefseqs:
    def test_none_returns_empty(self):
        assert _load_refseqs(None, {"c1": 4}) == {}

    def test_reads_only_wanted_chroms(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n>c2\nGG\n")
        assert _load_refseqs(f, {"c1": 4}) == {"c1": "ACGT"}

    def test_length_mismatch_raises(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACG\n")
        with pytest.raises(ValueError, match="does not match"):
            _load_refseqs(f, {"c1": 4})

    def test_concurrent_readers(self, tmp_path):
        ctx = multiprocessing.get_context("spawn")
        for i in range(3):
            f = _write(tmp_path / f"r{i}.fa", ">chr1\n" + "ACGT" * 2000 + "\n")
            with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as pool:
                futures = [
                    pool.submit(_load_refseqs, f, {"chr1": 8000})
                    for _ in range(16)
                ]
                results = [fut.result() for fut in futures]
            assert all(r == {"chr1": "ACGT" * 2000} for r in results)


class TestEnsureFastaIndex:
    def test_builds_missing_index(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n>c2\nGG\n")
        _ensure_fasta_index(f)
        lines = (tmp_path / "r.fa.fai").read_text().splitlines()
        assert [line.split("\t")[:2] for line in lines] == [
            ["c1", "4"],
            ["c2", "2"],
        ]
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "r.fa",
            "r.fa.fai",
        ]

    def test_rebuilds_stale_index(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n")
        fai = _write(tmp_path / "r.fa.fai", "stale\n")
        mtime = os.path.getmtime(f)
        os.utime(fai, (mtime - 10, mtime - 10))
        _ensure_fasta_index(f)
        assert fai.read_text().split("\t")[:2] == ["c1", "4"]

    def test_keeps_current_index(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n")
        fai = _write(tmp_path / "r.fa.fai", "current\n")
        mtime = os.path.getmtime(f)
        os.utime(fai, (mtime + 10, mtime + 10))
        _ensure_fasta_index(f)
        assert fai.read_text() == "current\n"
