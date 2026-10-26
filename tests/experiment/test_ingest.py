import pytest
from catella.experiment._ingest import _load_refseqs, _read_chromsize


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
