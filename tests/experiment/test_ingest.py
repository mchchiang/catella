import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from catella.experiment._ingest import (
    _ensure_fasta_index,
    _GapTracker,
    _keep_regions,
    _load_refseqs,
    _read_chromsize,
    _resolve_modkit_schema,
    _whole_bases,
)


def _write(path, text):
    path.write_text(text)
    return path


_REQUIRED = ["mol_id", "upos", "chrom", "strand", "mod_qual", "mod_code"]


def _track(chunks, nmol=1, min_gap=20, nbp=1000):
    tracker = _GapTracker(nmol, min_gap)
    for idx, ref, read in chunks:
        tracker.update(
            np.array(idx, dtype=np.int64),
            np.array(ref, dtype=np.int64),
            np.array(read, dtype=np.int64),
        )
    gaps, offset, n_bad = tracker.result(nbp)
    return gaps.tolist(), offset.tolist(), n_bad


class TestWholeBases:
    def _rows(self, keys):
        return pd.DataFrame(
            {
                "mol_id": [k[0] for k in keys],
                "upos": [k[1] for k in keys],
                "row": range(len(keys)),
            }
        )

    @pytest.mark.parametrize("size", [1, 2, 3, 7])
    def test_no_base_split_and_order_kept(self, size):
        keys = [("r0", 1), ("r0", 1), ("r0", 2), ("r1", 2), ("r1", 2),
                ("r1", 2), ("r1", 5)]
        df = self._rows(keys)
        chunks = [df.iloc[i:i + size] for i in range(0, len(df), size)]
        out = list(_whole_bases(chunks))
        assert pd.concat(out)["row"].tolist() == list(range(len(keys)))
        seen = set()
        for chunk in out:
            bases = set(zip(chunk["mol_id"], chunk["upos"]))
            assert not bases & seen
            seen |= bases

    def test_empty_chunks_skipped(self):
        df = self._rows([("r0", 1), ("r0", 2)])
        out = list(_whole_bases([df.iloc[:0], df, df.iloc[:0]]))
        assert pd.concat(out)["row"].tolist() == [0, 1]


class TestGapTracker:
    def test_plus_read_deletion(self):
        gaps, offset, n_bad = _track([([0, 0, 0], [10, 14, 60], [5, 9, 15])])
        assert gaps == [[15, 60]]
        assert offset == [0, 1]
        assert n_bad == 0

    def test_minus_read_deletion(self):
        gaps, _, _ = _track([([0, 0, 0], [10, 14, 60], [90, 86, 80])])
        assert gaps == [[15, 60]]

    def test_below_min_gap_ignored(self):
        gaps, offset, _ = _track([([0, 0], [10, 40], [0, 11])])
        assert gaps == []
        assert offset == [0, 0]

    def test_exactly_min_gap_recorded(self):
        gaps, _, _ = _track([([0, 0], [10, 40], [0, 10])])
        assert gaps == [[11, 40]]

    def test_insertion_offsets_deletion(self):
        gaps, _, _ = _track([([0, 0], [10, 40], [0, 25])])
        assert gaps == []

    def test_gap_across_chunk_boundary(self):
        gaps, _, _ = _track(
            [([0, 0], [10, 14], [5, 9]), ([0, 0], [60, 64], [15, 19])]
        )
        assert gaps == [[15, 60]]

    def test_interleaved_molecules(self):
        gaps, offset, _ = _track(
            [([0, 1, 0, 1], [10, 100, 60, 104], [0, 0, 5, 4])], nmol=2
        )
        assert gaps == [[11, 60]]
        assert offset == [0, 1, 1]

    def test_same_position_rows_ignored(self):
        gaps, _, n_bad = _track([([0, 0, 0], [10, 10, 11], [3, 3, 4])])
        assert gaps == []
        assert n_bad == 0

    def test_unordered_molecule_gets_no_gaps(self):
        gaps, offset, n_bad = _track(
            [([0, 0, 0, 1, 1], [10, 60, 30, 5, 50], [0, 5, 3, 0, 4])],
            nmol=2,
        )
        assert n_bad == 1
        assert gaps == [[6, 50]]
        assert offset == [0, 0, 1]

    def test_direction_change_across_chunks_is_unordered(self):
        _, _, n_bad = _track([([0, 0], [10, 20], [0, 10]), ([0], [5], [15])])
        assert n_bad == 1

    def test_gaps_sorted_per_molecule(self):
        gaps, offset, _ = _track(
            [([0, 0, 0], [10, 60, 200], [0, 5, 10])], min_gap=20
        )
        assert gaps == [[11, 60], [61, 200]]
        assert offset == [0, 2]

    def test_clipped_to_nbp(self):
        gaps, _, _ = _track([([0, 0], [10, 90], [0, 5])], nbp=50)
        assert gaps == [[11, 50]]

    def test_no_rows(self):
        gaps, offset, n_bad = _track([], nmol=3)
        assert gaps == []
        assert offset == [0, 0, 0, 0]
        assert n_bad == 0


class TestResolveModkitSchema:
    def test_header_with_alignment_columns(self, tmp_path):
        f = _write(
            tmp_path / "d.tsv",
            "read_id\tref_position\tchrom\tref_strand\tmod_qual\t"
            "mod_code\talignment_start\talignment_end\n",
        )
        header, name_for = _resolve_modkit_schema(f, "\t", None)
        assert header is None
        assert name_for["align_start"] == "alignment_start"
        assert name_for["align_end"] == "alignment_end"

    def test_header_without_alignment_columns(self, tmp_path):
        f = _write(
            tmp_path / "d.tsv",
            "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n",
        )
        _, name_for = _resolve_modkit_schema(f, "\t", None)
        assert sorted(name_for) == sorted(_REQUIRED)

    def test_colidx_six_entries(self, tmp_path):
        f = _write(tmp_path / "d.tsv", "a\t1\tc\t+\t0.5\ta\t0\t9\n")
        header, name_for = _resolve_modkit_schema(f, "\t", [0, 1, 2, 3, 4, 5])
        assert sorted(name_for) == sorted(_REQUIRED)
        assert header[:6] == _REQUIRED

    def test_colidx_eight_entries(self, tmp_path):
        f = _write(tmp_path / "d.tsv", "a\t1\tc\t+\t0.5\ta\t0\t9\n")
        header, name_for = _resolve_modkit_schema(
            f, "\t", [0, 1, 2, 3, 4, 5, 6, 7]
        )
        assert sorted(name_for) == sorted(
            _REQUIRED + ["align_start", "align_end"]
        )
        assert header[6:] == ["align_start", "align_end"]

    def test_colidx_wrong_length_raises(self, tmp_path):
        f = _write(tmp_path / "d.tsv", "a\t1\tc\t+\t0.5\ta\t0\t9\n")
        with pytest.raises(ValueError, match="6 or 8 entries, got 7"):
            _resolve_modkit_schema(f, "\t", [0, 1, 2, 3, 4, 5, 6])


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

    @pytest.mark.parametrize("header", ["chrom\tlength\n", ""])
    def test_numeric_names_are_text(self, tmp_path, header):
        f = _write(tmp_path / "s.tsv", f"{header}601\t100\n1\t51\n")
        full, _ = _read_chromsize(f, False, ["601"])
        assert full == {"601": 100}

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

    def test_index_installed_first_by_other_process_kept(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n")
        fai = tmp_path / "r.fa.fai"

        def other_process_wins(src, dst):
            _write(fai, "other\n")
            raise FileExistsError(dst)

        with (
            patch(
                "catella.experiment._ingest.os.rename",
                side_effect=other_process_wins,
            ),
            patch("catella.experiment._ingest.os.replace") as replace,
        ):
            _ensure_fasta_index(f)

        replace.assert_not_called()
        assert fai.read_text() == "other\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "r.fa",
            "r.fa.fai",
        ]

    def test_missing_index_never_replaced(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n")
        with patch("catella.experiment._ingest.os.replace") as replace:
            _ensure_fasta_index(f)
        replace.assert_not_called()
        assert (tmp_path / "r.fa.fai").read_text().split("\t")[:2] == [
            "c1",
            "4",
        ]

    def _stale_index(self, tmp_path):
        f = _write(tmp_path / "r.fa", ">c1\nACGT\n")
        fai = _write(tmp_path / "r.fa.fai", "stale\n")
        mtime = os.path.getmtime(f)
        os.utime(fai, (mtime - 10, mtime - 10))
        return f, fai

    def test_refused_replace_keeps_refreshed_index(self, tmp_path):
        f, fai = self._stale_index(tmp_path)

        def other_process_refreshes(src, dst):
            _write(fai, "other\n")
            raise PermissionError("Access is denied")

        with patch(
            "catella.experiment._ingest.os.replace",
            side_effect=other_process_refreshes,
        ):
            _ensure_fasta_index(f)

        assert fai.read_text() == "other\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "r.fa",
            "r.fa.fai",
        ]

    def test_refused_replace_of_stale_index_raises(self, tmp_path):
        f, fai = self._stale_index(tmp_path)

        with (
            patch(
                "catella.experiment._ingest.os.replace",
                side_effect=PermissionError("Access is denied"),
            ),
            pytest.raises(PermissionError),
        ):
            _ensure_fasta_index(f)

        assert fai.read_text() == "stale\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "r.fa",
            "r.fa.fai",
        ]


class TestKeepRegions:
    def test_keeps_regions_in_every_file(self, recwarn):
        keep = _keep_regions([{"a", "b"}, {"a", "b", "c"}], ["a", "b"])
        assert keep == {"a", "b"}
        assert not recwarn

    def test_drops_with_warning(self):
        with pytest.warns(UserWarning, match="dropped 1 region.*'b'"):
            keep = _keep_regions([{"a", "b"}, {"a"}], ["a", "b"])
        assert keep == {"a"}

    def test_long_list_truncated(self):
        requested = ["k", *(f"r{i}" for i in range(7))]
        with pytest.warns(UserWarning, match=r"'r4', \.\.\. \(\+2 more\)"):
            _keep_regions([{"k"}], requested)

    def test_none_kept_raises(self):
        with pytest.raises(ValueError, match="no region has molecules"):
            _keep_regions([{"a"}, {"b"}], ["a", "b"])
