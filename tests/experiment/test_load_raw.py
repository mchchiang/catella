# test_load_raw.py

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from catella.experiment.methdata import MethPrintData, MethPrintExperiment


_HEADER = "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"


def _write_tsv(path, rows, header=_HEADER):
    with open(path, "w") as f:
        if header:
            f.write(header)
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        for chrom, length in sizes.items():
            f.write(f"{chrom}\t{length}\n")


def _write_fasta(path, records):
    with open(path, "w") as f:
        for chrom, seq in records.items():
            f.write(f">{chrom}\n{seq}\n")


def _make_rows(chrom, mol_ids, positions, seed=0, strand="+"):
    rng = np.random.default_rng(seed)
    rows = []
    for mol in mol_ids:
        for pos in positions:
            rows.append((mol, pos, chrom, strand,
                        round(float(rng.random()), 4), "a"))
    return rows


def _legacy_reference(test_rows, chromsize, wrap=False, max_nmol=None,
                      seed=None):
    # Standalone re-implementation of the pre-streaming algorithm, used
    # only to verify the streaming version reproduces it exactly.
    df = pd.DataFrame(test_rows, columns=["mol_id", "upos", "chrom",
                                          "strand", "mod_qual", "mod_code"])
    df_size = pd.DataFrame(list(chromsize.items()),
                           columns=["chrom", "length"])
    rng = None if max_nmol is None else np.random.default_rng(seed)
    if max_nmol is not None:
        mol_map = df[["chrom", "mol_id"]].drop_duplicates()
        sampled = []
        for chrom, group in mol_map.groupby("chrom"):
            mols = group["mol_id"].values
            if len(mols) > max_nmol:
                mols = rng.choice(mols, max_nmol, replace=False)
            sampled.append(mols)
        sampled = np.concatenate(sampled)
        df = df.set_index("mol_id").loc[sampled].reset_index()
    df = pd.merge(df, df_size, on="chrom")
    if wrap:
        df["pos"] = np.where(df["upos"] > (df["length"]-1)/2,
                             df["length"]-df["upos"]-1, df["upos"])
    else:
        df["pos"] = df["upos"]
    df = df.sort_values(["chrom", "mol_id", "pos"]).reset_index(drop=True)
    df["mol_index"] = df.groupby("chrom")["mol_id"].transform(
        lambda x: pd.factorize(x)[0])
    mol_ids = {c: g["mol_id"].unique() for c, g in df.groupby("chrom")}
    keep = ["chrom", "mol_index", "pos", "strand", "mod_qual", "mod_code"]
    dfs = {c: g[keep].drop(columns="chrom")
          for c, g in df.groupby("chrom")}
    return mol_ids, dfs


class TestBasicIngestion:
    def test_multi_chromosome(self, tmp_path):
        rows = (_make_rows("chr1", ["m0", "m1", "m2"], [10, 20, 30], seed=0)
               + _make_rows("chr2", ["m0", "m1"], [5, 15], seed=1))
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 50})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, chunk_size=4)
        assert set(exp.chroms) == {"chr1", "chr2"}

        raw1 = exp.raw["chr1"]
        assert raw1.nbp == 100
        assert sorted(raw1.test_mol_id) == ["m0", "m1", "m2"]
        assert set(raw1.test_data["pos"]) == {10, 20, 30}

        raw2 = exp.raw["chr2"]
        assert raw2.nbp == 50
        assert sorted(raw2.test_mol_id) == ["m0", "m1"]
        exp.close()

    def test_chunk_size_does_not_change_result(self, tmp_path):
        rows = _make_rows("chr1", [f"m{i}" for i in range(12)],
                          [1, 2, 3, 4], seed=3)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp_a = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, chunk_size=10_000)
        exp_b = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, chunk_size=3)

        a = exp_a.raw["chr1"].test_data.sort_values(
            ["mol_index", "pos"]).reset_index(drop=True)
        b = exp_b.raw["chr1"].test_data.sort_values(
            ["mol_index", "pos"]).reset_index(drop=True)
        np.testing.assert_array_equal(a["mol_index"].to_numpy(),
                                      b["mol_index"].to_numpy())
        np.testing.assert_array_equal(a["pos"].to_numpy(),
                                      b["pos"].to_numpy())
        np.testing.assert_array_equal(a["mod_qual"].to_numpy(),
                                      b["mod_qual"].to_numpy())
        exp_a.close()
        exp_b.close()


class TestDownsamplingAndWrap:
    def test_max_nmol_matches_legacy_algorithm(self, tmp_path):
        rows = (_make_rows("chr1", [f"m{i}" for i in range(20)],
                           [1, 2, 3], seed=5)
               + _make_rows("chr2", [f"n{i}" for i in range(8)],
                            [1, 2], seed=6))
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        sizes = {"chr1": 1000, "chr2": 500}
        _write_chromsize(chromsize, sizes)

        seed, max_nmol = 123, 7
        legacy_ids, legacy_dfs = _legacy_reference(
            rows, sizes, max_nmol=max_nmol, seed=seed)

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, max_nmol=max_nmol,
            seed=seed, chunk_size=5)

        for chrom in ("chr1", "chr2"):
            raw = exp.raw[chrom]
            assert np.array_equal(np.sort(legacy_ids[chrom]),
                                  raw.test_mol_id)
            l = legacy_dfs[chrom].sort_values(
                ["mol_index", "pos"]).reset_index(drop=True)
            n = raw.test_data.sort_values(
                ["mol_index", "pos"]).reset_index(drop=True)
            np.testing.assert_array_equal(l["mol_index"].to_numpy(),
                                          n["mol_index"].to_numpy())
            np.testing.assert_array_equal(l["pos"].to_numpy(),
                                          n["pos"].to_numpy())
            np.testing.assert_array_equal(l["mod_qual"].to_numpy(),
                                          n["mod_qual"].to_numpy())
        exp.close()

    def test_wrap_folds_positions_around_center(self, tmp_path):
        # length=100 -> fold threshold at (100-1)/2=49.5; upos>49.5 folds
        rows = [("m0", 10, "chr1", "+", 0.1, "a"),
               ("m0", 90, "chr1", "+", 0.2, "a")]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True)
        raw = exp.raw["chr1"]
        assert raw.nbp == 50
        positions = sorted(raw.test_data["pos"].tolist())
        # upos=10 stays 10 (below threshold); upos=90 -> 100-90-1=9
        assert positions == [9, 10]
        exp.close()

    def test_chrom_mismatch_raises(self, tmp_path):
        test_rows = _make_rows("chr1", ["m0"], [1, 2])
        unmeth_rows = _make_rows("chr2", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        unmeth_file = tmp_path / "unmeth.tsv"
        _write_tsv(test_file, test_rows)
        _write_tsv(unmeth_file, unmeth_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 100})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file,
                unmeth_file=unmeth_file)


class TestColidx:
    def test_headerless_file_with_colidx(self, tmp_path):
        # File columns: extra, chrom, upos, strand, mod_qual, mod_code, mol_id
        rows = [
            ("junk", "chr1", 5, "+", 0.5, "a", "mol1"),
            ("junk", "chr1", 6, "-", 0.6, "h", "mol1"),
        ]
        test_file = tmp_path / "test.tsv"
        with open(test_file, "w") as f:
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        # colidx positional order: read_id, ref_position, chrom,
        # ref_strand, mod_qual, mod_code
        colidx = [6, 2, 1, 3, 4, 5]
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, colidx=colidx)
        raw = exp.raw["chr1"]
        assert list(raw.test_mol_id) == ["mol1"]
        assert sorted(raw.test_data["pos"].tolist()) == [5, 6]
        exp.close()


class TestChromSelection:
    def test_selects_subset_of_chromosomes(self, tmp_path):
        rows = (_make_rows("chr1", ["m0"], [1, 2])
               + _make_rows("chr2", ["m0"], [1, 2])
               + _make_rows("chr3", ["m0"], [1, 2]))
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 100, "chr3": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file,
            chroms=["chr1", "chr3"])
        assert set(exp.chroms) == {"chr1", "chr3"}
        with pytest.raises(KeyError):
            exp.raw["chr2"]
        exp.close()

    def test_unknown_chromosome_raises(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file,
                chroms=["chrX"])


class TestRefseq:
    def test_refseq_loaded_into_raw_data(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "ACGTACGTAC"})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file)
        assert exp.raw["chr1"].refseq == "ACGTACGTAC"
        exp.close()

    def test_refseq_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "ACGTACGTAC"})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file)
        exp_file = tmp_path / "exp.h5"
        exp.save(exp_file)
        exp.close()

        exp2 = MethPrintExperiment.load(exp_file)
        assert exp2.raw["chr1"].refseq == "ACGTACGTAC"

    def test_length_mismatch_raises(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "ACGT"})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file,
                fasta_file=fasta_file)

    def test_chrom_missing_from_fasta_stays_none(self, tmp_path):
        rows = (_make_rows("chr1", ["m0"], [1, 2])
               + _make_rows("chr2", ["m0"], [1, 2]))
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10, "chr2": 8})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "ACGTACGTAC"})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file)
        assert exp.raw["chr1"].refseq == "ACGTACGTAC"
        assert exp.raw["chr2"].refseq is None
        exp.close()

    def test_no_fasta_file_backward_compatible(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file)
        assert exp.raw["chr1"].refseq is None
        exp.close()

    def test_wrap_validates_against_full_length(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 20})

        full_fasta = tmp_path / "full.fa"
        _write_fasta(full_fasta, {"chr1": "A" * 20})
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True,
            fasta_file=full_fasta)
        assert exp.raw["chr1"].nbp == 10
        assert exp.raw["chr1"].refseq == "A" * 20
        exp.close()

        halved_fasta = tmp_path / "halved.fa"
        _write_fasta(halved_fasta, {"chr1": "A" * 10})
        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file, wrap=True,
                fasta_file=halved_fasta)

    def test_chroms_filter_ignores_excluded_fasta_records(self, tmp_path):
        rows = (_make_rows("chr1", ["m0"], [1, 2])
               + _make_rows("chr3", ["m0"], [1, 2]))
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10, "chr2": 8, "chr3": 6})
        fasta_file = tmp_path / "ref.fa"
        # chr2 is excluded via chroms=, and deliberately wrong-length
        _write_fasta(fasta_file, {"chr1": "ACGTACGTAC", "chr2": "AAA",
                                  "chr3": "TTTTTT"})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            chroms=["chr1", "chr3"])
        assert exp.raw["chr1"].refseq == "ACGTACGTAC"
        assert exp.raw["chr3"].refseq == "TTTTTT"
        exp.close()


class TestStrandConsistency:
    def test_drops_row_inconsistent_with_mtase_and_strand(self, tmp_path):
        # seq: A(0) T(1) C(2) G(3) A(4) T(5) C(6) G(7)
        seq = "ATCGATCG"
        rows = [("m0", 0, "chr1", "+", 0.9, "a"),   # valid: ref 'A' on '+'
               ("m0", 1, "chr1", "+", 0.9, "a"),   # invalid: ref 'T' on '+'
               ("m0", 1, "chr1", "-", 0.9, "a")]   # valid: ref 'T' on '-'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="A")
        assert sorted(exp.raw["chr1"].test_data["pos"].tolist()) == [0, 1]
        exp.close()

    def test_respects_dinucleotide_context(self, tmp_path):
        seq = "ACAT"  # 'C' at index 1 not followed by 'G' -- not a CpG site
        rows = [("m0", 1, "chr1", "+", 0.9, "a")]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="CG")
        assert len(exp.raw["chr1"].test_data) == 0
        exp.close()

    def test_unmapped_strand_rows_kept(self, tmp_path):
        seq = "ATCG"
        rows = [("m0", 2, "chr1", ".", 0.9, "a")]  # ref 'C', can't validate
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="A")
        assert len(exp.raw["chr1"].test_data) == 1
        exp.close()

    def test_no_mtase_skips_filtering(self, tmp_path):
        seq = "ATCG"
        rows = [("m0", 1, "chr1", "+", 0.9, "a")]  # invalid for mtase 'A'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file)
        assert len(exp.raw["chr1"].test_data) == 1
        exp.close()

    def test_composes_with_wrap_using_unfolded_position(self, tmp_path):
        seq = "ATCGATCG"
        rows = [("m0", 5, "chr1", "-", 0.9, "a")]  # ref[5]='T', valid on '-'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="A", wrap=True)
        raw = exp.raw["chr1"]
        assert len(raw.test_data) == 1
        assert raw.test_data["pos"].iloc[0] == 2  # folded: 8-5-1=2
        exp.close()

    def test_ignore_strand_keeps_row_valid_on_opposite_strand(
            self, tmp_path):
        seq = "ATCGATCG"
        rows = [("m0", 1, "chr1", "+", 0.9, "a")]  # ref 'T': valid on '-'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="A", ignore_strand=True)
        assert len(exp.raw["chr1"].test_data) == 1
        exp.close()

    def test_ignore_strand_still_drops_row_invalid_on_both_strands(
            self, tmp_path):
        seq = "ATCGATCG"
        rows = [("m0", 2, "chr1", "+", 0.9, "a")]  # ref 'C': invalid on both
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="A", ignore_strand=True)
        assert len(exp.raw["chr1"].test_data) == 0
        exp.close()

    def test_ignore_strand_defaults_to_false(self, tmp_path):
        seq = "ATCGATCG"
        rows = [("m0", 1, "chr1", "+", 0.9, "a")]  # ref 'T': invalid on '+'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
            mtase="A")
        assert len(exp.raw["chr1"].test_data) == 0
        exp.close()


class TestMtase:
    def test_no_mtase_defaults_to_none(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file)
        assert exp.mtase is None
        exp.close()

    def test_single_string_normalized_to_tuple(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase="CG")
        assert exp.mtase == ("CG",)
        exp.close()

    def test_list_of_labels_preserved(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase=["CG", "GC"])
        assert exp.mtase == ("CG", "GC")
        exp.close()

    @pytest.mark.parametrize("bad", ["X", "AT", "XX", "cg", "ga"])
    def test_unknown_label_raises(self, tmp_path, bad):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file, mtase=bad)

    def test_duplicate_labels_raise(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file,
                mtase=["CG", "CG"])

    def test_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase=["A", "GC"])
        exp_file = tmp_path / "exp.h5"
        exp.save(exp_file)
        exp.close()

        exp2 = MethPrintExperiment.load(exp_file)
        assert exp2.mtase == ("A", "GC")


class TestWrapAttribute:
    def test_defaults_to_false(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file)
        assert exp.wrap is False
        exp.close()

    def test_true_when_requested(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True)
        assert exp.wrap is True
        exp.close()

    def test_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True)
        exp_file = tmp_path / "exp.h5"
        exp.save(exp_file)
        exp.close()

        exp2 = MethPrintExperiment.load(exp_file)
        assert exp2.wrap is True


class TestIgnoreStrandAttribute:
    def test_defaults_to_false(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file)
        assert exp.ignore_strand is False
        exp.close()

    def test_true_when_requested(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, ignore_strand=True)
        assert exp.ignore_strand is True
        exp.close()

    def test_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, ignore_strand=True)
        exp_file = tmp_path / "exp.h5"
        exp.save(exp_file)
        exp.close()

        exp2 = MethPrintExperiment.load(exp_file)
        assert exp2.ignore_strand is True


class TestLazyAndScratchLifecycle:
    def test_raw_data_loaded_lazily(self, tmp_path):
        rows = (_make_rows("chr1", ["m0"], [1, 2])
               + _make_rows("chr2", ["m0"], [1, 2]))
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 100})

        with patch.object(MethPrintData, "_load",
                          wraps=MethPrintData._load) as spy:
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file)
            assert spy.call_count == 0
            exp.raw["chr1"]
            assert spy.call_count == 1
            exp.raw["chr2"]
            assert spy.call_count == 2
        exp.close()

    def test_scratch_dir_cleaned_up_on_close(self, tmp_path):
        import os
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file)
        tmp_dir = exp.resolve_tmp_dir()
        assert os.path.isdir(tmp_dir)
        exp.close()
        assert not os.path.exists(tmp_dir)

    def test_staging_dir_cleaned_up_when_load_raw_raises(self, tmp_path):
        import os
        import tempfile

        test_rows = _make_rows("chr1", ["m0"], [1, 2])
        unmeth_rows = _make_rows("chr2", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        unmeth_file = tmp_path / "unmeth.tsv"
        _write_tsv(test_file, test_rows)
        _write_tsv(unmeth_file, unmeth_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 100})

        before = set(os.listdir(tempfile.gettempdir()))
        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file,
                unmeth_file=unmeth_file)
        after = set(os.listdir(tempfile.gettempdir()))
        leaked = [d for d in after - before if d.startswith("catella_")]
        assert not leaked

    def test_staging_dir_cleaned_up_on_keyboard_interrupt(self, tmp_path):
        import os
        import tempfile

        test_rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, test_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        before = set(os.listdir(tempfile.gettempdir()))
        with patch(
            "catella.experiment.methdata._stream_rows_to_staging",
            side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                MethPrintExperiment.load_raw(
                    chromsize=chromsize, test_file=test_file)
        after = set(os.listdir(tempfile.gettempdir()))
        leaked = [d for d in after - before if d.startswith("catella_")]
        assert not leaked


def _raw_snapshot(exp, chroms):
    # Comparable snapshot of an experiment's raw data across sources
    # actually present (test, and unmeth/meth if given).
    snapshot = {}
    for chrom in chroms:
        raw = exp.raw[chrom]
        snapshot[chrom] = {}
        for src in ("test", "unmeth", "meth"):
            mol_id = getattr(raw, f"{src}_mol_id")
            if mol_id is None:
                continue
            data = getattr(raw, f"{src}_data").sort_values(
                ["mol_index", "pos"]).reset_index(drop=True)
            snapshot[chrom][src] = (sorted(mol_id), data)
    return snapshot


def _assert_snapshots_equal(a, b):
    assert a.keys() == b.keys()
    for chrom in a:
        assert a[chrom].keys() == b[chrom].keys()
        for src in a[chrom]:
            ids_a, df_a = a[chrom][src]
            ids_b, df_b = b[chrom][src]
            assert ids_a == ids_b
            pd.testing.assert_frame_equal(df_a, df_b)


class TestParallelReading:
    def test_nworker_matches_sequential(self, tmp_path):
        test_rows = (_make_rows("chr1", [f"t{i}" for i in range(6)],
                                [1, 2, 3], seed=0)
                    + _make_rows("chr2", [f"t{i}" for i in range(4)],
                                [1, 2], seed=1))
        unmeth_rows = (_make_rows("chr1", [f"u{i}" for i in range(6)],
                                  [1, 2, 3], seed=2)
                      + _make_rows("chr2", [f"u{i}" for i in range(4)],
                                  [1, 2], seed=3))
        meth_rows = (_make_rows("chr1", [f"m{i}" for i in range(6)],
                                [1, 2, 3], seed=4)
                    + _make_rows("chr2", [f"m{i}" for i in range(4)],
                                [1, 2], seed=5))
        test_file = tmp_path / "test.tsv"
        unmeth_file = tmp_path / "unmeth.tsv"
        meth_file = tmp_path / "meth.tsv"
        _write_tsv(test_file, test_rows)
        _write_tsv(unmeth_file, unmeth_rows)
        _write_tsv(meth_file, meth_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 50})

        exp1 = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file,
            unmeth_file=unmeth_file, meth_file=meth_file, chunk_size=3,
            nworker=1)
        snap1 = _raw_snapshot(exp1, ["chr1", "chr2"])
        exp1.close()

        exp3 = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file,
            unmeth_file=unmeth_file, meth_file=meth_file, chunk_size=3,
            nworker=3)
        snap3 = _raw_snapshot(exp3, ["chr1", "chr2"])
        exp3.close()

        _assert_snapshots_equal(snap1, snap3)

    def test_seeded_downsampling_deterministic_across_nworker(
            self, tmp_path):
        test_rows = _make_rows("chr1", [f"t{i}" for i in range(20)],
                               [1, 2], seed=0)
        unmeth_rows = _make_rows("chr1", [f"u{i}" for i in range(20)],
                                 [1, 2], seed=1)
        test_file = tmp_path / "test.tsv"
        unmeth_file = tmp_path / "unmeth.tsv"
        _write_tsv(test_file, test_rows)
        _write_tsv(unmeth_file, unmeth_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        def run(nworker):
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file,
                unmeth_file=unmeth_file, max_nmol=5, seed=7, chunk_size=3,
                nworker=nworker)
            snapshot = _raw_snapshot(exp, ["chr1"])
            exp.close()
            return snapshot

        snap_1a = run(1)
        snap_1b = run(1)
        snap_3 = run(3)
        _assert_snapshots_equal(snap_1a, snap_1b)
        _assert_snapshots_equal(snap_1a, snap_3)

    def test_gzip_input_matches_plain(self, tmp_path):
        import gzip

        rows = (_make_rows("chr1", ["t0", "t1", "t2"], [1, 2, 3], seed=0)
               + _make_rows("chr2", ["t0", "t1"], [5, 15], seed=1))
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 50})

        plain_file = tmp_path / "test.tsv"
        _write_tsv(plain_file, rows)
        exp_plain = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=plain_file, chunk_size=3)
        snap_plain = _raw_snapshot(exp_plain, ["chr1", "chr2"])
        exp_plain.close()

        gz_file = tmp_path / "test.tsv.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write(_HEADER)
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        exp_gz = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=gz_file, chunk_size=3)
        snap_gz = _raw_snapshot(exp_gz, ["chr1", "chr2"])
        exp_gz.close()

        _assert_snapshots_equal(snap_plain, snap_gz)

