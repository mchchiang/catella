# test_load_raw.py

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nucmc.experiment.methdata import MethPrintData, MethPrintExperiment


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

    def test_scratch_file_cleaned_up_on_close(self, tmp_path):
        import os
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file)
        scratch_path = exp._scratch_path
        assert os.path.exists(scratch_path)
        exp.close()
        assert not os.path.exists(scratch_path)

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

