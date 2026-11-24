# test_load_raw.py

from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
from catella._regions import Region
from catella.experiment.methdata import MethPrintData, MethPrintExperiment

_HEADER = "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"


def _write_tsv(path, rows, header=_HEADER):
    with open(path, "w") as f:
        if header:
            f.write(header)
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        f.writelines(f"{chrom}\t{length}\n" for chrom, length in sizes.items())


def _write_fasta(path, records):
    with open(path, "w") as f:
        f.writelines(f">{region}\n{seq}\n" for region, seq in records.items())


def _make_rows(region, mol_ids, positions, seed=0, strand="+"):
    rng = np.random.default_rng(seed)
    rows = []
    for mol in mol_ids:
        for pos in positions:
            rows.append(
                (mol, pos, region, strand, round(float(rng.random()), 4), "a")
            )
    return rows


def _legacy_reference(
    test_rows, chromsize, wrap=False, max_nmol=None, seed=None
):
    # Standalone re-implementation of the pre-streaming algorithm, used
    # only to verify the streaming version reproduces it exactly.
    df = pd.DataFrame(
        test_rows,
        columns=["mol_id", "upos", "region", "strand", "mod_qual", "mod_code"],
    )
    df_size = pd.DataFrame(
        list(chromsize.items()), columns=["region", "length"]
    )
    rng = None if max_nmol is None else np.random.default_rng(seed)
    if max_nmol is not None:
        mol_map = df[["region", "mol_id"]].drop_duplicates()
        sampled = []
        for region, group in mol_map.groupby("region"):
            mols = group["mol_id"].values
            if len(mols) > max_nmol:
                mols = rng.choice(mols, max_nmol, replace=False)
            sampled.append(mols)
        sampled = np.concatenate(sampled)
        df = df.set_index("mol_id").loc[sampled].reset_index()
    df = pd.merge(df, df_size, on="region")
    if wrap:
        df["pos"] = np.where(
            df["upos"] > (df["length"] - 1) / 2,
            df["length"] - df["upos"] - 1,
            df["upos"],
        )
    else:
        df["pos"] = df["upos"]
    df = df.sort_values(["region", "mol_id", "pos"]).reset_index(drop=True)
    df["mol_index"] = df.groupby("region")["mol_id"].transform(
        lambda x: pd.factorize(x)[0]
    )
    mol_ids = {r: g["mol_id"].unique() for r, g in df.groupby("region")}
    keep = ["region", "mol_index", "pos", "strand", "mod_qual", "mod_code"]
    dfs = {r: g[keep].drop(columns="region") for r, g in df.groupby("region")}
    return mol_ids, dfs


class TestBasicIngestion:
    def test_multi_region(self, tmp_path):
        rows = _make_rows(
            "chr1", ["m0", "m1", "m2"], [10, 20, 30], seed=0
        ) + _make_rows("chr2", ["m0", "m1"], [5, 15], seed=1)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 50})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, chunk_size=4
        )
        assert set(exp.regions) == {"chr1", "chr2"}

        raw1 = exp.raw["chr1"]
        assert raw1.nbp == 100
        assert sorted(raw1.test.mol_id) == ["m0", "m1", "m2"]
        assert set(raw1.test.data["pos"]) == {10, 20, 30}

        raw2 = exp.raw["chr2"]
        assert raw2.nbp == 50
        assert sorted(raw2.test.mol_id) == ["m0", "m1"]
        exp.close()

    def test_pos_and_mol_index_are_integers_after_reload(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, _make_rows("chr1", ["m0", "m1"], [10, 20]))
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        df = exp.raw["chr1"].test.data
        assert np.issubdtype(df["pos"].dtype, np.integer)
        assert np.issubdtype(df["mol_index"].dtype, np.integer)

        saved = tmp_path / "exp.h5"
        exp.save(saved)
        exp.close()
        reloaded = MethPrintExperiment.load(saved)
        df = reloaded.raw["chr1"].test.data
        assert np.issubdtype(df["pos"].dtype, np.integer)
        assert np.issubdtype(df["mol_index"].dtype, np.integer)
        assert set(df["pos"]) == {10, 20}
        reloaded.close()

    def test_multi_character_mod_code_is_preserved(self, tmp_path):
        test_file = tmp_path / "test.tsv"
        _write_tsv(
            test_file,
            [
                ("m0", 10, "chr1", "+", 0.5, "76792"),
                ("m0", 20, "chr1", "+", 0.5, "a"),
            ],
        )
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        codes = exp.raw["chr1"].test.data["mod_code"].tolist()
        assert sorted(codes) == ["76792", "a"]
        exp.close()

    def test_numeric_region_names(self, tmp_path):
        rows = _make_rows("601", ["m0", "m1"], [10, 20]) + _make_rows(
            "1", ["m0"], [5], seed=1
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"601": 100, "1": 50})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        assert set(exp.regions) == {"601", "1"}
        assert sorted(exp.raw["601"].test.mol_id) == ["m0", "m1"]
        assert exp.raw["1"].nbp == 50
        exp.close()

    def test_chunk_size_does_not_change_result(self, tmp_path):
        rows = _make_rows(
            "chr1", [f"m{i}" for i in range(12)], [1, 2, 3, 4], seed=3
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp_a = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, chunk_size=10_000
        )
        exp_b = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, chunk_size=3
        )

        a = (
            exp_a.raw["chr1"]
            .test.data.sort_values(["mol_index", "pos"])
            .reset_index(drop=True)
        )
        b = (
            exp_b.raw["chr1"]
            .test.data.sort_values(["mol_index", "pos"])
            .reset_index(drop=True)
        )
        np.testing.assert_array_equal(
            a["mol_index"].to_numpy(), b["mol_index"].to_numpy()
        )
        np.testing.assert_array_equal(a["pos"].to_numpy(), b["pos"].to_numpy())
        np.testing.assert_array_equal(
            a["mod_qual"].to_numpy(), b["mod_qual"].to_numpy()
        )
        exp_a.close()
        exp_b.close()


class TestDownsamplingAndWrap:
    def test_max_nmol_matches_legacy_algorithm(self, tmp_path):
        rows = _make_rows(
            "chr1", [f"m{i}" for i in range(20)], [1, 2, 3], seed=5
        ) + _make_rows("chr2", [f"n{i}" for i in range(8)], [1, 2], seed=6)
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        sizes = {"chr1": 1000, "chr2": 500}
        _write_chromsize(chromsize, sizes)

        seed, max_nmol = 123, 7
        legacy_ids, legacy_dfs = _legacy_reference(
            rows, sizes, max_nmol=max_nmol, seed=seed
        )

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            max_nmol=max_nmol,
            seed=seed,
            chunk_size=5,
        )

        for region in ("chr1", "chr2"):
            raw = exp.raw[region]
            assert np.array_equal(np.sort(legacy_ids[region]), raw.test.mol_id)
            l = (
                legacy_dfs[region]
                .sort_values(["mol_index", "pos"])
                .reset_index(drop=True)
            )
            n = raw.test.data.sort_values(["mol_index", "pos"]).reset_index(
                drop=True
            )
            np.testing.assert_array_equal(
                l["mol_index"].to_numpy(), n["mol_index"].to_numpy()
            )
            np.testing.assert_array_equal(
                l["pos"].to_numpy(), n["pos"].to_numpy()
            )
            np.testing.assert_array_equal(
                l["mod_qual"].to_numpy(), n["mod_qual"].to_numpy()
            )
        exp.close()

    def test_wrap_folds_positions_around_center(self, tmp_path):
        # length=100 -> fold threshold at (100-1)/2=49.5; upos>49.5 folds
        rows = [
            ("m0", 10, "chr1", "+", 0.1, "a"),
            ("m0", 90, "chr1", "+", 0.2, "a"),
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True
        )
        raw = exp.raw["chr1"]
        assert raw.nbp == 50
        positions = sorted(raw.test.data["pos"].tolist())
        # upos=10 stays 10 (below threshold); upos=90 -> 100-90-1=9
        assert positions == [9, 10]
        exp.close()

    def test_region_mismatch_raises(self, tmp_path):
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
                chromsize=chromsize,
                test_file=test_file,
                unmeth_file=unmeth_file,
            )


def _strand_counts(raw):
    df = raw.test.data
    first = df.groupby("mol_index")["strand"].first()
    return first.value_counts().to_dict()


class TestBalanceStrand:
    def _setup(self, tmp_path, nplus, nminus, nunmapped=0, size=100):
        rows = (
            _make_rows("chr1", [f"p{i}" for i in range(nplus)], [10, 90])
            + _make_rows(
                "chr1", [f"q{i}" for i in range(nminus)], [10, 90], strand="-"
            )
            + _make_rows(
                "chr1", [f"u{i}" for i in range(nunmapped)], [10], strand="."
            )
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": size})
        return chromsize, test_file

    def test_equal_counts_when_both_strands_large(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 20, 30)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            max_nmol=10,
            balance_strand=True,
            seed=1,
        )
        assert _strand_counts(exp.raw["chr1"]) == {"+": 5, "-": 5}
        exp.close()

    @pytest.mark.filterwarnings("error")
    def test_no_warning_when_both_strands_large(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 20, 30)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            max_nmol=10,
            balance_strand=True,
        )
        exp.close()

    def test_shortfall_keeps_min_per_strand_and_warns(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 3, 20)
        with pytest.warns(UserWarning, match="only 3 '\\+' strand"):
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize,
                test_file=test_file,
                max_nmol=10,
                balance_strand=True,
            )
        assert _strand_counts(exp.raw["chr1"]) == {"+": 3, "-": 3}
        exp.close()

    def test_odd_max_nmol_rounds_down(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 10, 10)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            max_nmol=7,
            balance_strand=True,
        )
        assert _strand_counts(exp.raw["chr1"]) == {"+": 3, "-": 3}
        exp.close()

    def test_unmapped_strand_dropped(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 5, 5, nunmapped=10)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            max_nmol=10,
            balance_strand=True,
        )
        raw = exp.raw["chr1"]
        assert not any(m.startswith("u") for m in raw.test.mol_id)
        assert _strand_counts(raw) == {"+": 5, "-": 5}
        exp.close()

    def test_same_seed_same_molecules(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 20, 30)
        kwargs = {
            "chromsize": chromsize,
            "test_file": test_file,
            "max_nmol": 10,
            "balance_strand": True,
            "seed": 3,
        }
        exp_a = MethPrintExperiment.load_raw(**kwargs)
        exp_b = MethPrintExperiment.load_raw(**kwargs)
        assert np.array_equal(
            exp_a.raw["chr1"].test.mol_id, exp_b.raw["chr1"].test.mol_id
        )
        exp_a.close()
        exp_b.close()

    def test_wrap_keeps_balance_and_molecules(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 20, 30)
        kwargs = {
            "chromsize": chromsize,
            "test_file": test_file,
            "max_nmol": 10,
            "balance_strand": True,
            "seed": 4,
        }
        exp_plain = MethPrintExperiment.load_raw(**kwargs)
        exp_wrap = MethPrintExperiment.load_raw(wrap=True, **kwargs)
        raw = exp_wrap.raw["chr1"]
        assert _strand_counts(raw) == {"+": 5, "-": 5}
        assert np.array_equal(
            exp_plain.raw["chr1"].test.mol_id, raw.test.mol_id
        )
        exp_plain.close()
        exp_wrap.close()

    def test_one_strand_only_gives_empty_region(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 10, 0)
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "A" * 100})
        with pytest.warns(UserWarning, match="only 0 '-' strand"):
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize,
                test_file=test_file,
                fasta_file=fasta_file,
                mtase="A",
                max_nmol=10,
                balance_strand=True,
            )
        assert len(exp.raw["chr1"].test.mol_id) == 0
        assert len(exp.qc()) == 1
        exp.close()

    def test_requires_max_nmol(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 2, 2)
        with pytest.raises(ValueError, match="requires max_nmol"):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file, balance_strand=True
            )

    def test_max_nmol_below_two_raises(self, tmp_path):
        chromsize, test_file = self._setup(tmp_path, 2, 2)
        with pytest.raises(ValueError, match="at least 2"):
            MethPrintExperiment.load_raw(
                chromsize=chromsize,
                test_file=test_file,
                max_nmol=1,
                balance_strand=True,
            )


class TestCanonicalCallFiltering:
    def test_drops_canonical_mod_code_rows(self, tmp_path):
        rows = [
            ("m0", 1, "chr1", "+", 0.9, "a"),
            ("m0", 2, "chr1", "+", 0.8, "-"),
            ("m0", 3, "chr1", "+", 0.7, "m"),
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        raw = exp.raw["chr1"]
        assert sorted(raw.test.data["pos"]) == [1, 3]
        assert "-" not in raw.test.data["mod_code"].to_numpy()
        exp.close()

    def test_all_canonical_leaves_molecule_with_no_rows(self, tmp_path):
        rows = [("m0", 1, "chr1", "+", 0.9, "-")]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        raw = exp.raw["chr1"]
        assert len(raw.test.data) == 0
        exp.close()


class TestOutOfBoundsRows:
    def _setup(self, tmp_path, with_fasta):
        rows = [
            ("m0", -1, "chr1", "+", 0.9, "a"),
            ("m0", 1, "chr1", "+", 0.8, "a"),
            ("m0", 8, "chr1", "+", 0.7, "a"),
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 8})
        kwargs = {"chromsize": chromsize, "test_file": test_file}
        if with_fasta:
            fasta = tmp_path / "ref.fa"
            _write_fasta(fasta, {"chr1": "AAAAAAAA"})
            kwargs.update(fasta_file=fasta, mtase="A")
        return kwargs

    @pytest.mark.parametrize("with_fasta", [False, True])
    def test_rows_dropped_with_warning(self, tmp_path, with_fasta):
        kwargs = self._setup(tmp_path, with_fasta)
        with pytest.warns(UserWarning, match="dropped 2 row"):
            exp = MethPrintExperiment.load_raw(**kwargs)
        assert exp.raw["chr1"].test.data["pos"].tolist() == [1]
        exp.close()

    def test_no_warning_when_in_bounds(self, tmp_path, recwarn):
        rows = [("m0", 1, "chr1", "+", 0.8, "a")]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 8})
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        exp.close()
        assert not [w for w in recwarn if "dropped" in str(w.message)]


_FLAG_HEADER = (
    "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\tflag\n"
)


class TestNonPrimaryRows:
    def _load(self, tmp_path, rows):
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows, header=_FLAG_HEADER)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})
        return MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )

    def test_secondary_and_supplementary_rows_dropped(self, tmp_path):
        rows = [
            ("m0", 10, "chr1", "+", 0.9, "a", 0),
            ("m0", 20, "chr1", "+", 0.8, "a", 16 | 0),
            ("m0", 30, "chr1", "+", 0.7, "a", 2048),
            ("m0", 40, "chr1", "-", 0.6, "a", 256 | 16),
        ]
        with pytest.warns(UserWarning, match="dropped 2 row"):
            exp = self._load(tmp_path, rows)
        raw = exp.raw["chr1"]
        assert raw.test.data["pos"].tolist() == [10, 20]
        assert raw.test.mol_end.tolist() == [21]
        exp.close()

    def test_molecule_with_only_supplementary_rows_not_kept(self, tmp_path):
        rows = [
            ("m0", 10, "chr1", "+", 0.9, "a", 0),
            ("m1", 50, "chr1", "+", 0.8, "a", 2048),
        ]
        with pytest.warns(UserWarning, match="supplementary"):
            exp = self._load(tmp_path, rows)
        assert list(exp.raw["chr1"].test.mol_id) == ["m0"]
        exp.close()

    def test_no_warning_for_primary_rows(self, tmp_path, recwarn):
        rows = [
            ("m0", 10, "chr1", "+", 0.9, "a", 0),
            ("m1", 50, "chr1", "-", 0.8, "a", 16),
        ]
        exp = self._load(tmp_path, rows)
        assert sorted(exp.raw["chr1"].test.mol_id) == ["m0", "m1"]
        exp.close()
        assert not [w for w in recwarn if "supplementary" in str(w.message)]


class TestColidx:
    def test_headerless_file_with_colidx(self, tmp_path):
        # File columns: extra, region, upos, strand, mod_qual, mod_code, mol_id
        rows = [
            ("junk", "chr1", 5, "+", 0.5, "a", "mol1"),
            ("junk", "chr1", 6, "-", 0.6, "h", "mol1"),
        ]
        test_file = tmp_path / "test.tsv"
        with open(test_file, "w") as f:
            f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        # colidx positional order: read_id, ref_position, region,
        # ref_strand, mod_qual, mod_code
        colidx = [6, 2, 1, 3, 4, 5]
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, colidx=colidx
        )
        raw = exp.raw["chr1"]
        assert list(raw.test.mol_id) == ["mol1"]
        assert sorted(raw.test.data["pos"].tolist()) == [5, 6]
        exp.close()


class TestRegionSelection:
    def test_selects_subset_of_regions(self, tmp_path):
        rows = (
            _make_rows("chr1", ["m0"], [1, 2])
            + _make_rows("chr2", ["m0"], [1, 2])
            + _make_rows("chr3", ["m0"], [1, 2])
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 100, "chr3": 100})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, regions=["chr1", "chr3"]
        )
        assert set(exp.regions) == {"chr1", "chr3"}
        with pytest.raises(KeyError):
            exp.raw["chr2"]
        exp.close()

    def test_unknown_region_raises(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file, regions=["chrX"]
            )

    def _scoped_setup(self, tmp_path, with_chr0):
        rows = _make_rows(
            "chr1", [f"p{i}" for i in range(20)], [10, 90]
        ) + _make_rows(
            "chr1", [f"q{i}" for i in range(20)], [10, 90], strand="-"
        )
        if with_chr0:
            rows = (
                _make_rows("chr0", [f"r{i}" for i in range(20)], [10, 90])
                + rows
            )
        test_file = tmp_path / f"test_{with_chr0}.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr0": 100, "chr1": 100})
        return chromsize, test_file

    @pytest.mark.filterwarnings("error")
    def test_balance_strand_ignores_excluded_regions(self, tmp_path):
        chromsize, test_file = self._scoped_setup(tmp_path, True)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            regions=["chr1"],
            max_nmol=10,
            balance_strand=True,
        )
        assert set(exp.regions) == {"chr1"}
        exp.close()

    @pytest.mark.parametrize("balance_strand", [False, True])
    def test_excluded_regions_do_not_affect_sampling(
        self, tmp_path, balance_strand
    ):
        kwargs = {
            "regions": ["chr1"],
            "max_nmol": 10,
            "balance_strand": balance_strand,
            "seed": 5,
        }
        chromsize, file_a = self._scoped_setup(tmp_path, False)
        _, file_b = self._scoped_setup(tmp_path, True)
        exp_a = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=file_a, **kwargs
        )
        exp_b = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=file_b, **kwargs
        )
        assert np.array_equal(
            exp_a.raw["chr1"].test.mol_id, exp_b.raw["chr1"].test.mol_id
        )
        exp_a.close()
        exp_b.close()


class TestRegionCoords:
    def _load(self, tmp_path, **kwargs):
        rows = _make_rows("chr1", ["m0"], [1, 5]) + _make_rows(
            "chr2", ["m1"], [2]
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 60})
        return MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, **kwargs
        )

    def test_whole_chrom_coords(self, tmp_path):
        exp = self._load(tmp_path)
        assert exp.raw["chr1"].coords == Region("chr1", 0, 100, "chr1", ".")
        table = exp.region_coords
        assert table.index.tolist() == ["chr1", "chr2"]
        assert table.loc["chr2"].tolist() == ["chr2", 0, 60, ".", 60]
        exp.close()

    def test_wrap_keeps_full_length(self, tmp_path):
        exp = self._load(tmp_path, wrap=True)
        row = exp.region_coords.loc["chr1"]
        assert (row["ref_end"], row["nbp"]) == (100, 50)
        exp.close()

    def test_save_and_subset_load(self, tmp_path):
        exp = self._load(tmp_path)
        saved = tmp_path / "exp.h5"
        exp.save(saved)
        exp.close()
        reloaded = MethPrintExperiment.load(saved, regions=["chr2"])
        assert reloaded.region_coords.index.tolist() == ["chr2"]
        assert reloaded.raw["chr2"].coords.end == 60
        reloaded.close()

    def test_multi_file_merge_keeps_coords(self, tmp_path):
        unmeth = tmp_path / "unmeth.tsv"
        _write_tsv(
            unmeth,
            _make_rows("chr1", ["u0"], [3]) + _make_rows("chr2", ["u1"], [4]),
        )
        exp = self._load(tmp_path, unmeth_file=unmeth)
        assert exp.region_coords["ref_end"].tolist() == [100, 60]
        exp.close()

    def test_old_file_without_coords(self, tmp_path):
        exp = self._load(tmp_path)
        saved = tmp_path / "exp.h5"
        exp.save(saved)
        exp.close()
        with h5py.File(saved, "a") as h5:
            for region in ("chr1", "chr2"):
                attrs = h5[f"raw_data/{region}/metadata"].attrs
                for key in ("ref_chrom", "ref_start", "ref_end", "ref_strand"):
                    del attrs[key]
        reloaded = MethPrintExperiment.load(saved)
        assert reloaded.raw["chr1"].coords is None
        table = reloaded.region_coords
        assert table["ref_start"].isna().all()
        assert table["nbp"].tolist() == [100, 60]
        reloaded.close()

    def test_in_memory_experiment(self):
        raw = MethPrintData._create(
            region="r1",
            nbp=10,
            coords=Region("chr9", 5, 15, "r1", "-"),
            test_mol_id=np.array(["m0"], dtype=object),
            test_data=pd.DataFrame(
                {
                    "mol_index": [0],
                    "pos": [1],
                    "strand": ["+"],
                    "mod_qual": [0.5],
                    "mod_code": ["a"],
                }
            ),
        )
        exp = MethPrintExperiment._create(_raw_data={"r1": raw})
        assert exp.region_coords.loc["r1"].tolist() == [
            "chr9", 5, 15, "-", 10
        ]

    def test_returns_a_copy(self, tmp_path):
        exp = self._load(tmp_path)
        table = exp.region_coords
        table.loc["chr1", "nbp"] = -1
        assert exp.region_coords.loc["chr1", "nbp"] == 100
        exp.close()


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
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
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
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
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
                chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
            )

    def test_region_missing_from_fasta_stays_none(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2]) + _make_rows(
            "chr2", ["m0"], [1, 2]
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10, "chr2": 8})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": "ACGTACGTAC"})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
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
            chromsize=chromsize, test_file=test_file
        )
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
            chromsize=chromsize,
            test_file=test_file,
            wrap=True,
            fasta_file=full_fasta,
        )
        assert exp.raw["chr1"].nbp == 10
        assert exp.raw["chr1"].refseq == "A" * 20
        exp.close()

        halved_fasta = tmp_path / "halved.fa"
        _write_fasta(halved_fasta, {"chr1": "A" * 10})
        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize,
                test_file=test_file,
                wrap=True,
                fasta_file=halved_fasta,
            )

    def test_regions_filter_ignores_excluded_fasta_records(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2]) + _make_rows(
            "chr3", ["m0"], [1, 2]
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10, "chr2": 8, "chr3": 6})
        fasta_file = tmp_path / "ref.fa"
        # chr2 is excluded via regions=, and deliberately wrong-length
        _write_fasta(
            fasta_file, {"chr1": "ACGTACGTAC", "chr2": "AAA", "chr3": "TTTTTT"}
        )

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            regions=["chr1", "chr3"],
        )
        assert exp.raw["chr1"].refseq == "ACGTACGTAC"
        assert exp.raw["chr3"].refseq == "TTTTTT"
        exp.close()


class TestStrandConsistency:
    def test_drops_row_inconsistent_with_mtase_and_strand(self, tmp_path):
        # seq: A(0) T(1) C(2) G(3) A(4) T(5) C(6) G(7)
        seq = "ATCGATCG"
        rows = [
            ("m0", 0, "chr1", "+", 0.9, "a"),  # valid: ref 'A' on '+'
            ("m0", 1, "chr1", "+", 0.9, "a"),  # invalid: ref 'T' on '+'
            ("m0", 1, "chr1", "-", 0.9, "a"),
        ]  # valid: ref 'T' on '-'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
        )
        assert sorted(exp.raw["chr1"].test.data["pos"].tolist()) == [0, 1]
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
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="CG",
        )
        assert len(exp.raw["chr1"].test.data) == 0
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
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
        )
        assert len(exp.raw["chr1"].test.data) == 1
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
            chromsize=chromsize, test_file=test_file, fasta_file=fasta_file
        )
        assert len(exp.raw["chr1"].test.data) == 1
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
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
            wrap=True,
        )
        raw = exp.raw["chr1"]
        assert len(raw.test.data) == 1
        assert raw.test.data["pos"].iloc[0] == 2  # folded: 8-5-1=2
        exp.close()

    def test_ignore_strand_keeps_row_valid_on_opposite_strand(self, tmp_path):
        seq = "ATCGATCG"
        rows = [("m0", 1, "chr1", "+", 0.9, "a")]  # ref 'T': valid on '-'
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
            ignore_strand=True,
        )
        assert len(exp.raw["chr1"].test.data) == 1
        exp.close()

    def test_ignore_strand_still_drops_row_invalid_on_both_strands(
        self, tmp_path
    ):
        seq = "ATCGATCG"
        rows = [("m0", 2, "chr1", "+", 0.9, "a")]  # ref 'C': invalid on both
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": len(seq)})
        fasta_file = tmp_path / "ref.fa"
        _write_fasta(fasta_file, {"chr1": seq})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
            ignore_strand=True,
        )
        assert len(exp.raw["chr1"].test.data) == 0
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
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta_file,
            mtase="A",
        )
        assert len(exp.raw["chr1"].test.data) == 0
        exp.close()


class TestMtase:
    def test_no_mtase_defaults_to_none(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file
        )
        assert exp.mtase is None
        exp.close()

    def test_single_string_normalized_to_tuple(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase="CG"
        )
        assert exp.mtase == ("CG",)
        exp.close()

    def test_list_of_labels_preserved(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase=["CG", "GC"]
        )
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
                chromsize=chromsize, test_file=test_file, mtase=bad
            )

    def test_duplicate_labels_raise(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        with pytest.raises(ValueError):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file, mtase=["CG", "CG"]
            )

    def test_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase=["A", "GC"]
        )
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
            chromsize=chromsize, test_file=test_file
        )
        assert exp.wrap is False
        exp.close()

    def test_true_when_requested(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True
        )
        assert exp.wrap is True
        exp.close()

    def test_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, wrap=True
        )
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
            chromsize=chromsize, test_file=test_file
        )
        assert exp.ignore_strand is False
        exp.close()

    def test_true_when_requested(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, ignore_strand=True
        )
        assert exp.ignore_strand is True
        exp.close()

    def test_persists_through_save_load_roundtrip(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 10})

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, ignore_strand=True
        )
        exp_file = tmp_path / "exp.h5"
        exp.save(exp_file)
        exp.close()

        exp2 = MethPrintExperiment.load(exp_file)
        assert exp2.ignore_strand is True


class TestLazyAndScratchLifecycle:
    def test_raw_data_loaded_lazily(self, tmp_path):
        rows = _make_rows("chr1", ["m0"], [1, 2]) + _make_rows(
            "chr2", ["m0"], [1, 2]
        )
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 100})

        with patch.object(
            MethPrintData, "_load", wraps=MethPrintData._load
        ) as spy:
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file
            )
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
            chromsize=chromsize, test_file=test_file
        )
        tmp_dir = exp.resolve_tmp_dir()
        assert os.path.isdir(tmp_dir)
        exp.close()
        assert not os.path.exists(tmp_dir)

    @pytest.mark.usefixtures("isolated_tempdir")
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
                chromsize=chromsize,
                test_file=test_file,
                unmeth_file=unmeth_file,
            )
        after = set(os.listdir(tempfile.gettempdir()))
        leaked = [d for d in after - before if d.startswith("catella_")]
        assert not leaked

    @pytest.mark.usefixtures("isolated_tempdir")
    def test_staging_dir_cleaned_up_on_keyboard_interrupt(self, tmp_path):
        import os
        import tempfile

        test_rows = _make_rows("chr1", ["m0"], [1, 2])
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, test_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        before = set(os.listdir(tempfile.gettempdir()))
        with (
            patch(
                "catella.experiment._ingest._stream_rows_to_staging",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=test_file
            )
        after = set(os.listdir(tempfile.gettempdir()))
        leaked = [d for d in after - before if d.startswith("catella_")]
        assert not leaked


def _raw_snapshot(exp, regions):
    # Comparable snapshot of an experiment's raw data across sources
    # actually present (test, and unmeth/meth if given).
    snapshot = {}
    for region in regions:
        raw = exp.raw[region]
        snapshot[region] = {}
        for src, source in raw.sources.items():
            data = source.data.sort_values(["mol_index", "pos"]).reset_index(
                drop=True
            )
            snapshot[region][src] = (sorted(source.mol_id), data)
    return snapshot


def _assert_snapshots_equal(a, b):
    assert a.keys() == b.keys()
    for region in a:
        assert a[region].keys() == b[region].keys()
        for src in a[region]:
            ids_a, df_a = a[region][src]
            ids_b, df_b = b[region][src]
            assert ids_a == ids_b
            pd.testing.assert_frame_equal(df_a, df_b)


class TestParallelReading:
    def test_nworker_matches_sequential(self, tmp_path):
        test_rows = _make_rows(
            "chr1", [f"t{i}" for i in range(6)], [1, 2, 3], seed=0
        ) + _make_rows("chr2", [f"t{i}" for i in range(4)], [1, 2], seed=1)
        unmeth_rows = _make_rows(
            "chr1", [f"u{i}" for i in range(6)], [1, 2, 3], seed=2
        ) + _make_rows("chr2", [f"u{i}" for i in range(4)], [1, 2], seed=3)
        meth_rows = _make_rows(
            "chr1", [f"m{i}" for i in range(6)], [1, 2, 3], seed=4
        ) + _make_rows("chr2", [f"m{i}" for i in range(4)], [1, 2], seed=5)
        test_file = tmp_path / "test.tsv"
        unmeth_file = tmp_path / "unmeth.tsv"
        meth_file = tmp_path / "meth.tsv"
        _write_tsv(test_file, test_rows)
        _write_tsv(unmeth_file, unmeth_rows)
        _write_tsv(meth_file, meth_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 50})

        exp1 = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            unmeth_file=unmeth_file,
            meth_file=meth_file,
            chunk_size=3,
            nworker=1,
        )
        snap1 = _raw_snapshot(exp1, ["chr1", "chr2"])
        exp1.close()

        exp3 = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            unmeth_file=unmeth_file,
            meth_file=meth_file,
            chunk_size=3,
            nworker=3,
        )
        snap3 = _raw_snapshot(exp3, ["chr1", "chr2"])
        exp3.close()

        _assert_snapshots_equal(snap1, snap3)

    def test_seeded_downsampling_deterministic_across_nworker(self, tmp_path):
        test_rows = _make_rows(
            "chr1", [f"t{i}" for i in range(20)], [1, 2], seed=0
        )
        unmeth_rows = _make_rows(
            "chr1", [f"u{i}" for i in range(20)], [1, 2], seed=1
        )
        test_file = tmp_path / "test.tsv"
        unmeth_file = tmp_path / "unmeth.tsv"
        _write_tsv(test_file, test_rows)
        _write_tsv(unmeth_file, unmeth_rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100})

        def run(nworker):
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize,
                test_file=test_file,
                unmeth_file=unmeth_file,
                max_nmol=5,
                seed=7,
                chunk_size=3,
                nworker=nworker,
            )
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

        rows = _make_rows(
            "chr1", ["t0", "t1", "t2"], [1, 2, 3], seed=0
        ) + _make_rows("chr2", ["t0", "t1"], [5, 15], seed=1)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 100, "chr2": 50})

        plain_file = tmp_path / "test.tsv"
        _write_tsv(plain_file, rows)
        exp_plain = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=plain_file, chunk_size=3
        )
        snap_plain = _raw_snapshot(exp_plain, ["chr1", "chr2"])
        exp_plain.close()

        gz_file = tmp_path / "test.tsv.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write(_HEADER)
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        exp_gz = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=gz_file, chunk_size=3
        )
        snap_gz = _raw_snapshot(exp_gz, ["chr1", "chr2"])
        exp_gz.close()

        _assert_snapshots_equal(snap_plain, snap_gz)


_ALIGN_HEADER = (
    "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\t"
    "alignment_start\talignment_end\n"
)


def _extent_setup(tmp_path, rows, header=_HEADER, sizes=None, name="t"):
    data_file = tmp_path / f"{name}.tsv"
    _write_tsv(data_file, rows, header=header)
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, sizes or {"chr1": 100})
    return chromsize, data_file


def _extent(raw, src="test"):
    source = raw.sources[src]
    return source.mol_start.tolist(), source.mol_end.tolist()


class TestReadExtents:
    def test_from_alignment_columns(self, tmp_path):
        rows = [
            ("m0", 12, "chr1", "+", 0.9, "a", 10, 40),
            ("m0", 30, "chr1", "+", 0.8, "a", 10, 40),
            ("m1", 60, "chr1", "-", 0.7, "a", 55, 90),
        ]
        chromsize, f = _extent_setup(tmp_path, rows, header=_ALIGN_HEADER)
        exp = MethPrintExperiment.load_raw(chromsize=chromsize, test_file=f)
        assert _extent(exp.raw["chr1"]) == ([10, 55], [40, 90])
        exp.close()

    def test_fallback_to_first_and_last_row(self, tmp_path):
        rows = [
            ("m0", 12, "chr1", "+", 0.9, "a"),
            ("m0", 30, "chr1", "+", 0.8, "a"),
            ("m1", 60, "chr1", "+", 0.7, "a"),
        ]
        chromsize, f = _extent_setup(tmp_path, rows)
        exp = MethPrintExperiment.load_raw(chromsize=chromsize, test_file=f)
        assert _extent(exp.raw["chr1"]) == ([12, 60], [31, 61])
        exp.close()

    def test_filtered_rows_keep_extent(self, tmp_path):
        rows = [
            ("m0", 5, "chr1", "+", 0.9, "-"),
            ("m0", 20, "chr1", "+", 0.8, "a"),
            ("m0", 50, "chr1", "+", 0.7, "-"),
            ("m1", 70, "chr1", "+", 0.7, "-"),
        ]
        chromsize, f = _extent_setup(tmp_path, rows)
        exp = MethPrintExperiment.load_raw(chromsize=chromsize, test_file=f)
        raw = exp.raw["chr1"]
        assert _extent(raw) == ([5, 70], [51, 71])
        assert raw.test.data["pos"].tolist() == [20]
        exp.close()

    def test_clipped_to_region(self, tmp_path):
        rows = [("m0", 5, "chr1", "+", 0.9, "a", -3, 140)]
        chromsize, f = _extent_setup(tmp_path, rows, header=_ALIGN_HEADER)
        exp = MethPrintExperiment.load_raw(chromsize=chromsize, test_file=f)
        assert _extent(exp.raw["chr1"]) == ([0], [100])
        exp.close()

    def test_out_of_bounds_rows_do_not_widen_fallback(self, tmp_path):
        rows = [
            ("m0", -1, "chr1", "+", 0.9, "a"),
            ("m0", 7, "chr1", "+", 0.8, "a"),
        ]
        chromsize, f = _extent_setup(tmp_path, rows)
        with pytest.warns(UserWarning, match="dropped 1 row"):
            exp = MethPrintExperiment.load_raw(
                chromsize=chromsize, test_file=f
            )
        assert _extent(exp.raw["chr1"]) == ([7], [8])
        exp.close()

    def test_chunks_do_not_change_extent(self, tmp_path):
        rows = [("m0", p, "chr1", "+", 0.5, "a") for p in (40, 3, 77, 15)]
        chromsize, f = _extent_setup(tmp_path, rows)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, chunk_size=1
        )
        assert _extent(exp.raw["chr1"]) == ([3], [78])
        exp.close()

    def test_colidx_with_alignment_columns(self, tmp_path):
        rows = [("m0", 12, "chr1", "+", 0.9, "a", 10, 40)]
        chromsize, f = _extent_setup(tmp_path, rows, header=None)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=f,
            colidx=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        assert _extent(exp.raw["chr1"]) == ([10], [40])
        exp.close()

    def test_colidx_without_alignment_columns(self, tmp_path):
        rows = [("m0", 12, "chr1", "+", 0.9, "a", 10, 40)]
        chromsize, f = _extent_setup(tmp_path, rows, header=None)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, colidx=[0, 1, 2, 3, 4, 5]
        )
        assert _extent(exp.raw["chr1"]) == ([12], [13])
        exp.close()

    def test_none_with_wrap(self, tmp_path):
        rows = [("m0", 12, "chr1", "+", 0.9, "a", 10, 40)]
        chromsize, f = _extent_setup(tmp_path, rows, header=_ALIGN_HEADER)
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, wrap=True
        )
        raw = exp.raw["chr1"]
        assert raw.test.mol_start is None
        assert raw.test.mol_end is None
        exp.close()

    def test_multi_file_stores_each_source(self, tmp_path):
        test_rows = [("m0", 12, "chr1", "+", 0.9, "a", 10, 40)]
        ctrl_rows = [("u0", 50, "chr1", "+", 0.1, "a", 45, 95)]
        chromsize, f = _extent_setup(
            tmp_path, test_rows, header=_ALIGN_HEADER
        )
        _, uf = _extent_setup(
            tmp_path, ctrl_rows, header=_ALIGN_HEADER, name="u"
        )
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, unmeth_file=uf
        )
        raw = exp.raw["chr1"]
        assert _extent(raw) == ([10], [40])
        assert _extent(raw, "unmeth") == ([45], [95])
        assert raw.meth is None
        exp.close()

    def test_save_load_roundtrip(self, tmp_path):
        rows = [
            ("m0", 12, "chr1", "+", 0.9, "a", 10, 40),
            ("m1", 60, "chr1", "-", 0.7, "a", 55, 90),
        ]
        chromsize, f = _extent_setup(tmp_path, rows, header=_ALIGN_HEADER)
        exp = MethPrintExperiment.load_raw(chromsize=chromsize, test_file=f)
        saved = tmp_path / "exp.h5"
        exp.save(saved)
        exp.close()
        reloaded = MethPrintExperiment.load(saved)
        raw = reloaded.raw["chr1"]
        assert _extent(raw) == ([10, 55], [40, 90])
        assert raw.test.mol_start.dtype == np.int64
        reloaded.close()

    def test_old_file_without_extents_loads(self, tmp_path):
        import h5py

        rows = [("m0", 12, "chr1", "+", 0.9, "a", 10, 40)]
        chromsize, f = _extent_setup(tmp_path, rows, header=_ALIGN_HEADER)
        exp = MethPrintExperiment.load_raw(chromsize=chromsize, test_file=f)
        saved = tmp_path / "exp.h5"
        exp.save(saved)
        exp.close()
        with h5py.File(saved, "a") as h5:
            gdata = h5["raw_data/chr1/data"]
            del gdata["test_mol_start"]
            del gdata["test_mol_end"]
        reloaded = MethPrintExperiment.load(saved)
        raw = reloaded.raw["chr1"]
        assert raw.test.mol_start is None
        assert raw.test.mol_end is None
        assert list(raw.test.mol_id) == ["m0"]
        reloaded.close()


_GAP_HEADER = (
    "read_id\tforward_read_position\tref_position\tchrom\tref_strand\t"
    "mod_qual\tmod_code\n"
)


def _gap_rows():
    return [
        ("m0", 0, 10, "chr1", "+", 0.9, "a"),
        ("m0", 4, 14, "chr1", "+", 0.8, "-"),
        ("m0", 10, 60, "chr1", "+", 0.7, "a"),
        ("m1", 50, 20, "chr1", "-", 0.6, "a"),
        ("m1", 45, 25, "chr1", "-", 0.5, "a"),
    ]


def _gaps(raw, src="test"):
    source = raw.sources[src]
    return source.gaps.tolist(), source.gap_offset.tolist()


class TestInternalGaps:
    def _load(self, tmp_path, rows=None, header=_GAP_HEADER, **kwargs):
        chromsize, f = _extent_setup(
            tmp_path, rows or _gap_rows(), header=header
        )
        return MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, **kwargs
        )

    def test_inferred_from_read_positions(self, tmp_path):
        exp = self._load(tmp_path)
        assert _gaps(exp.raw["chr1"]) == ([[15, 60]], [0, 1, 1])
        exp.close()

    def test_canonical_rows_used_before_filtering(self, tmp_path):
        rows = _gap_rows()
        rows[1] = ("m0", 4, 50, "chr1", "+", 0.8, "-")
        exp = self._load(tmp_path, rows)
        raw = exp.raw["chr1"]
        assert _gaps(raw) == ([[11, 50]], [0, 1, 1])
        assert 50 not in raw.test.data["pos"].tolist()
        exp.close()

    def test_min_gap_threshold(self, tmp_path):
        exp = self._load(tmp_path, min_gap=50)
        assert _gaps(exp.raw["chr1"]) == ([], [0, 0, 0])
        exp.close()

    def test_min_gap_none_records_nothing(self, tmp_path):
        exp = self._load(tmp_path, min_gap=None)
        raw = exp.raw["chr1"]
        assert raw.test.gaps is None
        assert raw.test.gap_offset is None
        exp.close()

    @pytest.mark.parametrize("bad", [0, -5, 2.5, True, "20"])
    def test_invalid_min_gap_raises(self, tmp_path, bad):
        with pytest.raises(ValueError, match="min_gap must be None"):
            self._load(tmp_path, min_gap=bad)

    def test_no_read_position_column_records_nothing(self, tmp_path):
        rows = [r[:1] + r[2:] for r in _gap_rows()]
        exp = self._load(tmp_path, rows, header=_HEADER)
        assert exp.raw["chr1"].test.gaps is None
        exp.close()

    def test_none_with_wrap(self, tmp_path):
        exp = self._load(tmp_path, wrap=True)
        assert exp.raw["chr1"].test.gaps is None
        exp.close()

    def test_unordered_molecule_warns(self, tmp_path):
        rows = _gap_rows() + [("m0", 12, 30, "chr1", "+", 0.4, "a")]
        with pytest.warns(UserWarning, match="not ordered"):
            exp = self._load(tmp_path, rows)
        assert _gaps(exp.raw["chr1"]) == ([], [0, 0, 0])
        exp.close()

    def test_chunks_do_not_change_gaps(self, tmp_path):
        exp = self._load(tmp_path, chunk_size=1)
        assert _gaps(exp.raw["chr1"]) == ([[15, 60]], [0, 1, 1])
        exp.close()

    def test_multi_file_and_save_load(self, tmp_path):
        chromsize, f = _extent_setup(tmp_path, _gap_rows(), header=_GAP_HEADER)
        ctrl = [
            ("u0", 0, 5, "chr1", "+", 0.1, "a"),
            ("u0", 3, 70, "chr1", "+", 0.1, "a"),
        ]
        _, uf = _extent_setup(tmp_path, ctrl, header=_GAP_HEADER, name="u")
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, unmeth_file=uf
        )
        saved = tmp_path / "exp.h5"
        exp.save(saved)
        exp.close()
        reloaded = MethPrintExperiment.load(saved)
        raw = reloaded.raw["chr1"]
        assert _gaps(raw) == ([[15, 60]], [0, 1, 1])
        assert _gaps(raw, "unmeth") == ([[6, 70]], [0, 1])
        assert raw.meth is None
        reloaded.close()


def _bed_line(region, start, end, name, sizes, starts):
    return (
        f"{region}\t{start}\t{end}\t{name}\t60\t+\t{start}\t{end}\t"
        f"0\t{len(sizes)}\t{','.join(map(str, sizes))},\t"
        f"{','.join(map(str, starts))},\n"
    )


def _write_bed(path, lines):
    with open(path, "w") as f:
        f.writelines(_bed_line(*line) for line in lines)
    return path


class TestGapsBed:
    def _load(self, tmp_path, bed_lines, rows=None, header=_GAP_HEADER, **kw):
        chromsize, f = _extent_setup(
            tmp_path, rows or _gap_rows(), header=header
        )
        bed = _write_bed(tmp_path / "reads.bed", bed_lines)
        return MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=f, gaps_bed=bed, **kw
        )

    def test_listed_molecule_uses_bed_blocks(self, tmp_path):
        lines = [
            ("chr1", 0, 70, "m0", [10, 5], [0, 65]),
            ("chr1", 20, 120, "m1", [10, 30], [0, 70]),
        ]
        exp = self._load(tmp_path, lines)
        assert _gaps(exp.raw["chr1"]) == ([[10, 65], [30, 90]], [0, 1, 2])
        exp.close()

    def test_unlisted_molecule_keeps_inferred_gaps(self, tmp_path):
        lines = [("chr1", 20, 120, "m1", [10, 30], [0, 70])]
        with pytest.warns(UserWarning, match="inferred from rows"):
            exp = self._load(tmp_path, lines)
        assert _gaps(exp.raw["chr1"]) == ([[15, 60], [30, 90]], [0, 1, 2])
        exp.close()

    def test_short_bed_gap_ignored(self, tmp_path):
        lines = [
            ("chr1", 0, 70, "m0", [10, 50], [0, 20]),
            ("chr1", 20, 120, "m1", [10, 30], [0, 70]),
        ]
        exp = self._load(tmp_path, lines)
        assert _gaps(exp.raw["chr1"]) == ([[30, 90]], [0, 0, 1])
        exp.close()

    def test_multi_line_molecule_counts_as_unlisted(self, tmp_path):
        lines = [
            ("chr1", 0, 70, "m0", [10, 5], [0, 65]),
            ("chr1", 20, 120, "m1", [10, 30], [0, 70]),
            ("chr1", 20, 60, "m1", [40], [0]),
        ]
        with pytest.warns(UserWarning, match="1 listed more than once"):
            exp = self._load(tmp_path, lines)
        assert _gaps(exp.raw["chr1"]) == ([[10, 65]], [0, 1, 1])
        exp.close()

    def test_without_read_position_column(self, tmp_path):
        rows = [r[:1] + r[2:] for r in _gap_rows()]
        lines = [("chr1", 20, 120, "m1", [10, 30], [0, 70])]
        with pytest.warns(UserWarning, match="not recorded"):
            exp = self._load(tmp_path, lines, rows=rows, header=_HEADER)
        assert _gaps(exp.raw["chr1"]) == ([[30, 90]], [0, 0, 1])
        exp.close()

    @pytest.mark.parametrize(
        "kw", [{"wrap": True}, {"min_gap": None}], ids=["wrap", "no_gap"]
    )
    def test_invalid_combination_raises(self, tmp_path, kw):
        with pytest.raises(ValueError, match="gaps_bed requires"):
            self._load(tmp_path, [], **kw)
