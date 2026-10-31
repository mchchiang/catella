# test_run_real.py

import gzip
import json

import callers
import catella
import numpy as np
import pandas as pd
import pytest
import run_real
import synthgen

COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(seq):
    return seq.translate(COMP)[::-1]


def _fibre_with_601(starts, length=1200, seed=0):
    seq = list(synthgen.random_sequence(length, seed=seed))
    core = _revcomp(run_real.CORE_601)
    for start in starts:
        seq[start : start + len(core)] = core
    return "".join(seq)


def _modkit_rows(read_id, doubled, rng, codes=("a", "m"), chrom="601"):
    rows = []
    for code in codes:
        mtase = run_real.MTASE_OF_CODE[code]
        for pos in synthgen.eligible_sites(doubled, mtase, "+"):
            rows.append((read_id, pos, chrom, "+", rng.random(), code))
    return rows


def _write_raw(raw, fibre, nread, seed=0):
    raw.mkdir(parents=True, exist_ok=True)
    doubled = fibre + _revcomp(fibre)
    synthgen.write_fasta({"601": doubled}, raw / "segments.fa")
    rng = np.random.default_rng(seed)
    cols = ["read_id", "ref_position", "chrom", "ref_strand", "mod_qual",
            "mod_code"]
    for sample in run_real.SAMPLES:
        rows = []
        for i in range(nread):
            rows += _modkit_rows(f"{sample}_{i}", doubled, rng)
        table = pd.DataFrame(rows, columns=cols)
        with gzip.open(raw / f"{sample}.tsv.gz", "wt") as fh:
            table.to_csv(fh, sep="\t", index=False)
    return doubled


class TestFind601Dyads:
    def test_both_orientations(self):
        seq = list(synthgen.random_sequence(1500, seed=1))
        seq[100:247] = run_real.CORE_601
        seq[700:847] = _revcomp(run_real.CORE_601)
        dyads = run_real.find_601_dyads("".join(seq))
        assert dyads.tolist() == [173, 773]

    def test_variant_beyond_anchor(self):
        core = list(_revcomp(run_real.CORE_601))
        core[105], core[106] = core[106], core[105]
        seq = list(synthgen.random_sequence(600, seed=2))
        seq[200:347] = core
        assert run_real.find_601_dyads("".join(seq)).tolist() == [273]

    def test_none(self):
        seq = synthgen.random_sequence(600, seed=3)
        assert run_real.find_601_dyads(seq).size == 0


class TestUnwrapCalls:
    def _calls(self, rows):
        return pd.DataFrame(
            rows,
            columns=["read_id", "ref_position", "ref_strand", "mod_qual",
                     "mod_code"],
        )

    def test_fold_and_filter(self):
        fibre = synthgen.random_sequence(200, seed=4)
        doubled = fibre + _revcomp(fibre)
        n = len(fibre)
        lower = int(synthgen.eligible_sites(doubled, "A", "+")[0])
        upper = int(synthgen.eligible_sites(doubled, "A", "+")[-1])
        wrong = int(np.flatnonzero(np.array(list(doubled)) == "C")[0])
        calls = self._calls(
            [
                ("r", lower, "+", 0.9, "a"),
                ("r", upper, "+", 0.2, "a"),
                ("r", wrong, "+", 0.5, "a"),
            ]
        )
        out = run_real.unwrap_calls(calls, doubled)
        assert out["pos"].tolist() == sorted([lower, 2 * n - 1 - upper])
        assert fibre[2 * n - 1 - upper] == "T"
        assert set(out["mod_qual"]) == {0.9, 0.2}

    def test_duplicates_averaged(self):
        fibre = synthgen.random_sequence(200, seed=5)
        doubled = fibre + _revcomp(fibre)
        pos = int(synthgen.eligible_sites(doubled, "A", "+")[0])
        calls = self._calls(
            [("r", pos, "+", 0.2, "a"), ("r", pos, "+", 0.6, "a")]
        )
        out = run_real.unwrap_calls(calls, doubled)
        assert out["mod_qual"].tolist() == pytest.approx([0.4])

    def test_junction_dropped(self):
        fibre = synthgen.random_sequence(200, seed=6)
        n = len(fibre)
        j = next(i for i in range(n) if fibre[i] in "CG")
        doubled = list(fibre + _revcomp(fibre))
        doubled[2 * n - 1 - j] = "A"
        doubled = "".join(doubled)
        calls = self._calls([("r", 2 * n - 1 - j, "+", 0.7, "a")])
        assert run_real.unwrap_calls(calls, doubled).empty

    def test_matches_catella_wrap(self, tmp_path):
        fibre = synthgen.random_sequence(300, seed=7)
        doubled = fibre + _revcomp(fibre)
        rng = np.random.default_rng(0)
        rows = []
        for i in range(3):
            rows += _modkit_rows(
                f"r{i}", doubled, rng, codes=("a",), chrom="fib"
            )
        cols = ["read_id", "ref_position", "chrom", "ref_strand",
                "mod_qual", "mod_code"]
        table = pd.DataFrame(rows, columns=cols)
        table.to_csv(tmp_path / "test.tsv", sep="\t", index=False)
        synthgen.write_fasta({"fib": doubled}, tmp_path / "ref.fa")
        synthgen.write_chromsize(
            {"fib": len(doubled)}, tmp_path / "ref.size"
        )
        exp = catella.load_raw(
            chromsize=tmp_path / "ref.size",
            test_file=tmp_path / "test.tsv",
            fasta_file=tmp_path / "ref.fa",
            mtase="A",
            wrap=True,
            tmp_dir=tmp_path,
        )
        raw = exp.raw["fib"]
        ids = np.asarray(raw.test_mol_id)
        theirs = raw.test_data.assign(
            mol_id=ids[raw.test_data["mol_index"].to_numpy(dtype=int)],
            pos=raw.test_data["pos"].astype(int),
        )
        ours = run_real.unwrap_calls(
            table.drop(columns="chrom"), doubled
        )
        key = ["mol_id", "pos"]
        theirs = theirs.sort_values(key)[key + ["mod_qual"]]
        ours = ours.sort_values(key)[key + ["mod_qual"]]
        np.testing.assert_array_equal(
            theirs[key].to_numpy(), ours[key].to_numpy()
        )
        np.testing.assert_allclose(
            theirs["mod_qual"].to_numpy(), ours["mod_qual"].to_numpy()
        )


class TestSelectMolecules:
    def _calls(self):
        return pd.DataFrame(
            {
                "mol_id": ["full"] * 2 + ["part"] * 2 + ["full2"] * 2,
                "pos": [0, 999, 0, 499, 5, 990],
                "mod_code": "a",
                "mod_qual": 0.5,
            }
        )

    def test_coverage_filter(self):
        ids = run_real.select_molecules(
            self._calls(), 1000, nmol=None, min_cover=0.8, seed=0
        )
        assert ids == ["full", "full2"]

    def test_nmol_deterministic(self):
        kwargs = {"nmol": 1, "min_cover": 0.0, "seed": 3}
        a = run_real.select_molecules(self._calls(), 1000, **kwargs)
        b = run_real.select_molecules(self._calls(), 1000, **kwargs)
        assert a == b and len(a) == 1

    def test_invalid_min_cover(self):
        with pytest.raises(ValueError):
            run_real.select_molecules(
                self._calls(), 1000, nmol=None, min_cover=1.5, seed=0
            )


class TestSplitMolecules:
    def test_disjoint_and_deterministic(self):
        ids = [f"m{i}" for i in range(11)]
        first, second = run_real.split_molecules(ids, seed=1)
        assert len(first) == 6 and len(second) == 5
        assert set(first).isdisjoint(second)
        assert set(first) | set(second) == set(ids)
        assert run_real.split_molecules(ids[::-1], seed=1) == (first, second)


class TestPrepare:
    def test_outputs(self, tmp_path):
        fibre = _fibre_with_601([100, 297, 494])
        _write_raw(tmp_path / "raw", fibre, nread=6)
        annotation = run_real.prepare(
            tmp_path / "raw", tmp_path / "out", nmol=4, convert=False
        )
        assert annotation["dyads"] == [173, 370, 567]
        assert annotation["fibre_length"] == len(fibre)
        assert all(len(v) == 4 for v in annotation["molecules"].values())
        calib = set(annotation["meth_calibration"])
        evaluation = set(annotation["meth_evaluation"])
        assert calib.isdisjoint(evaluation)
        assert calib | evaluation == set(annotation["molecules"]["methylated"])
        saved = json.loads((tmp_path / "out" / "annotation.json").read_text())
        assert saved == annotation
        for name, test_ids in (
            ("test", set(annotation["molecules"]["chromatin"])),
            ("meth_eval", evaluation),
        ):
            ddir = tmp_path / "out" / name
            test = pd.read_csv(ddir / "synth.tsv", sep="\t")
            meth = pd.read_csv(ddir / "synth.meth.tsv", sep="\t")
            assert set(test["read_id"]) == test_ids
            assert set(meth["read_id"]) == calib
            assert set(test["mod_code"]) == {"a", "m"}
            assert test["ref_position"].max() < len(fibre)
            sizes = pd.read_csv(ddir / "synth.size", sep="\t")
            assert sizes["length"].tolist() == [len(fibre)]
            for stem in ("synth", "synth.meth", "synth.unmeth"):
                assert (ddir / f"{stem}.sam").exists()
                assert (ddir / f"{stem}.gpc.sam").exists()


DYADS = np.array([300, 497, 694])


def _calls(centres_per_mol, nbp=1000):
    occ = np.zeros((len(centres_per_mol), nbp), dtype=np.float32)
    for row, centres in enumerate(centres_per_mol):
        for c in centres:
            occ[row, c - 73 : c + 74] = 1
    ids = np.array([f"m{i}" for i in range(len(centres_per_mol))])
    return callers.Calls(ids, occ, occ.sum(axis=1) / 147)


def _spans(n, lo=0, hi=999):
    index = pd.Index([f"m{i}" for i in range(n)], name="mol_id")
    return pd.DataFrame({"lo": lo, "hi": hi}, index=index)


class TestOccupancyRuns:
    def test_runs(self):
        occ = np.array([0, 1, 1, 0, 0.6, 0.6, 0.4, 1])
        runs = run_real.occupancy_runs(occ)
        assert runs.tolist() == [[1, 3], [4, 6], [7, 8]]


class TestMoleculeMetrics:
    def test_perfect_calls(self):
        table = run_real.molecule_metrics(
            _calls([DYADS.tolist()]), _spans(1), DYADS, trim=0
        )
        row = table.loc["m0"]
        assert row["n_calls"] == 3 and row["n_hits"] == 3
        assert row["core"] == 1 and row["linker"] == 0
        assert row["n_pairs"] == 2 and row["n_regular"] == 2
        assert row["kb"] == 1.0
        assert row["lengths"] == [147, 147, 147]

    def test_shifted_calls(self):
        table = run_real.molecule_metrics(
            _calls([(DYADS + 50).tolist()]), _spans(1), DYADS, trim=0
        )
        assert table.loc["m0", "n_hits"] == 0
        assert table.loc["m0", "linker"] > 0

    def test_trim_excludes_edge_calls(self):
        table = run_real.molecule_metrics(
            _calls([DYADS.tolist()]), _spans(1, lo=0, hi=799), DYADS,
            trim=250,
        )
        assert table.loc["m0", "n_calls"] == 2
        assert table.loc["m0", "kb"] == 0.3


class TestAggregate:
    def test_pooled_values(self):
        table = run_real.molecule_metrics(
            _calls([DYADS.tolist(), (DYADS + 50).tolist()]),
            _spans(2), DYADS, trim=0,
        )
        values = run_real.aggregate(table)
        assert values["precision"] == pytest.approx(0.5)
        assert values["false_per_kb"] == pytest.approx(3.0)
        assert values["spacing"] == pytest.approx(1.0)
        assert values["per_mol"] == pytest.approx(3.0)


class TestEquivalenceLabel:
    def test_labels(self):
        rng = np.random.default_rng(0)
        same = np.zeros(1000)
        up = rng.normal(0.1, 0.01, 1000)
        noisy = rng.normal(0.0, 0.1, 1000)
        assert run_real.equivalence_label(same, 0.02, True) == "equivalent"
        assert run_real.equivalence_label(up, 0.02, True) == "better"
        assert run_real.equivalence_label(up, 0.02, False) == "worse"
        assert run_real.equivalence_label(-up, 0.02, False) == "better"
        assert run_real.equivalence_label(noisy, 0.02, True) == (
            "inconclusive"
        )
        assert run_real.equivalence_label(
            np.array([np.nan]), 0.02, True
        ) == "inconclusive"


class TestBootstrap:
    def test_paired_resamples(self):
        table = run_real.molecule_metrics(
            _calls([DYADS.tolist(), (DYADS + 50).tolist()] * 5),
            _spans(10), DYADS, trim=0,
        )
        boots = run_real.bootstrap(
            {"a": table, "b": table.copy()}, nboot=50, seed=1
        )
        assert boots["a"].shape == (50, 5)
        pd.testing.assert_frame_equal(boots["a"], boots["b"])


class TestScore:
    def test_outputs_and_labels(self, tmp_path):
        nmol = 20
        (tmp_path / "annotation.json").write_text(
            json.dumps({"dyads": DYADS.tolist()})
        )
        for dataset in ("test", "meth_eval"):
            ddir = tmp_path / dataset
            ddir.mkdir()
            pd.DataFrame(
                {
                    "read_id": [f"m{i}" for i in range(nmol)] * 2,
                    "ref_position": [0] * nmol + [999] * nmol,
                }
            ).to_csv(ddir / "synth.tsv", sep="\t", index=False)
        good = _calls([DYADS.tolist()] * nmol)
        bad = _calls([(DYADS + 50).tolist()] * nmol)
        empty = _calls([[]] * nmol)
        for tool, test, meth in (("catella", good, empty), ("ft", bad, bad)):
            for dataset, calls in (("test", test), ("meth_eval", meth)):
                callers.save_calls(
                    calls,
                    tmp_path / "calls" / f"{tool}_tuned" / f"{dataset}.h5",
                )
        out = run_real.score(tmp_path, nboot=100)
        for name in ("real_results.tsv", "real_paired.tsv",
                     "real_summary.md"):
            assert (tmp_path / name).exists()
        paired = out["paired"].set_index(["dataset", "metric"])["label"]
        assert paired[("test", "precision")] == "better"
        assert paired[("test", "contrast")] == "better"
        assert paired[("meth_eval", "false_per_kb")] == "better"
        results = out["results"]
        assert set(results["tool"]) == {"catella", "ft"}
        assert set(results["variant"]) == {"tuned"}
