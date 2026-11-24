# test_dropout_masks.py

from types import SimpleNamespace

import numpy as np
import pytest
from catella.experiment._contexts import (
    CG,
    GC,
    GCG,
    GCH,
    HCG,
    A,
    _channel_site_masks,
    _merge_contexts,
    _reference_contexts,
    _strand_site_masks,
)
from catella.experiment._ingest import _methylatable_positions


def _idx(mask):
    return np.flatnonzero(mask).tolist()


def _masks(seq, codes):
    sites = {
        s: np.logical_or.reduce(
            [_methylatable_positions(seq, m, s) for m in ("A", "CG", "GC")]
        )
        for s in ("+", "-")
    }
    ctx = _merge_contexts(_reference_contexts(seq), codes)
    return _channel_site_masks(sites, ctx, codes)


def _exp(seq, mtase, wrap=False, nbp=None):
    raw = SimpleNamespace(refseq=seq, nbp=nbp or len(seq))
    return SimpleNamespace(mtase=mtase, wrap=wrap, raw={"r": raw})


class TestChannelSiteMasks:
    def test_keys_are_channel_names(self):
        masks = _masks("ACGTGCA", (A, GCH, HCG, GCG))
        assert list(masks) == ["A", "GCH", "HCG", "GCG"]

    def test_masks_have_sequence_length(self):
        seq = "ACGTGCAGCGC"
        for plus, minus in _masks(seq, (A, GCH)).values():
            assert plus.shape == (len(seq),)
            assert minus.shape == (len(seq),)
            assert plus.dtype == bool

    def test_a_channel_plus_is_a_minus_is_t(self):
        plus, minus = _masks("AATGCTTA", (A,))["A"]
        assert _idx(plus) == [0, 1, 7]
        assert _idx(minus) == [2, 5, 6]

    def test_gch_sites(self):
        plus, minus = _masks("AGCA", (GCH,))["GCH"]
        assert _idx(plus) == [2]
        assert _idx(minus) == [1]

    def test_hcg_sites(self):
        plus, minus = _masks("ACGA", (HCG,))["HCG"]
        assert _idx(plus) == [1]
        assert _idx(minus) == [2]

    def test_gcg_and_gch_split(self):
        masks = _masks("GCGC", (GCH, GCG))
        assert _idx(masks["GCG"][0]) == [1]
        assert _idx(masks["GCG"][1]) == [2]
        assert _idx(masks["GCH"][0]) == [3]
        assert _idx(masks["GCH"][1]) == [0]

    def test_channels_are_disjoint(self):
        rng = np.random.default_rng(0)
        seq = "".join(rng.choice(list("ACGT"), 500))
        masks = _masks(seq, (A, GCH, HCG, GCG))
        for s in (0, 1):
            total = sum(m[s].astype(int) for m in masks.values())
            assert total.max() <= 1

    @pytest.mark.parametrize(
        ("merged", "parts"), [(GC, ("GCH", "GCG")), (CG, ("HCG", "GCG"))]
    )
    def test_merged_channel_pools_parts(self, merged, parts):
        rng = np.random.default_rng(1)
        seq = "".join(rng.choice(list("ACGT"), 500))
        name = "GC" if merged == GC else "CG"
        got = _masks(seq, (merged,))[name]
        sep = _masks(seq, (GCH, HCG, GCG))
        for s in (0, 1):
            expected = sep[parts[0]][s] | sep[parts[1]][s]
            np.testing.assert_array_equal(got[s], expected)

    def test_merged_gc_has_equal_strand_counts(self):
        rng = np.random.default_rng(2)
        seq = "".join(rng.choice(list("ACGT"), 500))
        plus, minus = _masks(seq, (GC,))["GC"]
        assert plus.sum() == minus.sum()


class TestStrandSiteMasks:
    def test_only_experiment_mtase(self):
        sites = _strand_site_masks(_exp("ACGT", ("A",)), "r", 4)
        assert _idx(sites["+"]) == [0]
        assert _idx(sites["-"]) == [3]

    def test_all_labels_without_mtase(self):
        sites = _strand_site_masks(_exp("ACGT", None), "r", 4)
        assert _idx(sites["+"]) == [0, 1]
        assert _idx(sites["-"]) == [2, 3]

    def test_wrap_folds_both_halves(self):
        exp = _exp("AAAAAC", ("A",), wrap=True, nbp=3)
        sites = _strand_site_masks(exp, "r", 3)
        assert _idx(sites["+"]) == [0, 1, 2]
        assert _idx(sites["-"]) == []
