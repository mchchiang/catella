# test_parallel.py

import math
import operator
from types import SimpleNamespace

import pytest
from catella.simulation._parallel import (
    _job_label,
    _process_map,
    _progress_bar,
    _report_crash,
    _threaded_map,
)
from rich.console import Console


def _square(x):
    return x * x


def _fail_on_three(x):
    if x == 3:
        raise RuntimeError("boom")
    return x


class TestThreadedMap:
    def test_serial_in_order(self):
        out = list(_threaded_map(_square, [3, 1, 2]))
        assert out == [(3, 9), (1, 1), (2, 4)]

    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    def test_threaded_yields_every_item_once(self, batch_size):
        items = (i for i in range(7))
        out = list(
            _threaded_map(_square, items, nworker=3, batch_size=batch_size)
        )
        assert sorted(out) == [(i, i * i) for i in range(7)]

    def test_batches_follow_input_order(self):
        out = list(
            _threaded_map(_square, range(6), nworker=2, batch_size=2)
        )
        batches = [sorted(x for x, _ in out[i : i + 2]) for i in (0, 2, 4)]
        assert batches == [[0, 1], [2, 3], [4, 5]]

    def test_empty(self):
        assert list(_threaded_map(_square, [], nworker=2)) == []

    @pytest.mark.parametrize("nworker", [1, 2])
    def test_exception_propagates(self, nworker):
        with pytest.raises(RuntimeError, match="boom"):
            list(_threaded_map(_fail_on_three, range(5), nworker=nworker))


class TestProcessMap:
    def test_serial_in_order(self):
        out = list(_process_map(operator.neg, [3, 1, 2]))
        assert out == [(3, -3), (1, -1), (2, -2)]

    def test_workers_yield_every_item_once(self):
        items = (i for i in range(9))
        out = list(_process_map(operator.neg, items, nworker=2))
        assert sorted(out) == sorted((i, -i) for i in range(9))

    def test_on_crash_gets_in_flight_items(self):
        crashed = []
        with pytest.raises(ValueError):
            list(
                _process_map(
                    math.sqrt,
                    [-1.0],
                    nworker=2,
                    on_crash=crashed.append,
                )
            )
        assert crashed == [[-1.0]]

    def test_serial_does_not_call_on_crash(self):
        crashed = []
        with pytest.raises(ValueError):
            list(_process_map(math.sqrt, [-1.0], on_crash=crashed.append))
        assert crashed == []


class TestProgressBar:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_disabled_unless_verbose(self, verbose):
        progress = _progress_bar(verbose)
        assert progress.disable is (not verbose)


class TestJobLabel:
    def test_with_run(self):
        assert _job_label("chr1", 3, 0) == "chrom=chr1, mol=3, run=0"

    def test_without_run(self):
        assert _job_label("chr1", 3) == "chrom=chr1, mol=3"


class TestReportCrash:
    def test_prints_status_of_each_job(self):
        statuses = {("a", 0, 1): "missing", ("a", 1, 0): None}

        def classify(chrom, mol, run):
            if (chrom, mol, run) == ("b", 0, 0):
                raise OSError("unreadable")
            return statuses[(chrom, mol, run)]

        dataset = SimpleNamespace(_classify_run=classify)
        jobs = [
            SimpleNamespace(chrom="a", mol=0, run=1),
            SimpleNamespace(chrom="a", mol=1, run=0),
            SimpleNamespace(chrom="b", mol=0),
        ]
        console = Console(record=True, width=120)
        _report_crash(console, dataset, jobs)
        text = console.export_text()
        assert "Worker pool crashed" in text
        assert "chrom=a, mol=0, run=1: missing" in text
        assert "chrom=a, mol=1, run=0: ok" in text
        assert "chrom=b, mol=0: unknown" in text
