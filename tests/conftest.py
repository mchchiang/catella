# conftest.py

import tempfile

import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def isolated_tempdir(tmp_path_factory, monkeypatch):
    """Point `tempfile.gettempdir` at a fresh per-test directory.

    Tests that check for leftover scratch directories compare the
    contents of the temp directory before and after a call. Under
    pytest-xdist, other workers create scratch directories in the
    shared system temp directory concurrently, so such tests need a
    private one.
    """
    path = tmp_path_factory.mktemp("tempdir")
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test.

    Runs for every test automatically, so figures left open by
    plotting functions (which do not close their own figures) do not
    accumulate across the session and trigger matplotlib's
    too-many-open-figures warning.
    """
    yield
    plt.close("all")
