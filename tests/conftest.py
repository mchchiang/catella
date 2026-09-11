# conftest.py

import matplotlib.pyplot as plt
import pytest


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
