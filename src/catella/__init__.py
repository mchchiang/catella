# __init__.py

import warnings

# catella_cpp doesn't declare free-threading support, so Python 3.13+
# auto-enables the GIL on import and warns; suppress the warning.
warnings.filterwarnings(
    "ignore",
    message=".*global interpreter lock.*",
    category=RuntimeWarning,
)

from catella.api import *

__all__ = [
    "analyze",
    "compute_empirical_prob",
    "compute_model_prob",
    "downsample",
    "estimate_start_temp",
    "filter_dropout",
    "load_raw",
    "plot_dropout_ecdf",
    "plot_energy",
    "plot_meth_energy",
    "plot_meth_prob",
    "plot_nuc_pos",
    "plot_occup",
    "run",
    "sort_by_linkage",
    "summarize_dropout",
]
