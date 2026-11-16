# __init__.py

import warnings

# catella_cpp doesn't declare free-threading support, so Python 3.13+
# auto-enables the GIL on import and warns; suppress the warning.
warnings.filterwarnings(
    "ignore",
    message=".*global interpreter lock.*",
    category=RuntimeWarning,
)

from catella.__about__ import __version__ as __version__
from catella.api import *

__all__ = [
    "compute_config",
    "compute_empirical_prob",
    "compute_model_prob",
    "compute_occup",
    "downsample",
    "estimate_llink",
    "estimate_start_temp",
    "extract_config",
    "filter_dropout",
    "load_raw",
    "plot_dropout_ecdf",
    "plot_meth_energy",
    "plot_meth_prob",
    "plot_occup",
    "qc",
    "run",
    "sort_by_linkage",
]
