# catella

**catella** is a Python/C++ package for predicting nucleosome positions
from single-molecule DNA methylation footprinting data (e.g., fiber-seq).
It combines a biophysical model of nucleosome arrays with a statistical
model of the footprinting assay, and samples the resulting posterior
over nucleosome configurations using either an exact transfer matrix
(TM) solver or Markov Chain Monte Carlo (MCMC) simulations.

## Overview

Given a table of methylation calls per molecule (such as ModKit output),
catella:

- loads and quality-controls the raw footprinting data;
- estimates, per site, the probability that it is occupied by a
  nucleosome, either from a statistical emission model or from an
  empirical, control-normalized signal;
- samples likely nucleosome configurations for each molecule, using a
  performance-critical C++ engine (`catella_cpp`, exposed via pybind11)
  that offers two methods: an exact transfer matrix solver (`tm`), or
  an MCMC simulation (`mcmc`); and
- provides analysis and plotting utilities for the resulting samples.

The theoretical background is described in full in the
[model documentation](https://mchchiang.github.io/catella/model.html).

## Installation

catella requires a C++ compiler in addition to Python, so the
recommended way to install it is via the provided Conda environment,
which manages both.

```bash
git clone https://github.com/mchchiang/catella.git
cd catella
conda env create -f environment.yml
conda activate catella
```

This installs catella in editable mode, which triggers a
scikit-build-core/CMake build of the `catella_cpp` extension. Verify
the installation with:

```bash
catella --help
```

See the
[installation guide](https://mchchiang.github.io/catella/install.html)
for further details.

## Quick Start

catella can be used either as a Python library or from the command
line. The
[example pipeline](https://mchchiang.github.io/catella/example_pipeline.html)
walks through a full run, from raw methylation calls to nucleosome
position predictions, and the
[tutorials](https://mchchiang.github.io/catella/tutorials/tut00_data_prep.html)
cover each stage in more depth.

```python
import catella
from catella.simulation.config import SimSettings

# Load methylation calls and compute methylation probabilities
exp = catella.load_raw(chromsize="segments.size",
                       test_file="chromatin.tsv.gz",
                       fasta_file="segments.fa",
                       mtase=["A", "GC"])
exp = catella.compute_model_prob(exp)

# Draw 100 exact samples per molecule with the transfer matrix solver
settings = SimSettings(nucbp=147, llink=50, elink=25.0, mu=0.0,
                       end_temp=1.0, emax=25.0)
meth_prob = {c: exp.analysis[c]["meth_prob"] for c in exp.chroms}
sim = catella.run(chroms=exp.chroms,
                  nsim=100,
                  method="tm",
                  settings=settings,
                  meth_prob=meth_prob,
                  out_dir="tm_result",
                  seed=1234)

# Compute occupancy and mean nucleosome count for each molecule
catella.analyze(sim)
occup = sim.analysis["601"]["occup"]

# Nucleosome positions in sample 0 of molecule 0
positions = sim.raw["601", 0, 0].position
```

Here `"601"` is a chromosome named in `segments.size`. The same
operations are available as CLI subcommands, e.g., `catella load_raw`,
`catella compute_model_prob`, and `catella run`; run `catella --help`
for the full list.

## Documentation

The full documentation, including the model derivation, data format
reference, CLI and API reference, is available at
https://mchchiang.github.io/catella/.

## Citation

If you use catella in your research, please cite it as described in
[CITATION.cff](CITATION.cff).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
setting up a development environment and running the test suite.

## License

catella is distributed under the [MIT License](LICENSE).
