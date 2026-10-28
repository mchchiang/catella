# config.py

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

from catella_cpp import NucPosMCMC


# Store parameters relevant for a batch of runs
@dataclass(frozen=True, slots=True, kw_only=True)
class SimSettings:
    """
    Configuration for the physical model and simulation protocol.

    This class encapsulates the 'recipe' for a simulation batch, ensuring
    reproducibility by grouping physical constants and annealing
    schedules. Shared by both engines (see `SimManager.run`'s `method`
    parameter); the MCMC-only protocol fields (`start_temp`,
    `cool_option`, `nsweep`, `print_freq`) are optional and unused for
    method='tm'.
    """

    nucbp: int
    """The number of base pairs occupied by a single nucleosome."""

    llink: int
    """The DNA linker length between nucleosomes."""

    elink: float
    """The energy scale of the linker repulsion potential."""

    mu: float
    """The chemical potential; energy gained by adding a nucleosome."""

    end_temp: float
    """Ending temperature for the simulation annealing (method='mcmc'),
    or the temperature the exact equilibrium distribution is evaluated
    at (method='tm')."""

    start_temp: float | None = None
    """Starting temperature for the simulation annealing. Required for
    method='mcmc'; unused for method='tm'."""

    cool_option: str | None = None
    """The protocol for updating the temperature in the simulation from
    the start value to the end value. Required for method='mcmc';
    unused for method='tm'."""

    nsweep: int | None = None
    """Number of Monte Carlo sweeps to perform per simulation. Required
    for method='mcmc'; unused for method='tm'."""

    print_freq: int | None = None
    """Frequency (in sweeps) at which to record simulation data.
    Required for method='mcmc'; unused for method='tm'."""

    emax: float
    """The maximum methylation energy (in $k_BT$)."""

    _cool_map: ClassVar[dict[str, NucPosMCMC.Cooling]] = {
        "linear": NucPosMCMC.Cooling.Linear,
        "geometric": NucPosMCMC.Cooling.Geometric,
        "constant": NucPosMCMC.Cooling.Constant,
    }

    @classmethod
    def load(cls, config_file: str | Path) -> Self:
        """
        Load simulation settings from a configuration file.

        Parameters
        ----------
        config_file : str or Path
            The directory of the configuration file. Supported formats include
            JSON (.json) and YAML (.yaml, .yml).

        Returns
        -------
        SimSettings
            A new instance of the settings dataclass populated with values
            from the file.

        Raises
        ------
        FileNotFoundError
            If the specified configuration file does not exist.
        ValueError
            If the file extension is unsupported or the file content
            cannot be parsed.
        TypeError
            If the file contains keys that do not match the expected
            SimSettings attributes.


        .. note::

           The input file must be a flat mapping where keys correspond
           exactly to the field names defined in this class (e.g., 'nucbp',
           'mu'). YAML support requires the `PyYAML` package to be installed.
        """
        config_file = Path(config_file)
        with config_file.open("r") as f:
            if config_file.suffix == ".json":
                import json

                data = json.load(f)
            elif config_file.suffix in (".yaml", ".yml"):
                import yaml

                data = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"unsupported file format: {config_file.suffix}"
                )
        return cls(**data)
