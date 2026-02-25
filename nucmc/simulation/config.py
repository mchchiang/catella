# config.pyA

from dataclasses import dataclass
from nucmc_cpp import NucPosModel
from pathlib import Path
from typing import Self

# Store parameters relevant for a batch of runs
@dataclass(frozen=True, slots=True, kw_only=True)
class SimSettings:
    """
    Configuration for the physical model and Monte Carlo simulation protocol.

    This class encapsulates the 'recipe' for a simulation batch, ensuring 
    reproducibility by grouping physical constants and annealing schedules.
    """
    
    nucbp : int
    """The number of base pairs occupied by a single nucleosome."""
    
    llink : int
    """The DNA linker length between nucleosomes."""
    
    mu : float
    """The chemical potential; energy gained by adding a nucleosome."""    

    start_temp : float
    """Starting temperature for the simulation annealing."""
    
    end_temp : float
    """Ending temperature for the simulation annealing."""
    
    cool_option : str
    """The protocol for updating the temperature in the simulation from the
    start value to the end value."""

    nsweep : int
    """Number of Monte Carlo sweeps to perform per simulation."""
    
    print_freq : int
    """Frequency (in sweeps) at which to record simulation data."""

    emax : float
    """The maximum methylation energy (in $k_BT$)."""

    _cool_map = {"linear" : NucPosModel.Cooling.Linear,
                 "geometric" : NucPosModel.Cooling.Geometric,
                 "constant" : NucPosModel.Cooling.Constant}
    
    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load simulation settings from a configuration file.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the configuration file. Supported formats include 
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
        path = Path(path)
        with path.open("r") as f:
            if path.suffix == ".json":
                import json
                data = json.load(f)
            elif path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        return cls(**data)
