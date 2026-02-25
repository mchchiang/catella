# methdata.py

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, List, Mapping, Dict, Self, Any
from ..containers import DataFrameMap, FixedKeyMap
from .. import utils
from .. import h5_utils
import numpy as np
import pandas as pd
import h5py

@utils.add_frozen_properties
@dataclass(frozen=True, slots=True, init=False)
class MethPrintData:
    """
    Container for MethPrint experimental data for a single chromosome.

    This class stores methylation signal data for test samples and optional 
    control samples (unmethylated and fully methylated). Data are stored 
    as Pandas DataFrames and retrieved via read-only properties that 
    provide defensive copies.

    .. note::
       This class is intended for internal use within a 
       :class:`MethPrintExperiment`. Use the experiment's loading 
       mechanisms rather than instantiating this class directly.
    """
    
    chrom : str
    """The chromosome identifier for this data block."""
    
    nbp : int
    """The total number of base pairs in the chromatin fiber."""
    
    _frozen_test_mol_id : np.ndarray = field(
        metadata={"doc": "np.ndarray: Array of molecule identifiers for the "
                  "test samples."})
    
    _frozen_test_data : pd.DataFrame = field(
        metadata={"doc": "pd.DataFrame: Raw methylation quality scores for "
                  "test samples."})
    
    _frozen_unmeth_mol_id : np.ndarray | None = field(
        metadata={"doc": "np.ndarray: Array of molecule identifiers for "
                  "unmethylated controls."})
    
    _frozen_unmeth_data : pd.DataFrame | None = field(
        metadata={"doc": "pd.DataFrame: Raw methylation quality scores for "
                  "unmethylated controls."})
    
    _frozen_meth_mol_id : np.ndarray | None = field(
        metadata={"doc": "np.ndarray: Array of molecule identifiers for "
                  "methylated controls."})
    
    _frozen_meth_data : pd.DataFrame | None = field(
        metadata={"doc": "pd.DataFrame: Raw methylation quality scores for "
                  "methylated controls."})

    def __init__(self, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use MethyPrintData._load() to instantiate ",
                            "this class.")
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is None:
                val = kwargs.get(f.name.removeprefix("_frozen_"))
            object.__setattr__(self, f.name, val)
    
    def _save(self, group):
        dt = h5py.string_dtype(encoding="utf-8") # For storing strings        
        gchrom = group.create_group(self.chrom)
        gmeta = gchrom.create_group("metadata")
        gmeta.attrs["chrom"] = self.chrom
        gmeta.attrs["nbp"] = self.nbp            
        gdata = gchrom.create_group("data")
        def save_data(name, mol_id, df, gdata):
            if mol_id is not None and df is not None:
                gdata.create_dataset(name+"_mol_id", data=mol_id, dtype=dt)
                h5_utils.save_df(name+"_data", df, gdata)
        save_data("test", self._frozen_test_mol_id,
                  self._frozen_test_data, gdata)
        save_data("unmeth", self._frozen_unmeth_mol_id,
                  self._frozen_unmeth_data, gdata)
        save_data("meth", self._frozen_meth_mol_id,
                  self._frozen_meth_data, gdata)        
        
    @classmethod
    def _load(cls, group, chrom) -> Self:
        if chrom not in group:
            raise ValueError(f"Cannot find data for the chrom {chrom}.")
        gchrom = group[chrom]
        gmeta = gchrom["metadata"]
        chrom = gmeta.attrs["chrom"]
        nbp = int(gmeta.attrs["nbp"])
        def load_data(name, gdata):
            id_name = name+"_mol_id"
            data_name = name+"_data"
            if id_name in gdata and data_name in gdata:
                mol_id = gdata[id_name].asstr()[()]
                df = h5_utils.load_df(data_name, gdata)
                return mol_id, df
            return None, None
        gdata = gchrom["data"]
        test_mol_id, test_data = load_data("test", gdata)        
        unmeth_mol_id, unmeth_data = load_data("unmeth", gdata)
        meth_mol_id, meth_data = load_data("meth", gdata)
        return cls._create(chrom=chrom, nbp=nbp, test_mol_id=test_mol_id,
                           test_data=test_data, unmeth_mol_id=unmeth_mol_id,
                           unmeth_data=unmeth_data, meth_mol_id=meth_mol_id,
                           meth_data=meth_data)
    
    @classmethod
    def _create(cls, **kwargs) -> Self:
        # Some validation
        if kwargs.get("nbp") <= 0:
            raise ValueError("nbp must be positive")
        return cls(**kwargs, _internal=True)

    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [attr for attr in self.__slots__ if
                        not attr.startswith('_')]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith('_'):
                public_attrs.append(attr)        
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str


@dataclass(slots=True, init=False)
class MethPrintExperiment:
    """
    Manager for MethPrint experimental data and analysis results.

    This class provides tools to load raw methylation signals from sequencing 
    files, manage control datasets (unmethylated and methylated, which are
    optional), and store the results of normalization and downstream analyses.

    .. note::
       Direct instantiation is disabled. Use :meth:`load_raw` to process new 
       sequencing data or :meth:`load` to open an existing HDF5 dataset.
    """
    
    _raw_data : Mapping[str,MethPrintData]
    _analysis : Mapping[str,DataFrameMap]
    _global_analysis : DataFrameMap

    def __init__(self, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use MethPrintExperiment.load() or .load_raw()",
                            "to instantiate this class")

        # Bulk assigmnet of fields
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is not None:
                setattr(self, f.name, val)

        # Create the dicts for analysis
        self._analysis = {chrom : DataFrameMap()
                          for chrom in self._raw_data.keys()}
        self._global_analysis = DataFrameMap()
    
        
    @classmethod
    def load_raw(cls,
                 chromsize : str | Path,
                 test_file : str | Path,
                 unmeth_file : str | Path | None = None,
                 meth_file : str | Path | None = None,
                 wrap : bool = False,
                 colidx : List | None = None) -> Self:
        """
        Create an experiment by processing raw sequencing data files.

        This method reads chromosome sizes and experimental data (typically 
        modkit CSV output), re-orients the data if requested, and splits 
        the signals by chromosome.

        Parameters
        ----------
        chromsize : str or pathlib.Path
            Path to a tab-separated file containing chromosome names and 
            their lengths (bp).
        test_file : str or pathlib.Path
            Path to the raw experimental (test) data file.
        unmeth_file : str or pathlib.Path, optional
            Path to the unmethylated control data file.
        meth_file : str or pathlib.Path, optional
            Path to the fully methylated control data file.
        wrap : bool, default False
            If True, calculates positions relative to the fiber center 
            (useful for circular or symmetrical fibers).
        colidx : list of int, optional
            Specific column indices to use if the input file does not follow 
            the standard modkit format.

        Returns
        -------
        MethPrintExperiment
            A newly initialized experiment object containing the processed
            data.

        Raises
        ------
        ValueError
            If chromosome names are not unique or if control datasets contain 
            chromosomes not found in the test dataset.
        """        
        
        # Read chromosome sizes
        df_size = pd.read_csv(chromsize, sep="\t")
        if "chrom" not in df_size.columns:
            raise ValueError("Column 'chrom' containing the chromosome "
                             "identifier is missing.")
        if "length" not in df_size.columns:
            raise ValueError("Column 'length' containing the chromosome "
                             "length is missing.")        
        if not df_size["chrom"].is_unique:
            raise ValueError("Chromosomes must be unique in chromsize file")
        

        # Adjust the chromosome size if wrapped
        if wrap:
            sizes = dict(zip(df_size["chrom"],
                             (df_size["length"]/2).astype(int)))
        else:
            sizes = dict(zip(df_size["chrom"], df_size["length"].astype(int)))
        
        # Read the raw data as a data frame
        def read_data(data_file, df_size, wrap=False, colidx=None, sep="\t"):
            """
            Internal parser for raw methylation sequencing files.

            This helper handles the conversion from Modkit-style CSV/TSV 
            formats into the internal DataFrame structure, performs 
            coordinate re-orientation, and indexes molecules.

            Parameters
            ----------
            data_file : str or pathlib.Path
                The raw data file to be parsed.
            df_size : pd.DataFrame
                A lookup table containing 'chrom' and 'length' columns.
            wrap : bool, default False
                If True, positions are 'wrapped' to represent distance from 
                the fiber center rather than absolute genomic coordinates.
            colidx : list of int, optional
                Explicit column indices if the file format deviates from 
                standard Modkit output.
            sep : str, default "\\t"
                The delimiter used in the input file.

            Returns
            -------
            tuple of (dict, dict)
                A pair of mappings: (mol_ids, dataframes), both keyed by 
                chromosome name.

            Notes
            -----
            The function maps Modkit columns to internal identifiers:
            
            * ``read_id`` -> ``mol_id``
            * ``ref_position`` -> ``upos`` (unprocessed position)
            * ``mod_qual`` -> Methylation quality score
            
            It also generates a zero-based ``mol_index`` for each chromosome 
            to allow for efficient array-based downstream analysis.
            """
            
            # Column names from modkit documentation
            modkit_colnames = ["read_id", "ref_position", "chrom", "mod_qual",
                               "mod_code"]
            colnames = ["mol_id", "upos", "chrom", "mod_qual", "mod_code"]
            usecols = lambda c : c.lstrip("#").strip() in modkit_colnames
            print(f"Reading {data_file} ...")
            if colidx is None:
                df = pd.read_csv(data_file, sep=sep, usecols=usecols)
            else:
                df = pd.read_csv(data_file, header=None, sep=sep,
                                 usecols=colidx)
            df.columns = colnames
            # Add chromosome length column
            df = pd.merge(df, df_size, on="chrom") 
            if wrap:
                df["pos"] = np.where(df["upos"] > (df["length"]-1)/2,
                                     df["length"]-df["upos"]-1, df["upos"])
            else:
                df["pos"] = df["upos"]
            
            # Split the data by chromosomes
            chroms = df["chrom"].unique()
            dfs = {c:df[df["chrom"] == c] for c in chroms}

            # Sort by position. Note that each molecule only has one chrom
            mol_ids = dict()
            for c in chroms:
                dfs[c] = dfs[c].sort_values(["mol_id","pos"]).reset_index()
                mol_ids[c] = dfs[c]["mol_id"].unique()
                id2idx = {rid:i for i,rid in enumerate(mol_ids[c])}
                dfs[c]["mol_index"] = dfs[c]["mol_id"].map(id2idx)
                dfs[c] = dfs[c][["mol_index","pos","mod_qual","mod_code"]]
            return mol_ids, dfs
        
        # Read footprinting data and re-orientate the data with pos as index
        # and mod_qual score at each pos for each molecule as columns        
        test_ids, dfs_test = read_data(test_file, df_size, wrap, colidx)
        chroms = dfs_test.keys()

        unmeth_ids = None
        meth_ids = None
        dfs_unmeth = None
        dfs_meth = None
        if unmeth_file is not None:
            unmeth_ids, dfs_unmeth = read_data(unmeth_file, df_size, wrap,
                                               colidx)
            if set(chroms) != set(dfs_unmeth.keys()):
                raise ValueError("Different number of chromosomes in test and"
                                 "unmeth datasets")            
        if meth_file is not None:
            meth_ids, dfs_meth = read_data(meth_file, df_size, wrap, colidx)
            if set(chroms) != set(dfs_meth.keys()):
                raise ValueError("Different number of chromosomes in test and"
                                 "meth datasets")

        # Create the raw data object for each chromosome
        raw_data = {}
        for chrom in chroms:
            raw_data[chrom] = MethPrintData._create(
                chrom=chrom, nbp=sizes[chrom], test_mol_id=test_ids[chrom],
                test_data=dfs_test[chrom], unmeth_mol_id=unmeth_ids[chrom],
                unmeth_data=dfs_unmeth[chrom], meth_mol_id=meth_ids[chrom],
                meth_data=dfs_meth[chrom])
        return cls._create(_raw_data=raw_data)
                
    def save(self, path: str | Path):
        """
        Save the experiment data and analysis to an HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            The output file path. Raw data is only written if it does not 
            already exist in the file.

        Raises
        ------
        OSError
            If the file cannot be written to disk.
        """
        with h5py.File(path, "a") as h5stream:
            # Save the raw data - write once if raw_data does not exist
            if not "raw_data" in h5stream:
                graw = h5stream.create_group("raw_data")
                for chrom, rdata in self._raw_data.items():
                    rdata._save(graw)

            # Save any analysis data
            if "analysis" in h5stream: del h5stream["analysis"]
            gana = h5stream.create_group("analysis")
            for chrom, data in self._analysis.items():
                gchrom = gana.create_group(chrom)
                for name, df in data.items():
                    h5_utils.save_df(name, df, gchrom)
            if "global_analysis" in h5stream: del h5stream["global_analysis"]
            gana = h5stream.create_group("global_analysis")
            for name, df in self._global_analysis.items():
                h5_utils.save_df(name, df, gana)
                
    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load an experiment from a persistent HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the HDF5 file containing the experiment.

        Returns
        -------
        MethPrintExperiment
            The loaded experiment object with all data and analysis maps.
        """
        with h5py.File(path, "r") as h5stream:
            # Load the raw data
            graw = h5stream["raw_data"]
            raw_data = {}
            for chrom in graw:
                raw_data[chrom] = MethPrintData._load(graw, chrom)
            obj = cls._create(_raw_data=raw_data)
            
            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                gchrom = gana[chrom]
                for name in gchrom:
                    obj._analysis[chrom][name] = h5_utils.load_df(name, gchrom)
            gana = h5stream["global_analysis"]
            for name in gana:
                obj._global_analysis[name] = h5_utils.load_df(name, gana)
            return obj
                
    @classmethod
    def _create(cls, **kwargs) -> Self:
        return cls(**kwargs, _internal=True)

    @property
    def chroms(self) -> Tuple[str,...]:
        """
        Get the identifiers of all chromosomes in the experiment.

        Returns
        -------
        tuple of str
            A sorted tuple of chromosome names.
        """
        return tuple(self._raw_data.keys())

    @property
    def raw(self) -> Mapping[str,MethPrintData]:
        """
        Provide read-only access to the raw experimental data.

        Returns
        -------
        MappingProxyType
            A frozen mapping where keys are chromosome names and values 
            are :class:`MethPrintData` objects.
        """
        return MappingProxyType(self._raw_data)

    # Allow iterating over the raw data by chromosome
    def __iter__(self):
        return iter(self._raw_data)

    def __len__(self):
        return len(self._raw_data)

    @property
    def analysis(self) -> Mapping[str, DataFrameMap]:
        """
        Access the analysis results for each chromosome.
        
        Returns
        -------
        FixedKeyMap
            A mapping where chromosome keys are fixed, but the analysis 
            DataFrames remain mutable.
        """
        return FixedKeyMap(self._analysis)
    
    @property
    def global_analysis(self) -> DataFrameMap:
        """
        Access analysis results aggregated across the entire experiment.

        Returns
        -------
        DataFrameMap
            A mutable map containing global analysis results.
        """
        return self._global_analysis
    
    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [attr for attr in self.__slots__ if
                        not attr.startswith('_')]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith('_'):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str
