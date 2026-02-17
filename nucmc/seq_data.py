# seq_data.py

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import List, Mapping, Dict, Self, Any
from .util import DataFrameMap, FixedKeyMap
from . import util
import numpy as np
import pandas as pd
import inspect
import h5py

@util.copy_array_properties
@dataclass(frozen=True, slots=True, init=False)
class FiberSeqRawData:
    chrom : str
    nbp : int
    _test_mol_id : np.ndarray
    _test_data : pd.DataFrame
    _unmeth_mol_id : np.ndarray
    _unmeth_data : pd.DataFrame
    _meth_mol_id : np.ndarray
    _meth_data : pd.DataFrame

    def __init__(self, *args : Any, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use FiberSeqRawData._load() to instantiate",
                            "this class")
        sig = inspect.signature(self._create)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for key,val in bound.arguments.items():
            object.__setattr__(self, key, val)
    
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
                util.save_df(name+"_data", df, gdata)
        save_data("test", self._test_mol_id, self._test_data, gdata)
        save_data("unmeth", self._unmeth_mol_id, self._unmeth_data, gdata)
        save_data("meth", self._meth_mol_id, self._meth_data, gdata)        
        
    @classmethod
    def _load(cls, group, chrom):
        if chrom not in group:
            raise ValueError(f"Cannot find data for the chrom {chrom}")
        gchrom = group[chrom]
        gmeta = gchrom["metadata"]
        chrom = gmeta.attrs["chrom"]
        nbp = int(gmeta.attrs["nbp"])
        def load_data(name, gdata):
            id_name = name+"_mol_id"
            data_name = name+"_data"
            if id_name in gdata and data_name in gdata:
                mol_id = gdata[id_name].asstr()[()]
                df = util.load_df(data_name, gdata)
                return mol_id, df
            return None, None
        gdata = gchrom["data"]
        test_mol_id, test_data = load_data("test", gdata)        
        unmeth_mol_id, unmeth_data = load_data("unmeth", gdata)
        meth_mol_id, meth_data = load_data("meth", gdata)
        return cls._create(chrom, nbp, test_mol_id, test_data, unmeth_mol_id,
                           unmeth_data, meth_mol_id, meth_data)
    
    @classmethod
    def _create(cls, chrom, nbp, _test_mol_id, _test_data,
                _unmeth_mol_id=None, _unmeth_data=None, _meth_mol_id=None,
                _meth_data=None) -> Self:
        # Some validation
        if nbp <= 0:
            raise ValueError("nbp must be positive")
        return cls(chrom, nbp, _test_mol_id, _test_data, _unmeth_mol_id,
                   _unmeth_data, _meth_mol_id, _meth_data, _internal=True)

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
class FiberSeqExperiment:
    _raw_data : Mapping[str,FiberSeqRawData]
    _analysis : Mapping[str,DataFrameMap]
    _global_analysis : DataFrameMap

    def __init__(self, *args : Any, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use FiberSeqExperiment.load() or .load_raw()",
                            "to instantiate this class")
        sig = inspect.signature(self._create)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for key,val in bound.arguments.items():
            setattr(self, key, val)

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
                 colidx : List | None = None):
    
        # Read chromosome sizes
        df_size = pd.read_csv(chromsize, header=None, sep="\t",
                              names=["chrom", "length"])
        if not df_size["chrom"].is_unique:
            raise ValueError("Chromosomes must be unique in chromsize file")
        sizes = dict(zip(df_size["chrom"], df_size["length"]))

        # Read the raw data as a data frame
        def read_data(data_file, df_size, wrap=False, colidx=None, sep="\t"):
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
        
        # Read fiber-seq data and re-orientate the data with pos as index and
        # mod_qual score at each pos for each molecule as columns        
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
        for c in chroms:
            raw_data[c] = FiberSeqRawData._create(
                c, sizes[c], test_ids[c], dfs_test[c], unmeth_ids[c],
                dfs_unmeth[c], meth_ids[c], dfs_meth[c])
        return cls._create(raw_data)
                
    def save(self, path: str | Path):
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
                    util.save_df(name, df, gchrom)
            if "global_analysis" in h5stream: del h5stream["global_analysis"]
            gana = h5stream.create_group("global_analysis")
            for name, df in self._global_analysis.items():
                util.save_df(name, df, gana)
                
    @classmethod
    def load(cls, path: str | Path):
        with h5py.File(path, "r") as h5stream:
            # Load the raw data
            graw = h5stream["raw_data"]
            raw_data = {}
            for chrom in graw:
                raw_data[chrom] = FiberSeqRawData._load(graw, chrom)
            obj = cls._create(raw_data)
            
            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                gchrom = gana[chrom]
                for name in gchrom:
                    obj._analysis[chrom][name] = util.load_df(name, gchrom)
            gana = h5stream["global_analysis"]
            for name in gana:
                obj._global_analysis[name] = util.load_df(name, gana)
            return obj
                
    @classmethod
    def _create(cls, _raw_data):
        return cls(_raw_data, _internal=True)

    @property
    def chroms(self):
        return self._raw_data.keys()

    @property
    def raw(self) -> Dict[str,FiberSeqRawData]:
        # Make sure the raw data map is immutable
        return MappingProxyType(self._raw_data)

    # Allow iterating over the raw data by chromosome
    def __iter__(self):
        return iter(self._raw_data)

    def __len__(self):
        return len(self._raw_data)

    @property
    def analysis(self) -> Mapping[str, DataFrameMap]:
        return FixedKeyMap(self._analysis)
    
    @property
    def global_analysis(self) -> DataFrameMap:
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
