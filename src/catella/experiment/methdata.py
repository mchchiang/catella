# methdata.py

import os
import shutil
import tempfile
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field, fields
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import h5py
import numpy as np
import pandas as pd

from catella import h5_utils
from catella._analysis_io import (
    _analysis_group,
    _read_entries,
    _save_analysis,
)
from catella._batches import _row_batches
from catella._dataclasses import _add_frozen_properties, _init_frozen_fields
from catella._regions import (
    Region,
    _check_region_ends,
    _check_region_lengths,
    _check_region_names,
    _parse_regions,
)
from catella._warnings import _warn
from catella.containers import DataFrameMap, FixedKeyMap
from catella.experiment._calibration import _fold_contexts
from catella.experiment._contexts import (
    _channel_site_masks,
    _resolve_channels,
    _strand_site_masks,
)
from catella.experiment._dropout import (
    _apply_keep_mask,
    _dropout_fracs,
    _resolve_sources,
    _source_coverage,
)
from catella.experiment._ingest import (
    _PER_MOL_KEYS,
    _ingest_into,
    _ingest_sources,
    _keep_regions,
    _load_refseqs,
    _load_region_refseqs,
    _merge_staging,
    _normalize_mtase,
    _read_chromsize,
    _read_coords,
    _write_coords,
)
from catella.experiment._qc import _check_percentiles, _region_qc_rows
from catella.experiment._reads import _check_span
from catella.h5_array import H5Array

# Estimated peak memory of to_dense(as_h5array=False) per molecule and
# base pair: the pivot needs ~9x the 8-byte result (measured on 1000 and
# 4000 molecules of 5.6 kb)
_DENSE_BYTES_PER_CELL = 72

# to_dense(as_h5array=False) warns above this estimated peak memory
_DENSE_WARN_BYTES = 10**9


# Data sources of an experiment, in their canonical order
_SOURCE_NAMES = ("test", "unmeth", "meth")


@_add_frozen_properties
@dataclass(frozen=True, slots=True, init=False, eq=False)
class SourceData:
    """
    Raw molecules and calls of one data source (test, unmeth or meth).

    .. note::
       Built by `MethPrintData`; not meant to be created directly.
    """

    _frozen_mol_id: np.ndarray = field(
        metadata={"doc": "np.ndarray: Read id of each molecule."}
    )

    _frozen_data: pd.DataFrame = field(
        metadata={
            "doc": "pd.DataFrame: Calls, with columns `mol_index`, `pos`, "
            "`strand`, `mod_qual` and `mod_code`."
        }
    )

    _frozen_mol_start: np.ndarray | None = field(
        metadata={
            "doc": "np.ndarray: Start (bp) of each molecule's read "
            "extent, or None if not recorded."
        }
    )

    _frozen_mol_end: np.ndarray | None = field(
        metadata={
            "doc": "np.ndarray: End (bp, exclusive) of each molecule's "
            "read extent, or None if not recorded."
        }
    )

    _frozen_gaps: np.ndarray | None = field(
        metadata={
            "doc": "np.ndarray: Internal gaps, shape (n_gap, 2), grouped "
            "by molecule, or None if not recorded."
        }
    )

    _frozen_gap_offset: np.ndarray | None = field(
        metadata={
            "doc": "np.ndarray: Length nmol + 1; the gaps of molecule i "
            "are `gaps[gap_offset[i]:gap_offset[i + 1]]`, or None."
        }
    )

    def __init__(self, **kwargs: Any):
        _init_frozen_fields(
            self, kwargs, "SourceData is built by MethPrintData"
        )

    @property
    def nmol(self) -> int:
        """Number of molecules."""
        return len(self._frozen_mol_id)

    @classmethod
    def _create(cls, name="source", **kwargs) -> Self:
        """
        Build and check one source.

        Raises
        ------
        ValueError
            If the per-molecule arrays do not match the number of
            molecules, or `gaps`/`gap_offset` do not fit together.
        """
        obj = cls(**kwargs, _internal=True)
        nmol = obj.nmol
        for key in ("mol_start", "mol_end"):
            arr = getattr(obj, f"_frozen_{key}")
            if arr is not None and len(arr) != nmol:
                raise ValueError(
                    f"{name} {key} has length {len(arr)}, expected {nmol}"
                )
        gaps, offset = obj._frozen_gaps, obj._frozen_gap_offset
        if (gaps is None) != (offset is None):
            raise ValueError(
                f"{name} gaps and gap_offset must both be given or both "
                "be None"
            )
        if offset is not None and (
            len(offset) != nmol + 1 or offset[-1] != len(gaps)
        ):
            raise ValueError(
                f"{name} gap_offset must have length {nmol + 1} and end at "
                f"{len(gaps)}"
            )
        return obj


@_add_frozen_properties
@dataclass(frozen=True, slots=True, init=False, eq=False)
class MethPrintData:
    """
    Container for MethPrint experimental data for a single region.

    This class stores methylation signal data for test samples and optional
    control samples (unmethylated and fully methylated). Data are stored
    as Pandas DataFrames and retrieved via read-only properties that
    provide defensive copies. An optional reference nucleotide sequence
    for the region may also be stored, via `refseq`.

    .. note::
       This class is intended for internal use within a
       :class:`MethPrintExperiment`. Use the experiment's loading
       mechanisms rather than instantiating this class directly.
    """

    region: str
    """The region identifier for this data block."""

    nbp: int
    """The total number of base pairs in the chromatin fiber."""

    _frozen_refseq: str | None = field(
        metadata={
            "doc": "str: The reference nucleotide sequence for this "
            "region, or None if not provided."
        }
    )

    _frozen_coords: Region | None = field(
        metadata={
            "doc": "Region: Genomic coordinates of this region, or None "
            "if not stored."
        }
    )

    _frozen_test: SourceData = field(
        metadata={"doc": "SourceData: The test (chromatin) molecules."}
    )

    _frozen_unmeth: SourceData | None = field(
        metadata={
            "doc": "SourceData: The unmethylated control molecules, or "
            "None if not given."
        }
    )

    _frozen_meth: SourceData | None = field(
        metadata={
            "doc": "SourceData: The methylated control molecules, or "
            "None if not given."
        }
    )

    def __init__(self, **kwargs: Any):
        _init_frozen_fields(
            self, kwargs, "use MethPrintData._load() to instantiate this class"
        )

    @property
    def sources(self) -> Mapping[str, SourceData]:
        """
        The sources present, by name, in the order test, unmeth, meth.

        Returns
        -------
        Mapping of str to SourceData
            Read-only; a control that was not given is absent.
        """
        return MappingProxyType(
            {
                name: getattr(self, f"_frozen_{name}")
                for name in _SOURCE_NAMES
                if getattr(self, f"_frozen_{name}") is not None
            }
        )

    def _save(self, group):
        dt = h5py.string_dtype(encoding="utf-8")  # For storing strings
        gregion = group.create_group(self.region)
        gmeta = gregion.create_group("metadata")
        gmeta.attrs["chrom"] = self.region
        gmeta.attrs["nbp"] = self.nbp
        _write_coords(gmeta, self._frozen_coords)
        if self._frozen_refseq is not None:
            gmeta.create_dataset("refseq", data=self._frozen_refseq, dtype=dt)
        gdata = gregion.create_group("data")

        for name, src in self.sources.items():
            gdata.create_dataset(
                name + "_mol_id", data=src._frozen_mol_id, dtype=dt
            )
            h5_utils.save_df(name + "_data", src._frozen_data, gdata)
            for key in _PER_MOL_KEYS:
                arr = getattr(src, f"_frozen_{key}")
                if arr is not None:
                    gdata.create_dataset(f"{name}_{key}", data=arr)

    @classmethod
    def _load(cls, group, region) -> Self:
        if region not in group:
            raise ValueError(f"cannot find data for the region {region}")
        gregion = group[region]
        gmeta = gregion["metadata"]
        region = gmeta.attrs["chrom"]
        nbp = int(gmeta.attrs["nbp"])
        refseq = gmeta["refseq"].asstr()[()] if "refseq" in gmeta else None
        coords = _read_coords(gmeta, region)

        gdata = gregion["data"]
        kwargs = {}
        for name in _SOURCE_NAMES:
            if f"{name}_mol_id" not in gdata or f"{name}_data" not in gdata:
                continue
            kwargs[f"{name}_mol_id"] = gdata[f"{name}_mol_id"].asstr()[()]
            kwargs[f"{name}_data"] = h5_utils.load_df(f"{name}_data", gdata)
            for key in _PER_MOL_KEYS:
                if f"{name}_{key}" in gdata:
                    kwargs[f"{name}_{key}"] = gdata[f"{name}_{key}"][
                        ()
                    ].astype(np.int64)
        return cls._create(
            region=region, nbp=nbp, refseq=refseq, coords=coords, **kwargs
        )

    @classmethod
    def _create(cls, **kwargs) -> Self:
        """
        Build from flat keyword arguments.

        Besides `region`, `nbp`, `refseq` and `coords`, each source is
        given as `{source}_mol_id`, `{source}_data` and, optionally,
        `{source}_mol_start`, `{source}_mol_end`, `{source}_gaps` and
        `{source}_gap_offset`; a source without `mol_id` and `data` is
        absent.
        """
        if kwargs.get("nbp") <= 0:
            raise ValueError("nbp must be positive")
        for name in _SOURCE_NAMES:
            fields_ = {
                key: kwargs.pop(f"{name}_{key}", None)
                for key in ("mol_id", "data", *_PER_MOL_KEYS)
            }
            if fields_["mol_id"] is None or fields_["data"] is None:
                continue
            kwargs[name] = SourceData._create(name=name, **fields_)
        return cls(**kwargs, _internal=True)

    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [
            attr for attr in self.__slots__ if not attr.startswith("_")
        ]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith("_"):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str


class LazyRawDataMap(ABCMapping):
    """
    Load per-region MethPrintData from an HDF5 file on demand.

    Region data is read from disk the first time it is accessed and
    kept in a small least-recently-used cache. Accessing a different
    region once the cache is full evicts the least-recently-used
    entry, bounding how many regions' raw data can be resident in
    memory at once, while repeated access to the same region avoids
    re-reading it from disk.
    """

    def __init__(self, path, regions, *, max_cached=1):
        self._path = str(path)
        self._regions = tuple(regions)
        self._max_cached = max_cached
        self._cache = OrderedDict()

    def __getitem__(self, region):
        if region not in self._regions:
            raise KeyError(region)
        if region in self._cache:
            self._cache.move_to_end(region)
            return self._cache[region]
        with h5py.File(self._path, "r") as h5stream:
            data = MethPrintData._load(h5stream["raw_data"], region)
        self._cache[region] = data
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return data

    def __iter__(self):
        return iter(self._regions)

    def __len__(self):
        return len(self._regions)

    def coords(self):
        """
        Coordinates and length of each region, read from the region
        metadata only (no raw call table is loaded).

        Returns
        -------
        list of (str, Region or None, int)
            `(region, coords, nbp)` per region, in region order.
        """
        with h5py.File(self._path, "r") as h5stream:
            graw = h5stream["raw_data"]
            out = []
            for region in self._regions:
                gmeta = graw[region]["metadata"]
                out.append(
                    (
                        region,
                        _read_coords(gmeta, region),
                        int(gmeta.attrs["nbp"]),
                    )
                )
        return out


def _coords_table(rows):
    """
    The `region_coords` table from `(region, coords, nbp)` rows.

    Parameters
    ----------
    rows : iterable of (str, Region or None, int)
        Per region, its coordinates (None if not stored) and length.

    Returns
    -------
    pd.DataFrame
        See `MethPrintExperiment.region_coords`.
    """
    records = []
    for region, coords, nbp in rows:
        records.append(
            {
                "region": region,
                "ref_chrom": None if coords is None else coords.chrom,
                "ref_start": None if coords is None else coords.start,
                "ref_end": None if coords is None else coords.end,
                "ref_strand": None if coords is None else coords.strand,
                "nbp": nbp,
            }
        )
    table = pd.DataFrame(
        records,
        columns=[
            "region", "ref_chrom", "ref_start", "ref_end", "ref_strand",
            "nbp",
        ],
    )
    table = table.astype(
        {
            "ref_chrom": "string",
            "ref_start": "Int64",
            "ref_end": "Int64",
            "ref_strand": "string",
            "nbp": "int64",
        }
    )
    return table.set_index("region")


# Valid values for MethPrintExperiment.filter_dropout()'s
# 'unmapped_strand' and 'method' arguments.
_VALID_UNMAPPED_STRAND = {"union", "drop", "+", "-"}
_VALID_FILTER_DROPOUT_METHOD = {"separate", "aggregate"}


@dataclass(slots=True, init=False, weakref_slot=True)
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

    _raw_data: Mapping[str, MethPrintData]
    _analysis: Mapping[str, DataFrameMap]
    _global_analysis: DataFrameMap
    _finalizer: Any | None
    _tmp_dir: str | None
    _exp_file: str | None
    _mtase: tuple[str, ...] | None
    _wrap: bool
    _ignore_strand: bool
    _region_coords: pd.DataFrame | None

    def __init__(self, **kwargs: Any):
        if not kwargs.pop("_internal", False):
            raise TypeError(
                "use MethPrintExperiment.load() or .load_raw() "
                "to instantiate this class"
            )

        # Bulk assignment of fields
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is not None:
                setattr(self, f.name, val)

        # Create the dicts for analysis
        self._analysis = {region: DataFrameMap() for region in self._raw_data}
        self._global_analysis = DataFrameMap()

        self._finalizer = kwargs.get("_finalizer")

        # Scratch directory shared by this experiment's own staging
        # file and any later smooth()/meth_prob() calls that don't
        # specify their own tmp_dir (set by load_raw() when it
        # generates one, or lazily by resolve_tmp_dir()).
        self._tmp_dir = kwargs.get("_tmp_dir")

        # File this experiment was loaded from, or last saved to (None
        # if never saved); used as save()'s default destination.
        self._exp_file = kwargs.get("_exp_file")

        # Methyltransferase(s) used, or None; set at load_raw() time.
        self._mtase = kwargs.get("_mtase")

        # Whether raw positions were folded around the fiber center;
        # set at load_raw() time.
        self._wrap = kwargs.get("_wrap", False)

        # Whether strand was ignored when resolving methylatable
        # positions (both strands' motif context OR'd together); set
        # at load_raw() time.
        self._ignore_strand = kwargs.get("_ignore_strand", False)

        # Table of region coordinates, built on first use
        self._region_coords = None

    @staticmethod
    def _cleanup_tmp_dir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def resolve_tmp_dir(self):
        """
        Return this experiment's scratch directory for temporary HDF5
        files, creating and caching one under the system default
        temporary directory the first time it's needed. The directory
        is removed once the experiment is closed or garbage-collected;
        see `close`.

        Returns
        -------
        str
            Path to the resolved scratch directory.
        """
        if self._tmp_dir is None:
            self._tmp_dir = h5_utils.fresh_tmp_dir()
            self._finalizer = weakref.finalize(
                self, MethPrintExperiment._cleanup_tmp_dir, self._tmp_dir
            )
        return self._tmp_dir

    def close(self):
        """
        Delete this experiment's scratch directory (all temporary HDF5
        files created for it, e.g. by `load_raw`, `smooth`, or
        `meth_prob`), if one was created. No-op for experiments with no
        scratch directory (e.g. from `.load()` with no later
        `smooth`/`meth_prob` calls).
        """
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @classmethod
    def load_raw(
        cls,
        *,
        chromsize: str | Path,
        test_file: str | Path,
        unmeth_file: str | Path | None = None,
        meth_file: str | Path | None = None,
        fasta_file: str | Path | None = None,
        mtase: str | list[str] | None = None,
        regions: str | Path | Iterable | None = None,
        wrap: bool = False,
        ignore_strand: bool = False,
        colidx: list | None = None,
        max_nmol: int | None = None,
        balance_strand: bool = False,
        seed: int | None = None,
        chunk_size: int = 1000000,
        tmp_dir: str | Path | None = None,
        max_cached_regions: int = 1,
        nworker: int = 1,
        min_gap: int | None = 20,
        gaps_bed: str | Path | Iterable | None = None,
        min_overlap: int = 0,
    ) -> Self:
        """
        Create an experiment by processing raw sequencing data files.

        Reads region sizes and experimental data (typically modkit
        output), drops canonical calls (`mod_code` of '-'), re-orients
        data if requested, and splits signals by region into a
        staging HDF5 file that the returned experiment lazily loads
        (see `load`). If more than one of `test_file`/`unmeth_file`/
        `meth_file` is given, each is read independently (see
        `nworker`) and merged afterward.

        Parameters
        ----------
        chromsize : str or pathlib.Path
            Tab-separated file of chromosome names and lengths (bp).
            Unless `regions` gives intervals, each chromosome is
            analyzed as one region.
        test_file : str or pathlib.Path
            Raw experimental (test) data file.
        unmeth_file : str or pathlib.Path, optional
            Unmethylated control data file.
        meth_file : str or pathlib.Path, optional
            Fully methylated control data file.
        fasta_file : str or pathlib.Path, optional
            Multi-FASTA file of reference sequences (record id
            matching `chromsize`); each region's sequence is stored on
            `MethPrintData.refseq`.
        mtase : str or list of str, optional
            Methyltransferase(s) used to generate the test data: 'A'
            (any-context adenine, e.g., EcoGII), 'CG' (CpG, e.g.,
            M.SssI), 'GC' (GpC, e.g., M.CviPI). One label, a list of
            labels, or None (default) if unspecified. If given, a call
            is kept only if some label accepts both its position (with
            `fasta_file`) and its `mod_code`: 'a' or '28871' (6mA) for
            'A', 'm' or '27551' (5mC) for 'CG' and 'GC'. Kept calls of
            different mod codes at one read and position are summed
            into one (`mod_qual` at most 1).
        regions : str, pathlib.Path, or iterable, optional
            Regions to analyze: None (default) for every chromosome in
            `chromsize`; chromosome name(s), each a whole region; a
            BED file; or `(chrom, start, end[, name[, strand]])`
            tuples. Intervals are 0-based and half-open, named
            `chrom_start_end` by default (BED column 4 otherwise);
            positions are relative to `start` (see `region_coords`).
        wrap : bool, default False
            If True, calculates positions relative to the fiber center
            (useful for circular or symmetrical fibers). No read
            extents are recorded (see Notes).
        ignore_strand : bool, default False
            If True, ignore each row's recorded `strand` for `mtase`
            context filtering, and in `filter_dropout`,
            `dropout_fractions` and `qc`. Independent of
            `norm_by_strand`.
        colidx : list of int, optional
            Explicit column indices if the input file has no header
            matching Modkit's column names, in canonical field order:
            `read_id`, `ref_position`, `region`, `ref_strand`,
            `mod_qual`, `mod_code`, and optionally `alignment_start`,
            `alignment_end` (6 or 8 entries).
        max_nmol : int or None
            Maximum number of molecules to extract for each region.
        balance_strand : bool, default False
            If True, keep an equal number of '+' and '-' strand
            molecules per region: at most `max_nmol`; fewer if
            either strand has fewer than `max_nmol // 2` molecules
            (warns). Molecules with unmapped strand ('.') are dropped.
            Uses the recorded strand regardless of `ignore_strand`.
            Requires `max_nmol` of at least 2.
        seed : int or None
            Random number generator seed for downsampling when
            `max_nmol` is given; independent per data file when more
            than one is given.
        chunk_size : int, default 1000000
            Approximate number of rows read (and held in memory) per
            streamed chunk.
        tmp_dir : str or Path, optional
            Directory for the scratch staging file backing the
            returned experiment's raw data (removed once the
            experiment is closed or garbage-collected; see `close`).
            A fresh `catella_<timestamp>_<hex>` subfolder is created
            under it (or under the system default temporary directory)
            and reused by later `smooth`/`meth_prob` calls.
        max_cached_regions : int, default 1
            Maximum number of regions' raw data kept in memory at
            once by the returned experiment (forwarded to `load`).
        nworker : int, default 1
            Maximum number of threads used to read test/unmeth/meth
            files concurrently. Only relevant when more than one is
            given; never changes the resulting data.
        min_gap : int or None, default 20
            Minimum net deletion (bp) inside a read recorded as an
            internal gap, a stretch the read did not measure (see
            Notes). None records no gaps.
        gaps_bed : str, pathlib.Path, or list of them, optional
            BED12 file(s) from `bedtools bamtobed -bed12 -splitD`
            (filter out non-primary alignments first, e.g.,
            `samtools view -F 2304`). For the reads listed, gaps are
            taken from the alignment blocks instead of being inferred.
        min_overlap : int, default 0
            Minimum overlap (bp) of a molecule's read extent with a
            region (internal gaps not subtracted); molecules
            overlapping less are dropped before `max_nmol` sampling.

        Returns
        -------
        MethPrintExperiment
            A newly initialized experiment object containing the processed
            data.

        Raises
        ------
        ValueError
            If region names are not unique, if control datasets
            contain regions not found in the test dataset, if
            `regions` contains a region not found in `chromsize`,
            if a sequence in `fasta_file` does not match its
            region's length in `chromsize`, if `mtase`
            contains an unknown or duplicate label, or if
            `balance_strand` is True and `max_nmol` is None or less
            than 2, if `colidx` has neither 6 nor 8 entries, if
            `min_gap` is neither None nor a positive integer, if
            `gaps_bed` is given with `wrap` True or `min_gap` None, if
            a region is longer than 1 Mb, if a region name is not
            safe as a file name, if `regions` gives intervals with
            `wrap` True, or if an interval's chromosome is not in
            `chromsize` or its end is past the chromosome's end, or
            if no interval region has molecules in every given file, or
            if `min_overlap` is negative, not an integer, or positive
            with `wrap` True, or if `mtase` is given and a file has
            calls but none with a `mod_code` of `mtase`, or if a read
            has two calls of the same `mod_code` at one position.
        TypeError
            If `regions` has an unsupported type or mixes names and
            tuples.

        Warns
        -----
        UserWarning
            If a region is longer than 100 kb. If `balance_strand` is
            True and a region has fewer
            than `max_nmol // 2` molecules on either strand, or if
            rows have a `ref_position` outside the region or come
            from secondary or supplementary alignments (per Modkit's
            `flag` column, present with `--allow-non-primary`); those
            rows are dropped. Also if some molecules' rows are not
            ordered along the reference; those get no internal gaps.
            Also if interval regions are dropped for lacking molecules
            in some given file. Also if calls at positions of `mtase`
            are dropped for their `mod_code` (e.g., 5mC calls at read
            mismatches over a reference A with `mtase` 'A').

        Notes
        -----
        Unless `wrap` is True, each molecule's read extent is stored as
        `mol_start`/`mol_end` of its source (e.g.
        `exp.raw[region].test.mol_start`), from the
        `alignment_start`/`alignment_end` columns, or else from its
        first and last row.

        Internal gaps (deletions, which leave no rows) are stored as
        `gaps`/`gap_offset` of each source. They are inferred from the
        `forward_read_position` column: a net deletion of at least
        `min_gap` between consecutive rows marks the stretch between
        them as a gap.
        """
        if min_gap is not None and (
            isinstance(min_gap, bool)
            or not isinstance(min_gap, Integral)
            or min_gap < 1
        ):
            raise ValueError(
                f"min_gap must be None or a positive integer, got {min_gap!r}"
            )
        if isinstance(gaps_bed, (str, Path)):
            gaps_bed = [gaps_bed]
        elif gaps_bed is not None:
            gaps_bed = list(gaps_bed)
        if gaps_bed and (wrap or min_gap is None):
            raise ValueError(
                "gaps_bed requires wrap=False and min_gap not None"
            )
        if (
            isinstance(min_overlap, bool)
            or not isinstance(min_overlap, Integral)
            or min_overlap < 0
        ):
            raise ValueError(
                "min_overlap must be a non-negative integer, got "
                f"{min_overlap!r}"
            )
        if min_overlap and wrap:
            raise ValueError("min_overlap requires wrap=False")
        if balance_strand:
            if max_nmol is None:
                raise ValueError("balance_strand requires max_nmol")
            if max_nmol < 2:
                raise ValueError(
                    "max_nmol must be at least 2 when balance_strand is "
                    f"True, got {max_nmol}"
                )
        mtase = _normalize_mtase(mtase)
        kind, parsed = _parse_regions(regions)
        if kind == "regions":
            if wrap:
                raise ValueError(
                    "regions given as intervals require wrap=False"
                )
            chrom_sizes, _ = _read_chromsize(chromsize, False, None)
            sizes = {r.name: r.end - r.start for r in parsed}
            _check_region_lengths(sizes)
            _check_region_ends(parsed, chrom_sizes)
            coords = {r.name: r for r in parsed}
            refseq_by_region = _load_region_refseqs(
                fasta_file, parsed, chrom_sizes
            )
        else:
            chrom_sizes, sizes = _read_chromsize(chromsize, wrap, parsed)
            _check_region_lengths(sizes)
            _check_region_names(list(sizes))
            coords = {
                r: Region(r, 0, chrom_sizes[r], r, ".") for r in sizes
            }
            refseq_by_region = _load_refseqs(fasta_file, chrom_sizes)

        tmp_dir = h5_utils.fresh_tmp_dir(base_dir=tmp_dir)
        fd, tmp_file = tempfile.mkstemp(suffix=".h5", dir=tmp_dir)
        os.close(fd)

        files = [("test", test_file)]
        if unmeth_file is not None:
            files.append(("unmeth", unmeth_file))
        if meth_file is not None:
            files.append(("meth", meth_file))
        ingest_kwargs = {
            "sep": "\t",
            "colidx": colidx,
            "chunk_size": chunk_size,
            "block_size": 64 * 1024 * 1024,
            "sizes": sizes,
            "coords": coords,
            "chrom_sizes": chrom_sizes,
            "wrap": wrap,
            "refseq_by_region": refseq_by_region,
            "mtase": mtase,
            "ignore_strand": ignore_strand,
            "max_nmol": max_nmol,
            "balance_strand": balance_strand,
            "tmp_dir": tmp_dir,
            "min_gap": min_gap,
            "gaps_bed": gaps_bed,
            "min_overlap": min_overlap,
        }

        try:
            with h5py.File(tmp_file, "w") as h5stream:
                graw = h5stream.create_group("raw_data")
                h5stream.create_group("analysis")
                h5stream.create_group("global_analysis")
                if mtase is not None:
                    h5stream.attrs["mtase"] = ",".join(mtase)
                h5stream.attrs["wrap"] = bool(wrap)
                h5stream.attrs["ignore_strand"] = bool(ignore_strand)

                if len(files) == 1:
                    rng = (
                        None
                        if max_nmol is None
                        else np.random.default_rng(seed)
                    )
                    found = _ingest_into(
                        graw, *files[0], rng=rng, **ingest_kwargs
                    )
                    if kind == "regions":
                        _keep_regions([found], sizes)
                else:
                    ingested = _ingest_sources(
                        files, nworker=nworker, seed=seed, **ingest_kwargs
                    )
                    keep = None
                    if kind == "regions":
                        found = [r for _, r, _ in ingested]
                        keep = _keep_regions(found, sizes)
                    _merge_staging(graw, ingested, keep)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        exp = cls.load(tmp_file, max_cached_regions=max_cached_regions)
        # tmp_file is an internal staging file, not a real save target
        exp._exp_file = None
        exp._tmp_dir = tmp_dir
        exp._finalizer = weakref.finalize(
            exp, MethPrintExperiment._cleanup_tmp_dir, tmp_dir
        )
        return exp

    def save(self, exp_file: str | Path | None = None):
        """
        Save the experiment data and analysis to an HDF5 file.

        Saving to an existing file updates it in place. Analysis arrays
        already stored in the file are relinked rather than copied, so
        re-saving needs no extra disk space for them, and loaded arrays
        held elsewhere stay usable afterwards.

        Parameters
        ----------
        exp_file : str or pathlib.Path, optional
            The output file path. Raw data is skipped only if this is
            the file the experiment is already backed by. If None,
            saves to the file this experiment was loaded from or last
            saved to.

        Raises
        ------
        ValueError
            If `exp_file` is None and this experiment has no file to
            default to (e.g. fresh from `load_raw()` and never saved).
        OSError
            If the file is open elsewhere (e.g. by another loaded
            experiment) or cannot be written to disk.

        Notes
        -----
        An interrupted save keeps the previous analysis. A hard kill or
        power loss mid-save can corrupt the file; save to a new path to
        be safe.
        """
        if exp_file is None:
            exp_file = self._exp_file
            if exp_file is None:
                raise ValueError(
                    "no exp_file given and this experiment has not "
                    "been saved before; pass 'exp_file' explicitly"
                )

        dest_path = Path(exp_file)
        same_file = (
            self._exp_file is not None
            and dest_path.exists()
            and Path(self._exp_file).exists()
            and os.path.samefile(self._exp_file, dest_path)
        )

        def write_raw(h5stream):
            # Other destinations always get a fresh copy of raw data
            if same_file and "raw_data" in h5stream:
                return
            if "raw_data" in h5stream:
                del h5stream["raw_data"]
            graw = h5stream.create_group("raw_data")
            for rdata in self._raw_data.values():
                rdata._save(graw)
            if self._mtase is not None:
                h5stream.attrs["mtase"] = ",".join(self._mtase)
            h5stream.attrs["wrap"] = self._wrap
            h5stream.attrs["ignore_strand"] = self._ignore_strand

        _save_analysis(
            dest_path,
            self._analysis,
            self._global_analysis,
            write_extra=write_raw,
            keep_unlisted=same_file,
        )
        self._exp_file = str(dest_path.resolve())

    @classmethod
    def load(
        cls,
        exp_file: str | Path,
        regions: list[str] | None = None,
        max_cached_regions: int = 1,
    ) -> Self:
        """
        Load an experiment from a persistent HDF5 file.

        Raw per-region data is loaded lazily: `MethPrintData` for a
        region is only read from disk when accessed via `raw`, and
        is cached for at most `max_cached_regions` regions at a time
        (least-recently-used eviction), so not all regions need to
        be resident in memory simultaneously.

        Parameters
        ----------
        exp_file : str or pathlib.Path
            HDF5 file containing the experiment.
        regions : list of str, optional
            Restrict the loaded experiment to these regions. If
            None (default), all regions in the file are exposed.
        max_cached_regions : int, default 1
            Maximum number of regions' raw data kept in memory at
            once. Accessing more distinct regions than this evicts
            the least-recently-used one.

        Returns
        -------
        MethPrintExperiment
            The loaded experiment object with all data and analysis maps.

        Raises
        ------
        ValueError
            If `regions` contains a region not found in the file.
        """
        with h5py.File(exp_file, "r") as h5stream:
            # Set up lazy, memory-bounded access to the raw data
            available_regions = list(h5stream["raw_data"].keys())
            if regions is not None:
                unknown = set(regions) - set(available_regions)
                if unknown:
                    raise ValueError(
                        f"regions not found in {exp_file}: "
                        f"{sorted(unknown)}"
                    )
                selected = list(regions)
            else:
                selected = available_regions
            raw_data = LazyRawDataMap(
                exp_file, selected, max_cached=max_cached_regions
            )
            mtase = (
                tuple(h5stream.attrs["mtase"].split(","))
                if "mtase" in h5stream.attrs
                else None
            )
            wrap = bool(h5stream.attrs.get("wrap", False))
            ignore_strand = bool(h5stream.attrs.get("ignore_strand", False))
            obj = cls._create(
                _raw_data=raw_data,
                _exp_file=str(Path(exp_file).resolve()),
                _mtase=mtase,
                _wrap=wrap,
                _ignore_strand=ignore_strand,
            )

            # Load any analysis data
            gana = _analysis_group(h5stream, "analysis")
            for region in gana:
                if region not in selected:
                    continue
                entries = _read_entries(gana[region], exp_file)
                for name, entry in entries.items():
                    obj._analysis[region][name] = entry
            gglobal = _analysis_group(h5stream, "global_analysis")
            entries = _read_entries(gglobal, exp_file)
            for name, entry in entries.items():
                obj._global_analysis[name] = entry
            return obj

    @classmethod
    def _create(cls, **kwargs) -> Self:
        return cls(**kwargs, _internal=True)

    @property
    def regions(self) -> tuple[str, ...]:
        """
        Get the identifiers of all regions in the experiment.

        Returns
        -------
        tuple of str
            A sorted tuple of region names.
        """
        return tuple(self._raw_data.keys())

    @property
    def region_coords(self) -> pd.DataFrame:
        """
        Genomic coordinates of every region.

        Returns
        -------
        pd.DataFrame
            Indexed by region, with columns `ref_chrom`, `ref_start`,
            `ref_end`, `ref_strand` and `nbp`. Position `pos` of a
            region is at `ref_start + pos` on `ref_chrom`.
        """
        if self._region_coords is None:
            rows = (
                self._raw_data.coords()
                if isinstance(self._raw_data, LazyRawDataMap)
                else [
                    (region, data.coords, data.nbp)
                    for region, data in self._raw_data.items()
                ]
            )
            self._region_coords = _coords_table(rows)
        return self._region_coords.copy()

    @property
    def mtase(self) -> tuple[str, ...] | None:
        """
        Get the methyltransferase(s) used for this experiment.

        Returns
        -------
        tuple of str or None
            One or more of 'A', 'CG', 'GC', or None if unspecified.
        """
        return self._mtase

    @property
    def wrap(self) -> bool:
        """
        Get whether raw positions were folded around the fiber center.

        Returns
        -------
        bool
            True if `load_raw` was called with `wrap=True`.
        """
        return self._wrap

    @property
    def ignore_strand(self) -> bool:
        """
        Get whether strand was ignored for methylatable positions.

        Returns
        -------
        bool
            True if `load_raw` was called with `ignore_strand=True`.
        """
        return self._ignore_strand

    @property
    def raw(self) -> Mapping[str, MethPrintData]:
        """
        Provide read-only access to the raw experimental data.

        Returns
        -------
        MappingProxyType
            A frozen mapping where keys are region names and values
            are :class:`MethPrintData` objects.
        """
        return MappingProxyType(self._raw_data)

    # Allow iterating over the raw data by region
    def __iter__(self):
        return iter(self._raw_data)

    def __len__(self):
        return len(self._raw_data)

    @property
    def analysis(self) -> Mapping[str, DataFrameMap]:
        """
        Access the analysis results for each region.

        Returns
        -------
        FixedKeyMap
            A mapping where region keys are fixed, but the analysis
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

    def to_dense(
        self,
        region,
        which="test",
        *,
        mols=None,
        as_h5array=True,
        dtype=np.float64,
        batch_size=20000,
        mask_name=None,
    ):
        """
        Convert a raw long-format methylation table to a dense
        representation.

        Parameters
        ----------
        region : str
            Region identifier.
        which : {"test", "meth", "unmeth"}, default "test"
            Which raw table to convert.
        mols : int or sequence of int, optional
            Molecule index/indices (values of `mol_index`) to include.
            If None (default), every molecule is included.
        as_h5array : bool, default True
            Whether to stream the result to a disk-backed `H5Array`
            (True) or materialize it as an in-memory `pd.DataFrame`
            (False). Independent of `mols` -- e.g. `mols=[3, 7]` with
            `as_h5array=True` (the default) returns a 2-row `H5Array`.
        dtype : data-type, default np.float64
            Numeric dtype of the returned dense data. Must be a
            floating dtype, since missing (mol_index, pos) combinations
            are represented as NaN.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per
            batch. Only used when `as_h5array` is True.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `filter_dropout(mask_name=mask_name)` call for this
            source are returned as all-NaN rows. Looked up as
            `analysis[region][f"{which}_{mask_name}"]`.

        Returns
        -------
        H5Array or pd.DataFrame
            `H5Array` if `as_h5array` is True; otherwise a
            `pd.DataFrame`, indexed by the selected `mol_index` values.

        Raises
        ------
        ValueError
            If no data of the requested kind exists for `region`, or if
            `dtype` is not a floating dtype.

        Warns
        -----
        UserWarning
            If `as_h5array` is False and building the DataFrame is
            estimated to need more than 1 GB of memory (about 72 bytes
            per molecule and base pair, as the pivot needs ~9x the
            result).

        Notes
        -----
        Scratch files backing a resulting `H5Array` are written to this
        experiment's scratch directory (see `resolve_tmp_dir`), shared
        with any other scratch files from the same experiment (e.g.
        from `load_raw`, `smooth`, or `meth_prob`).
        """
        if not np.issubdtype(np.dtype(dtype), np.floating):
            raise ValueError(
                "'dtype' must be a floating dtype to "
                f"represent missing values as NaN, got "
                f"{dtype}"
            )

        raw = self.raw[region]
        source = raw.sources.get(which)
        if source is None:
            raise ValueError(
                f"no '{which}' data available for region '{region}'"
            )
        df, mol_id = source.data, source.mol_id
        nbp = raw.nbp
        all_pos = pd.Index(range(nbp))

        keep = None
        if mask_name is not None:
            keep = self._analysis[region][f"{which}_{mask_name}"][
                "keep"
            ].to_numpy()

        mol_ids = (
            np.arange(len(mol_id)) if mols is None else np.atleast_1d(mols)
        )
        if mols is not None:
            df = df[df["mol_index"].isin(mol_ids)]

        if not as_h5array:
            nbytes = len(mol_ids) * nbp * _DENSE_BYTES_PER_CELL
            if nbytes > _DENSE_WARN_BYTES:
                _warn(
                    f"materializing {len(mol_ids)} molecules as a "
                    f"pd.DataFrame ({len(mol_ids)}x{nbp}) needs about "
                    f"{nbytes / 1e9:.1f} GB of memory; pass "
                    "as_h5array=True to stream to disk instead"
                )
            piv = df.pivot(
                index="mol_index", columns="pos", values="mod_qual"
            ).reindex(index=mol_ids, columns=all_pos)
            piv = piv.astype(dtype)
            if keep is not None:
                piv = _apply_keep_mask(piv, keep[mol_ids])
            return piv

        df = df.sort_values("mol_index", kind="stable")
        mol_index = df["mol_index"].to_numpy()
        nmol_out = len(mol_ids)
        out = H5Array.create(
            (nmol_out, nbp),
            dtype=dtype,
            index=(mol_ids if mols is not None else None),
            dir=self.resolve_tmp_dir(),
        )
        for start, stop in _row_batches(0, nmol_out, batch_size):
            chunk_ids = mol_ids[start:stop]
            if mols is None:
                # mol_ids == arange(nmol_out): contiguous, sorted --
                # take the fast searchsorted range slice.
                lo, hi = np.searchsorted(mol_index, [start, stop])
                batch_df = df.iloc[lo:hi]
            else:
                # chunk_ids may be an arbitrary (unsorted,
                # non-contiguous) subset -- select by membership.
                batch_df = df[df["mol_index"].isin(chunk_ids)]
            piv = batch_df.pivot(
                index="mol_index", columns="pos", values="mod_qual"
            ).reindex(index=chunk_ids, columns=all_pos)
            arr = piv.to_numpy()
            if keep is not None:
                arr = _apply_keep_mask(arr, keep[chunk_ids])
            out.write_batch(start, stop, arr)
        return out

    def _resolve_dropout_channels(self, channels):
        """
        Validate and resolve the channels to evaluate dropout on.

        Parameters
        ----------
        channels : str, list of str, or None
            Channel names, as in `qc`. None uses one channel per
            `self.mtase` label (A, GC, CG), or every assayed channel
            unmerged if both "GC" and "CG" are present, so GCG sites
            are counted once.

        Returns
        -------
        tuple of int
            Channel codes, as from `_resolve_channels`.

        Raises
        ------
        ValueError
            If `self.mtase` is unset, or `channels` is invalid.
        """
        if self._mtase is None:
            raise ValueError(
                "this experiment has no 'mtase' set (specify it via "
                "load_raw()); required to determine methylatable "
                "positions"
            )
        if channels is None and not {"GC", "CG"} <= set(self._mtase):
            channels = list(self._mtase)
        return _resolve_channels(self, channels)

    def _resolve_regions_subset(self, regions):
        """
        Validate and resolve a `regions` subset.

        Parameters
        ----------
        regions : list of str or None
            Subset of `self.regions` to use. None resolves to all of
            `self.regions`.

        Returns
        -------
        Tuple[str, ...]

        Raises
        ------
        ValueError
            If `regions` contains a region not in `self.regions`.
        """
        if regions is None:
            return self.regions
        unknown = set(regions) - set(self.regions)
        if unknown:
            raise ValueError(
                "'regions' contains region(s) not in this "
                f"experiment's regions {list(self.regions)}: "
                f"{sorted(unknown)}"
            )
        return tuple(regions)

    def _channel_masks(self, region, codes):
        """
        Per-strand site masks of each channel, on stored positions.

        Parameters
        ----------
        region : str
            Region with a `refseq`.
        codes : tuple of int
            Channel codes, as from `_resolve_dropout_channels`.

        Returns
        -------
        dict of str to (np.ndarray, np.ndarray)
            As `_channel_site_masks`, with the same sites and contexts
            (folded if `self.wrap`) as `model_prob`.
        """
        ctx = _fold_contexts(self, region, codes)
        sites = _strand_site_masks(self, region, len(ctx))
        return _channel_site_masks(sites, ctx, codes)

    def filter_dropout(
        self,
        *,
        which: str | None = None,
        channels: str | list | None = None,
        regions: list | None = None,
        thres_min: float = 0.0,
        thres_max: float = 1.0,
        unmapped_strand: str = "union",
        method: str = "separate",
        mask_name: str = "dropout_mask",
        span: str = "full",
    ) -> None:
        """
        Flag molecules with poor coverage at methylatable positions.

        Computes, per molecule, the fraction of methylatable positions
        (per channel, on that molecule's own strand, within `span`,
        outside internal gaps) with no signal, and stores a keep/drop
        mask in `analysis`. Raw data is
        never modified -- pass `mask_name` to `to_dense`/
        `MethPrintAnalysis.smooth`/`.meth_prob` to apply it downstream.
        If this experiment was loaded with `ignore_strand=True`,
        `unmapped_strand` governs coverage for every molecule.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to evaluate. None evaluates every source present
            for each region.
        channels : str, list of str, or None, default None
            Channels to evaluate, as in `qc`: among "A", "GCH", "HCG",
            "GCG", "GC" (GCH and GCG merged) and "CG" (HCG and GCG
            merged). None uses one channel per `self.mtase` label, or
            every assayed channel unmerged if both "GC" and "CG" are
            present, so GCG sites are counted once.
        regions : list of str, optional
            Subset of `self.regions` to evaluate. None (default)
            evaluates every region.
        thres_min : float, default 0.0
            Min allowed no-signal fraction (per channel, or of the
            pooled total under `method="aggregate"`) to be kept.
        thres_max : float, default 1.0
            Max allowed no-signal fraction (per channel, or of the
            pooled total under `method="aggregate"`) to be kept.
        unmapped_strand : {"union", "drop", "+", "-"}, default "union"
            How to evaluate unmapped ('.') strand molecules: union of
            '+'/'-' position sets, always dropout, or treat as that
            strand.
        method : {"separate", "aggregate"}, default "separate"
            How multiple channels combine into `keep`. "separate":
            must clear `thres_min`/`thres_max` per channel.
            "aggregate": site counts pooled across channels into one
            fraction first. Irrelevant for a single channel.
        mask_name : str, default "dropout_mask"
            Key for the mask in `analysis[region]`, as
            `f"{source}_{mask_name}"` -- a single-column DataFrame
            with a boolean `keep` column, indexed by `mol_index`.
            A channel without positions to measure is skipped; a
            molecule without any is dropped.
        span : {"full", "calls", "read"}, default "full"
            As in `MethPrintAnalysis.model_prob`; pass the same value.

        Raises
        ------
        ValueError
            If `mtase` is unset on this experiment, `channels` is
            invalid (see `qc`), `regions` contains a region
            not in `self.regions`, `thres_min` or `thres_max` is not
            in [0, 1], `thres_min` is greater than `thres_max`,
            `unmapped_strand`/`method`/`span` is invalid, a requested
            source is missing for some region, `refseq` is missing, or
            `span` is "read" without read extents.

        Warns
        -----
        UserWarning
            If a region has no positions of any channel.
        """
        codes = self._resolve_dropout_channels(channels)
        regions = self._resolve_regions_subset(regions)
        if not (0.0 <= thres_min <= 1.0):
            raise ValueError(
                f"'thres_min' must be in [0, 1], got {thres_min}"
            )
        if not (0.0 <= thres_max <= 1.0):
            raise ValueError(
                f"'thres_max' must be in [0, 1], got {thres_max}"
            )
        if thres_min > thres_max:
            raise ValueError(
                "'thres_min' must be <= 'thres_max', got "
                f"thres_min={thres_min}, thres_max={thres_max}"
            )
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}"
            )
        if method not in _VALID_FILTER_DROPOUT_METHOD:
            raise ValueError(
                "'method' must be one of "
                f"{sorted(_VALID_FILTER_DROPOUT_METHOD)}, got "
                f"{method!r}"
            )
        _check_span(span)

        for region in regions:
            raw = self.raw[region]
            if raw.refseq is None:
                raise ValueError(
                    f"no refseq available for region '{region}'; "
                    "required to determine methylatable positions"
                )
            sources = _resolve_sources(raw, which)
            masks = self._channel_masks(region, codes)
            names = list(masks)
            if not any(p.any() or m.any() for p, m in masks.values()):
                _warn(
                    f"region '{region}' has no sites of channel(s) "
                    f"{names}; all molecules dropped"
                )

            for src in sources:
                cov = _source_coverage(
                    raw,
                    region,
                    src,
                    masks,
                    unmapped_strand,
                    self._ignore_strand,
                    span,
                )
                dropout_frac, combined_frac = _dropout_fracs(cov, names)

                if method == "separate":
                    # NaN channels (no sites to measure) are skipped
                    valid = ~np.isnan(dropout_frac)
                    in_range = (dropout_frac >= thres_min) & (
                        dropout_frac <= thres_max
                    )
                    keep = valid.any(axis=0) & (in_range | ~valid).all(
                        axis=0
                    )
                else:
                    keep = (combined_frac >= thres_min) & (
                        combined_frac <= thres_max
                    )

                # Stored as int, not bool: bool isn't a numeric dtype
                # to h5_utils' save/load, so it would round-trip
                # through the string block as literal "True"/"False"
                # text -- and casting that back to bool makes every
                # non-empty string truthy, silently breaking the mask.
                self._analysis[region][f"{src}_{mask_name}"] = pd.DataFrame(
                    {"keep": keep.astype(np.int8)}
                )

    def is_kept(
        self,
        region: str,
        *,
        which: str = "test",
        mask_name: str | None = "dropout_mask",
    ) -> np.ndarray:
        """
        Return whether each molecule is kept by a molecule filter.

        Reads the per-molecule flags that a filter stored in
        `analysis[region]` under `f"{which}_{mask_name}"`, with a "keep"
        column; `filter_dropout` stores them this way.

        Parameters
        ----------
        region : str
            The region.
        which : {"test", "meth", "unmeth"}, default "test"
            The source of the molecules.
        mask_name : str or None, default "dropout_mask"
            The name under which the filter stored its flags, as passed
            to `filter_dropout`. If None, every molecule is kept.

        Returns
        -------
        np.ndarray
            bool, one entry per molecule of `which`; True if kept.

        Raises
        ------
        KeyError
            If no flags are stored under `mask_name` for `which` and
            `region`.
        """
        if mask_name is None:
            nmol = self.raw[region].sources[which].nmol
            return np.ones(nmol, dtype=bool)
        key = f"{which}_{mask_name}"
        if key not in self.analysis[region]:
            raise KeyError(
                f"'{key}' not found in analysis['{region}']; run "
                "filter_dropout() for this source first, matching "
                "mask_name"
            )
        return self.analysis[region][key]["keep"].to_numpy().astype(bool)

    def dropout_fractions(
        self,
        *,
        which: str | None = None,
        channels: str | list | None = None,
        regions: list | None = None,
        unmapped_strand: str = "union",
        span: str = "full",
    ) -> dict:
        """
        Return each molecule's dropout fraction (QC check).

        If this experiment was loaded with `ignore_strand=True`,
        `unmapped_strand` governs coverage for every molecule.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to evaluate. None evaluates every source
            present for each region.
        channels : str, list of str, or None, default None
            Channels to evaluate, as in `filter_dropout`.
        regions : list of str, optional
            Subset of `self.regions` to evaluate. None (default)
            evaluates every region. Restricting this avoids the
            cost (time and memory) of computing coverage for
            regions not needed by the caller, e.g. a single
            region about to be plotted.
        unmapped_strand : {"union", "drop", "+", "-"}, default
            "union"
            How to evaluate unmapped ('.') strand molecules.
        span : {"full", "calls", "read"}, default "full"
            As in `filter_dropout`.

        Returns
        -------
        dict of (str, str, str) to np.ndarray
            Maps `(region, source, channel)` to that group's
            per-molecule dropout fraction array (NaN without positions
            to measure). Includes a `channel="all"` entry per
            (region, source) pooling all channels, if more than one
            channel is evaluated.

        Raises
        ------
        ValueError
            As in `filter_dropout`.
        """
        codes = self._resolve_dropout_channels(channels)
        regions = self._resolve_regions_subset(regions)
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}"
            )
        _check_span(span)

        result = {}
        for region in regions:
            raw = self.raw[region]
            if raw.refseq is None:
                raise ValueError(
                    f"no refseq available for region '{region}'; "
                    "required to determine methylatable positions"
                )
            sources = _resolve_sources(raw, which)
            masks = self._channel_masks(region, codes)
            names = list(masks)

            for src in sources:
                cov = _source_coverage(
                    raw,
                    region,
                    src,
                    masks,
                    unmapped_strand,
                    self._ignore_strand,
                    span,
                )
                frac, pooled = _dropout_fracs(cov, names)

                for i, name in enumerate(names):
                    result[(region, src, name)] = frac[i]
                if len(names) > 1:
                    result[(region, src, "all")] = pooled

        return result

    def qc(
        self,
        *,
        channels: str | list | None = None,
        regions: list | None = None,
        percentiles: float | list = (50,),
        mask_name: str | None = None,
        batch_size: int = 20000,
        overwrite: bool = False,
        span: str = "full",
    ) -> pd.DataFrame:
        """
        Summarize data quality per region and channel.

        Model-free: computed from the raw data only. The table is
        returned and stored in `global_analysis["qc"]`.

        Parameters
        ----------
        channels : str, list of str, or None, default None
            Channels to summarize, as in `model_prob`: among "A",
            "GCH", "HCG", "GCG", "GC" (GCH and GCG merged) and "CG"
            (HCG and GCG merged). None uses every assayed channel,
            unmerged.
        regions : list of str, optional
            Subset of `self.regions`. None (default) uses all.
        percentiles : float or list of float, default (50,)
            Dropout percentiles to report, each in [0, 100].
        mask_name : str, optional
            Mask from `filter_dropout(mask_name=...)`. Masked
            molecules are left out of every column except
            `frac_masked_<src>`.
        batch_size : int, default 20000
            Molecules per batch when counting calls.
        overwrite : bool, default False
            Whether to replace an existing `global_analysis["qc"]`.
        span : {"full", "calls", "read"}, default "full"
            As in `filter_dropout`, for the dropout and rate columns.

        Returns
        -------
        pd.DataFrame
            One row per (region, channel), plus a `channel="all"` row
            per region pooling the channels if there is more than one.
            `<src>` is one of test, meth, unmeth. Columns:

            - `nsite_plus`, `nsite_minus`: reference sites of the
              channel per strand. A single `nsite` (both strands)
              instead if `ignore_strand`.
            - `nmol`: kept test molecules.
            - `frac_plus_<src>`, `frac_minus_<src>`,
              `frac_unmapped_<src>`: strand balance of kept molecules.
              Not included if `ignore_strand`.
            - `frac_masked_<src>`: share of molecules removed by
              `mask_name` (0 without a mask).
            - `frac_read_<src>`: mean share of the region read by kept
              molecules (read extent minus gaps). Not included without
              read extents (e.g. `wrap`).
            - `dropout_<src>_p<q>`: percentile `q` of the per-molecule
              dropout fraction, i.e. the share of the channel's sites
              on the molecule's strand (within `span`, outside gaps)
              with no call.
            - `meth_frac_<src>`: share of the channel's calls that are
              methylated. `meth_frac_unmeth` is the false positive
              rate, and 1 - `meth_frac_meth` the false negative rate.
            - `accessible_frac`: (`meth_frac_test` -
              `meth_frac_unmeth`) / (`meth_frac_meth` -
              `meth_frac_unmeth`), the test accessibility relative to
              the controls.

        Notes
        -----
        Only test data is required. Columns of a missing control are
        left out, and `accessible_frac` needs both controls. The
        `meth_frac_<src>` and `accessible_frac` columns are not
        included in the "all" row.

        Rates pool all calls, so molecules with more calls weigh more.
        They use the same counts as `model_prob` with the same `span`.
        `accessible_frac`
        assumes the methylated control is fully accessible; it is not
        clipped, so values outside [0, 1] flag a control problem, and
        it is noisy when `meth_frac_meth` - `meth_frac_unmeth` is
        small.

        Dropout is computed on the channel's sites, as in
        `filter_dropout` with the same `channels`. Molecules of
        unmapped ('.') strand are left out of the dropout columns,
        since their sites are not defined; if a source has no strand
        information at all, its dropout columns are NaN, and
        `ignore_strand=True` at `load_raw` should be used. If
        `ignore_strand`, dropout uses the sites of both strands.

        Raises
        ------
        ValueError
            If `channels` or `regions` is invalid, `percentiles` is
            empty or outside [0, 100], `global_analysis["qc"]`
            exists and `overwrite` is False, a region has no
            `refseq` or no test data, `span` is invalid, or `span` is
            "read" without read extents.
        KeyError
            If `mask_name` is not found for some source.
        """
        if "qc" in self._global_analysis and not overwrite:
            raise ValueError(
                "global_analysis already has 'qc'; set overwrite=True "
                "to replace it"
            )
        codes = _resolve_channels(self, channels)
        regions = self._resolve_regions_subset(regions)
        qs = _check_percentiles(percentiles)
        _check_span(span)
        rows = []
        for region in regions:
            rows += _region_qc_rows(
                self, region, codes, qs, mask_name, batch_size, span
            )
        table = pd.DataFrame(rows)
        self._global_analysis["qc"] = table
        return table

    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [
            attr for attr in self.__slots__ if not attr.startswith("_")
        ]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith("_"):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str
