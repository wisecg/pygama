"""
Routines for high-level data loading and skimming.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
from itertools import product
from keyword import iskeyword

import numpy as np
import pandas as pd
from tqdm import tqdm

from pygama.flow.file_db import FileDB
from pygama.lgdo import (
    Array,
    ArrayOfEqualSizedArrays,
    LH5Store,
    Struct,
    Table,
    WaveformTable,
)
from pygama.lgdo.vovutils import build_cl, explode_arrays, explode_cl

log = logging.getLogger(__name__)


class DataLoader:
    """Facilitate loading of processed data across several tiers.

    Where possible, uses a :class:`FileDB` object so that a user can quickly
    select a subset of cycle files for interest, and access information at each
    processing tier.

    Example JSON configuration file:

    .. code-block:: json

        {
            "levels": {
                "hit": {
                    "tiers": ["raw", "dsp", "hit"]
                },
                "tcm": {
                    "tiers": ["tcm"],
                    "parent": "hit",
                    "child": "evt",
                    "tcm_cols": {
                        "child_idx": "coin_idx",
                        "parent_tb": "array_id",
                        "parent_idx": "array_idx"
                    }
                },
                "evt": {
                    "tiers": ["evt"]
                }
            },
            "channel_map": {}
        }


    Examples
    --------

    >>> from pygama.flow import DataLoader
    >>> dl = DataLoader("loader-config.json", "filedb-config.json")
    >>> dl.set_files("file_status == 26 and timestamp == '20220716T130443Z'")
    >>> dl.set_datastreams([3, 6, 8], "ch")
    >>> dl.set_cuts({"hit": "daqenergy > 1000 and AoE > 3", "evt": "muon_veto == False"})
    >>> dl.set_output(fmt="pd.DataFrame", columns=["daqenergy", "channel"])
    >>> data = dl.load()


    Advanced Usage:

    >>> from pygama.flow import DataLoader
    >>> dl = DataLoader("loader-config.json", "filedb-config.json")
    >>> dl.set_files("all")
    >>> dl.set_datastreams([0], "ch")
    >>> dl.set_cuts({"hit": "wf_max > 30000"})
    >>> el = dl.build_entry_list(tcm_level="tcm", mode="any")
    >>> el.query("hit_table == 20", inplace=True)
    >>> dl.set_output(fmt="pd.DataFrame", columns=["daqenergy", "channel"])
    >>> data = dl.load(el)
    """

    def __init__(
        self,
        config: str | dict,
        filedb: str | dict | FileDB,
        file_query: str = None,
    ) -> None:
        """
        Parameters
        ----------
        config
            configuration dictionary or JSON file, see above for specifications.

        filedb
            the loader needs a file database. It can be specified in multiple ways:

            - an instance of :class:`.FileDB`.
            - an LH5 file containing a :class:`.FileDB` (see also
              :meth:`.FileDB.to_disk`).
            - a :class:`.FileDB` configuration dictionary or JSON file.

        file_query
            string query that should operate on columns of a :class:`.FileDB`.

        Note
        ----
        No data is loaded in memory at this point.
        """
        # declare all member variables
        self.config: dict = None
        self.filedb: FileDB = None
        self.file_list: list = None
        self.table_list = None
        self.cuts = None
        self.merge_files = True
        self.output_format = "lgdo.Table"
        self.output_columns = None
        self.data = None

        if isinstance(filedb, FileDB):
            self.filedb = filedb
        else:
            self.filedb = FileDB(filedb)

        # load things if available
        if config is not None:
            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)
            self.set_config(config)

        # set the file_list
        if file_query is not None:
            self.file_list = list(self.filedb.df.query(file_query).index)

    # --------- Get/Set/Reset Functions ----------#

    def set_config(self, config: dict) -> None:
        """Load configuration dictionary."""

        if not os.path.isdir(self.filedb.data_dir):
            raise FileNotFoundError(
                f"{self.filedb.data_dir} (path to data root directory in FileDB) does not exist"
            )

        self.config = config
        self.data_dir = self.filedb.data_dir
        self.levels = list(config["levels"].keys())
        self.tiers = {}
        self.cut_priority = {}
        self.evts = {}
        self.tcms = {}
        for level in self.levels:
            self.tiers[level] = config["levels"][level]["tiers"]
            # Set cut priority
            if "parent" in config["levels"][level].keys():  # This level is a TCM
                parent = config["levels"][level]["parent"]
                child = config["levels"][level]["child"]
                self.tcms[level] = config["levels"][level]
                self.cut_priority[level] = self.cut_priority[parent] + 2
                self.cut_priority[child] = self.cut_priority[parent] + 1
                self.evts[child] = {"tcm": level, "parent": parent}

                # Set TCM columns to lookup
                if "tcm_cols" not in config["levels"][level].keys():
                    log.warning(
                        f"TCM levels, e.g. {level}, need to specify the TCM lookup columns"
                    )
            else:
                self.cut_priority[level] = 0

        # Set channel map
        if isinstance(config["channel_map"], dict):
            self.channel_map = config["channel_map"]
        elif isinstance(config["channel_map"], str):
            with open(config["channel_map"]) as f:
                self.channel_map = json.load(f)
        else:
            log.warning("Channel map must be dict or path to JSON file")

    def set_files(self, query: str | list[str]) -> None:
        """Apply a file selection.

        Sets `self.file_list`, which is a list of indices corresponding to the
        rows in the file database.

        Parameters
        ----------
        query
            if single string, defines an operation on the file database columns
            supported by :meth:`pandas.DataFrame.query`. In addition, the
            ``all`` keyword is supported to select all files in the database.
            If list of strings, will be interpreted as key (cycle timestamp) list.

        Note
        ----
        Call this function before any other operation. A second call to
        :meth:`set_files` does not replace the current file list, which gets
        instead integrated with the new list. Use :meth:`reset` to reset the
        file query.

        Example
        -------
        >>> dl.set_files("file_status == 26 and timestamp == '20220716T130443Z'")
        """

        if isinstance(query, (list, tuple)) and query:
            query = " or ".join([f"timestamp == '{key}'" for key in query])

        if isinstance(query, str):
            if query.replace(" ", "") == "all":
                inds = list(self.filedb.df.index)
            else:
                inds = list(
                    self.filedb.df.query(query, engine="python", inplace=False).index
                )
        else:
            raise ValueError("bad query format")

        if not inds:
            log.warning("no files matching selection found")

        if self.file_list is None:
            self.file_list = inds
        else:
            self.file_list += inds

    def get_file_list(self) -> pd.DataFrame:
        """
        Returns a copy of the file database with its dataframe pared down to
        the current file list.
        """
        return self.filedb.df.iloc[self.file_list]

    # TODO Make this able to handle more complicated requests
    def set_datastreams(self, ds: list | tuple | np.ndarray, word: str) -> None:
        """Apply selection on data streams (or channels).

        Sets `self.table_list`.

        Parameters
        ----------
        ds
            identifies the detectors of interest. Can be a list of detector
            names, serial numbers, or channels or a list of subsystems of
            interest e.g.  ``ged``.
        word
            the type of identifier used in ds. Should be a key in the given
            channel map or a word defined in the configuration file.

        Example
        -------
        >>> dl.set_datastreams(np.arange(40, 45), "ch")
        """
        if self.table_list is None:
            self.table_list = {}

        ds = list(ds)

        found = False
        for level in self.levels:
            tier = self.tiers[level][0]

            template = self.filedb.table_format[tier]
            fm = string.Formatter()
            parse_arr = np.array(list(fm.parse(template)))
            names = list(parse_arr[:, 1])  # fields required to generate file name
            if len(names) > 0:
                keyword = names[0]

            if word == keyword:
                found = True
                if level in self.table_list.keys():
                    self.table_list[level] += ds
                else:
                    self.table_list[level] = ds

        if not found:
            # look for word in channel map
            raise NotImplementedError

    def set_cuts(self, cuts: dict | list) -> None:
        """Apply a selection on columns in the data tables.

        Parameters
        ----------
        cut
            the cuts on the columns of the data table, e.g. ``trapEftp_cal >
            1000``.  If passing a dictionary, the dictionary should be
            structured as ``dict[tier] = cut_expr``. If passing a list, each
            item in the array should be able to be applied on one level of
            tables. The cuts at different levels will be joined with an AND.

        Example
        -------
        >>> dl.set_cuts({"raw": "daqenergy > 1000", "hit": "AoE > 3"})
        """
        if self.cuts is None:
            self.cuts = {}
        if isinstance(cuts, dict):
            # verify the correct structure
            for key, value in cuts.items():
                if not (key in self.levels and isinstance(value, str)):
                    raise ValueError(
                        r"cuts dictionary must be in the format \{ level: string \}"
                    )
                if key in self.cuts.keys():
                    self.cuts[key] += " and " + value
                else:
                    self.cuts[key] = value
        elif isinstance(cuts, list):
            raise NotImplementedError
            self.cuts = {}
            # TODO Parse strings to match column names so you don't have to specify which level it is

    def set_output(
        self, fmt: str = None, merge_files: bool = None, columns: list = None
    ) -> None:
        """
        Set the parameters for the output format of load

        Parameters
        ---------
        fmt
            ``lgdo.Table`` or ``pd.DataFrame``.
        merge_files
            If ``True``, information from multiple files will be merged into
            one table.
        columns
            The columns that should be copied into the output.

        Example
        -------
        >>> dl.set_output(fmt="pd.DataFrame", merge_files=False, columns=["daqenergy", "trapEmax", "channel"])
        """
        if fmt not in ["lgdo.Table", "pd.DataFrame", None]:
            raise ValueError(f"'{fmt}' output format not supported")

        if fmt is not None:
            self.output_format = fmt
        if merge_files is not None:
            self.merge_files = merge_files
        if columns is not None:
            self.output_columns = columns

    def reset(self):
        """Resets all fields to their default values, as if this is a newly
        created data loader.
        """
        self.file_list = None
        self.table_list = None
        self.cuts = None
        self.merge_files = True
        self.output_format = "lgdo.Table"
        self.output_columns = None
        self.data = None

    # ------------- Applying Cuts/Loading Data --------------#

    # TODO: mode
    def build_entry_list(
        self,
        tcm_level: str = None,
        tcm_table: int | str = None,
        mode: str = "only",
        save_output_columns: bool = False,
        in_memory: bool = True,
        output_file: str = None,
    ) -> dict[int, pd.DataFrame] | pd.DataFrame | None:
        """Applies cuts to the tables and files of interest.

        Can only load up to two levels, those joined by `tcm_level`.

        Parameters
        ----------
        tcm_level
            the type of TCM to be used. If ``None``, will only return
            information from lowest level.
        tcm_table
            the identifier of the table inside this TCM level that you want to
            use.  If unspecified, there must only be one table inside a TCM
            file in `tcm_level`.
        mode
            if ``any``, returns every hit in the event if any hit in the event
            passes the cuts. If ``only``, only returns hits that pass the cuts.
        save_output_columns
            if ``True``, saves any columns needed for both the cut and the
            output to the `self.entry_list`.
        in_memory
            if ``True``, returns the generated entry list in memory.
        output_file
            HDF5 file name to write the entry list to.

        Returns
        -------
        entries
            the entry list containing columns for ``{parent}_idx``,
            ``{parent}_table``, ``{child}_idx`` and output columns if
            applicable.  Only returned if `in_memory` is ``True``.

        Note
        ----
        Does *not* load the column information into memory. This is done by
        :meth:`.load`.
        """
        if self.file_list is None:
            raise ValueError("You need to make a query on filedb, use set_files()")

        if not in_memory and output_file is None:
            raise ValueError("If in_memory is False, need to specify an output file")

        if in_memory:
            entries = {}

        log.debug("generating entry list determined by cuts")

        # Default columns in the entry_list
        if tcm_level is None:
            return self.build_hit_entries(save_output_columns, in_memory, output_file)

        # Assumes only one tier in a TCM level
        parent = self.tcms[tcm_level]["parent"]
        child = self.tcms[tcm_level]["child"]
        entry_cols = [
            f"{parent}_table",
            f"{parent}_idx",
            f"{child}_idx",
        ]
        if save_output_columns:
            for_output = []

        # Find out which columns are needed for any cuts
        cut_cols = {}

        for level in [child, parent]:
            cut_cols[level] = []
            if self.cuts is not None and level in self.cuts.keys():
                cut = self.cuts[level]
            else:
                cut = ""
            # String parsing to determine which columns need to be loaded
            # Matches any valid python variable name
            terms = re.findall(r"[^\W0-9]\w*", cut)
            for term in terms:
                # Assumes that column names are valid python variable names
                if term.isidentifier() and not iskeyword(term):
                    cut_cols[level].append(term)
                    # Add column to entry_cols if they are needed for both the cut and the output
                    if self.output_columns is not None:
                        if (
                            term in self.output_columns
                            and term not in entry_cols
                            and save_output_columns
                        ):
                            for_output.append(term)

        if save_output_columns:
            entry_cols += for_output

        sto = LH5Store()

        # Make the entry list for each file
        for file in self.file_list:
            # Get the TCM specified
            # Assumes that each TCM level only has one tier
            tcm_tier = self.tiers[tcm_level][0]
            tcm_tables = self.filedb.df.loc[file, f"{tcm_tier}_tables"]
            if len(tcm_tables) > 1 and tcm_table is None:
                raise ValueError(
                    f"There are {len(tcm_tables)} TCM tables, need to specify which to use"
                )
            else:
                if tcm_table is not None:
                    if tcm_table in tcm_tables:
                        tcm_tb = tcm_table
                    else:
                        raise ValueError(
                            f"Table {tcm_table} doesn't exist in {tcm_level}"
                        )
                else:
                    tcm_tb = tcm_tables[0]

            tcm_path = os.path.join(
                self.data_dir,
                self.filedb.tier_dirs[tcm_tier].lstrip("/"),
                self.filedb.df.iloc[file][f"{tcm_tier}_file"].lstrip("/"),
            )

            if not os.path.exists(tcm_path):
                raise FileNotFoundError(f"Can't find TCM file for {tcm_level}")

            tcm_table_name = self.get_table_name(tcm_tier, tcm_tb)
            tcm_lgdo, _ = sto.read_object(tcm_table_name, tcm_path)
            # Have to do some hacky stuff until I get a get_dataframe() method
            tcm_lgdo[self.tcms[tcm_level]["tcm_cols"]["child_idx"]] = Array(
                nda=explode_cl(tcm_lgdo["cumulative_length"].nda)
            )
            tcm_lgdo.pop("cumulative_length")
            tcm_tb = Table(col_dict=tcm_lgdo)
            f_entries = tcm_tb.get_dataframe()
            renaming = {
                self.tcms[tcm_level]["tcm_cols"]["child_idx"]: f"{child}_idx",
                self.tcms[tcm_level]["tcm_cols"]["parent_tb"]: f"{parent}_table",
                self.tcms[tcm_level]["tcm_cols"]["parent_idx"]: f"{parent}_idx",
            }
            f_entries.rename(columns=renaming, inplace=True)
            if self.merge_files:
                f_entries["file"] = file
            # At this point, should have a list of all available hits/evts joined by tcm

            if mode == "any":
                drop_idx = None

            # Perform cuts specified for child or parent level, in that order
            for level in [child, parent]:
                if self.cuts is None or level not in self.cuts.keys():
                    continue
                cut = self.cuts[level]
                col_tiers = self.get_tiers_for_col(cut_cols[level], merge_files=False)

                # Tables in first tier of event should be the same for all tiers in one level
                tables = self.filedb.df.loc[file, f"{self.tiers[level][0]}_tables"]
                if self.table_list is not None:
                    if level in self.table_list.keys():
                        tables = self.table_list[level]
                # Cut any rows of TCM not relating to requested tables
                if level == parent:
                    f_entries.query(f"{level}_table in {tables}", inplace=True)

                for tb in tables:
                    tb_table = None
                    if level == parent:
                        tcm_idx = f_entries.query(f"{level}_table == {tb}").index
                    else:
                        tcm_idx = f_entries.index
                    idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"]
                    for tier in self.tiers[level]:
                        tier_path = os.path.join(
                            self.data_dir,
                            self.filedb.tier_dirs[tier].lstrip("/"),
                            self.filedb.df.loc[file, f"{tier}_file"].lstrip("/"),
                        )
                        if tier in col_tiers[file]["tables"].keys():
                            if tb in col_tiers[file]["tables"][tier]:
                                table_name = self.get_table_name(tier, tb)
                                tier_table, _ = sto.read_object(
                                    table_name,
                                    tier_path,
                                    field_mask=cut_cols[level],
                                    idx=idx_mask.tolist(),
                                )
                                if tb_table is None:
                                    tb_table = tier_table
                                else:
                                    tb_table.join(tier_table)
                    if tb_table is None:
                        continue
                    tb_df = tb_table.get_dataframe()
                    tb_df.query(cut, inplace=True)
                    idx_match = f_entries.query(f"{level}_idx in {list(tb_df.index)}")
                    if level == parent:
                        idx_match = idx_match.query(f"{level}_table == {tb}")
                    if mode == "only":
                        keep_idx = idx_match.index
                        drop_idx = set.symmetric_difference(
                            set(tcm_idx), list(keep_idx)
                        )
                        f_entries.drop(drop_idx, inplace=True)
                    elif mode == "any":
                        evts = idx_match[f"{child}_idx"].unique()
                        keep_idx = f_entries.query(f"{child}_idx in {evts}").index
                        drop = set.symmetric_difference(
                            set(f_entries.index), list(keep_idx)
                        )
                        if drop_idx is None:
                            drop_idx = drop
                        else:
                            drop_idx = set.intersection(drop_idx, drop)
                    else:
                        raise ValueError("mode must be either 'any' or 'only'")

                    if save_output_columns:
                        for col in tb_df.columns:
                            if col in for_output:
                                f_entries.loc[keep_idx, col] = tb_df[col].tolist()
                    # end for each table loop
                # end for each level loop
            if mode == "any":
                f_entries.drop(drop_idx, inplace=True)
            f_entries.reset_index(inplace=True, drop=True)
            if in_memory:
                entries[file] = f_entries
            if output_file:
                # Convert f_entries DataFrame to Struct
                f_dict = f_entries.to_dict("list")
                f_struct = Struct(f_dict)
                if self.merge_files:
                    sto.write_object(f_struct, "entries", output_file, wo_mode="a")
                else:
                    sto.write_object(
                        f_struct, f"entries/{file}", output_file, wo_mode="a"
                    )
            # end for each file loop

        if in_memory:
            if self.merge_files:
                entries = pd.concat(entries.values(), ignore_index=True)
            return entries

    def build_hit_entries(
        self,
        save_output_columns: bool = False,
        in_memory: bool = True,
        output_file: str = None,
    ) -> dict[int, pd.DataFrame] | pd.DataFrame | None:
        """Called by :meth:`.build_entry_list` to handle the case when
        `tcm_level` is unspecified.

        Ignores any cuts set on levels above lowest level.

        Parameters
        ----------
        save_output_columns
            If ``True``, saves any columns needed for both the cut and the
            output to the entry list.
        in_memory
            If ``True``, returns the generated entry list in memory.
        output_file
            HDF5 file name to write the entry list to.

        Returns
        -------
        entries
            the entry list containing columns for ``{low_level}_idx``,
            ``{low_level}_table``, and output columns if
            applicable.  Only returned if `in_memory` is ``True``.
        """
        low_level = self.levels[0]
        if in_memory:
            entries = {}
        entry_cols = [f"{low_level}_table", f"{low_level}_idx"]

        log.debug(
            f"generating (hit-oriented) entry list corresponding to cuts on the {low_level} level"
        )

        # find out which columns are needed for the cut
        cut_cols = []
        cut = ""
        if self.cuts is not None and low_level in self.cuts.keys():
            cut = self.cuts[low_level]
            # do cut expression string parsing to determine which columns need to be loaded
            # this matches any valid python variable name
            terms = re.findall(r"[^\W0-9]\w*", cut)
            for term in terms:
                if not iskeyword(term):
                    cut_cols.append(term)
                    # Add column to entry_cols if they are needed for both the cut and the output
                    if self.output_columns is not None:
                        if (
                            term in self.output_columns
                            and term not in entry_cols
                            and save_output_columns
                        ):
                            entry_cols.append(term)

        log.debug(f"need to load {cut_cols} columns for applying cuts")
        col_tiers = self.get_tiers_for_col(cut_cols, merge_files=False)

        sto = LH5Store()

        if log.getEffectiveLevel() >= logging.INFO:
            progress_bar = tqdm(
                desc="Building entry list",
                total=len(self.file_list),
                delay=2,
                unit=" keys",
            )

        # now we loop over the files in our list
        for file in self.file_list:

            if log.getEffectiveLevel() >= logging.INFO:
                progress_bar.update()
                progress_bar.set_postfix(key=self.filedb.df.iloc[file]["timestamp"])

            log.debug(
                f"building entry list for cycle {self.filedb.df.iloc[file]['timestamp']}"
            )

            # this dataframe will be associated with the file and will contain
            # the columns needed for the cut as well as the columns requested
            # for the output
            f_entries = pd.DataFrame(columns=entry_cols)

            # check if we were asked to consider only certain data streams
            if self.table_list is not None:
                if low_level in self.table_list.keys():
                    tables = self.table_list[low_level]
            else:
                tables = self.filedb.df.loc[file, f"{self.tiers[low_level][0]}_tables"]

            # now loop over these data streams
            for tb in tables:
                tb_table = None

                # no cut is set, let's include all indices for the data stream
                if cut == "":
                    tier = self.tiers[low_level][0]
                    # reconstruct absolute path to tier file
                    tier_path = os.path.join(
                        self.data_dir,
                        self.filedb.tier_dirs[tier].lstrip("/"),
                        self.filedb.df.iloc[file][f"{tier}_file"].lstrip("/"),
                    )
                    # now read how many rows are there in the file
                    table_name = self.get_table_name(tier, tb)
                    n_rows = sto.read_n_rows(table_name, tier_path)
                    tb_df = pd.DataFrame(
                        {
                            f"{low_level}_idx": np.arange(n_rows),
                            f"{low_level}_table": tb,
                        }
                    )
                    if self.merge_files:
                        tb_df["file"] = file
                else:
                    # loop over tiers in the lowest level
                    for tier in self.tiers[low_level]:
                        # is the tier involved, considered the columns on which cuts are applied?
                        if (
                            tier in col_tiers[file]["tables"].keys()
                            and tb in col_tiers[file]["tables"][tier]
                        ):
                            # reconstruct absolute path to tier file
                            tier_path = os.path.join(
                                self.data_dir,
                                self.filedb.tier_dirs[tier].lstrip("/"),
                                self.filedb.df.iloc[file][f"{tier}_file"].lstrip("/"),
                            )

                            # load the data from the tier file, just the columns needed for the cut
                            table_name = self.get_table_name(tier, tb)
                            tier_tb, _ = sto.read_object(
                                table_name, tier_path, field_mask=cut_cols
                            )
                            # join eveything in one table
                            if tb_table is None:
                                tb_table = tier_tb
                            else:
                                tb_table.join(tier_tb)

                    if tb_table is None:
                        continue
                    # convert to DataFrame and apply cuts
                    tb_df = tb_table.get_dataframe()
                    tb_df.query(cut, inplace=True)
                    tb_df[f"{low_level}_table"] = tb
                    tb_df[f"{low_level}_idx"] = tb_df.index

                # final DataFrame
                f_entries = pd.concat((f_entries, tb_df), ignore_index=True)[entry_cols]
                # end tb loop
            if self.merge_files:
                f_entries["file"] = file
            if in_memory:
                entries[file] = f_entries
            if output_file:
                # Convert f_entries DataFrame to Struct
                f_dict = f_entries.to_dict("list")
                f_struct = Struct(f_dict)
                if self.merge_files:
                    sto.write_object(f_struct, "entries", output_file, wo_mode="a")
                else:
                    sto.write_object(
                        f_struct, f"entries/{file}", output_file, wo_mode="a"
                    )
            # end file loop

        if log.getEffectiveLevel() >= logging.INFO:
            progress_bar.close()

        if in_memory:
            if self.merge_files:
                entries = pd.concat(entries.values(), ignore_index=True)
            return entries

    # TODO : support chunked reading of entry_list from disk
    def load(
        self,
        entry_list: pd.DataFrame = None,
        in_memory: bool = True,
        output_file: str = None,
        orientation: str = "hit",
        tcm_level: str = None,
    ) -> None | Table | Struct | pd.DataFrame:
        """Loads the requested columns in `self.output_columns` for the entries
        in the given `entry_list`.

        Parameters
        ----------
        entry_list
            the output of :meth:`.build_entry_list`.
        in_memory
            if ``True``, returns the loaded data in memory and stores in
            `self.data`.
        output_file
            if not ``None``, writes the loaded data to the specified file.
        orientation
            specifies the orientation of the output table. Can be ``hit`` or
            ``evt``.
        tcm_level
            which TCM was used to create the ``entry_list``.

        Returns
        -------
        data
            The data loaded from disk, as specified by `self.output_format`,
            `self.output_columns`, and `self.merge_files`. Only returned if
            `in_memory` is ``True``.
        """
        # set save_output_columns=True to avoid wasting time
        if entry_list is None:
            entry_list = self.build_entry_list(
                tcm_level=tcm_level, save_output_columns=True
            )

        if not in_memory and output_file is None:
            raise ValueError("if in_memory is False, need to specify an output file")

        if self.output_columns is None or not self.output_columns:
            raise ValueError("need to set output columns to load data")

        if orientation == "hit":
            self.data = self.load_hits(entry_list, in_memory, output_file, tcm_level)
        elif orientation == "evt":
            if tcm_level is None:
                if len(self.tcms) == 1:
                    tcm_level = list(self.tcms.keys())[0]
                else:
                    raise ValueError(
                        "need to specify which coincidence map to use to return event-oriented data"
                    )
            self.data = self.load_evts(entry_list, in_memory, output_file, tcm_level)
        else:
            raise ValueError(
                f"I don't understand what orientation={orientation} means!"
            )

        return self.data

    def load_hits(
        self,
        entry_list: pd.DataFrame,
        in_memory: bool = False,
        output_file: str = None,
        tcm_level: str = None,
    ) -> None | Table | Struct | pd.DataFrame:
        """Called by :meth:`.load` when orientation is ``hit``."""
        if tcm_level is None:
            parent = self.levels[0]
            child = None
            load_levels = [parent]
        else:
            parent = self.tcms[tcm_level]["parent"]
            child = self.tcms[tcm_level]["child"]
            load_levels = [parent, child]

        def explode_evt_cols(el, tier_table):
            # Explode columns from "evt"-style levels, untested
            # Will only work if column has an "nda" attribute
            cum_length = build_cl(el[f"{child}_idx"])
            exp_cols = explode_arrays(
                cum_length,
                [a.nda for a in tier_table.values()],
            )
            tier_table.update(zip(tier_table.keys(), exp_cols))
            return tier_table

        def fill_col_dict(tier_table, col_dict, tcm_idx):
            # Put the information from the tier_table (after the columns have been exploded)
            # into col_dict, which will be turned into the final Table
            for col in tier_table.keys():
                if isinstance(tier_table[col], Array):
                    # Allocate memory for column for all channels
                    if col not in col_dict.keys():
                        col_dict[col] = np.empty(
                            table_length,
                            dtype=tier_table[col].dtype,
                        )
                    col_dict[col][tcm_idx] = tier_table[col].nda

                elif isinstance(tier_table[col], WaveformTable):
                    wf_table = tier_table[col]
                    if isinstance(
                        wf_table["values"],
                        ArrayOfEqualSizedArrays,
                    ):
                        # Allocate memory for columns for all channels
                        if "wf_dt" not in col_dict.keys():
                            col_dict["wf_t0"] = np.empty(
                                table_length,
                                dtype=wf_table["t0"].dtype,
                            )
                            col_dict["wf_dt"] = np.empty(
                                table_length,
                                dtype=wf_table["dt"].dtype,
                            )
                            col_dict["wf_values"] = np.empty(
                                (
                                    table_length,
                                    wf_table["values"].nda.shape[1],
                                ),
                                dtype=wf_table["values"].dtype,
                            )
                        col_dict["wf_t0"][tcm_idx] = wf_table["t0"].nda
                        col_dict["wf_dt"][tcm_idx] = wf_table["dt"].nda
                        col_dict["wf_values"][tcm_idx] = wf_table["values"].nda
                    else:  # wf_values is a VectorOfVectors
                        log.warning(
                            "not sure how to handle waveforms with values "
                            f"of type {type(tier_table[col]['values'])} yet"
                        )
                else:
                    log.warning(
                        f"not sure how to handle column {col} "
                        f"of type {type(tier_table[col])} yet"
                    )
            return col_dict

        sto = LH5Store()

        if self.merge_files:
            tables = entry_list[f"{parent}_table"].unique()
            field_mask = []
            for col in self.output_columns:
                if col not in entry_list.columns:
                    field_mask.append(col)

            col_tiers = self.get_tiers_for_col(field_mask)
            col_dict = entry_list.to_dict("list")
            table_length = len(entry_list)

            for tb, level in product(tables, load_levels):
                gb = entry_list.query(f"{parent}_table == {tb}").groupby("file")
                files = list(gb.groups.keys())
                el_idx = list(gb.groups.values())
                idx_mask = [list(entry_list.loc[i, f"{level}_idx"]) for i in el_idx]

                for tier in self.tiers[level]:
                    if tb not in col_tiers[tier]:
                        continue
                    tb_name = self.get_table_name(tier, tb)
                    tier_paths = [
                        os.path.join(
                            self.data_dir,
                            self.filedb.tier_dirs[tier].lstrip("/"),
                            self.filedb.df.iloc[file][f"{tier}_file"].lstrip("/"),
                        )
                        for file in files
                    ]

                    tier_table, _ = sto.read_object(
                        name=tb_name,
                        lh5_file=tier_paths,
                        idx=idx_mask,
                        field_mask=field_mask,
                    )

                    if level == child:
                        explode_evt_cols(entry_list, tier_table)

                    col_dict = fill_col_dict(
                        tier_table,
                        col_dict,
                        [idx for idx_list in el_idx for idx in idx_list],
                    )
            # Convert col_dict to lgdo.Table
            for col in col_dict.keys():
                nda = np.array(col_dict[col])
                col_dict[col] = Array(nda=nda)
            f_table = Table(col_dict=col_dict)

            if output_file:
                sto.write_object(f_table, "merged_data", output_file, wo_mode="o")
            if in_memory:
                if self.output_format == "lgdo.Table":
                    return f_table
                elif self.output_format == "pd.DataFrame":
                    return f_table.get_dataframe()
                else:
                    raise ValueError(
                        f"'{self.output_format}' output format not supported"
                    )
        else:  # not merge_files
            if in_memory:
                load_out = {}

            if log.getEffectiveLevel() >= logging.INFO:
                progress_bar = tqdm(
                    desc="Loading data",
                    total=len(entry_list),
                    delay=2,
                    unit=" keys",
                )

            # now loop over the output of build_entry_list()
            for file, f_entries in entry_list.items():

                if log.getEffectiveLevel() >= logging.INFO:
                    progress_bar.update()
                    progress_bar.set_postfix(key=self.filedb.df.iloc[file]["timestamp"])

                log.debug(
                    f"loading data for cycle key {self.filedb.df.iloc[file]['timestamp']}"
                )

                field_mask = []

                for col in self.output_columns:
                    if col not in f_entries.columns:
                        field_mask.append(col)

                col_tiers = self.get_tiers_for_col(field_mask)
                col_dict = f_entries.to_dict("list")
                table_length = len(f_entries)

                log.debug(f"will load new columns {field_mask}")

                # loop through each table in entry list and
                # loop through each level we're asked to load from
                for tb, level in product(
                    f_entries[f"{parent}_table"].unique(), load_levels
                ):
                    tcm_idx = f_entries.query(f"{parent}_table == {tb}").index
                    idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"].tolist()

                    # loop over tiers in the level
                    for tier in self.tiers[level]:

                        if tb not in col_tiers[file]["tables"][tier]:
                            continue

                        log.debug(
                            f"...for stream '{self.get_table_name(tier, tb)}' (at {level} level)"
                        )

                        # path to tier file
                        tier_path = os.path.join(
                            self.data_dir,
                            self.filedb.tier_dirs[tier].lstrip("/"),
                            self.filedb.df.iloc[file][f"{tier}_file"].lstrip("/"),
                        )
                        # this should not happen
                        if not os.path.exists(tier_path):
                            raise FileNotFoundError(tier_path)

                        table_name = self.get_table_name(tier, tb)
                        tier_table, _ = sto.read_object(
                            table_name,
                            tier_path,
                            idx=idx_mask,
                            field_mask=field_mask,
                        )
                        if level == child:
                            explode_evt_cols(f_entries, tier_table)

                        col_dict = fill_col_dict(tier_table, col_dict, tcm_idx)
                        # end tb loop

                # Convert col_dict to lgdo.Table
                for col in col_dict.keys():
                    nda = np.array(col_dict[col])
                    col_dict[col] = Array(nda=nda)
                f_table = Table(col_dict=col_dict)

                if in_memory:
                    load_out[file] = f_table
                if output_file:
                    sto.write_object(f_table, f"file{file}", output_file, wo_mode="o")
                # end file loop

            if log.getEffectiveLevel() >= logging.INFO:
                progress_bar.close()

            if in_memory:
                if self.output_format == "lgdo.Table":
                    return load_out
                elif self.output_format == "pd.DataFrame":
                    for file in load_out.keys():
                        load_out[file] = load_out[file].get_dataframe()
                    return load_out
                else:
                    raise ValueError(
                        f"'{self.output_format}' output format not supported"
                    )

    def load_evts(
        self,
        entry_list: pd.DataFrame = None,
        in_memory: bool = False,
        output_file: str = None,
        tcm_level: str = None,
    ) -> None | Table | Struct | pd.DataFrame:
        """Called by :meth:`load` when orientation is ``evt``."""
        raise NotImplementedError

        parent = self.tcms[tcm_level]["parent"]
        child = self.tcms[tcm_level]["child"]
        load_levels = [parent, child]

        sto = LH5Store()

        if self.merge_files:  # Try to load all information at once
            raise NotImplementedError
        else:  # Not merge_files
            if in_memory:
                load_out = {}
            for file, f_entries in entry_list.items():
                field_mask = []
                f_table = None
                # Pre-allocate memory for output columns
                for col in self.output_columns:
                    if col not in f_entries.columns:
                        f_entries[col] = None
                        field_mask.append(col)
                col_tiers = self.get_tiers_for_col(field_mask)
                col_dict = f_entries.to_dict("list")
                for col in col_dict.keys():
                    nda = np.array(col_dict[col])
                    col_dict[col] = Array(nda=nda)

                # Loop through each table in entry list
                for tb in f_entries[f"{parent}_table"].unique():
                    tcm_idx = f_entries.query(f"{parent}_table == {tb}").index
                    for level in load_levels:
                        idx_mask = f_entries.loc[tcm_idx, f"{level}_idx"]

                        for tier in self.tiers[level]:
                            if tb in col_tiers["tables"][tier]:
                                tier_path = os.path.join(
                                    self.data_dir,
                                    self.filedb.tier_dirs[tier].lstrip("/"),
                                    self.filedb.df.iloc[file][f"{tier}_file"].lstrip(
                                        "/"
                                    ),
                                )
                                if os.path.exists(tier_path):
                                    table_name = self.get_table_name(tier, tb)
                                    tier_table, _ = sto.read_object(
                                        table_name,
                                        tier_path,
                                        idx=idx_mask,
                                        field_mask=field_mask,
                                    )
                                    for col in tier_table.keys():
                                        f_table[col].nda[tcm_idx] = tier_table[
                                            col
                                        ].tolist()
                    # end tb loop
                if in_memory:
                    load_out[file] = f_table
                if output_file:
                    sto.write_object(f_table, f"file{file}", output_file, wo_mode="o")
                # end file loop

            if in_memory:
                if self.output_format == "lgdo.Table":
                    return load_out
                elif self.output_format == "pd.DataFrame":
                    for file in load_out.keys():
                        load_out[file] = load_out[file].get_dataframe()
                    return load_out
                else:
                    raise ValueError(
                        f"'{self.output_format}' output format not supported"
                    )

    def load_detector(self, det_id):
        """
        special version of `load` designed to retrieve all file files, tables,
        column names, and potentially calibration/dsp parameters relevant to one
        single detector.
        """
        raise NotImplementedError

    def load_settings(self):
        """
        get metadata stored in raw files, usually from a DAQ machine.
        """
        raise NotImplementedError

    def load_dsp_pars(self, query):
        """
        access the dsp_pars parameter database (probably JSON format) and do
        some kind of query to retrieve parameters of interest for our file list,
        and return some tables.
        """
        raise NotImplementedError

    def load_cal_pars(self, query):
        """
        access the cal_pars parameter database, run a query, and return some tables.
        """
        raise NotImplementedError

    def skim_waveforms(self, mode: str = "hit", hit_list=None, evt_list=None):
        """
        handle this one separately because waveforms can easily fill up memory.
        """
        raise NotImplementedError

    def browse(self, query, dsp_config=None):
        """
        Interface between DataLoader and WaveformBrowser.
        """
        raise NotImplementedError

    # -------------- Helper Functions ----------------#
    def get_tiers_for_col(
        self, columns: list | np.ndarray, merge_files: bool = None
    ) -> dict:
        """For each column given, get the tiers and tables in that tier where
        that column can be found.

        Parameters
        ----------
        columns
            the columns to look for.

        Returns
        -------
        col_tiers
            ``col_tiers[file]["tables"][tier]`` gives a list of tables in
            ``tier`` that contain a column of interest.
            ``col_tiers[file]["columns"][column]`` gives the tier that
            ``column`` can be found in. If `self.merge_file`s then `
            col_tiers[tier]`` is a list of tables in ``tier`` that contain a
            column of interest.
        """

        col_tiers = {}

        # filedb.columns is needed, generate it now if not available
        if self.filedb.columns is None:
            self.filedb.scan_tables_columns()

        if merge_files is None:
            merge_files = self.merge_files

        if merge_files:
            for file in self.file_list:
                col_inds = set()
                for i, col_list in enumerate(self.filedb.columns):
                    if not set(col_list).isdisjoint(columns):
                        col_inds.add(i)

                for level in self.levels:
                    for tier in self.tiers[level]:
                        col_tiers[tier] = set()
                        if self.filedb.df.loc[file, f"{tier}_col_idx"] is not None:
                            for i in range(
                                len(self.filedb.df.loc[file, f"{tier}_col_idx"])
                            ):
                                if (
                                    self.filedb.df.loc[file, f"{tier}_col_idx"][i]
                                    in col_inds
                                ):
                                    col_tiers[tier].add(
                                        self.filedb.df.loc[file, f"{tier}_tables"][i]
                                    )
        else:
            # loop over selected files (db entries)
            for file in self.file_list:
                # this is the output object
                col_tiers[file] = {"tables": {}, "columns": {}}
                # Rows of FileDB.columns that include columns that we are interested in
                col_inds = set()
                for i, col_list in enumerate(self.filedb.columns):
                    if not set(list(col_list)).isdisjoint(columns):
                        col_inds.add(i)

                # Loop over tiers
                for level in self.levels:
                    for tier in self.tiers[level]:
                        col_tiers[file]["tables"][tier] = []
                        tier_col_idx = self.filedb.df.loc[file, f"{tier}_col_idx"]
                        if tier_col_idx is not None:
                            # Loop over tables
                            for i in range(len(tier_col_idx)):
                                col_idx = self.filedb.df.loc[file, f"{tier}_col_idx"][i]
                                if col_idx in col_inds:
                                    col_tiers[file]["tables"][tier].append(
                                        self.filedb.df.loc[file, f"{tier}_tables"][i]
                                    )
                                    col_in_tier = set.intersection(
                                        set(self.filedb.columns[col_idx]), set(columns)
                                    )
                                    for c in col_in_tier:
                                        col_tiers[file]["columns"][c] = tier

        return col_tiers

    def get_table_name(self, tier: str, tb: str) -> str:
        """Get the table name for a tier given its table identifier.

        Parameters
        ----------
        tier
            specify the tier whose table format will be used.
        tb
            the table identifier that will be passed to the table format.

        Returns
        -------
        table_name
            the name of the table in `tier` with table identifier `tb`
        """
        template = self.filedb.table_format[tier]
        fm = string.Formatter()
        parse_arr = np.array(list(fm.parse(template)))
        names = list(parse_arr[:, 1])
        if len(names) > 0:
            keyword = names[0]
            args = {keyword: tb}
            table_name = template.format(**args)
        else:
            table_name = template
        return table_name
