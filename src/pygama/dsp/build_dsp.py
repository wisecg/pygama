"""
This module provides high-level routines for running signal processing chains
on waveform data.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import h5py
import numpy as np
from tqdm import tqdm

import pygama
import pygama.lgdo as lgdo
import pygama.lgdo.lh5_store as lh5
from pygama.dsp.errors import DSPFatal
from pygama.dsp.processing_chain import build_processing_chain
from pygama.lgdo.lgdo_utils import expand_path

log = logging.getLogger(__name__)


def build_dsp(
    f_raw: str,
    f_dsp: str,
    dsp_config: str | dict = None,
    lh5_tables: list[str] | str = None,
    database: str | dict = None,
    outputs: list[str] = None,
    n_max: int = np.inf,
    write_mode: str = None,
    buffer_len: int = 3200,
    block_width: int = 16,
    chan_config: dict[str, str] = None,
) -> None:
    """Convert raw-tier LH5 data into dsp-tier LH5 data by running a sequence
    of processors via the :class:`~.processing_chain.ProcessingChain`.

    Parameters
    ----------
    f_raw
        name of raw-tier LH5 file to read from.
    f_dsp
        name of dsp-tier LH5 file to write to.
    dsp_config
        :class:`dict` or name of JSON file containing
        :class:`~.processing_chain.ProcessingChain` config. See
        :func:`~.processing_chain.build_processing_chain` for details.
    lh5_tables
        list of HDF5 groups to consider in the input file. If ``None``, process
        all valid groups.
    database
        dictionary or name of JSON file containing a parameter database. See
        :func:`~.processing_chain.build_processing_chain` for details.
    outputs
        list of parameter names to write to the output file. If not provided,
        use list provided under ``"outputs"`` in the DSP configuration file.
    n_max
        number of waveforms to process.
    write_mode
        - ``None`` -- create new output file if it does not exist
        - `'r'` -- delete existing output file with same name before writing
        - `'a'` -- append to end of existing output file
        - `'u'` -- update values in existing output file
    buffer_len
        number of waveforms to read/write from/to disk at a time.
    block_width
        number of waveforms to process at a time.
    chan_config
        contains JSON DSP configuration file names for every table in
        `lh5_tables`.
    """

    if chan_config is not None:
        # clear existing output files
        if write_mode == "r":
            if os.path.isfile(f_dsp):
                os.remove(f_dsp)
            write_mode = "a"

        for tb, dsp_config in chan_config.items():
            log.debug(f"processing table: {tb} with DSP config file {dsp_config}")
            try:
                build_dsp(
                    f_raw,
                    f_dsp,
                    dsp_config,
                    [tb],
                    database,
                    outputs,
                    n_max,
                    write_mode,
                    buffer_len,
                    block_width,
                )
            except RuntimeError:
                log.debug(f"table {tb} not found")
        return

    raw_store = lh5.LH5Store()
    lh5_file = raw_store.gimme_file(f_raw, "r")
    if lh5_file is None:
        raise ValueError(f"input file not found: {f_raw}")
        return

    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = lh5.ls(f_raw)
    elif isinstance(lh5_tables, str):
        lh5_tables = [lh5_tables]
    elif not (
        hasattr(lh5_tables, "__iter__")
        and all(isinstance(el, str) for el in lh5_tables)
    ):
        raise RuntimeError("lh5_tables must be None, a string, or a list of strings")

    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if "raw" not in tb and lh5.ls(lh5_file, f"{tb}/raw"):
            lh5_tables[i] = f"{tb}/raw"
        elif not lh5.ls(lh5_file, tb):
            del lh5_tables[i]

    if len(lh5_tables) == 0:
        raise RuntimeError(f"could not find any valid LH5 table in {f_raw}")

    # get the database parameters. For now, this will just be a dict in a json
    # file, but eventually we will want to interface with the metadata repo
    if isinstance(database, str):
        with open(expand_path(database)) as db_file:
            database = json.load(db_file)

    if database and not isinstance(database, dict):
        raise ValueError("input database is not a valid JSON file or dict")

    if write_mode is None and os.path.isfile(f_dsp):
        raise FileExistsError(
            f"output file {f_dsp} exists. Set the 'write_mode' keyword"
        )

    # clear existing output files
    if write_mode == "r":
        if os.path.isfile(f_dsp):
            os.remove(f_dsp)

    # write processing metadata
    dsp_info = lgdo.Struct()
    dsp_info.add_field("timestamp", lgdo.Scalar(np.uint64(time.time())))
    dsp_info.add_field("python_version", lgdo.Scalar(sys.version))
    dsp_info.add_field("numpy_version", lgdo.Scalar(np.version.version))
    dsp_info.add_field("h5py_version", lgdo.Scalar(h5py.version.version))
    dsp_info.add_field("hdf5_version", lgdo.Scalar(h5py.version.hdf5_version))
    dsp_info.add_field("pygama_version", lgdo.Scalar(pygama.__version__))

    # loop over tables to run DSP on
    for tb in lh5_tables:
        # load primary table and build processing chain and output table
        tot_n_rows = raw_store.read_n_rows(tb, f_raw)
        if n_max and n_max < tot_n_rows:
            tot_n_rows = n_max

        chan_name = tb.split("/")[0]
        db_dict = database.get(chan_name) if database else None
        tb_name = tb.replace("/raw", "/dsp")

        write_offset = 0
        raw_store.gimme_file(f_dsp, "a")
        if write_mode == "a" and lh5.ls(f_dsp, tb_name):
            write_offset = raw_store.read_n_rows(tb_name, f_dsp)

        # Main processing loop
        lh5_it = lh5.LH5Iterator(f_raw, tb, buffer_len=buffer_len)
        proc_chain = None
        for lh5_in, start_row, n_rows in lh5_it:
            # Initialize
            if proc_chain is None:
                proc_chain, lh5_it.field_mask, tb_out = build_processing_chain(
                    lh5_in, dsp_config, db_dict, outputs, block_width
                )
                if log.getEffectiveLevel() <= logging.INFO:
                    progress_bar = tqdm(
                        desc=f"Processing table {tb}",
                        total=tot_n_rows,
                        delay=2,
                        unit=" rows",
                    )

            n_rows = min(tot_n_rows - start_row, n_rows)
            try:
                proc_chain.execute(0, n_rows)
            except DSPFatal as e:
                # Update the wf_range to reflect the file position
                e.wf_range = f"{e.wf_range[0]+start_row}-{e.wf_range[1]+start_row}"
                raise e

            raw_store.write_object(
                obj=tb_out,
                name=tb_name,
                lh5_file=f_dsp,
                n_rows=n_rows,
                wo_mode="o" if write_mode == "u" else "a",
                write_start=write_offset + start_row,
            )

            if log.getEffectiveLevel() <= logging.INFO:
                progress_bar.update(n_rows)

            if start_row + n_rows >= tot_n_rows:
                break

        if log.getEffectiveLevel() <= logging.INFO:
            progress_bar.close()

    raw_store.write_object(dsp_info, "dsp_info", f_dsp, wo_mode="o")
