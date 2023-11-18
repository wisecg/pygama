from __future__ import annotations

import glob
import json
import logging
import os
import time

import numpy as np
from tqdm import tqdm

from pygama import lgdo
from pygama.math.utils import sizeof_fmt

from .fc.fc_streamer import FCStreamer
from .orca.orca_streamer import OrcaStreamer
from .raw_buffer import RawBufferLibrary, write_to_lh5_and_clear

log = logging.getLogger(__name__)


def build_raw(
    in_stream: int,
    in_stream_type: str = None,
    out_spec: str | dict | RawBufferLibrary = None,
    buffer_size: int = 8192,
    n_max: int = np.inf,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """Convert data into LEGEND HDF5 raw-tier format.

    Takes an input stream of a given type and writes to output file(s)
    according to the user's a specification.

    Parameters
    ----------
    in_stream
        the name of the input stream to be converted. Typically a filename,
        including path. Can use environment variables. Some streamers may be
        able to (eventually) accept e.g. streaming over a port as an input.

    in_stream_type : 'ORCA', 'FlashCam', 'LlamaDaq', 'Compass' or 'MGDO'
        type of stream used to write the input file.

    out_spec
        Specification for the output stream.

        - if None, uses ``{in_stream}.lh5`` as the output filename.
        - if a str not ending in ``.json``, interpreted as the output filename.
        - if a str ending in ``.json``, interpreted as a filename containing
          json-shorthand for the output specification (see :mod:`.raw_buffer`).
        - if a JSON dict, should be a dict loaded from the json shorthand
          notation for RawBufferLibraries (see :mod:`.raw_buffer`), which is
          then used to build a :class:`.RawBufferLibrary`.
        - if a :class:`.RawBufferLibrary`, the mapping of data to output file /
          group is taken from that.

    buffer_size
        default size to use for data buffering.

    n_max
        maximum number of rows of data to process from the input file.

    overwrite
        sets whether to overwrite the output file(s) if it (they) already exist.

    **kwargs
        sent to :class:`.RawBufferLibrary` generation as `kw_dict`.
    """

    # convert any environment variables in in_stream so that we can check for readability
    in_stream = os.path.expandvars(in_stream)
    # later: fix if in_stream is not a file
    # (e.g. a socket, which exists if open and has size = np.inf)
    if not os.path.exists(in_stream):
        raise FileNotFoundError(f"file {in_stream} not found")

    in_stream_size = os.stat(in_stream).st_size

    # try to guess the input stream type if it's not provided
    if in_stream_type is None:
        i_ext = in_stream.split("/")[-1].rfind(".")
        if i_ext == -1:
            if OrcaStreamer.is_orca_stream(in_stream):
                in_stream_type = "ORCA"
            else:
                raise RuntimeError("unknown file type. Specify in_stream_type")
        else:
            ext = in_stream.split("/")[-1][i_ext + 1 :]
            if ext == "fcio":
                in_stream_type = "FlashCam"
            elif OrcaStreamer.is_orca_stream(in_stream):
                in_stream_type = "ORCA"
            else:
                raise RuntimeError(
                    f"unknown file extension {ext}. Specify in_stream_type"
                )

    # process out_spec and setup rb_lib if specified
    rb_lib = None
    if isinstance(out_spec, str) and out_spec.endswith(".json"):
        with open(out_spec) as json_file:
            out_spec = json.load(json_file)
    if isinstance(out_spec, dict):
        out_spec = RawBufferLibrary(json_dict=out_spec, kw_dict=kwargs)
    if isinstance(out_spec, RawBufferLibrary):
        rb_lib = out_spec
    # if no rb_lib, write all data to file
    if out_spec is None:
        out_spec = in_stream
        i_ext = out_spec.rfind(".")
        if i_ext != -1:
            out_spec = out_spec[:i_ext]
        out_spec += ".lh5"
    # by now, out_spec should be a str or a RawBufferLibrary
    if not isinstance(out_spec, str) and not isinstance(out_spec, RawBufferLibrary):
        raise TypeError(f"unknown out_spec type {type(out_spec).__name__}")

    # modify buffer_size if necessary for n_max
    if buffer_size < 1:
        raise ValueError(f"bad buffer_size {buffer_size}")
    if buffer_size > n_max:
        buffer_size = n_max

    log.info(f"input: {in_stream}")
    out_files = [out_spec]
    if isinstance(out_spec, RawBufferLibrary):
        out_files = out_spec.get_list_of("out_stream")
    if len(out_files) == 1:
        log.info(f"output: {out_files[0]}")
    else:
        log.info("output:")
        for out_file in out_files:
            log.info(f" -> {out_file}")
    log.info(f"buffer size: {buffer_size}")
    if n_max < np.inf:
        log.info(f"maximum number of events: {n_max}")
    if log.getEffectiveLevel() >= logging.INFO:
        if n_max < np.inf:
            progress_bar = tqdm(desc="Decoding", total=n_max, delay=2, unit=" rows")
        else:
            progress_bar = tqdm(
                desc="Decoding",
                total=in_stream_size,
                delay=2,
                unit=" B",
                unit_scale=True,
            )

    # start a timer and a byte counter
    t_start = time.time()

    # select the appropriate streamer for in_stream
    streamer = None
    if in_stream_type == "ORCA":
        streamer = OrcaStreamer()
    elif in_stream_type == "FlashCam":
        streamer = FCStreamer()
    elif in_stream_type == "LlamaDaq":
        raise NotImplementedError("LlamaDaq streaming not yet implemented")
    elif in_stream_type == "Compass":
        raise NotImplementedError("Compass streaming not yet implemented")
    elif in_stream_type == "MGDO":
        raise NotImplementedError("MGDO streaming not yet implemented")
    else:
        raise NotImplementedError("unknown input stream type {in_stream_type}")

    # initialize the stream and read header. Also initializes rb_lib
    if log.getEffectiveLevel() <= logging.INFO:
        progress_bar.update(0)

    out_stream = out_spec if isinstance(out_spec, str) else ""
    header_data = streamer.open_stream(
        in_stream,
        rb_lib=rb_lib,
        buffer_size=buffer_size,
        chunk_mode="full_only",
        out_stream=out_stream,
    )
    rb_lib = streamer.rb_lib
    if log.getEffectiveLevel() <= logging.INFO and n_max == np.inf:
        progress_bar.update(streamer.n_bytes_read)

    # rb_lib should now be fully initialized. Check if files need to be
    # overwritten or if we need to stop to avoid overwriting
    out_files = rb_lib.get_list_of("out_stream")
    for out_file in out_files:
        colpos = out_file.find(":")
        if colpos != -1:
            out_file = out_file[:colpos]
        out_file_glob = glob.glob(out_file)
        if len(out_file_glob) == 0:
            continue
        if len(out_file_glob) > 1:
            raise RuntimeError(
                f"got multiple matches for out_file {out_file}: {out_file_glob}"
            )
        if not overwrite:
            raise FileExistsError(
                f"file {out_file_glob[0]} exists. Use option overwrite to proceed."
            )

        os.remove(out_file_glob[0])

    # Write header data
    lh5_store = lgdo.LH5Store(keep_open=True)
    write_to_lh5_and_clear(header_data, lh5_store)

    # Now loop through the data
    n_bytes_last = streamer.n_bytes_read
    while True:
        chunk_list = streamer.read_chunk()
        if log.getEffectiveLevel() <= logging.INFO and n_max == np.inf:
            progress_bar.update(streamer.n_bytes_read - n_bytes_last)
            n_bytes_last = streamer.n_bytes_read
        if len(chunk_list) == 0:
            break
        n_read = 0
        for rb in chunk_list:
            if rb.loc > n_max:
                rb.loc = n_max
            n_max -= rb.loc
            n_read += rb.loc
        if log.getEffectiveLevel() <= logging.INFO and n_max < np.inf:
            progress_bar.update(n_read)
        write_to_lh5_and_clear(chunk_list, lh5_store)
        if n_max <= 0:
            break

    streamer.close_stream()
    progress_bar.close()

    out_files = rb_lib.get_list_of("out_stream")
    if len(out_files) == 1:
        out_file = out_files[0].split(":", 1)[0]
        if os.path.exists(out_file):
            file_size = os.stat(out_file).st_size
            log.info(f"output file: {out_file} ({sizeof_fmt(file_size)})")
        else:
            log.info(f"output file: {out_file} (not written)")
    else:
        log.info("output files:")
        unique_outfilenames = {file.split(":", 1)[0] for file in out_files}
        for out_file in unique_outfilenames:
            if os.path.exists(out_file):
                file_size = os.stat(out_file).st_size
                log.info(f" -> {out_file} ({sizeof_fmt(file_size)})")
            else:
                log.info(f" -> {out_file} (not written)")
    log.info(f"total converted: {sizeof_fmt(streamer.n_bytes_read)}")
    elapsed = time.time() - t_start
    log.info(f"conversion speed: {sizeof_fmt(streamer.n_bytes_read/elapsed)}ps")
