#!/usr/bin/env python3
import os
import time
import argparse
import h5py
import numpy as np
import pandas as pd
from pprint import pprint

import pygama.lh5 as lh5

def hit_to_evt(f_hit:str, f_evt:str=None, lh5_tables:list=None, copy_cols:list=None,
               overwrite:bool=True, evt_tb_name:str=None, builder_config:dict=None):
    """
    Given an input file containing separate Tables for each active data taker,
    create an output Table which merges all channels together, sorting everything
    by ascending timestamp, and grouping coincident hits in each channel together
    as "events".  The output table will contain columns ['ix_evt', 'ix_sub']
    denoting the event and sub-event (hit), in addition to any other columns
    we desire to copy over into the output file.

    Parameters
    ----------
    f_hit : str
        Input file containing multiple LH5 tables, one for each channel.
        Usually a RAW or DSP file.
    f_evt : str
        Output file name.
    lh5_tables : list (optional)
        Specify the list of tables to analyze, instead of detecting them
        automatically by default.
    copy_cols : list (optional)
        Specify the list of columns to copy into the output file.  For a DSP
        file, it should be all columns by default, while for a RAW file, we
        probably do NOT want to copy the waveforms into the output file.
    evt_tb_name : str (optional)
        Specify the name of the output table instead of automatically setting it.
    builder_config : dict (optional)
        Give a dict of kwargs for `cluster_events`, including coincidence window
        times, conversion to seconds, and names of timestamp and channel columns.

    Returns
    -------
    tcm : pd.Dataframe (conditional!)
        If f_evt is not set, don't write to LH5 file, and instead return a
        DataFrame.
    """
    t_start = time.time()

    # get a list of LH5 tables to decode in the input file
    if lh5_tables is None:
        lh5_tables, lh5_cols = [], {}

        def find_tables(name, obj):
            for attr, val in obj.attrs.items():
                if 'table' in val and 'waveform' not in name: # ignore waveform tables
                    lh5_tables.append(name)
                    col_list = val.replace('table{', '').replace('}','').split(',')
                    if 'waveform' in col_list: col_list.remove('waveform')
                    lh5_cols[name] = col_list

        with h5py.File(f_hit) as hf:
            # to iterate through all groups, you have to pass 'visititems' a function
            hf.visititems(find_tables)

        # for now, assume all column names are the same!
        if copy_cols is None:
            tb_0 = list(lh5_cols.keys())[0]
            copy_cols = lh5_cols[tb_0]

    # get event builder options
    if builder_config is None:
        builder_config = {
        'ts_unit' : 1e-8,       # give the conversion of timestamps to seconds
        'ch_col' : 'channel',   # name of column with channel ID (should be int)
        'ts_col' : 'timestamp', # name of column with timestamps
        'coin_window' : 4e-6,   # length of coincidence window in seconds
        'data_cols' : copy_cols # columns to copy over into output table
        }

    # show status before running
    print('Tables to merge:\n', lh5_tables)
    print('Event builder config:')
    print(builder_config)

    # read LH5 tables and convert to DataFrame.  This copy and conversion is
    # a required step!  (Clint will be happy to discuss the reasons with you.)
    print('Loading dataframes ...')
    dfs = []
    for tb in lh5_tables:
        df = lh5.load_dfs(f_hit, copy_cols, lh5_group=tb, idx_list=None, verbose=False)
        print(f'Table: {tb}, rows:{len(df)}')
        dfs.append(df)
        # print(df)

    # # sanity check - test cluster_events with an extra fake channel, #7, with slightly changed timestamps
    # from scipy.ndimage.filters import gaussian_filter
    # df_extra = dfs[1].copy()
    # df_extra['timestamp'] = gaussian_filter(df_extra['timestamp'], sigma=0.1)
    # df_extra['channel'] = 7
    # dfs.append(df_extra)

    # run cluster_events
    print('Building time coincidence map ...')
    tcm = cluster_events(dfs, **builder_config)
    print('Done.  TCM output columns:')
    print(tcm.columns)

    # examine tcm
    # tcm_view = ['channel','tcm_sec','tcm_dt','ix_evt','ix_hit']
    # pd.options.display.max_rows = 50
    # pd.options.display.float_format = '{:,.3e}'.format
    # print(tcm[tcm_view].head(50))
    # print(tcm)
    # print(tcm.columns)

    # write to file if f_evt is set, or return in-memory DataFrame if it isn't.
    if f_evt is None:
        return tcm

    print('Writing to file:', f_evt)
    if os.path.exists(f_evt):
        os.remove(f_evt)

    sto = lh5.Store()
    col_dict = {col : lh5.Array(tcm[col].values, attrs={'units':''}) for col in tcm.columns}
    tb_tcm = lh5.Table(size=len(tcm), col_dict=col_dict)
    tb_name = 'events'
    sto.write_object(tb_tcm, tb_name, f_evt)

    # # sanity check, read it back in and convert back to DataFrame
    # df_check = lh5.load_dfs(f_evt, tcm.columns, lh5_group='events', idx_list=None, verbose=False)
    # print(tcm)
    # print(df_check)

    t_elap = time.time() - t_start
    print(f'Done!  Time elapsed: {t_elap:.2f} sec.')


def cluster_events(tb_list:list, ts_unit:float=1e-8, ch_col:str='channel',
                   ts_col:str='timestamp', coin_window:float=4e-6,
                   data_cols:list=None):
    """
    Create a time coincidence map (TCM), given a list of data tables from separate channels.
    Assume that all tables are from different channels in a SINGLE cycle file.
    Hopefully we won't need to extend this to event building across multiple cycles
    as in Majorana.
    - Sort events in DataFrames by strictly ascending timestamps
    - Group hits if the difference between times is less than `coin_window` (default length 4*us)
    - Don't allow events from the same channel into any event.
    - TODO: add a "t_offset" output column which is the time since the FIRST hit in the event.

    Parameters
    ----------
    tb_list : list
        A list of N tables containing 'channel' & 'timestamp' columns.
    ts_unit : float (optional)
        Conversion factor for timestamps in native units to SECONDS.
    ch_col : str (optional)
        Name of the common column to use as 'channel'.  Default is 'channel'.
    ts_col : str (optional)
        Name of the common column to use as 'timestamp'.  Default is 'timestamp'.
    coin_window : float (optional)
        The default clustering time in seconds. (4e-6 sec is good for HPGe detectors)
        If events in other channels occur within the first window, extend the window
        by this amount and search again.
    data_cols : list (optional)
        Copy over additional columns (such as DSP parameters) when we sort events together
        by timestamps.  This is handy for event building DSP files with pandas, then
        optionally converting back to LH5 tables for file i/o.

    Returns
    -------
    tcm : DataFrame
        Return a table with NEW columns: ['ix_evt','ix_hit','tcm_sec','tcm_dt','idx_row_{ch}'].
        If data_cols is set, these columns will be added to the input tables,
        usually dsp data.
    """
    if not isinstance(tb_list[0], pd.DataFrame):
        print("LH5 tables not supported yet, but easy, just need to convert them to DataFrame")
        return None

    # create 'ts_sec' timestamps for each channel, using the supplied conversion to seconds.
    for df in tb_list:
        df['tcm_sec'] = df[ts_col] * ts_unit

        # throw an error if we detect resetting timestamps
        ts = df['tcm_sec'].values
        tdiff = np.diff(ts)
        tdiff = np.insert(tdiff, 0 , 0)
        ix_resets = np.where(tdiff < 0)
        if len(ix_resets[0]) > 0:
            print('Warning! timestamps reset for this channel.  TC map will be total nonsense!  AAAAAHHH')

        # save the original index for reverse lookup
        chan = df[ch_col].unique()[0]
        df[f'ix_row'] = df.index.values

    # # make a list of columns from the input tables we want to copy over
    # copy_cols = ['tcm_sec'] + ch_rows
    # if data_cols is not None:
    #     copy_cols.extend(data_cols)
    # copy_cols = sorted(list(set(copy_cols))) # drop duplicates

    # create a new dataframe where we SORT ALL ROWS by a striclty ascending timestamp
    dfs = tb_list
    tcm = pd.concat(dfs).sort_values('tcm_sec')
    tcm.reset_index(inplace=True, drop=True)

    # create the event column by comparing the time since last event to the coincindence window
    tcm['tcm_dt'] = tcm['tcm_sec'].diff()
    tcm['ix_evt'] = (tcm.tcm_dt > coin_window).cumsum()

    # create the sub-event column (groupbys are easy with pandas, hard with LH5 tables.)
    tcm['ix_hit'] = tcm.groupby(tcm.ix_evt).cumcount()

    return tcm


def lookup_wfs(wf_files:list, tcm_files:list, pd_query:str=None, n_evts:int=None,
              ch_col:str='channel', return_lh5:bool=False):
    """
    Take the TCM and pull ~5-10 waveforms for events, using ix_row + channel as the lookup.
    We supply a list of waveform files (potentially from DataGroup), a TCM for each file,
    and an optional pandas query to select events using other columns in the TCM.
    Returns a DataFrame with waveform data (merged across cycle files?)

    Parameters
    ----------
    wf_files : list[str]
        Give a list of raw cycle files to look up waveforms in.
    tcm_files : list[DataFrame]
        Give a time coincidence map (generated by `cluster_events` for each file in `wf_files`.
    pd_query : str (optional)
        Supply an additional query to filter waveforms by one of the columns in the TCM,
        e.g. by energy, channel, or pulse shape parameter.  This is done as a standard df.query().
    n_evts : int (optional)
        Limit the number of waveforms returned, usually for testing purposes.
    ch_col : str (optional)
        specify the name of the channel column (default is 'channel')
    return_lh5 : bool (optional)
        If this

    Returns
    -------
    tcm_wfs : pd.DataFrame or LH5.Table (conditional!)
        DataFrame containing an additional column `waveform` with an embedded np.array for each row.
        If `return_lh5` is set to `True`, return an LH5 table with an embedded `waveform` table
        suitable for DSP with `ProcessingChain`.
    """

    # make sure we have 1-1 matching between wf_files and tcm_files
    if len(wf_files) != len(tcm_files):
        print('I need one TCM for each raw cycle file in the list!')
        return None

    # loop over files
    dfs = []
    for i, f_raw in enumerate(wf_files):

        # run query on the TCM if we have one
        # if pd_query is not None:
        #     df_tcm = tcm_files[i].query(pd_query).copy()
        # else:
        #     df_tcm = tcm_files[i]

        # hmm, this is hard to think about how to implement generally.
        # sometimes you want all wfs from the event where any hit in the event matches the query,
        # and sometimes you only want hits that all pass the query.  maybe a bool switch controls
        # whether you groupby ix_evt befor running the query?  idk.   come back to this.
        # df_tcm = tcm_files[i].groupby('ix_evt').apply(query)

        # limit the number of events
        if n_evts is not None:
            df_tcm = tcm_files[i][:n_evts]
        else:
            df_tcm = tcm_files[i]

        # print(df_tcm)

        # use `ch_col` and `ix_row` to pull out waveforms.  try a groupby over channel.
        df_tcm = df_tcm.groupby(ch_col).apply(get_wfs)


def get_wfs(df):
    """Helper function for the groupby in `lookup_wfs`."""
    print(df)


if __name__=='__main__':
    doc = """Demonstrate usage of the `hit_to_evt` function, to build a time
    coincidence map and organize data from many channels in a single cycle file,
    into an event-like structure where coincindent events are time-ordered and
    grouped together into sub-events."""

    # parse user args
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('input', type=str, help='input file name (required)')
    arg('-o', '--output', type=str, help='output file name')
    args = par.parse_args()

    # set i/o
    f_in = args.input
    f_out = None if not args.output else args.output

    # run hit_to_evt
    hit_to_evt(f_in, f_out)
