import pytest

from pygama import lgdo
from pygama.dsp.processing_chain import build_processing_chain
import numpy as np


def test_waveform_slicing(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_blsub"],
        "processors": {
            "wf_blsub": {
                "function": "bl_subtract",
                "module": "pygama.dsp.processors",
                "args": ["waveform[0:100]", "baseline", "wf_blsub"],
                "unit": "ADC",
            },
        },
    }
    proc_chain, _, tbl_out = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)

    assert list(tbl_out.keys()) == ["wf_blsub"]
    assert isinstance(tbl_out["wf_blsub"], lgdo.WaveformTable)
    assert tbl_out["wf_blsub"].wf_len == 100


def test_processor_none_arg(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_cum"],
        "processors": {
            "wf_cum": {
                "function": "cumsum",
                "module": "numpy",
                "args": ["waveform", 1, None, "wf_cum"],
                "kwargs": {"signature": "(n),(),()->(n)", "types": ["fii->f"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)

    dsp_config["processors"]["wf_cum"]["args"][2] = "None"
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_processor_kwarg_assignment(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_cum"],
        "processors": {
            "wf_cum": {
                "function": "cumsum",
                "module": "numpy",
                "args": ["waveform", "axis=1", "out=wf_cum"],
                "kwargs": {"signature": "(n),()->(n)", "types": ["fi->f"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)

    dsp_config["processors"]["wf_cum"]["args"][1] = "dtypo=None"
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    with pytest.raises(TypeError):
        proc_chain.execute(0, 1)


def test_processor_dtype_arg(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_cum"],
        "processors": {
            "wf_cum": {
                "function": "cumsum",
                "module": "numpy",
                "args": ["waveform", "axis=0", "dtype='int32'", "out=wf_cum"],
                "kwargs": {"signature": "(n),(),()->(n)", "types": ["fiU->i"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_scipy_gauss_filter(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_gaus"],
        "processors": {
            "wf_gaus": {
                "function": "gaussian_filter1d",
                "module": "scipy.ndimage",
                "args": [
                    "waveform",
                    "0.1*us",
                    "mode='reflect'",
                    "truncate=3",
                    "output=wf_gaus",
                ],
                "kwargs": {"signature": "(n),(),(),()->(n)", "types": ["ffUf->f"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_histogram_processor_fixed_witdth(spms_raw_tbl):
    dsp_config = {
        "outputs": ["hist_weights", "hist_borders"],
        "processors": {
            "hist_weights , hist_borders": {
                "function": "histogram",
                "module": "pygama.dsp.processors.histogram",
                "args": ["waveform", "hist_weights(100)", "hist_borders(101)"],
                "unit": ["none", "ADC"],
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_processor_variable_array_output(spms_raw_tbl):
    dsp_config = {
        "outputs": ["vt_max_out"],
        "processors": {
            "vt_max_out, vt_min_out, n_max_out, n_min_out": {
                "function": "get_multi_local_extrema",
                "module": "pygama.dsp.processors",
                "args": [
                    "waveform",
                    5,
                    5,
                    0,
                    10,
                    0,
                    "vt_max_out(10)",
                    "vt_min_out(10)",
                    "n_max_out",
                    "n_min_out",
                ],
                "unit": "ADC",
            }
        },
    }

    proc_chain, _, _ = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)



# Test that timing variables can be converted between multiple coordinate
# grids correctly. Also tests slicing with a stride. Pickoff a time from
# a windowed wf and a down-sampled waveform; they should be the same
def test_proc_chain_coordinate_grid(spms_raw_tbl):
    dsp_config = {
        "outputs": ["a_window", "a_downsample", "tp", "tp_window", "tp_downsample"],
        "processors": {
            "a_window": {
                "function": "fixed_time_pickoff",
                "module": "pygama.dsp.processors",
                "args": [
                    "waveform[2625:4025]",
                    "51.2*us + waveform.offset",
                    "'i'",
                    "a_window",
                ],
                "unit": ["ADC"],
            },
            "a_downsample": {
                "function": "fixed_time_pickoff",
                "module": "pygama.dsp.processors",
                "args": [
                    "waveform[0:8000:8]",
                    "51.2*us + waveform.offset",
                    "'i'",
                    "a_downsample",
                ],
                "unit": ["ADC"],
            },
            "tp": {
                "function": "time_point_thresh",
                "module": "pygama.dsp.processors",
                "args": ["waveform", "a_window", "52.48*us+waveform.offset", 0, "tp"],
                "unit": "ns",
            },
            "tp_window": {
                "function": "time_point_thresh",
                "module": "pygama.dsp.processors",
                "args": [
                    "waveform[2625:4025]",
                    "a_window",
                    "52.48*us+waveform.offset",
                    0,
                    "tp_window",
                ],
                "unit": "ns",
            },
            "tp_downsample": {
                "function": "time_point_thresh",
                "module": "pygama.dsp.processors",
                "args": [
                    "waveform[0:8000:8]",
                    "a_window",
                    "52.48*us+waveform.offset",
                    0,
                    "tp_downsample",
                ],
                "unit": "ns",
            },
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["a_window"].nda[0] == lh5_out["a_downsample"].nda[0]
    assert lh5_out["tp_window"].nda[0] == lh5_out["tp"].nda[0]
    assert -128 < lh5_out["tp_downsample"].nda[0] - lh5_out["tp"].nda[0] < 128


def test_proc_chain_round(spms_raw_tbl):
    dsp_config = {
        "outputs": ["waveform_round"],
        "processors": {"waveform_round": "round(waveform, 4)"},
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert np.all(
        np.rint(spms_raw_tbl["waveform"]["values"].nda[0] / 4) * 4
        == lh5_out["waveform_round"]["values"].nda[0]
    )


def test_proc_chain_as_type(spms_raw_tbl):
    dsp_config = {
        "outputs": ["waveform_32"],
        "processors": {"waveform_32": "astype(waveform, 'float32')"},
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert np.all(
        spms_raw_tbl["waveform"]["values"].nda[0] == lh5_out["waveform_32"]["values"].nda[0]
    )


def test_output_types(spms_raw_tbl):
    dsp_config = {
        "outputs": ["wf_out", "vov_max_out", "n_max_out", "aoa_out"],
        "processors": {
            "wf_out": "-waveform",
            "aoa_out": "n_max_out + [1, 3, 5, 7, 9]",
            "vov_max_out, vov_min_out, n_max_out, n_min_out": {
                "function": "get_multi_local_extrema",
                "module": "pygama.dsp.processors",
                "args": [
                    "waveform",
                    5,
                    0.1,
                    1,
                    10,
                    0,
                    "vov_max_out(20, vector_len=n_max_out)",
                    "vov_min_out(20, vector_len=n_min_out)",
                    "n_max_out",
                    "n_min_out",
                ],
                "unit": ["ns", "ns", "none", "none"],
            },
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert isinstance(lh5_out["n_max_out"], lgdo.Array)
    assert isinstance(lh5_out["wf_out"], lgdo.WaveformTable)
    assert isinstance(lh5_out["aoa_out"], lgdo.ArrayOfEqualSizedArrays)
    assert isinstance(lh5_out["vov_max_out"], lgdo.VectorOfVectors)


def test_output_attrs(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_blsub"],
        "processors": {
            "wf_blsub": {
                "function": "bl_subtract",
                "module": "pygama.dsp.processors",
                "args": ["waveform[0:100]", "baseline", "wf_blsub"],
                "unit": "ADC",
                "lh5_attrs": {"test_attr": "This is a test"},
            }
        },
    }
    proc_chain, _, lh5_out = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["wf_blsub"].attrs["test_attr"] == "This is a test"