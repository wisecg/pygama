import os
from pathlib import Path

import pygama.lgdo.lh5_store as store
from pygama.dsp import build_dsp

config_dir = Path(__file__).parent / "configs"
dsp_file = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"


def test_histogram_fixed_width(lgnd_test_data):
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
    build_dsp(
        f_raw=lgnd_test_data.get_path(
            "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
        ),
        f_dsp=dsp_file,
        dsp_config=dsp_config,
        write_mode="r",
    )
    assert os.path.exists(dsp_file)

    df = store.load_nda(dsp_file, ["hist_weights", "hist_borders"], "geds/dsp/")

    assert len(df["hist_weights"][0]) + 1 == len(df["hist_borders"][0])
    for i in range(2, len(df["hist_borders"][0])):
        a = df["hist_borders"][0][i - 1] - df["hist_borders"][0][i - 2]
        b = df["hist_borders"][0][i] - df["hist_borders"][0][i - 1]
        assert round(a, 2) == round(b, 2)