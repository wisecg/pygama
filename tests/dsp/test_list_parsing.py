import os
from pathlib import Path

import numpy as np

import pygama.lgdo.lh5_store as store
from pygama.dsp import build_dsp

config_dir = Path(__file__).parent / "configs"
dsp_file = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"


def test_list_parisng(lgnd_test_data):
    dsp_config = {
        "outputs": ["wf_out"],
        "processors": {
            "wf_out": {
                "function": "add",
                "module": "numpy",
                "args": ["[1,2,3,4,5]", "[6,7,8,9,10]", "out=wf_out"],
                "unit": "ADC",
            },
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

    df = store.load_nda(dsp_file, ["wf_out"], "geds/dsp/")

    assert np.all(df["wf_out"][:] == np.array([7, 9, 11, 13, 15]))
