import os
from pathlib import Path

import pytest

from pygama import lgdo
from pygama.dsp import build_dsp
from pygama.lgdo.lh5_store import LH5Store, ls

config_dir = Path(__file__).parent / "configs"


def test_build_dsp_json(lgnd_test_data, tmptestdir):
    out_name = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        out_name,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )
    assert os.path.exists(out_name)



def test_build_dsp_errors(lgnd_test_data, tmptestdir):
    with pytest.raises(FileExistsError):
        build_dsp(
            lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
            f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            dsp_config=f"{config_dir}/icpc-dsp-config.json",
        )

    with pytest.raises(FileNotFoundError):
        build_dsp(
            "non-existent-file.lh5",
            f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            dsp_config=f"{config_dir}/icpc-dsp-config.json",
            write_mode="r",
        )


@pytest.fixture(scope="session")
def dsp_test_file_spm(lgnd_test_data, tmptestdir):
    chan_config = {
        "ch0/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch1/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch2/raw": f"{config_dir}/sipm-dsp-config.json",
    }

    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/L200-comm-20211130-phy-spms.lh5"),
        out_file,
        {},
        n_max=5,
        lh5_tables=chan_config.keys(),
        chan_config=chan_config,
        write_mode="r",
    )

    assert os.path.exists(out_file)

    return out_file


def test_build_dsp_spms_channelwise(dsp_test_file_spm):
    assert ls(dsp_test_file_spm) == ["ch0", "ch1", "ch2", "dsp_info"]
    assert ls(dsp_test_file_spm, "ch0/") == ["ch0/dsp"]
    assert ls(dsp_test_file_spm, "ch0/dsp/") == [
        "ch0/dsp/energies",
        "ch0/dsp/trigger_pos",
    ]

    store = LH5Store()
    lh5_obj, n_rows = store.read_object("/ch0/dsp/energies", dsp_test_file_spm)
    assert isinstance(lh5_obj, lgdo.VectorOfVectors)
    assert len(lh5_obj) == 5