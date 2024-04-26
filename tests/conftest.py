import inspect
import os
import re
from pathlib import Path

import numpy as np
import pytest
from legend_testdata import LegendTestData

# from pygama.dsp import build_dsp
# from pygama.raw import build_raw

config_dir = Path(__file__).parent / "dsp" / "configs"


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("39f9927")
    return ldata


# @pytest.fixture(scope="session")
# def dsp_test_file(lgnd_test_data):
#     out_name = "/tmp/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5"
#     build_dsp(
#         lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
#         out_name,
#         dsp_config=f"{config_dir}/icpc-dsp-config.json",
#         database={"pz": {"tau": 27460.5}},
#         write_mode="r",
#     )
#     assert os.path.exists(out_name)

#     return out_name


# @pytest.fixture(scope="session")
# def multich_raw_file(lgnd_test_data):
#     out_file = "/tmp/L200-comm-20211130-phy-spms.lh5"
#     out_spec = {
#         "FCEventDecoder": {
#             "ch{key}": {
#                 "key_list": [[0, 6]],
#                 "out_stream": out_file + ":{name}",
#                 "out_name": "raw",
#             }
#         }
#     }

#     build_raw(
#         in_stream=lgnd_test_data.get_path("fcio/L200-comm-20211130-phy-spms.fcio"),
#         out_spec=out_spec,
#         overwrite=True,
#     )
#     assert os.path.exists(out_file)

#     return out_file


# @pytest.fixture(scope="session")
# def dsp_test_file_spm(multich_raw_file):
#     chan_config = {
#         "ch0/raw": f"{config_dir}/sipm-dsp-config.json",
#         "ch1/raw": f"{config_dir}/sipm-dsp-config.json",
#         "ch2/raw": f"{config_dir}/sipm-dsp-config.json",
#     }

#     out_file = "/tmp/L200-comm-20211130-phy-spms_dsp.lh5"
#     build_dsp(
#         multich_raw_file,
#         out_file,
#         {},
#         n_max=5,
#         lh5_tables=chan_config.keys(),
#         chan_config=chan_config,
#         write_mode="r",
#     )

#     assert os.path.exists(out_file)

#     return out_file


# @pytest.fixture(scope="session")
# def compare_numba_vs_python():
#     def numba_vs_python(func, *inputs):
#         """
#         Function for testing that the numba and python versions of a
#         function are equal.

#         Parameters
#         ----------
#         func
#             The Numba-wrapped function to be tested
#         *inputs
#             The various inputs to be passed to the function to be
#             tested.

#         Returns
#         -------
#         func_output
#             The output of the function to be used in a unit test.

#         """
#         # parse outputs from function signature
#         all_params = list(inspect.signature(func).parameters)
#         output_sizes = re.findall(r"(\(n*\))", func.signature.split("->")[-1])
#         noutputs = len(output_sizes)
#         output_names = all_params[-noutputs:]

#         # numba outputs
#         outputs_numba = func(*inputs)
#         if noutputs == 1:
#             outputs_numba = [outputs_numba]

#         # unwrapped python outputs
#         func_unwrapped = inspect.unwrap(func)
#         output_dict = {key: np.empty(len(inputs[0])) for key in output_names}
#         func_unwrapped(*inputs, **output_dict)
#         for spec, key in zip(output_sizes, output_dict):
#             if spec == "()":
#                 output_dict[key] = output_dict[key][0]
#         outputs_python = [output_dict[key] for key in output_names]

#         # assert that numba and python are the same up to floating point
#         # precision, setting nans to be equal
#         assert np.allclose(outputs_numba, outputs_python, equal_nan=True)

#         # return value for comparison with expected solution
#         return outputs_numba[0] if noutputs == 1 else outputs_numba

#     return numba_vs_python
