import copy
import inspect
import os
import re
import shutil
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pytest
from legendtestdata import LegendTestData
import pygama.dsp.processors  # noqa: F401
from pygama.lgdo import LH5Store
from pygama.raw.build_raw import build_raw


config_dir = Path(__file__).parent / "dsp" / "configs"
_tmptestdir = os.path.join(
    gettempdir(), f"dspeed-tests-{getuser()}-{str(uuid.uuid4())}"
)


@pytest.fixture(scope="session")
def tmptestdir():
    os.mkdir(_tmptestdir)
    return _tmptestdir


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        shutil.rmtree(_tmptestdir)


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("69aaeee")
    return ldata


@pytest.fixture(scope="session")
def geds_raw_tbl(lgnd_test_data):
    store = LH5Store()
    obj, _ = store.read_object(
        "/geds/raw",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        n_rows=10,
    )
    return obj


@pytest.fixture(scope="session")
def spms_raw_tbl(lgnd_test_data):
    store = LH5Store()
    obj, _ = store.read_object(
        "/ch0/raw",
        lgnd_test_data.get_path("lh5/L200-comm-20211130-phy-spms.lh5"),
        n_rows=10,
    )
    return obj


@pytest.fixture(scope="session")
def compare_numba_vs_python():
    def numba_vs_python(func, *inputs):
        """
        Function for testing that the numba and python versions of a
        function are equal.

        Parameters
        ----------
        func
            The Numba-wrapped function to be tested
        *inputs
            The various inputs to be passed to the function to be
            tested.

        Returns
        -------
        func_output
            The output of the function to be used in a unit test.

        """

        if "->" in func.signature:
            # parse outputs from function signature
            all_params = list(inspect.signature(func).parameters)
            output_sizes = re.findall(r"(\(n*\))", func.signature.split("->")[-1])
            noutputs = len(output_sizes)
            output_names = all_params[-noutputs:]

            # numba outputs
            outputs_numba = func(*inputs)
            if noutputs == 1:
                outputs_numba = [outputs_numba]

            # unwrapped python outputs
            func_unwrapped = inspect.unwrap(func)
            output_dict = {key: np.empty(len(inputs[0])) for key in output_names}
            func_unwrapped(*inputs, **output_dict)
            for spec, key in zip(output_sizes, output_dict):
                if spec == "()":
                    output_dict[key] = output_dict[key][0]
            outputs_python = [output_dict[key] for key in output_names]
        else:
            # we are testing a factory function output, which updates
            # a single output in-place
            noutputs = 1
            # numba outputs
            func(*inputs)
            outputs_numba = copy.deepcopy(inputs[-noutputs:])

            # unwrapped python outputs
            func_unwrapped = inspect.unwrap(func)
            func_unwrapped(*inputs)
            outputs_python = copy.deepcopy(inputs[-noutputs:])

        # assert that numba and python are the same up to floating point
        # precision, setting nans to be equal
        assert np.allclose(outputs_numba, outputs_python, equal_nan=True)

        # return value for comparison with expected solution
        return outputs_numba[0] if noutputs == 1 else outputs_numba

    return numba_vs_python