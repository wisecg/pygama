Digital Signal Processing
=========================

*Under construction...*

Writing custom processors
-------------------------

.. code-block:: python

    # 1) Import Python modules
    #
    import numpy as np
    from numba import guvectorize

    from pygama.dsp.errors import DSPFatal


    # 2) Provide instructions to Numba
    #
    # Documentation about Numba guvectorize decorator:
    # https://numba.pydata.org/numba-doc/latest/user/vectorize.html#the-guvectorize-decorator
    #
    # Notes:
    # - Use "nopython=True, cache=True"
    # - Use two declarations, one for 32-bit variables and one for 64-bit variables
    # - Do not use "int" as it does not support an NaN value
    # - Use [:] for all output parameters
    #
    @guvectorize(["void(float32[:], float32, float32, float32[:])",
                  "void(float64[:], float64, float64, float64[:])"],
                  "(n),(),()->()", nopython=True, cache=True)

    # 3) Define the processor interface
    #
    # Notes:
    # - Add the "_in"/"_out" suffix to the name of the input/output variables
    # - Use "w_" for waveforms, "t_" for indexes, "a_" for amplitudes
    # - Use underscore casing for the name of the processor and variables, e.g.,
    #   "a_trap_energy_in" or "t_trigger_in"
    #
    def the_processor_template(w_in, t_in, a_in, w_out, t_out):

        # 4) Document the algorithm
        #
        """
        Add here a complete description of what the processor does, including the
        meaning of input and output variables.
        """

        # 5) Initialize output parameters
        #
        # Notes:
        # - All output parameters should be initializes to a NaN.  If a processor
        #   fails, its output parameters should have the default NaN value
        # - Use np.nan for both variables and arrays
        #
        w_out[:] = np.nan # use [:] for arrays
        t_out[0] = np.nan # use [0] for scalar parameters

        # 6) Check inputs
        #
        # There are two kinds of checks:
        # - NaN checks.  A processor might depend on others, i.e., its input
        #   parameters are the output parameters of an other processors.  When a
        #   processor fails, all processors depending on its output should not run.
        #   Thus, skip this processor if a NaN value is detected and return NaN
        #   output parameters to propagate the failure throughout the processing chain.
        # - In-range checks.  Check if indexes are within 0 and len(waveform),
        #   amplitudes are positive, etc.  A failure of this check implies errors in
        #   the DSP JSON config file.  Abort the analysis immediately.
        #
        if np.isnan(w_in).any() or np.isnan(t_in) or np.isnan(a_in):
            return

        if a_in < 0:
            raise DSPFatal('The error message goes here')

        # 7) Algorithm
        #
        # Loop over waveforms by using a "for i in range(.., .., ..)" instruction.
        # Avoid loops based on a while condition which might lead to segfault.  Avoid
        # also enumerate/ndenumerate to keep code as similar as possible among all
        # processors.
        #
        # Example of an algorithm to find the first index above "t_in" in which the
        # signal crossed the value "a_in"
        #
        for i in range(t_in, 1, 1):
            if w_in[i] >= a_in and w_in[i-1] < a_in:
                t_out[0] = i
                return
