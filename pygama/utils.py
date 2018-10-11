"""
this is a miscellaneous functions folder.  i found one useful before (waveLibs)
so is it good practice to keep a file like this?
there are just two very short functions here, they would be easy to move,
unless we anticipate this file growing.
"""

import sys


def update_progress(progress, runNumber=None):
    """ adapted from from https://stackoverflow.com/a/15860757 """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    if runNumber is None:
        text = "\rPROGRESS : [{}] {:0.3f}% {}".format(
            "#" * block + "-" * (barLength - block), progress * 100, status)
    else:
        text = "\rPROGRESS : [{}] {:0.3f}% {} (Run {})".format(
            "#" * block + "-" * (barLength - block), progress * 100, status,
            runNumber)
    sys.stdout.write(text)
    sys.stdout.flush()


def get_bin_centers(bins):
    return bins[:-1] + 0.5 * (bins[1] - bins[0])
