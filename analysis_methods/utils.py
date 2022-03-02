""" Utilities used across sub-modules
"""

import numpy as np


def avg_repeated_timepoints(t_trial, x_trial):
    """ Get the average value of x at each unique time-point t.
    """
    t = np.sort(np.unique(t_trial))
    x = [np.mean(x_trial[t_trial == t_point]) for t_point in t]
    x = np.array(x)
    return t, x


def dft(x, fs, nfft, axis=-1):
    """
    Find the amplitude spectrum using a discrete Fourier transform.

    Parameters
    ----------
    x : np.ndarray
        The data time-course
    fs : float
        Sampling rate
    nfft : int
        The number of samples in the DFT
    axis : int
        The axis of the array along which to calculate the DFT

    Returns
    -------
    f : np.ndarray
        Frequencies for the spectrum
    y : np.ndarray
        The amplitude spectrum
    """
    f = np.fft.fftfreq(nfft, 1 / fs)
    y = np.abs(np.fft.fft(x, nfft, axis=axis))
    # Only keep positive frequencies
    f_keep = f > 0
    f = f[f_keep]
    y = y[f_keep]
    return f, y
