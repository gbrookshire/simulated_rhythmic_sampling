"""
Tools to perform analyses by shuffling in time, as in Landau & Fries (2012) and
Fiebelkorn et al. (2013).
"""

import os
import yaml
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .utils import avg_repeated_timepoints, dft

# Load the details of the behavioral studies
_pathname = os.path.dirname(os.path.abspath(__file__))
_behav_fname = os.path.join(_pathname, '../behav_details.yaml')
behav_details = yaml.safe_load(open(_behav_fname))


def landau(x, t, fs, k_perm):
    """
    Analyze the data as in Landau & Fries (2012)

    Parameters
    ----------
    x : nd.array
        Array of Hit (1) or Miss (0) for each trial
    t : nd.array
        Time-stamp (SOA) for each trial

    Returns
    -------
    res : dict
        The results of the randomization test as returned by
        `time_shuffled_perm`, plus these items:
        t : np.ndarray
            The time-stamps of the individual trials
        t_agg : np.ndarray
            The time-steps for the aggregated accuracy time-series
        x_agg : np.ndarray
            The aggregated accuracy time-series
        p_corr : np.ndarray
            P-values corrected for multiple comparisons using Bonforroni
            correction
    """

    def landau_spectrum_trialwise(x_perm):
        """ Helper to compute spectrum on shuffled data
        """
        _, x_avg = avg_repeated_timepoints(t, x_perm)
        f, y = landau_spectrum(x_avg, fs)
        return f, y

    # Compute the results
    res = time_shuffled_perm(landau_spectrum_trialwise, x, k_perm)
    res['t'] = t
    res['t_agg'], res['x_agg'] = avg_repeated_timepoints(t, x)

    # Correct for multiple comparisons across frequencies
    _, p_corr, _, _ = multipletests(res['p'], method='bonferroni')
    res['p_corr'] = p_corr
    return res


def landau_spectrum(x, fs, detrend_ord=1):
    """
    Get the spectrum of behavioral data as in Landau & Fries (2012)

    The paper doesn't specifically mention detrending, but A.L. says they
    always detrend with a 2nd-order polynomial. That matches the data --
    without detrending, there should have been a peak at freq=0 due to the
    offset from mean accuracy being above 0.
    2021-06-14: AL tells me they used linear detrending.

    The paper says the data were padded before computing the FFT, but doesn't
    specify the padding or NFFT. I've chosen a value to match the frequency
    resolution in the plots.

    Parameters
    ----------
    x : np.ndarray
        The data time-series

    Returns
    -------
    f : np.ndarray
        The frequencies of the amplitude spectrum
    y : np.ndarray
        The amplitude spectrum
    """
    details = behav_details['landau']
    # Detrend the data
    x = sm.tsa.tsatools.detrend(x, order=detrend_ord)
    # Window the data
    x = window(x, np.hanning(len(x)))
    # Get the spectrum
    f, y = dft(x, fs, details['nfft'])
    return f, y


def fiebelkorn(x, t, k_perm):
    """
    Search for statistically significant behavioral oscillations as in
    Fiebelkorn et al. (2013)

    Parameters
    ----------
    x : np.ndarray
        A sequence of accuracy (Hit: 1, Miss: 0) for each trial
    t : np.ndarray
        The time-stamps for each trial
    k_perm : int
        The number of times to randomly shuffle the data when computing the
        permuted surrogate distribution

    Returns
    -------
    res : dict
        The results as given by `time_shuffled_perm`plus these items:
        t : np.ndarray
            The original time-stamps of the raw data
        p_corr : np.ndarray
            P-values for each frequency, corrected for multiple comparisons
            using FDR
    """
    # Compute the results
    res = time_shuffled_perm(lambda xx: fiebelkorn_spectrum(xx, t), x, k_perm)
    res['t'] = t

    # Correct for multiple comparisons across frequencies
    _, p_corr, _, _ = multipletests(res['p'], method='fdr_bh')
    res['p_corr'] = p_corr
    return res


def fiebelkorn_binning(x_trial, t_trial):
    """
    Given accuracy and time-points, find the time-smoothed average accuracy

    Parameters
    ----------
    x_trial : np.ndarray
        Accuracy (Hit: 1, Miss: 0) of each trial
    t_trial : np.ndarray
        The time-stamp of each trial

    Returns
    -------
    x_bin : np.ndarray
        The average accuracy within each time bin
    t_bin : np.ndarray
        The centers of each time bin
    """
    details = behav_details['fiebelkorn']
    # Time-stamps of the center of each bin
    t_bin = np.arange(details['t_start'],
                      details['t_end'] + 1e-10,
                      details['bin_step'])
    # Accuracy within each bin
    x_bin = []
    for i_bin in range(len(t_bin)):
        bin_center = t_bin[i_bin]
        bin_start = bin_center - (details['bin_width'] / 2)
        bin_end = bin_center + (details['bin_width'] / 2)
        bin_sel = (bin_start <= t_trial) & (t_trial <= bin_end)
        x_bin_avg = np.mean(x_trial[bin_sel])
        x_bin.append(x_bin_avg)
    x_bin = np.array(x_bin)

    return x_bin, t_bin


def fiebelkorn_spectrum(x, t):
    """
    Compute the spectrum of accuracy data as in Fiebelkorn et al. (2013)

    Parameters
    ----------
    x : np.ndarray
        The data for each trial
    t : np.ndarray
        The time-stamp for each trial

    Returns
    -------
    f : np.ndarray
        The frequencies of the resulting spectrum
    y : np.ndarray
        The amplitude spectrum
    """
    details = behav_details['fiebelkorn']
    # Get the moving average of accuracy
    x_bin, t_bin = fiebelkorn_binning(x, t)
    # Detrend the binned data
    x_bin = sm.tsa.tsatools.detrend(x_bin, order=2)
    # Window the data
    x_bin = window(x_bin, np.hanning(len(x_bin)))
    # Get the spectrum
    f, y = dft(x_bin, 1 / details['bin_step'], details['nfft'])
    # Only keep frequencies that were reported in the paper
    f_keep = f <= details['f_max']
    f = f[f_keep]
    y = y[f_keep]
    return f, y


def time_shuffled_perm(analysis_fnc, x, k_perm):
    """
    Run a permutation test by shuffling the time-stamps of individual trials.

    Parameters
    ----------
    analysis_fnc : function
        The function that will be used to generate the spectrum
    x : np.ndarray
        The data time-series
    k_perm : int
        How many permutations to run

    Returns
    -------
    res : dict
        Dictionary of the results of the randomization analysis
        x : np.ndarray
            The raw data
        x_perm : np.ndarray
            The shuffled data
        f : np.ndarray
            The frequencies of the resulting spectrum
        y_emp : np.ndarray
            The spectrum of the empirical (unshuffled) data
        y_avg : np.ndarray
            The spectra of the shuffled permutations
        y_cis : np.ndarray
            Confidence intervals for the spectra, at the 2.5th, 95th, and
            97.5th percentile
        p : np.ndarray
            P-values (uncorrected for multiple comparisons) for each frequency
    """

    # Compute the empirical statistics
    f, y_emp = analysis_fnc(x)

    # Run a bootstrapped permutation test.
    # Create a surrogate distribution by randomly shuffling resps in time.
    x_perm = []
    y_perm = []
    x_shuff = x.copy()
    for k in range(k_perm):
        np.random.shuffle(x_shuff)
        _, y_perm_k = analysis_fnc(x_shuff)
        y_perm.append(y_perm_k)
        if k < 10:  # Keep a few permutations for illustration
            x_perm.append(x_shuff.copy())

    # Find statistically significant oscillations
    # Sometimes we get p=0 if no perms are larger than emp. Note that in this
    # case, a Bonferroni correction doesn't have any effect on the p-values.
    p = np.mean(np.vstack([y_perm, y_emp]) > y_emp, axis=0)

    # Get summary of simulated spectra
    y_avg = np.mean(y_perm, 1)
    y_cis = np.percentile(y_perm, [2.5, 95, 97.5], 1)

    # Bundle the results together
    res = {}
    res['x'] = x
    res['x_perm'] = np.array(x_perm)
    res['f'] = f
    res['y_emp'] = y_emp
    res['y_perm'] = np.array(y_perm)
    res['y_avg'] = y_avg
    res['y_cis'] = y_cis
    res['p'] = p

    return res


def window(x, win):
    """ Apply a window to a segment of data

    Parameters
    ----------
    x : np.ndarray
        The data
    win : np.ndarray
        The window

    Returns
    -------
    x : np.ndarray
        The windowed data
    """
    return np.multiply(win, x.T).T
