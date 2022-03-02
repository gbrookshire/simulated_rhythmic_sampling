"""
Different ways to analyze data to search for rhythms in behavior
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import median_filter
from skimage import measure
from mtspec import mtspec
from .utils import dft


def ar_surr(x, fs, k_perm, freq_cutoff=15, correction='cluster'):
    """
    Test for oscillations by comparing against a surrogate distribution
    generated using an autoregressive model.

    Parameters
    ----------
    x : np.ndarray
        The aggregated data time-course
    fs : float
        Sampling rate of the data
    k_perm : int
        The number of simulated datasets in the surrogate distribution
    freq_cutoff : float
        The maximum frequency at which to search for oscillations
    correction : str ('cluster', 'bonferroni', 'fdr')
        How to correct for multiple comparisons across frequencies

    Returns
    -------
    res : dict
        A dictionary of the results of the test.
        x : np.ndarray
            The original time-series
        x_perm : np.ndarray
            The surrogate time-series, simulated following the AR model
        f : np.ndarray
            The frequencies of the Fourier transform
        y_emp : np.ndarray
            The amplitude spectrum for the real empirical data
        y_perm : np.ndarray
            The amplitude spectra of the surrogate data
        y_avg : np.ndarray
            The average of the amplitude spectra of the surrogate data
        y_cis : np.ndarray
            The confidence intervals of the surrogate amplitude spectra.
            Includes the following percentiles: 2.5, 95, and 97.5 For 95% CIs,
            take the 2.5th and 97.5th percentiles.
        p_raw : np.ndarray
            Raw p-values for each frequency of the amplitude spectrum. Not
            corrected for multiple comparisons.
        p_corr : np.ndarray
            P-values corrected for multiple comparisons.
        cluster_info : dict
            Information about the cluster test to correct for multiple
            comparisons across frequencies. (See clusterstat_1d for details)
    """
    assert correction in ('cluster', 'bonferroni', 'fdr'), \
        'The value of `correction` must be "cluster", "bonferroni", or "fdr"'

    # Subtract out the mean and linear trend
    detrend_ord = 1
    x = sm.tsa.tsatools.detrend(x, order=detrend_ord)

    # Estimate an AR model
    mdl_order = (1, 0)
    mdl = sm.tsa.ARMA(x, mdl_order)
    result = mdl.fit(trend='c', disp=0)
    result.summary()
    # Make a generative model using the AR parameters
    arma_process = sm.tsa.ArmaProcess.from_coeffs(result.arparams)
    # Simulate a bunch of time-courses from the model
    x_sim = arma_process.generate_sample((len(x), k_perm),
                                         scale=result.resid.std())
    # Subtract out the mean and linear trend
    x_sim = sm.tsa.tsatools.detrend(x_sim, order=detrend_ord, axis=0)

    # Calculate the spectra
    nfft = len(x)
    f, y_emp = dft(x, fs, nfft)
    f_sim, y_sim = dft(x_sim, fs, nfft, axis=0)

    # Get summary of simulated spectra
    y_avg = np.mean(y_sim, 1)
    y_cis = np.percentile(y_sim, [2.5, 95, 97.5], 1)

    # Find statistically significant oscillations
    p_raw = np.mean(np.vstack([y_sim.T, y_emp]) > y_emp, axis=0)

    # Select the frequency range
    freq_sel = f <= freq_cutoff
    f = f[freq_sel]
    y_emp = y_emp[freq_sel]
    y_sim = y_sim[freq_sel, :]
    y_avg = y_avg[freq_sel]
    y_cis = y_cis[:, freq_sel]
    p_raw = p_raw[freq_sel]

    # Bundle the results together
    res = {}
    res['x'] = x
    res['x_perm'] = x_sim.T
    res['f'] = f
    res['y_emp'] = y_emp
    res['y_perm'] = y_sim.T  # Transpose for consistency w/ other methods
    res['y_avg'] = y_avg
    res['y_cis'] = y_cis
    res['p_raw'] = p_raw

    # Correct for multiple comparisons
    if correction == 'cluster':
        # A cluster test is more sensitive than FDR
        p_clust, cluster_info = clusterstat_1d(y_emp, y_sim.T)
        p_corr = np.ones(y_emp.size)
        # If no samples are a member of a significant cluster
        for i_clust in range(cluster_info['labels'].max() + 1):
            clust_sel = cluster_info['labels'] == i_clust
            p_corr[clust_sel] = cluster_info['p_cluster'][i_clust]
        res['cluster_info'] = cluster_info

    elif correction == 'bonferroni':
        _, p_corr, _, _ = multipletests(p_raw, method='bonferroni')

    elif correction == 'fdr':
        _, p_corr, _, _ = multipletests(p_raw, method='fdr_bh')

    else:
        raise Exception(f"correction method {correction} not recognized")

    res['p_corr'] = p_corr

    return res


def robust_est(x, fs, nw=1.5, n_tapers=None,
               med_filt_win=7, freq_cutoff=15,
               correction='bonferroni'):
    """
    Robust est. analysis as in Mann & Lees (1996).

    Parameters
    ----------
    x : np.ndarray
        The aggregated data time-course
    fs : float
        Sampling rate of the data
    nw : float
        Time bandwidth parameter for the multitapers
    n_tapers : int
        Number of DPSS tapers. If None, set to int(2 * nw) - 1
    med_filt_win : int
        Width of the median filter used to determine the background spectrum.
    freq_cutoff : float
        Maximum frequency to look for rhythms in the time-series
    correction : str ('bonferroni', 'fdr')
        How to correct for multiple comparisons across frequencies.

    Returns
    -------
    res : dict
        A dictionary of the results of the statistical tests.
        x : np.ndarray
            The original time-series
        f : np.ndarray
            The frequencies of the Fourier transform
        y_emp : np.ndarray
            The amplitude spectrum for the data
        p_raw : np.ndarray
            Raw p-values for each frequency of the amplitude spectrum. Not
            corrected for multiple comparisons.
        p_corr : np.ndarray
            P-values corrected for multiple comparisons.
    """
    assert correction in ('bonferroni', 'fdr'), \
        'The value of `correction` must be "bonferroni" or "fdr"'

    # Detrend the data
    x = sm.tsa.tsatools.detrend(x, order=0)

    # Compute spectrum using multitapers
    if n_tapers is None:  # Number of tapers
        n_tapers = int(2 * nw) - 1
    spec, freq = mtspec(data=x,
                        delta=fs ** -1,
                        time_bandwidth=nw,
                        number_of_tapers=n_tapers,
                        statistics=False, rshape=0)

    # Smooth the spectrum with a median filter
    spec_filt = median_filter(spec, med_filt_win)

    # Fit an AR1 model of the background noise
    spec_bg = fit_ar_spec(freq, spec_filt, freq[-1])

    # Compute significance using a chi-square dist with df = 2 * n_tapers
    spec_ratio = spec / spec_bg  # Ratio of the raw spect vs background noise
    p_raw = stats.chi2.sf(spec_ratio, 2 * n_tapers)

    # Only keep the lower frequencies, ignore DC
    freq_sel = freq <= freq_cutoff
    freq_sel[0] = False  # Exclude DC
    spec = spec[freq_sel]
    spec_filt = spec_filt[freq_sel]
    spec_bg = spec_bg[freq_sel]
    freq = freq[freq_sel]
    p_raw = p_raw[freq_sel]

    # Correct for multiple comparisons
    if correction == 'bonferroni':
        _, p_corr, _, _ = multipletests(p_raw, method='bonferroni')

    elif correction == 'fdr':
        _, p_corr, _, _ = multipletests(p_raw, method='fdr_bh')

    else:
        raise Exception(f"correction method {correction} not recognized")

    # Put the results together
    res = {}
    res['x'] = x
    res['f'] = freq
    res['y_emp'] = spec
    res['p_raw'] = p_raw
    res['p_corr'] = p_corr
    return res


def fit_ar_spec(freq, spec, nyquist):
    """
    Get a spectrum's best-fit to an AR(1) process.
    Mann & Lees (1996), eq. 4

    Parameters
    ----------
    freq : np.ndarray
        The frequencies of the spectrum
    spec : np.ndarray
        The amplitude of the spectrum
    nyquist : float
        The Nyquist frequency

    Returns
    spec_fit : np.ndarray
        The spectrum that best approximates an AR(1) process fit to the data
    """
    def ar_spec(f, rho, S_0):
        """
        Spectrum of an AR(1) process at given frequency, rho, and S_0 (average
        value of the spectrum).
        """
        num = 1 - (rho ** 2)
        den = 1 - (2 * rho * np.cos(np.pi * f / nyquist)) + (rho ** 2)
        S_f = S_0 * num / den
        return S_f

    (rho, S_0), _ = curve_fit(ar_spec, freq, spec,
                              bounds=([0, 0], [1, np.inf]))
    spec_fit = ar_spec(freq, rho, S_0)

    return spec_fit


def clusterstat_1d(x_emp, x_perm, a_thresh=0.05, a_clust=0.05):
    """
    Look for clusters in time or frequency where a signal (x_emp) reliably
    differs from a set of surrogate signals.

    Parameters
    ----------
    x_emp : np.ndarray (time|freq|space, )
        The data (1-dimensional)
    x_perm : np.ndarray (permutations, time|freq|space)
        The surrogate distribution (2-dimensional)
    a_thresh : float
        Alpha threshold for selecting each sample for inclusion in a cluster.
        Must be between (0, 1).
    a_clust : float
        Threshold for significant clusters included in the output.

    Returns
    -------
    p : float
        P-value. The proportion runs in the surrogate distribution that had a
        larger cluster than the largest empirical cluster.
    cluster_info : dict
        Information about the clusters found in the data. Fields:
        labels : np.ndarray (int)
            Label of which cluster each sample belongs to. -1 means it's not a
            member of any cluster.
        stat : np.ndarray (float)
            The cluster statistic associated with each cluster
        member_of_signif_cluster : np.ndarray (bool)
            Whether each sample is a member of a significant cluster
        p_cluster : list (float)
            P-value for each cluster in the empirical data

    Algorithm:
    Z-score amplitude across runs within each frequency
    Get the empirical difference between the amp and avg surrogate amps
        Get cluster stat
    For each run in the permuted data:
        Get the empirical difference between the amp and the other amps
            (Including or excluding the real data?)
        Get cluster stat
    Get p-value from the distribution of cluster stats

    To get cluster stats
        Take all samples above some z-value
        Look at connected
    """

    x = np.vstack([x_emp, x_perm])

    # Z-score amplitude across runs within each frequency
    x = stats.zscore(x, axis=0)

    # Threshold the z-scores
    thresh = stats.norm.ppf(1 - a_thresh)
    x_thresh = x > thresh

    # Find cluster stat for each run (empirical and permuted)
    clust_stat = []
    for k in range(x_thresh.shape[0]):  # First run is the empirical data
        labels = measure.label(x_thresh[k, :])  # Find clusters
        cluster_labels = np.unique(labels)[1:]  # Ignore non-clusters
        # Get the summed z-scores in each cluster
        summed_z = [np.sum(x[k, labels == c]) for c in cluster_labels]
        try:
            s = np.max(summed_z)
        except ValueError:
            s = 0
        clust_stat.append(s)
        if k == 0:  # Save the cluster info for the real data
            cluster_info = {'labels': labels - 1,
                            'stat': summed_z}
    clust_stat = np.array(clust_stat)

    # Compute the p-value
    # Because the clusters are often pretty small, it's important to use
    # greater-than-or-equal instead of just greater-than. Otherwise you end up
    # with significant results if there are no big clusters in the empirical
    # data.
    p_cluster = []
    for c in cluster_info['stat']:
        p_cluster.append(np.mean(clust_stat[1:] >= c))
    cluster_info['p_cluster'] = p_cluster
    try:
        p = min(p_cluster)
    except ValueError:
        p = 1.0

    # Return the indices of samples that belong to a significant cluster
    thresh = np.percentile(clust_stat[1:], 100 * (1 - a_clust))
    signif_sample = np.zeros(x_emp.shape)
    for i_clust, stat in enumerate(cluster_info['stat']):
        if stat > thresh:
            signif_sample[cluster_info['labels'] == i_clust] = 1
    cluster_info['member_of_signif_cluster'] = signif_sample.astype(bool)

    return p, cluster_info
