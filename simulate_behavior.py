"""
Simulate behavioral time-series in rhythmic sampling experiments
"""

import numpy as np
from colorednoise import powerlaw_psd_gaussian
from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess


def simulate_behavior(fs, t_start, t_end,
                      accuracy_norm=True,
                      noise_method='powerlaw',
                      f_osc=0.0, osc_amp=0.0, **noise_args):
    """
    Simulate a time-series of data.

    Parameters
    ----------
    fs : float
        Sampling rate of the simulated data.
    t_start, t_end : float
        The time points of the beginning and end of the simulated data.
    accuracy_norm : bool
        Bound the time-course from [0.25, 0.75] so it mirrors accuracy in a
        typical rhythmic sampling experiment.
    noise_method : str
        The method used to generate the behavioral data. Options: powerlaw,
        arma, fully_random.
    noise_args : dict
        A dictionary of keyword arguments for the noise generation. For
        `powerlaw`, these arguments are passed to
        colorednoise.powerlaw_psd_gaussian. For `arma`, they are passed to
        `generate_arma`.
    f_osc : float
        Frequency of the oscillation to add to the data.
    osc_amp : float
        Amplitude of the oscillation in units of proportion accurate.

    Returns
    -------
    x : np.ndarray
        Simulated data time-course.
    """
    n = n_behav_samples(fs, t_start, t_end)

    # Simulate data time-series
    if noise_method == 'powerlaw':  # powerlaw noise
        assert 'exponent' in noise_args.keys(), \
               'Must specify exponent for powerlaw noise'
        x = powerlaw_psd_gaussian(noise_args['exponent'], n)
    elif noise_method == 'arma':  # ARMA noise
        assert np.all(np.isin(['ar_coefs', 'ma_coefs'],
                              list(noise_args.keys()))), \
               'Some args for generate_arma are missing or misspelled'
        x = generate_arma(np.array(noise_args['ar_coefs']),
                          np.array(noise_args['ma_coefs']),
                          n)
    elif noise_method == 'fully_random':  # Every trial is random
        x = 0.5 * np.ones(n)
    else:  # Catch errors
        raise(Exception(f"noise_method `{noise_method}` not recognized"))

    # Adjust accuracy to bound it at realistic levels
    if accuracy_norm and noise_method != 'fully_random':
        # Make the time-course bounded to mirror accuracy in published studies
        x -= x.min()  # Set lowest value to 0
        x /= x.max()  # Set highest value to 1
        x *= 0.2  # Squeeze the range to a smaller range
        x += 0.5  # Minimum value here

    # Add an oscillation to the behavioral data
    assert 0 <= osc_amp <= 1, "osc_amp must be in the range [0, 1]"
    t = time_vector(fs, t_start, t_end)
    phi = np.random.uniform(0, 2 * np.pi)  # Random phase offset
    osc = osc_amp / 2 * np.sin(2 * np.pi * f_osc * t + phi)
    x = x + osc

    return x


def simulate_behavior_trialwise(details, **behav_kwargs):
    """
    Simulate behavioral accuracy data for each trial individually, to mirror
    behavioral experiments like Landau & Fries (2012) or Fiebelkorn et al.
    (2013).

    This function starts by generating an idealized accuracy trace of the
    accuracy that would be obtained at each time-point with an infinite number
    of trials. Then it simulates individual trials by choosing a time-point,
    and randomly assigning the trial to be a hit or miss with probabilities
    based on the idealized hit rate at that point in time.

    Parameters
    ----------
    details : dict
        A dictionary of information about the behavioral paradigm to be
        simulated. It should have the following keys:
        fs : float
            Sampling rate of the simulated data (the spacing between subsequent
            SOAs).
        t_start, t_end : float
            The time points of the beginning and end of the simulated data.
        n_subjects : int
            The number of subjects to simulate.
        n_trials : int
            The number of trials to simulate for each subject.
    **behav_kwargs
        Additional keyword arguments are passed to `simulate_behavior`.

    Returns
    -------
    x_trialwise : np.ndarray
        Accuracy for each simulated trial (1: hit, 0: miss).
    t_trialwise : np.ndarray
        The time-stamp of the SOA for each simulated trial.
    """
    # Simulated probability of a correct response at each time-point
    acc_prob = simulate_behavior(details['fs'],
                                 details['t_start'],
                                 details['t_end'],
                                 **behav_kwargs)
    # Make sure this is a possible sequence of accuracy probabilities
    assert acc_prob.min() >= 0, "Minimum accuracy can't be lower than 0"
    assert acc_prob.max() <= 1, "Maximum accuracy can't be greater than 1"
    t_acc_prob = time_vector(details['fs'],  # Time-stamps for accuracy changes
                             details['t_start'],
                             details['t_end'])
    assert t_acc_prob.shape == acc_prob.shape

    # Get a list of trial times with accuracy for each trial
    # Get the time-index of each trial
    n_trials = details['n_trials'] * details['n_subjects']
    t_inx_trialwise = np.repeat(range(len(acc_prob)),
                                int(n_trials / len(acc_prob)))
    # Get the time-stamps for each trial
    t_trialwise = t_acc_prob[t_inx_trialwise]
    # Get the idealized accuracy for each trial
    p_corr_trialwise = acc_prob[t_inx_trialwise]
    # Get simulated accuracy for each trial
    x_trialwise = []
    for p in p_corr_trialwise:
        resp = np.random.choice([0, 1], p=[1 - p, p])
        x_trialwise.append(resp)
    x_trialwise = np.array(x_trialwise)

    # Sort the two vectors
    sort_inx = np.argsort(t_trialwise)
    x_trialwise = x_trialwise[sort_inx]
    t_trialwise = t_trialwise[sort_inx]

    return x_trialwise, t_trialwise


def generate_arma(ar_coefs, ma_coefs, n):
    """
    Generate a sequence using with an ARMA model.

    Parameters
    ----------
    ar_coefs : np.ndarray or list of float
        AR coefficients for the ARMA model.
    ma_coefs : np.ndarray or list of float
        MA coefficients for the ARMA model.
    n : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Sequence of data.
    """
    ar_coefs = np.r_[1, -ar_coefs]  # Add zero-lag and negate
    ma_coefs = np.r_[1, ma_coefs]  # Add zero-lag

    # Make sure the model is sensible
    arma_t = ArmaProcess(ar_coefs, ma_coefs)
    assert arma_t.isinvertible, "Coefs don't give an invertible model"
    assert arma_t.isstationary, "Coefs don't give a stationary model"

    # Simulate the data
    x = arma_generate_sample(ar_coefs, ma_coefs, nsample=n)

    return x


def time_vector(fs, t_start, t_end):
    """
    Get the time-stamps of a simulated behavioral time-series.

    Parameters
    ----------
    fs : float
        Sampling rate of the simulated data
    t_start, t_end : float
        Starting and ending points of the simulated data

    Returns
    -------
    t : np.ndarray
        Sequence of time-stamps
    """
    n = n_behav_samples(fs, t_start, t_end)
    t = np.linspace(t_start, t_end, n)
    return t


def n_behav_samples(fs, t_start, t_end):
    """
    Get the number of samples in a simulated behavioral time-series

    Parameters
    ----------
    fs : float
        Sampling rate of the simulated data
    t_start, t_end : float
        Starting and ending points of the simulated data

    Returns
    -------
    n : int
        Number of samples in the time-series
    """
    n = int(np.ceil((t_end - t_start) * fs + 1))
    return n
