"""
Simulate experiments following published papers
"""

import os
import yaml
import simulate_behavior as behav
from analysis_methods import shuff_time, alternatives, utils

# Load the details of the behavioral studies
_pathname = os.path.dirname(os.path.abspath(__file__))
_behav_fname = os.path.join(_pathname, 'behav_details.yaml')
behav_details = yaml.safe_load(open(_behav_fname))


def ar_experiment(correction='cluster', **behav_kwargs):
    """
    Simulate an experiment and analyze it with the AR surrogate method.

    Parameters
    ----------
    correction : str ('cluster', 'bonferroni', 'fdr')
        How to correct for multiple comparisons across frequencies.
    Keyword arguments are passed to `simulate_behavior.simulate_behavior`

    Returns
    -------
    res : dict
        The results of the experiment returned by `alternatives.ar_surr`,
        plus the following items:
        details : dict
            The parameters of the simulated behavioral experiment
        t : np.ndarray
            The time-points of the aggregated time-series
    """
    details = behav_details['landau']
    x_trial, t_trial = behav.simulate_behavior_trialwise(details,
                                                         **behav_kwargs)
    t, x = utils.avg_repeated_timepoints(t_trial, x_trial)
    res = alternatives.ar_surr(x, details['fs'], details['k_perm'],
                               correction=correction)
    res['details'] = details
    res['t'] = t
    return res


def robust_est_experiment(correction='bonferroni',
                          re_params={},
                          **behav_kwargs):
    """
    Simulate an experiment and analyze it with the robust est. method.

    Parameters
    ----------
    correction : str ('bonferroni', 'fdr')
        How to correct for multiple comparisons across frequencies.
    re_params : dict
        Keyword arguments for the robust est. analysis, passed on to
        `analysis_methods.robust_est`
    Other keyword arguments are passed to `simulate_behavior.simulate_behavior`

    Returns
    -------
    res : dict
        The results of the experiment returned by
        `analysis_methods.robust_est`, plus the following items:
        details : dict
            The parameters of the simulated behavioral experiment
        t : np.ndarray
            The time-points of the aggregated time-series
    """
    details = behav_details['landau']
    x_trial, t_trial = behav.simulate_behavior_trialwise(details,
                                                         **behav_kwargs)
    t, x = utils.avg_repeated_timepoints(t_trial, x_trial)
    res = alternatives.robust_est(x, details['fs'],
                                  correction=correction,
                                  **re_params)
    res['details'] = details
    res['t'] = t
    return res


def landau_experiment(**behav_kwargs):
    """
    Simulate an experiment and analyze it with shuffling in time as in Landau
    and Fries (2012)

    Parameters
    ----------
    Keyword arguments are passed to `simulate_behavior.simulate_behavior`

    Returns
    -------
    res : dict
        The results of the experiment returned by `shuff_time.landau`,
        plus the following items:
    """
    details = behav_details['landau']
    x_trial, t_trial = behav.simulate_behavior_trialwise(details,
                                                         **behav_kwargs)
    res = shuff_time.landau(x_trial, t_trial, details['fs'], details['k_perm'])
    return res


def fiebelkorn_experiment(**behav_kwargs):
    """
    Simulate an experiment and analyze it with shuffling in time as in
    Fiebelkorn et al. (2013)

    Parameters
    ----------
    Keyword arguments are passed to `simulate_behavior.simulate_behavior`

    Returns
    -------
    res : dict
        The results of the experiment returned by
        `shuff_time.fiebelkorn`
    """
    details = behav_details['fiebelkorn']
    x_trial, t_trial = behav.simulate_behavior_trialwise(details,
                                                         **behav_kwargs)
    res = shuff_time.fiebelkorn(x_trial, t_trial, details['k_perm'])
    return res
