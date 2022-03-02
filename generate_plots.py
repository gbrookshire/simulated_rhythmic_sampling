#!/usr/bin/env python3

import itertools
import copy
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import simulate_experiments as sim_exp
import simulate_behavior as sim_behav
from analysis_methods import shuff_time
import analysis

plot_dir = 'plots/'
data_dir = 'results/'

plt.ion()


#############
# Utilities #
#############

def remove_topright_axes(ax=None):
    """
    Remove the box from a plot, leaving only the lower and left axes

    Parameters
    ----------
    ax: A matplotlib axis

    Returns
    ------
    None
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def conf_int_prop(x, a=0.05):
    """
    Confidence Interval for proportions
    CI = p  +/-  z*(√p(1-p) / n)

    Parameters
    ----------
    x : np.ndarray of bool
        The data
    a : float
        The alpha value to find confidence intervals for

    Returns
    -------
    ci : np.ndarray of floats
        Two-item array with the lower and upper CI

    """
    p = np.mean(x)
    n = len(x)
    z = stats.norm.ppf([a / 2,
                        1 - (a / 2)])
    ci = p + z * np.sqrt(p * (1 - p) / n)
    return ci


def plot_signif_bars(x, signif, y_pos, **kwargs):
    """
    Add a horizontal bar denoting significant regions
    """
    changes = np.diff(np.r_[False, signif.astype(int)])
    onsets = np.nonzero(changes == 1)[0]
    offsets = np.nonzero(changes == -1)[0] - 1
    if len(onsets) == 0 and len(offsets) == 0:
        return None

    # Add an onset on the first sample if necessary
    if offsets[0] < onsets[0]:
        onsets = np.r_[x.min(), onsets]
    # Add an offset on the last sample if necessary
    if offsets[-1] < onsets[-1]:
        offsets = np.r_[offsets, x.max()]

    assert len(onsets) == len(offsets)

    for on, off in zip(x[onsets], x[offsets]):
        plt.plot([on, off], np.ones(2) * y_pos,
                 solid_capstyle='round',
                 **kwargs)


def sci_notation(number, sig_fig=2):
    """ Convert a number to scientific notation for pasting into Latex.
        e.g. 4.32342342e-10 --> 4.3 * 10^{-10}
    """
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    return f"${a} \\times 10^{{{b}}}$"


# def bootstrap(x, k, func, alpha=0.05):
#     sim = [func(np.random.choice(x, size=len(x), replace=True))
#            for _ in range(k)]
#     percentiles = [(alpha / 2) * 100, (1 - (alpha / 2)) * 100]
#     ci = np.percentile(sim, percentiles)
#     return ci


#############################################
# Functions to generate plots for the paper #
#############################################

def illustrate_shuffle():
    """
    Illustrate how shuffling in time changes the autocorrelation
    """

    # Generate some data
    np.random.seed(2)

    details = sim_exp.behav_details['landau']
    t = sim_behav.time_vector(details['fs'],
                              details['t_start'],
                              details['t_end'])
    x = sim_behav.generate_arma(0.7, 0, len(t))
    # Adjust the time-series so that it looks like accuracy
    x -= x.min()
    x /= x.max()
    x /= 2
    x += 0.25

    plt.figure(figsize=(5, 4))
    colors = {'original': 'deepskyblue',
              'shuffled': 'maroon'}

    # Plot the real behavior
    plt.subplot(2, 2, 1)
    plt.plot(t, x, color=colors['original'])
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Original: AR(1)', color=colors['original'])
    plt.xticks([details['t_start'], details['t_end']])
    remove_topright_axes()

    # Plot one permutation of randomly shuffled data
    plt.subplot(2, 2, 2)
    x_shuff = x.copy()
    np.random.shuffle(x_shuff)
    plt.plot(t, x_shuff, color=colors['shuffled'])
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Shuffled', color=colors['shuffled'])
    plt.xticks([details['t_start'], details['t_end']])
    remove_topright_axes()

    # Plot the autocorrelation functions of each dataset
    # Normalize to get Pearson coefficients
    def autocorr(x, maxlag=25):
        x = (x - np.mean(x)) / (np.std(x))
        c = np.correlate(x, x / len(x), 'full')
        lags = np.arange(len(c))
        lags -= int(np.mean(lags))
        keep_lags = (lags >= 0) & (lags <= maxlag)
        c = c[keep_lags]
        lags = lags[keep_lags]
        return c, lags

    c_orig, lags = autocorr(x)
    c_shuf, _ = autocorr(x_shuff)

    plt.subplot(2, 2, 3)
    plt.axhline(0, linestyle='-', color='k', linewidth=0.5)
    plt.plot(lags, c_orig, 'o-', color=colors['original'], markersize=4)
    plt.plot(lags, c_shuf, 'o-', color=colors['shuffled'], markersize=4)
    plt.xlabel('Lag (sample)')
    plt.ylabel('R')
    plt.title('Autocorrelation', color='white')
    remove_topright_axes()
    plt.text(15, 0.75, 'Original', color=colors['original'])
    plt.text(15, 0.55, 'Shuffled', color=colors['shuffled'])

    plt.subplot(2, 2, 4)
    f, y_orig = shuff_time.landau_spectrum(x, details['fs'])
    f, y_shuf = shuff_time.landau_spectrum(x_shuff, details['fs'])
    plt.plot(f, y_orig, color=colors['original'])
    plt.plot(f, y_shuf, color=colors['shuffled'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Spectra', color='white')
    plt.xlim(0, 15)
    remove_topright_axes()

    plt.tight_layout()
    plt.savefig(f"{plot_dir}illustrate_shuffle.pdf")


def illustrate_consistency():
    """
    Illustrate the effects of shuffling in time on non-rhythmic patterns that
    are consistent across trials.
    """
    details = sim_exp.behav_details['landau']
    t = sim_behav.time_vector(details['fs'],
                              details['t_start'],
                              details['t_end'])
    n_trials_per_time_step = 100
    t_trial = np.tile(t, n_trials_per_time_step)
    # Make one peak of accuracy
    x_trial = np.zeros(t_trial.shape)  # Responses: All misses
    idealized_acc = stats.norm.pdf(t, loc=0.6, scale=0.05)
    idealized_acc /= idealized_acc.max()  # Limit between [0, 1]
    for i_t in range(len(t)):
        acc = idealized_acc[i_t]
        n_hits = int(acc * n_trials_per_time_step)
        n_misses = n_trials_per_time_step - n_hits
        # Get the responses for each trial
        responses = np.repeat([False, True], [n_misses, n_hits])
        x_trial[t_trial == t[i_t]] = responses
    # Get average accuracy at each time-point
    x = np.array([np.mean(x_trial[t_trial == t_point]) for t_point in t])
    # And get the time-series after shuffling the data in time
    np.random.shuffle(t_trial)
    x_shuff = np.array([np.mean(x_trial[t_trial == t_point]) for t_point in t])
    # Compute the spectra of the real and shuffled accuracy time-courses
    f, y_lin = shuff_time.landau_spectrum(x, details['fs'], 1)  # Lin detrend
    f, y_pol = shuff_time.landau_spectrum(x, details['fs'], 2)  # Polynomial
    f, y_shuff_lin = shuff_time.landau_spectrum(x_shuff, details['fs'], 1)
    f, y_shuff_pol = shuff_time.landau_spectrum(x_shuff, details['fs'], 2)

    colors = {'original': 'deepskyblue',
              'shuffled': 'maroon'}
    plt.figure(figsize=(5, 2))

    # Plot the raw time-series
    plt.subplot(1, 2, 1)
    plt.plot(t, x, color=colors['original'], label='Original')
    plt.plot(t, x_shuff, color=colors['shuffled'], label='Shuffled')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.ylim(-0.01, 1.01)
    remove_topright_axes()
    plt.text(0.15, 0.95, 'Original', color=colors['original'])
    plt.text(0.15, 0.8, 'Shuffled', color=colors['shuffled'])

    # Plot the spectra
    plt.subplot(1, 2, 2)
    plt.plot(f, y_lin, color=colors['original'], alpha=0.6)
    plt.plot(f, y_pol, color=colors['original'],
             alpha=0.5, linestyle='--')
    plt.plot(f, y_shuff_lin, color=colors['shuffled'], alpha=0.6)
    plt.plot(f, y_shuff_pol, color=colors['shuffled'],
             alpha=0.5, linestyle='--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 15)
    plt.ylim(0, 6)
    remove_topright_axes()
    plt.text(8, 5.5, 'Linear', color='grey')
    plt.plot([6, 7.5], [5.8, 5.8], '-', color='grey')
    plt.text(8, 4.6, '2nd order', color='grey')
    plt.plot([6, 7.5], [4.9, 4.9], '--', color='grey')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}illustrate_consistency.pdf")


def false_pos_summary():
    """
    Plot out the false positives for one type of noise.
    Separate plots for the overall false positives and by-frequency
    """

    analysis_details = {'landau': {'color': 'green'},
                        'fiebelkorn': {'color': 'darkturquoise'},
                        'mann_lees': {'color': 'midnightblue'},
                        'ar': {'color': 'mediumorchid'}}
    analysis_labels = {'landau': 'Time-shuffled (LF2012)',
                       'fiebelkorn': 'Time-shuffled (FSK2013)',
                       'mann_lees': 'Robust est.',
                       'ar': 'AR surrogate'}

    noise_details = [
        {'name': 'Fully random',
         'noise_method': 'fully_random'},
        {'name': 'White noise',
         'exponent': 0},
        # {'name': '$1/f$',
        #  'exponent': 1},
        {'name': 'Random walk',
         'exponent': 2},
        {'name': 'AR(1) noise',
         'noise_method': 'arma',
         'ar_coefs': 0.5, 'ma_coefs': 0.0}]

    p_thresh = 0.05  # Threshold for a positive result

    prop_signif_overall = {}
    ci_overall = {}
    prop_signif_by_freq = {}
    prop_signif_by_freq_raw = {}
    freqs = {}
    peaks = {}

    for i_noise in range(len(noise_details)):
        details = noise_details[i_noise]
        name = details.pop('name')
        prop_signif_overall[name] = {}
        ci_overall[name] = {}
        prop_signif_by_freq[name] = {}
        prop_signif_by_freq_raw[name] = {}
        peaks[name] = {}

        for analysis_type in analysis_details.keys():
            lit = analysis.load_simulation(analysis_type,
                                           f_osc=0, osc_amp=0,
                                           **details)
            res = lit['result']
            p = np.array([e['p_corr'] for e in res])  # Corrected for mult comp
            signif = np.any(p < p_thresh, axis=-1)
            ci = conf_int_prop(signif)
            signif_by_freq = np.mean(p < p_thresh, axis=0)
            # Save the proportion of significant results
            prop_signif_overall[name][analysis_type] = np.mean(signif)
            ci_overall[name][analysis_type] = ci
            prop_signif_by_freq[name][analysis_type] = signif_by_freq
            # Save the frequencies for each analysis method
            if i_noise == 0:
                freqs[analysis_type] = res[0]['f']
            # Uncorrected for mult comp over frequency
            if analysis_type in ('ar', 'mann_lees'):
                p_raw_field = 'p_raw'
            else:
                p_raw_field = 'p'
            p_raw = np.array([e[p_raw_field] for e in res])
            signif_by_freq_raw = np.mean(p_raw < p_thresh, axis=0)
            prop_signif_by_freq_raw[name][analysis_type] = signif_by_freq_raw

            # Find peak frequency in significant experiments  FIXME
            peak_freqs = []
            for res in lit['result']:
                if np.min(res['p_corr']) < p_thresh:
                    f_peak = res['f'][np.argmax(res['y_emp'])]
                    peak_freqs.append(f_peak)
            peak_freqs = np.array(peak_freqs)
            peaks[name][analysis_type] = peak_freqs.copy()

    # Make one big barplot of overall significance rates
    plt.figure(figsize=(11, 7))  # (8, 5) for the larger text size
    plt.clf()
    noise_types = list(prop_signif_overall.keys())
    analysis_types = list(prop_signif_overall['Fully random'].keys())
    psig = list(itertools.chain(*[prop_signif_overall[noise_type].values()
                                  for noise_type in noise_types]))
    cis = np.array(list(itertools.chain(*[ci_overall[noise_type].values()
                                          for noise_type in noise_types]))).T
    cis_dist = [psig - cis[0, :],
                cis[1, :] - psig]
    colors = [analysis_details[t]['color'] for t in analysis_types]
    plt.subplot(3, 1, 1)
    x_pos = [0, 1, 2, 3,
             6, 7, 8, 9,
             12, 13, 14, 15,
             18, 19, 20, 21]
    plt.bar(x_pos, psig,
            yerr=cis_dist,
            color=colors)
    plt.axhline(0.05, linestyle='--', color='k', linewidth=1)
    ypos = 0.9
    yadj = 0.12  # 0.2 for the larger text size
    for label, color in zip(analysis_types, colors):
        plt.text(-0.5, ypos, analysis_labels[label], color=color)
        ypos -= yadj
    plt.xticks([1.5, 7.5, 13.5, 19.5], prop_signif_overall.keys())
    plt.xlim(-1, 22)
    plt.ylim(0, 1)
    plt.ylabel('Prop. false pos.')
    remove_topright_axes()

    # Plot the results per freqeuency
    # NOT corrected for multiple comparisons across frequencies
    analysis_labels = {'landau': 'LF2012',
                       'fiebelkorn': 'FSK2013',
                       'mann_lees': 'Robust est.',
                       'ar': 'AR surrogate'}
    for i_noise_type, noise_type in enumerate(noise_types):
        plt.subplot(3, 4, 4 + 1 + i_noise_type)
        # plt.title(noise_type)
        # if i_noise_type == 0:
        #     ypos = 0.85
        #     yadj = 0.2
        #     for label, color in zip(analysis_types, colors):
        #         plt.text(0.5, ypos, analysis_labels[label], color=color)
        #         ypos -= yadj

        plt.axhline(0.05, linestyle='--', color='k', linewidth=1)
        for analysis_type in analysis_types:
            f = freqs[analysis_type]
            f_lim_lower = 0
            f_lim_upper = 12
            f_sel = (f_lim_lower < f) & (f < f_lim_upper)
            f = f[f_sel]
            psig = prop_signif_by_freq_raw[noise_type][analysis_type].copy()
            psig = psig[f_sel]
            plt.plot(f, psig,
                     linewidth=2,
                     color=analysis_details[analysis_type]['color'])
        plt.xlim(0, 12)
        plt.xticks([0, 5, 10])
        plt.xlabel('Frequency (Hz)')
        plt.ylim(0, 0.1)
        plt.yticks([0, 0.5, 1])
        remove_topright_axes()
        if i_noise_type == 0:
            plt.ylabel('Prop. false pos.\n(uncorrected)')

    # Plot the number of false-positive peaks per noise type
    bins = np.arange(0, 15.5)
    for i_noise_type, noise_type in enumerate(noise_types):
        plt.subplot(3, 4, 8 + 1 + i_noise_type)
        for i_meth, analysis_type in enumerate(analysis_details.keys()):
            plt.hist(peaks[noise_type][analysis_type],
                     bins=bins,  # + (i_meth - 1.5) * 0.05,
                     histtype='step',
                     color=analysis_details[analysis_type]['color'],
                     linewidth=2)
        plt.xlim(0, 12)
        plt.xticks([0, 5, 10])
        plt.ylim(0, 350)
        plt.xlabel('Frequency (Hz)')
        remove_topright_axes()
        if i_noise_type == 0:
            plt.ylabel('Count of\nspectral peaks')

    # # Plot the results per frequency
    # # Normalized so they can be sensibly compared across methods
    # for i_noise_type, noise_type in enumerate(noise_types):
    #     plt.subplot(3, 4, 8 + 1 + i_noise_type)
    #     for analysis_type in analysis_types:
    #         f = freqs[analysis_type]
    #         f_lim_lower = 0
    #         f_lim_upper = 12
    #         f_sel = (f_lim_lower < f) & (f < f_lim_upper)
    #         f = f[f_sel]
    #         psig = prop_signif_by_freq[noise_type][analysis_type].copy()
    #         psig = psig[f_sel]
    #         psig /= np.trapz(psig, f)  # Integrate to 1 so it shows density
    #         # Scale by overall prop. false pos per method
    #         psig *= prop_signif_overall[noise_type][analysis_type]
    #         plt.plot(f, psig,
    #                  linewidth=2,
    #                  color=analysis_details[analysis_type]['color'])
    #     plt.xlim(0, 12)
    #     # plt.yticks(np.linspace(0, np.round(plt.ylim()[1] + 0.05, 1), 2))
    #     plt.ylim(0, 0.25)
    #     plt.yticks([0, 0.25])
    #     plt.xlabel('Frequency (Hz)')
    #     remove_topright_axes()
    #     if i_noise_type == 0:
    #         plt.ylabel('False pos.\n(normed density)')

    plt.tight_layout(h_pad=3.0)
    plt.savefig(f"{plot_dir}false_pos_summary.png", dpi=200)
    plt.savefig(f"{plot_dir}false_pos_summary.eps")

    # Print a table of proportions of false positives
    props = pd.DataFrame(prop_signif_overall).T
    n = len(lit['result'])  # Number of experiments per simulation
    props.to_csv(data_dir + 'false_pos_props.csv')

    def binom_test_helper(p):
        s = stats.binom_test(p * n, n, p=.05, alternative='greater')
        return s

    pvals = props.applymap(binom_test_helper)
    print(props)
    print(pvals)


def example_spectra():
    """ Show example spectra with significant clusters
    """
    analysis_details = {'landau': {'color': 'green'},
                        'fiebelkorn': {'color': 'darkturquoise'}}

    noise_details = [
        {'name': 'Fully random',
         'noise_method': 'fully_random'},
        # {'name': 'White noise',
        #  'exponent': 0},
        # {'name': '$1/f$',
        #  'exponent': 1},
        # {'name': 'Random walk',
        #  'exponent': 2},
        {'name': 'AR(1)',
         'noise_method': 'arma',
         'ar_coefs': 0.5, 'ma_coefs': 0.0}
        ]

    np.random.seed(7)  # Set the seed for reproducible illustrative examples
    n_sims_to_plot = 3
    sims_to_plot = {}
    sims_to_plot['fiebelkorn'] = np.random.choice(1000,
                                                  size=n_sims_to_plot,
                                                  replace=False)
    sims_to_plot['landau'] = np.random.choice(1000,
                                              size=n_sims_to_plot,
                                              replace=False)
    colors = plt.cm.Dark2_r(np.linspace(0, 1, n_sims_to_plot))
    alpha = 0.05
    y_max_spectra = 1.5
    plt.figure(figsize=(5, 10/3))
    plt.clf()
    for i_noise in range(len(noise_details)):
        details = noise_details[i_noise]
        # noise_type = details.pop('name')

        for i_a_type, analysis_type in enumerate(analysis_details.keys()):
            i_plot = (1 + i_noise) + (i_a_type * len(noise_details))
            plt.subplot(len(analysis_details), len(noise_details), i_plot)
            # plt.title(f'{noise_type}; {analysis_type}')
            lit = analysis.load_simulation(analysis_type,
                                           f_osc=0, osc_amp=0,
                                           **details)
            for i_sim in range(n_sims_to_plot):
                n_sim = sims_to_plot[analysis_type][i_sim]
                f = lit['result'][0]['f']
                y = lit['result'][n_sim]['y_emp']
                signif = lit['result'][n_sim]['p_corr'] < alpha
                plt.plot(f, y, color=colors[i_sim])
                y_marker_pos = 1.4 - (i_sim * 0.1)
                plot_signif_bars(f, signif, y_marker_pos,
                                 color=colors[i_sim],
                                 linewidth=5)
            plt.xlim(0, 15)
            plt.ylim(0, y_max_spectra)
            plt.yticks([0, y_max_spectra])
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency (Hz)')
            remove_topright_axes()

    plt.tight_layout()
    plt.savefig(f"{plot_dir}example_spectra.png", dpi=200)
    plt.savefig(f"{plot_dir}example_spectra.eps")


def reconstructed_oscillations():
    """
    Plot the likelihood of detecting true oscillations
    """
    analysis_methods = ['landau', 'fiebelkorn',
                        'mann_lees', 'ar']
    freqs = np.arange(2, 13, 1)
    amps = np.arange(0.1, 0.7, 0.1)
    threshold = 0.05
    plot_params = dict(aspect='auto',
                       origin='lower')
    noise_params = dict(noise_method='powerlaw',
                        exponent=2)

    def _plot_helper(data, vmin=0, vmax=None, cb_label='', cm=plt.cm.magma):
        plt.figure(figsize=(4, 3))
        plt.imshow(data.T, vmin=vmin, vmax=vmax, cmap=cm, **plot_params)
        plt.colorbar(label=cb_label)
        plt.xticks(np.arange(0, len(freqs), 2), freqs[::2])
        plt.yticks(np.arange(0, len(amps), 2), [f"{a:.1f}" for a in amps[::2]])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amp. of behavioral oscillation\n(Prop. correct trials)')
        plt.tight_layout()

    p = {m: np.full([len(freqs), len(amps)], np.nan) for m in analysis_methods}
    p_ratio = {}
    mae = copy.deepcopy(p)
    freq_err = {}
    freq_err_by_param = {}
    peaks = {}
    for method in analysis_methods:
        # Load data without any oscillations as a baseline
        lit_base = analysis.load_simulation(method, f_osc=0, osc_amp=0,
                                            **noise_params)
        p_base = analysis.prop_sig(lit_base)

        freq_err[method] = []
        freq_err_by_param[method] = []
        peaks[method] = {}
        for i_freq, freq in enumerate(freqs):
            freq_err_by_param[method].append([])
            peaks[method][freq] = {}
            for i_amp, amp in enumerate(amps):

                # Load the data
                lit = analysis.load_simulation(method, f_osc=freq, osc_amp=amp,
                                               **noise_params)

                # Find the proportion of significant results
                p[method][i_freq, i_amp] = analysis.prop_sig(lit)

                # Find the Median Absolute Error of the peak signif frequency
                peak_freqs = []
                for res in lit['result']:
                    if np.min(res['p_corr']) < threshold:
                        f_peak = res['f'][np.argmax(res['y_emp'])]
                        peak_freqs.append(f_peak)
                peak_freqs = np.array(peak_freqs)
                peaks[method][freq][amp] = peak_freqs.copy()
                f_err = peak_freqs - freq

                # if freq >= 3 and amp >= 0.2:
                #     freq_err[method].extend(f_err)
                freq_err[method].extend(f_err)
                freq_err_by_param[method][i_freq].append(f_err)
                res_mse = np.median(np.abs(f_err))
                mae[method][i_freq, i_amp] = res_mse

        p_ratio[method] = p[method] / p_base

        # Plot results for this method

        # Proportion positive results
        _plot_helper(p[method],
                     vmax=1,
                     cb_label='Prop. positive results')
        plt.savefig(f"{plot_dir}prop_pos_results_{method}.eps")

        # Plot the peak frequency
        # Average
        peak_avg = np.full([len(freqs), len(amps)], np.nan)
        for i_freq, freq in enumerate(freqs):
            for i_amp, amp in enumerate(amps):
                peak_avg[i_freq, i_amp] = np.mean(peaks[method][freq][amp])
        _plot_helper(peak_avg,
                     vmin=0,
                     vmax=13,
                     cm=plt.cm.viridis,
                     cb_label='Mean peak (Hz)')
        plt.savefig(f"{plot_dir}peak_freq_{method}.eps")

        # # Mean absolute error of the significant frequencies
        # _plot_helper(mae[method],
        #              vmax=10,
        #              cm=plt.cm.viridis,
        #              cb_label='Median abs. err. (Hz)')
        # plt.savefig(f"{plot_dir}mse_pos_results_{method}.eps")

        # Ratio of true oscillations detected to baseline false positives
        _plot_helper(p_ratio[method], vmin=1, vmax=65,
                     cb_label='Detection ratio (veridical / false pos.)')
        plt.savefig(f"{plot_dir}prop_pos_results_baseline_{method}.eps")

    # Plot difference between methods
    p_diff = p['ar'] - p['mann_lees']
    _plot_helper(p_diff,
                 vmin=-np.max(np.abs(p_diff)),
                 vmax=np.max(np.abs(p_diff)),
                 cm=plt.cm.RdBu_r,
                 cb_label='Prop. positive results')
    plt.savefig(f"{plot_dir}prop_pos_results_ar-vs-mann_lees.eps")

    # mse_diff = mae['ar'] - mae['mann_lees']
    # _plot_helper(mse_diff,
    #              vmin=-10, vmax=10,
    #              cm=plt.cm.RdBu_r,
    #              cb_label='Median abs. err. (Hz)')
    # plt.savefig(f"{plot_dir}mse_pos_results_ar-vs-mann_lees.eps")

    # Plot the average MSE
    # Print the average abs. err. in reconstructed peak freq
    labels = {'landau': 'LF2012',
              'fiebelkorn': 'FSK2013',
              'mann_lees': 'Robust est.',
              'ar': 'AR surr.'}
    colors = {'landau': 'green',
              'fiebelkorn': 'darkturquoise',
              'mann_lees': 'midnightblue',
              'ar': 'mediumorchid'}

    peak_avg_by_cell = {}
    mae_by_cell = {}
    for method in labels.keys():
        peak_avg_tmp = np.full([len(freqs), len(amps)], np.nan)
        mae_tmp = peak_avg_tmp.copy()
        for i_freq, freq in enumerate(freqs):
            for i_amp, amp in enumerate(amps):
                mean_peak = np.mean(peaks[method][freq][amp])
                peak_avg_tmp[i_freq, i_amp] = mean_peak
                mae_tmp[i_freq, i_amp] = np.abs(mean_peak - freq)
        peak_avg_by_cell[method] = peak_avg_tmp
        mae_by_cell[method] = mae_tmp

    avg_mae = {k: np.mean(v[1:, 1:]) for k, v in mae_by_cell.items()}
    ci_mae = {k: 1.96 * np.std(v[1:, 1:]) / np.sqrt(v[1:, 1:].size)
              for k, v in mae_by_cell.items()}

    plt.figure(figsize=(3, 3))
    plt.bar(np.arange(len(mae)), avg_mae.values(), 0.5,
            yerr=ci_mae.values(),
            color=[colors[k] for k in avg_mae.keys()])
    plt.ylabel('Abs. Error (Hz)')
    plt.xticks(np.arange(len(mae)),
               [labels[k] for k in avg_mae.keys()],
               rotation=45)
    plt.yticks([0, 0.25, 0.5])
    remove_topright_axes()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}peak_error_comparison.eps")

    plt.figure(figsize=(3, 3))
    for i_meth, method in enumerate(labels.keys()):
        plt.boxplot(np.abs(freq_err[method]), positions=[i_meth],
                    widths=0.5,
                    boxprops={'color': colors[method]},
                    medianprops={'color': colors[method]},
                    whiskerprops={'color': colors[method]},
                    capprops={'color': colors[method]},
                    flierprops={'markeredgecolor': colors[method],
                                'alpha': 0.005})
    plt.gca().set_yscale('log')
    plt.ylabel('Abs. Error (Hz)')
    plt.xticks(np.arange(len(labels)),
               labels.values(),
               rotation=45)
    plt.yticks([0.1, 1, 10], ['0.1', '1', '10'])
    remove_topright_axes()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}peak_error_comparison_boxplot.png")

    plt.figure(figsize=(3, 3))
    plt.axhline(y=0, color='k', linestyle='--', zorder=0)
    parts = plt.violinplot(freq_err.values(),
                           widths=0.6,
                           showmeans=False,
                           showmedians=False,
                           showextrema=False)
    for meth, pc in zip(freq_err.keys(), parts['bodies']):
        pc.set_facecolor(colors[meth])
        pc.set_alpha(0.8)
    plt.ylabel('Error (Hz)')
    plt.xticks(np.arange(len(labels)) + 1,
               labels.values(),
               rotation=45)
    remove_topright_axes()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}peak_error_comparison_violin.pdf")

    # Plot the distribution of individual errors in the recovered frequency
    _, (ax0, ax1) = plt.subplots(1, 2,
                                 gridspec_kw={'width_ratios': [3, 1]},
                                 figsize=(7.5, 4))
    plt.clf()
    plt.subplot(ax0)
    parts = plt.violinplot(freq_err.values(),
                           showmeans=False,
                           showmedians=False,
                           showextrema=False)
    for meth, pc in zip(freq_err.keys(), parts['bodies']):
        pc.set_facecolor(colors[meth])
        pc.set_alpha(1)
    plt.ylabel('Error (Hz)')
    plt.xticks(np.arange(len(labels)) + 1,
               [labels[k] for k in freq_err.keys()])
    remove_topright_axes()
    plt.subplot(ax1)
    for i_meth, meth in enumerate(freq_err.keys()):
        x = np.abs(freq_err[meth])
        plt.plot(i_meth, np.median(x), 'o',
                 markerfacecolor=colors[meth],
                 color=colors[meth])
        # ci = bootstrap(x, 10, np.median)
        # plt.plot([i_meth, i_meth], ci, '-', color=colors[meth])
    plt.xlim(-0.5, 3.5)
    plt.ylim(0, 2)
    plt.ylabel('Median Abs. Error (Hz)')
    plt.xticks(np.arange(len(labels)), [labels[k] for k in freq_err.keys()],
               rotation=75)
    plt.yticks(np.arange(0, 2.1, 0.5))
    remove_topright_axes()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}mse_pos_results_comparison_violin.pdf")

    # Barplot of detection ratio
    avg_ratio = {k: np.mean(v[1:, 1:]) for k, v in p_ratio.items()}
    ci_ratio = {k: 1.96 * np.std(v[1:, 1:]) / np.sqrt(v[1:, 1:].size)
                for k, v in p_ratio.items()}
    plt.figure(figsize=(3, 3))
    plt.bar(np.arange(len(p_ratio)), avg_ratio.values(), 0.5,
            yerr=ci_ratio.values(),
            color=[colors[k] for k in avg_ratio.keys()])
    plt.ylabel('Detection ratio')
    plt.xticks(np.arange(len(p_ratio)),
               [labels[k] for k in avg_ratio.keys()],
               rotation=45)
    remove_topright_axes()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}prop_pos_results_baseline_comparison.eps")

    # Boxplot of detection ratio
    plt.figure(figsize=(3, 3))
    for i_meth, method in enumerate(labels.keys()):
        plt.boxplot(p_ratio[method].flatten(),
                    positions=[i_meth],
                    widths=0.5,
                    boxprops={'color': colors[method]},
                    medianprops={'color': colors[method]},
                    whiskerprops={'color': colors[method]},
                    capprops={'color': colors[method]},
                    flierprops={'markeredgecolor': colors[method],
                                'alpha': 0.005})
    plt.ylabel('Detection ratio')
    plt.ylim(0, None)
    plt.xticks(np.arange(len(p_ratio)),
               [labels[k] for k in avg_ratio.keys()],
               rotation=45)
    remove_topright_axes()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}prop_pos_results_baseline_comparison_boxplot.pdf")

    # Stats on the differences in detection ratio between analysis methods
    print('Stats: Detection ratio')
    for comp in itertools.combinations(p_ratio.keys(), 2):
        dfs = {}
        for c in comp:
            df = pd.DataFrame(p_ratio[c])
            df.columns = [f'{n:.1}' for n in amps]
            df['freq'] = freqs
            df = pd.melt(df.iloc[1:, 1:], id_vars='freq', var_name='amp')
            df['analysis'] = c
            dfs[c] = df
        df = pd.concat(dfs.values())
        df.amp = df.amp.astype(float)
        mdl = ols("value ~ freq + amp + analysis", df).fit()
        mdl = ols("value ~ freq + amp + analysis", df).fit()
        mdl_res = mdl.summary().tables[1].data[2]
        coef, sem, t, pval, ci025, ci975 = [float(el) for el in mdl_res[1:]]
        # Print message in Latex table format
        # Columns: Comparison, beta [95% CI], t, p, #df_model, df_resid
        label = ', '.join([labels[el] for el in comp])
        msg = f"{label} & {coef:.2f} & {ci025:.2f}, {ci975:.2f} & "\
              f"{t:.2f} & {sci_notation(mdl.pvalues[1], 1)} \\\\"
        print(msg)


def generate_all_plots():
    illustrate_shuffle()
    illustrate_consistency()
    false_pos_summary()
    example_spectra()
    reconstructed_oscillations()


if __name__ == '__main__':
    generate_all_plots()
