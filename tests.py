#!/usr/bin/env python3

import unittest
import numpy as np
from scipy import signal
import analysis
import simulate_behavior as sim_behav
import simulate_experiments as sim_exp
from analysis_methods import shuff_time, alternatives, utils

np.random.seed(0)  # Make sure the randomization tests are reproducible


class TestAnalysisTools(unittest.TestCase):

    def setUp(self):
        self.n_experiments = 10  # Tests are more robust for higher numbers
        print('\n')

    # @unittest.skip('Saving time by skipping this')
    def test_analysis_methods_negative(self):
        # All methods give null results when the data are completely random
        for analysis_method in analysis.analysis_funcs.keys():
            lit = analysis.simulate_lit(analysis_method, self.n_experiments,
                                        save=False,
                                        noise_method='fully_random',
                                        f_osc=0, osc_amp=0)
            print('Prop sig =', analysis.prop_sig(lit))
            self.assertTrue(analysis.prop_sig(lit) <= 0.2)

    # @unittest.skip('Saving time by skipping this')
    def test_analysis_methods_positive(self):
        # Make sure all methods give signif results when there are oscillations
        for analysis_method in analysis.analysis_funcs.keys():
            lit = analysis.simulate_lit(analysis_method, self.n_experiments,
                                        save=False,
                                        noise_method='fully_random',
                                        f_osc=8, osc_amp=0.8)
            print('Prop sig =', analysis.prop_sig(lit))
            self.assertTrue(analysis.prop_sig(lit) > 0.5)

    def test_prop_sig(self):
        # Simulate data where every experiment should give a significant result
        lit = analysis.simulate_lit('ar', self.n_experiments,
                                    save=False,
                                    noise_method='fully_random',
                                    f_osc=8, osc_amp=0.999)
        self.assertEqual(analysis.prop_sig(lit), 1)


class TestSimulateExperiments(unittest.TestCase):

    def setUp(self):
        self.behav_details = sim_exp.behav_details
        # Simulate experiments that should give positive results
        self.behav_kwargs = {'noise_method': 'fully_random',
                             'f_osc': 5,
                             'osc_amp': 0.99}

    def test_behav_details(self):
        self.assertIsInstance(self.behav_details, dict)
        self.assertIsInstance(self.behav_details['landau'], dict)

    def test_ar_experiment(self):
        res = sim_exp.ar_experiment(**self.behav_details,
                                    **self.behav_kwargs)
        self.assertIsInstance(res, dict)
        # This should give a significant result
        self.assertTrue(np.min(res['p_corr']) < .05)

    def test_robust_est_experiment(self):
        res = sim_exp.robust_est_experiment(**self.behav_details,
                                            **self.behav_kwargs)
        self.assertIsInstance(res, dict)
        # This should give a significant result
        self.assertTrue(np.min(res['p_corr']) < .05)

    def test_landau_experiment(self):
        res = sim_exp.landau_experiment(**self.behav_details,
                                        **self.behav_kwargs)
        self.assertIsInstance(res, dict)
        # This should give a significant result
        self.assertTrue(np.min(res['p_corr']) < .05)

    def test_fiebelkorn_experiment(self):
        res = sim_exp.fiebelkorn_experiment(**self.behav_details,
                                            **self.behav_kwargs)
        self.assertIsInstance(res, dict)
        # This should give a significant result
        self.assertTrue(np.min(res['p_corr']) < .05)


class TestSimulateBehavior(unittest.TestCase):

    def setUp(self):
        self.behav_kwargs = {'exponent': 0.0,
                             'f_osc': 0.0,
                             'osc_amp': 0.0}
        self.fs = 60
        self.t_start = 0
        self.t_end = 1
        self.expected_len = 61

    def test_n_behav_samples(self):
        n = sim_behav.n_behav_samples(self.fs,
                                      self.t_start,
                                      self.t_end)
        self.assertEqual(n, self.expected_len)
        self.assertIsInstance(n, int)

    def test_time_vector(self):
        t = sim_behav.time_vector(self.fs,
                                  self.t_start,
                                  self.t_end)
        self.assertEqual(t[0], self.t_start)
        self.assertEqual(t[-1], self.t_end)
        self.assertEqual(len(t), self.expected_len)

    def test_simulate_behavior(self):
        x = sim_behav.simulate_behavior(self.fs,
                                        self.t_start,
                                        self.t_end,
                                        **self.behav_kwargs)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), self.expected_len)

    def test_trialwise_behavior(self):
        # Test simulated data
        details = sim_exp.behav_details['fiebelkorn']
        x_trial, t_trial = sim_behav.simulate_behavior_trialwise(
                                                details,
                                                **self.behav_kwargs)
        self.assertEqual(len(x_trial),  # Time and Acc same length
                         len(t_trial))
        self.assertTrue(np.all(np.isin(x_trial, [0, 1])))  # All hits or misses


class TestAnalysisMethods(unittest.TestCase):

    # Methods using shuffling in time

    def setUp(self):
        self.n = 100
        self.fs = 100
        self.k_perm = 50

    def analysis_method_helper(self, res):
        """ Run tests on the analysis methods
        """
        # Make sure the p-values are in the expected range
        self.assertTrue(np.all(res['p_corr']) <= 1)
        self.assertTrue(np.all(res['p_corr']) >= 0)
        # Make sure there's one p-value per frequency
        self.assertEqual(len(res['y_emp']), len(res['f']))
        self.assertEqual(len(res['p_corr']), len(res['f']))

    def test_window(self):
        # FFT windowing function
        # Test with 1D data
        x = np.ones(self.n)
        win = np.hanning(self.n)
        # Windowing all ones should give back the window
        x_w = shuff_time.window(x, win)
        self.assertTrue(np.all(x_w == win))
        self.assertEqual(len(x_w), self.n)
        # Windowing all zeros should give back zeros
        x_w = shuff_time.window(np.zeros(win.shape), win)
        self.assertTrue(np.all(x_w == 0))
        # Test with 2D data
        n_timecourses = 5
        x = np.ones([self.n, n_timecourses])
        win = np.hanning(self.n)
        x_w = shuff_time.window(x, win)
        self.assertTrue(np.all(x_w[:, 0] == win))
        self.assertEqual(x_w.shape, x.shape)
        self.assertEqual(x_w.shape, (self.n, n_timecourses))

    def test_time_shuffled_perm(self):

        def analysis_fnc_helper(x):
            """ Helper function for testing """
            f, y = utils.dft(x, self.fs, len(x))
            return f, y

        x = np.arange(self.n)
        res = shuff_time.time_shuffled_perm(analysis_fnc_helper,
                                            x,
                                            self.k_perm)
        self.assertTrue(np.all(x == res['x']))  # Original sequence is intact
        # Make sure all the permutations have the same data points
        self.assertTrue(np.all(np.apply_along_axis(
            lambda x_row: np.all(np.unique(x_row) == x),
            1,
            res['x_perm']
            )))
        # Make sure the points are in a different order for each perm
        n_unique_per_perm = np.apply_along_axis(
            lambda x_col: len(np.unique(x_col)),
            0,
            res['x_perm'])
        self.assertTrue(np.all(n_unique_per_perm > 1))

    def test_landau(self):
        t = np.arange(self.n) / self.fs
        x = np.random.normal(size=self.n)
        res = shuff_time.landau(x, t, self.fs, self.k_perm)
        # Run standard tests
        self.analysis_method_helper(res)

    def test_landau_spectrum(self):
        details = sim_exp.behav_details['landau']
        t = np.arange(details['t_start'],
                      details['t_end'] + 0.0001,
                      details['fs'] ** -1)
        freq = 8
        x = np.sin(2 * np.pi * freq * t)
        f, y = shuff_time.landau_spectrum(x, details['fs'])
        peak_inx = np.argmax(y)
        peak_freq = f[peak_inx]
        # Does this recover the right frequency?
        tol = 0.5  # tolerance in Hz
        self.assertTrue(np.abs(peak_freq - freq) < tol)

    def test_fiebelkorn(self):
        behav_kwargs = {'noise_method': 'fully_random',
                        'f_osc': 0,
                        'osc_amp': 0}
        details = sim_exp.behav_details['fiebelkorn']
        x_trial, t_trial = sim_behav.simulate_behavior_trialwise(
                details, **behav_kwargs)
        res = shuff_time.fiebelkorn(x_trial, t_trial, details['k_perm'])
        # Run standard tests
        self.analysis_method_helper(res)

    def test_fiebelkorn_binning(self):
        # Test binning procedure
        details = sim_exp.behav_details['fiebelkorn']
        t = sim_behav.time_vector(details['fs'],
                                  details['t_start'],
                                  details['t_end'])
        t_trial = np.tile(t, 20)  # Repeated measures at each step
        # Make 100% accuracy at times at or above 0.5, 0% below that
        x_trial = t_trial >= 0.5
        # Compute binned accuracy
        x_bin, t_bin = shuff_time.fiebelkorn_binning(x_trial, t_trial)
        # Make sure the proportions correct are between 0 and 1
        self.assertTrue(np.min(x_bin) >= 0)
        self.assertTrue(np.max(x_bin) <= 1)
        # Check whether the accuracy time-series starts at zero and ends at one
        self.assertEqual(x_bin[0], 0)
        self.assertEqual(x_bin[-1], 1)
        # Check whether values increase monotonically
        self.assertTrue(np.all(np.diff(x_bin) >= 0))
        # Check whether the sampling rate is as expected
        self.assertEqual(np.round(np.mean(np.diff(t_bin)), 3),
                         details['bin_step'])

    def test_fiebelkorn_spectrum(self):
        behav_kwargs = {'noise_method': 'fully_random',
                        'f_osc': 8,
                        'osc_amp': 0.99}
        details = sim_exp.behav_details['fiebelkorn']
        x_trial, t_trial = sim_behav.simulate_behavior_trialwise(
                details, **behav_kwargs)
        f, y = shuff_time.fiebelkorn_spectrum(x_trial, t_trial)
        peak_inx = np.argmax(y)
        peak_freq = f[peak_inx]
        # Does this recover the right frequency?
        tol = 0.5  # tolerance in Hz
        self.assertTrue(np.abs(peak_freq - behav_kwargs['f_osc']) < tol)

    # Alternative methods
    def test_ar_surr(self):
        x = np.random.normal(size=self.n)
        res = alternatives.ar_surr(x, self.fs, self.k_perm)
        self.assertIsInstance(res, dict)
        # Make sure the frequencies are what we expect
        expected_freqs = np.arange(1, self.fs / 2)
        expected_freqs = expected_freqs[expected_freqs <= 15]
        self.assertTrue(np.all(np.isclose(res['f'], expected_freqs)))
        # Make sure we ran the right number of permutations
        expected_shape = (self.k_perm, len(expected_freqs))
        self.assertTrue(res['y_perm'].shape == expected_shape)
        # Run standard tests
        self.analysis_method_helper(res)

    def test_robust_est(self):
        x = np.random.normal(size=self.n)
        res = alternatives.robust_est(x, self.fs)
        # Run standard tests
        self.analysis_method_helper(res)

    def test_fit_ar_spec(self):
        n = int(1e6)
        nfft = 100
        fs = 100.
        # Does this give a flat spectrum for white noise?
        x = sim_behav.generate_arma(0, 0, n)
        f, y = signal.welch(x, fs=fs, nperseg=nfft, noverlap=nfft / 2)
        spec_fit = alternatives.fit_ar_spec(f, y, fs / 2)
        tol = 0.01  # 1% tolerance between higest and lowest value
        self.assertTrue(
                (spec_fit.min() - spec_fit.max()) < (spec_fit.max() * tol))
        # Does this give a downward sloping spectrum for AR(1) noise?
        x = sim_behav.generate_arma(0.9, 0, n)
        f, y = signal.welch(x, fs=fs, nperseg=nfft, noverlap=nfft / 2)
        spec_fit = alternatives.fit_ar_spec(f, y, fs / 2)
        self.assertTrue(np.all(np.diff(spec_fit) < 0))

    def test_clusterstat_1d(self):
        cluster_size = 10  # Length of the cluster
        cluster_amp = 10  # The size of the bump in the cluster

        # Simulate some data with a peak that SHOULD be significant
        x_emp = np.random.normal(size=self.n)
        x_emp[:cluster_size] += cluster_amp
        # Make a surrogate distribution without a consistent peak
        x_perm = np.random.normal(size=(self.k_perm, self.n))
        # Compute the cluster statistic (2-tailed)
        p_clust, cluster_info = alternatives.clusterstat_1d(x_emp, x_perm)
        self.assertEqual(p_clust, 0)  # Should be very significant

        # Simulate some data that should NOT be significant
        x_emp = np.random.normal(scale=0, size=self.n)
        p_clust, cluster_info = alternatives.clusterstat_1d(x_emp, x_perm)
        self.assertTrue(p_clust > 0.1)  # Should not be significant

    # Utils
    def test_avg_repeated_timepoints(self):
        t_trial = np.array([0, 0, 1, 1, 2, 2])  # Time-stamps
        x_trial = np.array([0, 0, 0, 1, 1, 1])  # Accuracy by trial
        t, x = utils.avg_repeated_timepoints(t_trial, x_trial)
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(t.tolist(), [0, 1, 2])
        self.assertEqual(x.tolist(), [0, 0.5, 1])

    def test_dft(self):
        # Test whether this reconstructs a pure sine wave
        n = 100
        fs = 100
        freq = 5
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * freq * t)
        f, y = utils.dft(x, fs, n)
        peak_inx = np.argmax(y)
        peak_freq = f[peak_inx]
        # Did it get the right peak?
        self.assertEqual(peak_freq, freq)
        # Is everything off the peak close to zero?
        off_peak = np.r_[y[:peak_inx], y[(peak_inx + 1):]]
        self.assertTrue(np.all(np.isclose(off_peak, 0)))


if __name__ == '__main__':
    unittest.main()
