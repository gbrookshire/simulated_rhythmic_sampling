#!/usr/bin/env python3
"""
Tools to run the simulation analyses
"""

import numpy as np
import sys
from tqdm import tqdm
import simulate_experiments as simex

# Set this variable to use status bars or not. Disable when running on sbatch
show_status_bar = False
if show_status_bar:
    status_bar = tqdm
else:
    def status_bar(arg):
        return arg

analysis_funcs = {
        'ar': simex.ar_experiment,
        'landau': simex.landau_experiment,
        'fiebelkorn': simex.fiebelkorn_experiment,
        'mann_lees': simex.robust_est_experiment
        }

save_dir = 'results/'


def simulate_lit(analysis_type, n_experiments, save=False, desc='',
                 **behav_kwargs):
    """
    Simulate a literature (collection of experiments) with specified params.

    Parameters
    ----------
    analysis_type : str, function
        The type of analysis used to identify significant oscillations.
        Options: landau, fiebelkorn, ar, mann_lees (or a function)
    n_experiments : int
        The number of experiments to simulate
    save : bool
        Save the results instead of returning them. This makes it easier to run
        lots of analyses on the computing cluster.
    desc : str
        Additional description of the analysis
    **behav_kwargs
        Additional keyword arguments are passed to the function that simulates
        a single experiment. These args describe the details of the noise
        simulation (see simulate_behavior.simulate_behavior).

    Returns
    -------
    lit : dict
        A dictionary holding the results of the simulation.
        type : str
            The analysis type
        result : list
            A list of simulated experiments. Each element holds the output of
            the experiment function simulate_experiment.*_experiment
    """
    summary = f"{analysis_type}, N={n_experiments}, details: {behav_kwargs}"
    print(summary)

    # Make a helper function to simulate a single experiment
    if isinstance(analysis_type, str):
        def exp_fnc():
            fnc_helper = analysis_funcs[analysis_type]
            res = fnc_helper(**behav_kwargs)
            return res
    elif callable(analysis_type):
        def exp_fnc():
            res = analysis_type(**behav_kwargs)
            return res

    # Simulate a literature of experiments
    lit_res = []
    for k in status_bar(range(n_experiments)):
        res = exp_fnc()
        lit_res.append(res)

    # Package the results together
    lit = {'type': analysis_type, 'result': lit_res}
    lit.update(behav_kwargs)  # Save the simulation details
    if save:
        save_simulation(lit, analysis_type, desc=desc, **behav_kwargs)
    return lit


def save_simulation(lit, analysis_type, desc='', **behav_kwargs):
    """
    Save the results of simulate_lit(). To save space, don't save the full
    permuted data, only save the summaries.

    Parameters
    ----------
    lit : dict
        The results of simulate_lit
    analysis_type : str
        The type of analysis used to identify significant oscillations.
        Options: landau, fiebelkorn, ar, mann_lees
    desc : str
        Additional description of the analysis
    **behav_kwargs
        Additional keyword arguments that describe the details of the noise
        simulation (see simulate_behavior.simulate_behavior). Used to generate
        the filename.

    Returns
    -------
    None
    """
    # Remove the full permutations from the data to save space
    for i in range(len(lit['result'])):
        d = lit['result'][i]
        try:
            d.pop('x_perm')
            d.pop('y_perm')
        except KeyError:
            pass
    fn = save_fname(analysis_type, desc=desc, **behav_kwargs)
    print('saving:', fn)
    np.save(fn, lit)


def load_simulation(analysis_type, desc='', **behav_kwargs):
    """
    Load the saved results of a simulation.

    Parameters
    ----------
    analysis_type : str
        The type of analysis used to identify significant oscillations.
        Options: landau, fiebelkorn, ar, mann_lees
    desc : str
        Additional description of the analysis
    **behav_kwargs
        Additional keyword arguments that describe the details of the noise
        simulation (see simulate_behavior.simulate_behavior). Used to generate
        the filename.

    Returns
    -------
    lit : dict
        A simulated literature output by simulate_lit
    """
    fn = save_fname(analysis_type, desc=desc, **behav_kwargs)
    print('loading:', fn)
    x = np.load(fn, allow_pickle=True).item()
    return x


def save_fname(analysis_type, desc='', **behav_kwargs):
    """
    Create the filename for a given type of simulation.
    Round all numeric args to 2 decimal places.

    Parameters
    ----------
    analysis_type : str
        The type of analysis used to identify significant oscillations.
        Options: landau, fiebelkorn, ar, mann_lees
    desc : str
        Additional description of the analysis
    **behav_kwargs
        Additional keyword arguments that describe the details of the noise
        simulation (see simulate_behavior.simulate_behavior). Used to generate
        the filename.

    Returns
    -------
    fn : str
        The filename where this simulation will be saved
    """
    a = analysis_type
    f = behav_kwargs['f_osc']
    s = behav_kwargs['osc_amp']
    if 'exponent' in behav_kwargs.keys():
        e = behav_kwargs['exponent']
        fn = f"{save_dir}{a}_exp_{e:.2f}_f_{f:.2f}_amp_{s:.2f}"
    elif behav_kwargs['noise_method'] == 'arma':
        coefs = {}
        arma_coef_types = ['ar_coefs', 'ma_coefs']
        for coef_type in arma_coef_types:
            try:
                c = behav_kwargs[coef_type]
                if type(c) not in (float, int):
                    msg = 'ARMA noise only implemented for models with up ' \
                          'to one AR term and one MA term'
                    raise(NotImplementedError(msg))
                coefs[coef_type] = c
            except KeyError:  # Set non-specified terms to 0
                coefs[coef_type] = 0
        fn = f"ar_{coefs['ar_coefs']:.2f}_ma_{coefs['ma_coefs']:.2f}"
        fn = f"{save_dir}{a}_{fn}_f_{f:.2f}_amp_{s:.2f}"
    elif behav_kwargs['noise_method'] == 'fully_random':
        fn = f"{save_dir}{a}_fully_random_f_{f:.2f}_amp_{s:.2f}"
    else:
        msg = 'Saving not implemented for these behavioral details'
        raise(NotImplementedError(msg))

    fn = fn + f'{desc}.npy'

    return fn


def prop_sig(lit):
    """
    Get the proportion of significant experiments in a simulated literature

    Parameters
    ----------
    lit : dict
        A simulated literature output by simulate_lit

    Returns
    -------
    prop_signif : float
        The proportion of experiments significant at p < .05
    """
    min_p_per_experiment = [np.min(e['p_corr']) for e in lit['result']]
    min_p_per_experiment = np.array(min_p_per_experiment)
    prop_signif = np.mean(min_p_per_experiment < 0.05)
    return prop_signif


def main():
    """
    Simulate a literature from the command line.

    This file's name should be followed by a single-quoted comma-separated list
    of arguments to be passed to simulate_lit().
    E.g.
    $ python analysis.py \
      'analysis_type="ar", n_experiments=10, exponent=0, f_osc=9, osc_amp=0.3'
    """
    # Make a dictionary of arguments for simulating the behavior
    arg_list = sys.argv[1].split(',')
    kwargs = {}
    for arg in arg_list:
        k, v = arg.split('=')
        k = k.strip(' ')  # Strip out spaces
        v = v.strip('"\' ')  # Strip out quotes and spaces
        try:
            v = float(v)
        except ValueError:
            pass
        kwargs[k] = v
    # Run the simulation with those arguments
    n_experiments = int(kwargs.pop('n_experiments'))
    analysis_type = kwargs.pop('analysis_type')
    # Round the values to 2 decimal places
    for k, v in kwargs.items():
        if type(v) in (int, float):
            kwargs[k] = round(v, 2)

    # Run the analysis
    lit = simulate_lit(analysis_type, n_experiments, save=True, **kwargs)

    return lit


if __name__ == '__main__':
    main()
