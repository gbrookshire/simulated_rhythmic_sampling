import numpy as np
from scipy import stats
import re
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

rcomp = importr('rcompanion')


def cramers_v(tbl):
    """
    Compute Cramer's V (measure of effect size).

    Parameters
    ----------
    tbl : np.ndarray of ints (2-dim)
        Contingency table like is used to compute a chi-squared test

    Returns
    -------
    v : float
        Cramer's V
    ci_low, ci_high : float
        The bootstrapped lower and upper 95% CI on Cramer's V
    """
    m = ro.r.matrix(ro.IntVector(tbl.flatten()),
                    ncol=tbl.shape[0])
    ro.globalenv['m'] = m
    res = ro.r('cramerV(m, ci=TRUE)')  # Compute effect size and CI
    res = [e.tolist()[0] for e in np.asarray(res)]  # Convert to a list
    v, ci_low, ci_high = res
    return v, ci_low, ci_high


def chi_square_report(tbl):
    """
    Print the statistical report of a chi-squared test on a contingency table,
    formatted for LaTeX.

    Parameters
    ----------
    tbl : np.ndarray of ints (2-dim)
        Contingency table like is used to compute a chi-squared test

    Returns
    -------
    msg : str
        The statistical report. Takes the following form:
    """
    try:
        # Run a chi-square test
        chi2, pval, dof, exp = stats.chi2_contingency(tbl)
        msg = f'$\\chi^2({dof}) = {chi2:.1f}$, '
        msg += f'$p = {pval:.1g}$'
        # Get the effect size and CI
        v, ci_low, ci_high = cramers_v(tbl)
        msg += fr', $\phi_C = {v:.2f}$ [{ci_low:.2f}, {ci_high:.2f}]'
        # Format the p-value for scientific notation
        msg = re.sub('([0-9])e([-0-9]+)',
                     r'\1 \\times 10^{\2}',
                     msg)
    except ValueError:
        msg = "Can't compute chi-square; at floor/ceiling"

    return msg
