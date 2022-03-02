# Construct Table 1
# Test whether each analysis method controls the rate of false positives for
# each type of noise.

data_dir <- 'results/'

n_exp <- 1000  # Number of experiments per simulated literature
alpha <- 0.05  # Alpha level to test against

false_pos_props <- read.csv(paste0(data_dir, 'false_pos_props.csv'))
colnames(false_pos_props) <- c('noise_type',
                               'LF2012',
                               'FSK2013',
                               'Robust est.',
                               'AR surr')

for (i_noise_type in 1:nrow(false_pos_props)) {
  for (i_meth in 2:5) {
    method <- colnames(false_pos_props)[i_meth]
    if (method == 'LF2012') {
      cat('\\hline \n')
    }
    noise_type <- ifelse(method == 'LF2012',
                         false_pos_props[i_noise_type, 1],
                         '           ')
    prop <- false_pos_props[i_noise_type, method]
    btest <- binom.test(prop * n_exp, n_exp, alpha, alternative = 'greater')
    btest_ci <- binom.test(prop * n_exp, n_exp, alpha)
    msg <- '%s & %s & %.3f & %.3f, %.3f & %.1g \\\\\n'
    msg <- sprintf(msg,
                    noise_type,
                    method,
                    prop,
                    btest_ci$conf.int[1],
                    btest_ci$conf.int[2],
                    btest$p.value)
    # Put the p-value into Latex-formatted scientific notation
    msg <- sub('([0-9])e([-0-9]+)',
               '$\\1 \\\\times 10^{\\2}$',
               msg)
    msg <- sub('& ([01]) ',
               '& $\\\\approx \\1$ ',
               msg)
    cat(msg)
  }
}
