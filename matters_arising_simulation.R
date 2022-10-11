# Run a simulation of a t-test with a suboptimal study design
# for the response to the "Matters arising" article by Re and colleages

set.seed(0)

# A function to simulate a group of experiments
simulate.t.test <- function(n.obs.per.group) {
  n.simulations <- 1e4  # Number of simulated experiments
  true.difference <- 0.1  # In units of standard deviations
  alpha <- 0.05  # The significance threshold
  res <-  replicate(n.simulations,
                    {
                      t.test(
                        x = rnorm(n.obs.per.group, mean = 0),
                        y = rnorm(n.obs.per.group, mean = true.difference),
                        paired = FALSE,
                        var.equal = TRUE
                      )$p.value
                    })
  return(mean(res < alpha))
}

# Run the simulations
n.obs <- c(5, 5000)  # Number of observations collected
sapply(n.obs, simulate.t.test)