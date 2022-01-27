// Negative binomial model.

data {
    int<lower=0> N;  // number of data points
    int<lower=1> K;  // number of covariates
    matrix[N, K] X;  // covariates
    array[N] int y;  // observed counts
}

parameters {
    vector[K] beta;
    real<lower=0> alpha;
}

transformed parameters {
    vector[N] mu = exp(X * beta);
}

model {
    // Priors
    beta ~ normal(0.0, 5.0);
    alpha ~ cauchy(0.0, 10.0);

    // Likelihood
    y ~ neg_binomial_2(mu, alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] real y_hat;

  y_hat = neg_binomial_2_rng(mu, alpha);

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu[n], alpha);
  }
}
