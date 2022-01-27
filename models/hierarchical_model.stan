// Flexible hierarchical linear regression model.

data {
  int<lower=0> N;                     // number of observations total
  int<lower=1> J;                     // number of groups
  int<lower=1> K;                     // number of features
  array[N] int<lower=1,upper=J> idx;  // group indices
  array[N] row_vector[K] X;           // model matrix)
  array[N] real y;                    // response variable
  }

  parameters {
  vector[K] mu_beta;
  vector<lower=0>[K] mu_sigma;
  array[J] vector[K] delta_beta;
  real<lower=0> sigma;
}

transformed parameters {
  array[N] real mu;
  array[J] vector[K] beta;

  for(j in 1:J) {
    beta[j] = mu_beta + mu_sigma .* delta_beta[j];
  }

  for(n in 1:N) {
    mu[n] = X[n] * beta[idx[n]];
  }
}

model {
  // Priors.
  mu_beta ~ normal(0, 5);
  mu_sigma ~ cauchy(0, 2.5);
  sigma ~ gamma(2, 0.1);

  for(j in 1:J) {
    delta_beta[j] ~ normal(0, 1);
  }

  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  array[N] real y_hat;

  y_hat = normal_rng(mu, sigma);

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
