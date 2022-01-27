// Simple linear regression.
data {
  int<lower=0> N;  // number of data points
  array[N] real x;
  array[N] real y;
}

parameters {
  real a;
  real b;
  real<lower=0> sigma;
}

transformed parameters {
   array[N] real mu;

   for (n in 1:N) {
     mu[n] = a + b * x[n];
   }
}

model {
  // Priors
  a ~ normal(0, 5);
  b ~ normal(0, 5);
  sigma ~ normal(0, 5);

  // Likelihood
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
