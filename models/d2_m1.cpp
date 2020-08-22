data {
    int<lower=0> N;       // number of data points
    vector[N] y;          // LFC data
}

parameters {
    real alpha;           // intercept
    real<lower=0> sigma;  // standard deviation
}

model {
    // Priors
    alpha ~ normal(0, 5);
    sigma ~ cauchy(0, 5);
    
    y ~ normal(alpha, sigma);
}

generated quantities {
    vector[N] y_pred;
    for (n in 1:N)
        y_pred[n] = normal_rng(alpha, sigma);
}