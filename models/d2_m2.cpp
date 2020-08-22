data {
    int<lower=0> N;               // number of data points
    int<lower=1> S;               // number of shRNAs
    
    int<lower=1, upper=S> shrna[N];  // index shRNA
    
    vector[N] y;                  // LFC data
}

parameters {
    
    real mu_alpha;
    real<lower=0> sigma_alpha;
    
    real alpha[S];               // intercept
    real<lower=0> sigma;         // standard deviation
}

model {
    // Hyperpriors
    mu_alpha ~ normal(0, 10.0);
    sigma_alpha ~ cauchy(0, 10.0);
    
    // Priors
    for (s in 1:S)
        alpha[s] ~ normal(mu_alpha, sigma_alpha);
    
    sigma ~ cauchy(0, 10.0);
    
    for (n in 1:N)
        y ~ normal(alpha[shrna[n]], sigma);
}

generated quantities {
    vector[N] y_pred;
    for (n in 1:N)
        y_pred[n] = normal_rng(alpha[shrna[n]], sigma);
}