data {
    int<lower=0> N;  // number of data points
    int<lower=1> I;  // number of shRNAs
    
    int<lower=1, upper=I> shrna[N];  // index for genes
    
    vector[N] y;  // LFC data
}

parameters {
    real mu_beta;
    real<lower=0> sigma_beta;
    vector[I] beta;
    
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    for (n in 1:N) {
        mu[n] = beta[shrna[n]];
    }
}

model {
    // Hyperpriors
    mu_beta ~ normal(0, 2.5);
    sigma_beta ~ exponential(1.0 / 0.01);
    
    // Priors
    sigma ~ exponential(1.0 / 0.01);
    
    // Population model
    beta ~ normal(mu_beta, sigma_beta);
    
    // Likelihood
    y ~ normal(mu, sigma);
}


generated quantities {
    vector[N] yhat;
    for (n in 1:N) {
        yhat[n] = normal_rng(mu[n], sigma);
    }
}