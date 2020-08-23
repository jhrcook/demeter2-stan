data {
    int<lower=0> N;                  // number of data points
    int<lower=1> S;                  // number of shRNAs
    int<lower=1> L;                  // number of genes
    
    int<lower=1, upper=S> shrna[N];  // index shRNA
    int<lower=1, upper=L> gene[N];   // index gene
    
    vector[N] y;                     // LFC data
}

parameters {
    
    real mu_alpha;
    real<lower=0> sigma_alpha;
    real mu_g;
    real<lower=0> sigma_g;
    
    real alpha[S];               // varying intercept by shRNA
    real g[L];                   // varying intercept by gene
    real<lower=0> sigma;         // standard deviation
}

model {
    // Hyperpriors
    mu_alpha ~ normal(0, 2.0);
    sigma_alpha ~ cauchy(0, 10.0);
    mu_g ~ normal(0, 2.0);
    sigma_g ~ cauchy(0, 10.0);
    
    // Priors
    alpha ~ normal(mu_alpha, sigma_alpha);
    g ~ normal(mu_g, sigma_g);
    sigma ~ cauchy(0, 10.0);
    
    {
        vector[N] y_hat;
        for (n in 1:N)
            y_hat[n] = alpha[shrna[n]] + g[gene[n]];
        y ~ normal(y_hat, sigma);
    }
}

generated quantities {
    vector[N] y_pred;
    for (n in 1:N)
        y_pred[n] = normal_rng(alpha[shrna[n]] + g[gene[n]], sigma);
}
