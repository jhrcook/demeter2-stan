data {
    int<lower=0> N;                     // number of data points
    int<lower=1> S;                     // number of shRNAs
    int<lower=1> L;                     // number of genes
    int<lower=1> J;                     // number of cell lines
    
    int<lower=1, upper=S> shrna[N];     // shRNA index
    int<lower=1, upper=L> gene[N];      // gene index
    int<lower=1, upper=J> cell_line[N]; // cell line index
    
    vector[N] y;                        // LFC data
}

parameters {
    real<lower=0> sigma_c;
    real mu_gbar;
    real<lower=0> sigma_gbar;
    real<lower=0> sigma_g;
    real<lower=0> sigma_a;
    
    real a[J];
    real gamma[J];
    real c[S];
    real<lower=0, upper=1> alpha[S];
    real gbar[L];
    real g[J,L];
    
    real<lower=0> mu_sigma;
    real<lower=0> sigma_sigma;
    real<lower=0> sigma[S];
}

model {
    // Hyperpriors
    sigma_c ~ normal(0, 2.0);
    mu_gbar ~ normal(0, 2.0);
    sigma_gbar ~ normal(0, 2.0);
    sigma_g ~ normal(0, 2.0);
    sigma_a ~ normal(0.0, 2.0);
    mu_sigma ~ normal(0, 2.0);
    sigma_sigma ~ normal(0, 1.0);
    
    // Priors
    a ~ normal(0, sigma_a);
    gamma ~ normal(1.0, 1.0);
    c ~ normal(0, sigma_c);
    alpha ~ uniform(0, 1);
    gbar ~ normal(mu_gbar, sigma_gbar);
    for (l in 1:L)
        g[,l] ~ normal(0, sigma_g);
    
    sigma ~ normal(mu_sigma, sigma_sigma);
    
    {
        vector[N] y_hat;
        for (n in 1:N) {
            y_hat[n] = a[cell_line[n]] + (gamma[cell_line[n]] * (c[shrna[n]] + alpha[shrna[n]] * (gbar[gene[n]] + g[cell_line[n], gene[n]])));
            y[n] ~ normal(y_hat[n], sigma[shrna[n]]);
        }
    }
        
}

generated quantities {
    vector[N] y_pred;
    
    // Posterior predictions
    for (n in 1:N)
        y_pred[n] = normal_rng(a[cell_line[n]] + (gamma[cell_line[n]] * (c[shrna[n]] + alpha[shrna[n]] * (gbar[gene[n]] + g[cell_line[n], gene[n]]))), sigma[shrna[n]]);
}
