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
    
    real c[S];
    real gbar[L];
    real g[J,L];
    
    real<lower=0> mu_sigma;
    real<lower=0> sigma_sigma;
    real<lower=0> sigma[S];
}

model {
    // Hyperpriors
    sigma_c ~ normal(0, 3.0);
    mu_gbar ~ normal(0, 2.0);
    sigma_gbar ~ normal(0, 5.0);
    sigma_g ~ normal(0, 5.0);
    mu_sigma ~ normal(0, 2.0);
    sigma_sigma ~ normal(0, 1.0);
    
    // Priors
    c ~ normal(0, sigma_c);
    gbar ~ normal(mu_gbar, sigma_gbar);
    for (l in 1:L)
        g[,l] ~ normal(0, sigma_g);
    
    for (s in 1:S)
        sigma[s] ~ normal(mu_sigma, sigma_sigma);
    
    {
        vector[N] y_hat;
        for (n in 1:N) {
            y_hat[n] = c[shrna[n]] + gbar[gene[n]] + g[cell_line[n], gene[n]];
            y[n] ~ normal(y_hat[n], sigma[shrna[n]]);
        }
    }
        
}

generated quantities {
    vector[N] y_pred;
    
    // Posterior predictions
    for (n in 1:N)
        y_pred[n] = normal_rng(c[shrna[n]] + gbar[gene[n]] + g[cell_line[n], gene[n]], sigma[shrna[n]]);
}
