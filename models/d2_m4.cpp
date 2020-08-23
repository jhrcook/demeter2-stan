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
    
    real<lower=0> sigma;
}

model {
    // Hyperpriors
    sigma_c ~ cauchy(0, 3.0);
    mu_gbar ~ normal(0, 2.0);
    sigma_gbar ~ cauchy(0, 10.0);
    sigma_g ~ cauchy(0, 5.0);
    
    // Priors
    c ~ normal(0, sigma_c);
    gbar ~ normal(mu_gbar, sigma_gbar);
    for (l in 1:L)
        g[,l] ~ normal(0, sigma_g);
    sigma ~ cauchy(0, 10.0);
    
    {
        vector[N] y_hat;
        for (n in 1:N)
            y_hat[n] = c[shrna[n]] + gbar[gene[n]] - g[cell_line[n], gene[n]];
        y ~ normal(y_hat, sigma);
    }
}

generated quantities {
    vector[N] y_pred;
    for (n in 1:N)
        y_pred[n] = normal_rng(c[shrna[n]] + gbar[gene[n]] - g[cell_line[n], gene[n]], sigma);
}
