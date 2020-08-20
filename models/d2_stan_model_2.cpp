data {
    int<lower=0> N;  // number of data points
    int<lower=1> I;  // number of shRNAs
    int<lower=1> J;  // number of cell lines
    int<lower=1> L;  // number of cell lines
    
    int<lower=1, upper=I> shrna[N];  // index for shRNAs
    int<lower=1, upper=J> cell_line[N];  // index for cell lines
    int<lower=1, upper=L> gene[N];  // index for genes
    
    vector[N] y;  // LFC data
}

parameters {
    vector[J] q;
    
    real mu_alpha;
    real<lower=0> sigma_alpha;
    vector[I] alpha;
    
    real<lower=0> sigma_gbar;
    vector[L] gbar;
    
    real<lower=0> sigma_g;
    matrix[L,J] g;
    
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    for (n in 1:N) {
        mu[n] = q[cell_line[n]] * alpha[shrna[n]] * (gbar[gene[n]] - g[gene[n], cell_line[n]]);
    }
}

model {
    // Hyperpriors
    mu_alpha ~ normal(0, 2.5);
    sigma_alpha ~ exponential(1.0 / 0.01);
    sigma_gbar ~ exponential(1.0);
    sigma_g ~ exponential(1.0);
    
    
    // Priors
    sigma ~ exponential(1.0 / 0.01);
    alpha ~ normal(mu_alpha, sigma_alpha);
    gbar ~ normal(0, sigma_gbar);
    for (l in 1:L) {
        g[l, ] ~ normal(0, sigma_g);
    }
    
    
    // Likelihood
    y ~ normal(mu, sigma);
}


generated quantities {
    vector[N] yhat;
    for (n in 1:N) {
        yhat[n] = normal_rng(mu[n], sigma);
    }
}