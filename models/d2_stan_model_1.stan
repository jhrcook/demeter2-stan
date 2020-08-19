data {
    int<lower=1> N;  // number of data points
    int<lower=1> I;  // number of shRNAs
    int<lower=1> J;  // number of cell lines
    int<lower=1> K;  // number of batches
    int<lower=1> L;  // number of genes
    
    int cell_line[N];  // index for cell line group
    int batch[N];      // index for batch
    
    vector[N] y;     // log-fold-change
}

parameters {
    real a_jk[J,K];
    real<lower=0> sigma_a;
    
    real theta_k[K];
    real<lower=0> sigma_theta;
    
    real<lower=0> sigma;
}

model {

    sigma_a ~ exponential(1.0 / 0.01);
    sigma_theta ~ exponential(1.0 / 0.01);
    
    for (j in 1:J) {
        for (k in 1:K) {
            a_jk[j,k] ~ normal(0, sigma_a);
        }
    }
    
    for (k in 1:K) {
        theta_k[k] ~ normal(0, sigma_theta);
    }
    
    sigma ~ cauchy(0, 2.5);
    
    vector[N] mu;
    for (n in 1:N) {
        mu[n] = a_jk[cell_line[n], batch[n]] + theta_k[batch[n]]
    }
    
    y ~ normal(mu, sigma);
}