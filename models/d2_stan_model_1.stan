data {
    int<lower=1> N;  // number of data points
    int<lower=1> I;  // number of shRNAs
    int<lower=1> J;  // number of cell lines
    int<lower=1> K;  // number of batches
    int<lower=1> L;  // number of genes
    
    int shrna[N];      // index for shRNA
    int cell_line[N];  // index for cell line group
    int batch[N];      // index for batch
    int gene[N];       // index for gene
    
    int<lower=0, upper=1> G[I,L];  // binary matrix mapping shRNA to genes
    
    vector[N] y;     // log-fold-change
}

parameters {
    real a[J,K];
    real<lower=0> sigma_a;
    
    real theta[I, K];
    real<lower=0> sigma_theta;
    
        
    real gamma[J,K];
    real<lower=0> sigma_gamma;
    
    real q[J];
    real<lower=0> sigma_q;
    
    vector[N] alpha;
    
    real gbar[L];
    real<lower=0> sigma_gbar;
    real g[L,J];
    real<lower=0> sigma_g;
    
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    vector[N] gene_level_mu;
    
    for (n in 1:N) {
        gene_level_mu[n] = q[cell_line[n]] * (gbar[gene[n]] + g[gene[n], cell_line[n]]);
        mu[n] = a[cell_line[n], batch[n]] + theta[shrna[n], batch[n]] + gamma[cell_line[n], batch[n]] * gene_level_mu[n];
    }
}

model {

    sigma_a ~ exponential(1.0 / 0.1);
    sigma_theta ~ exponential(1.0 / 0.1);
    sigma_gamma ~ exponential(1.0 / 0.1);
    sigma_q ~ exponential(1.0 / 0.1);
    sigma_gbar ~ exponential(1.0);
    sigma_g ~ exponential(1.0);
    
    for (k in 1:K) {
        for (i in 1:I) {
            theta[i, k] ~ normal(0, sigma_theta);
        }
        for (j in 1:J) {
            a[j,k] ~ normal(0, sigma_a);
            gamma[j,k] ~ normal(1, sigma_gamma);
        }
    }
    
    for (l in 1:L) {
        gbar[l] ~ normal(0, sigma_gbar);
        for (j in 1:J) {
            g[l,j] ~ normal(0, sigma_g);
        }
    }
    
    for (j in 1:J) {
        q[j] ~ normal(1, sigma_q);
    }
    
    // for (i in 1:I) {
    //     alpha[i] ~ beta(2, 2);
    // }
    
    sigma ~ cauchy(0, 2.5);
    y ~ normal(mu, sigma);
}