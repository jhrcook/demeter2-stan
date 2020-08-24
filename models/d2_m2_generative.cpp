data {
    int<lower=0> N;               // number of data points
    int<lower=1> S;               // number of shRNAs
    
    int<lower=1, upper=S> shrna[N];  // index shRNA
}

generated quantities {
    vector[S] alpha;
    vector[N] y_pred;
    
    real mu_alpha = normal_rng(0, 2.0);
    real<lower=0> sigma_alpha = abs(cauchy_rng(0, 10.0));
    
    real<lower=0> sigma = abs(cauchy_rng(0, 10.0));
    
    for (s in 1:S) {
        alpha[s] = normal_rng(mu_alpha, sigma_alpha);
    }
    
    for (n in 1:N)
        y_pred[n] = normal_rng(alpha[shrna[n]], sigma);
}
