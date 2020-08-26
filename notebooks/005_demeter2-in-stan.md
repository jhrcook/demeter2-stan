# DEMETER2 in Stan


```python
import pystan
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import arviz as az
from pathlib import Path
import seaborn as sns
from timeit import default_timer as timer
import warnings
import re
from notebook_modules.pystan_helpers import StanModel_cache

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (10.0, 7.0)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 15

modeling_data_dir = Path('../modeling_data')

warnings.filterwarnings(action='ignore', 
                        message='Argument backend_kwargs has not effect in matplotlib.plot_distSupplied value won\'t be used')
```

## Data preparation


```python
modeling_data = pd.read_csv(modeling_data_dir / 'subset_modeling_data.csv')
modeling_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>barcode_sequence</th>
      <th>cell_line</th>
      <th>lfc</th>
      <th>batch</th>
      <th>gene_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>ln215_central_nervous_system</td>
      <td>1.966515</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>ln382_central_nervous_system</td>
      <td>1.289606</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>efo21_ovary</td>
      <td>0.625725</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>jhesoad1_oesophagus</td>
      <td>1.392272</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>km12_large_intestine</td>
      <td>0.820838</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory data analysis


```python
genes = set(modeling_data.gene_symbol.to_list())
fig, axes = plt.subplots(5, 3, figsize=(9, 9))
for ax, gene in zip(axes.flat, genes):
    lfc = modeling_data[modeling_data.gene_symbol == gene].lfc
    sns.distplot(lfc, kde=True, hist=False, rug=True, ax=ax, kde_kws={'shade': True}, color='b')
    
    y_data = ax.lines[0].get_ydata()
    ax.vlines(x=0, ymin=0, ymax=np.max(y_data) * 1.05, linestyles='dashed')
    
    ax.set_title(gene, fontsize=12)
    ax.set_xlabel(None)


axes[4, 2].axis('off')
axes[4, 1].axis('off')
fig.tight_layout(pad=1.0)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_5_0.png)



```python
cell_lines = set(modeling_data.cell_line.to_list())
for cell_line in cell_lines:
    lfc = modeling_data[modeling_data.cell_line == cell_line].lfc
    sns.distplot(lfc, kde=True, hist=False, label=None, kde_kws={'alpha': 0.2})

plt.title('LFC distributions')
plt.xlabel('LFC')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_6_0.png)



```python
sns.distplot(modeling_data.lfc)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_7_0.png)



```python
modeling_data[['barcode_sequence', 'gene_symbol']].drop_duplicates().groupby('gene_symbol').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>barcode_sequence</th>
    </tr>
    <tr>
      <th>gene_symbol</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BRAF</th>
      <td>8</td>
    </tr>
    <tr>
      <th>COG3</th>
      <td>5</td>
    </tr>
    <tr>
      <th>COL8A1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>EGFR</th>
      <td>19</td>
    </tr>
    <tr>
      <th>EIF6</th>
      <td>5</td>
    </tr>
    <tr>
      <th>ESPL1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>GRK5</th>
      <td>5</td>
    </tr>
    <tr>
      <th>KRAS</th>
      <td>11</td>
    </tr>
    <tr>
      <th>PTK2</th>
      <td>23</td>
    </tr>
    <tr>
      <th>RC3H2</th>
      <td>4</td>
    </tr>
    <tr>
      <th>RHBDL2</th>
      <td>5</td>
    </tr>
    <tr>
      <th>SDHB</th>
      <td>5</td>
    </tr>
    <tr>
      <th>TRIM39</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
lfc_corr = modeling_data \
    .pivot(index='cell_line', columns='barcode_sequence', values='lfc') \
    .corr()

mask = np.triu(np.ones_like(lfc_corr, dtype=np.bool), k=0)
f, ax = plt.subplots(figsize=(15, 13))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(lfc_corr, mask=mask, 
            cmap=cmap, center=0, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
plt.xlabel('barcode')
plt.ylabel('barcode')
plt.title('Correlation of LFC of barcodes')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_9_0.png)


## Modeling


```python
models_dir = Path('..', 'models')
```


```python
modeling_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>barcode_sequence</th>
      <th>cell_line</th>
      <th>lfc</th>
      <th>batch</th>
      <th>gene_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>ln215_central_nervous_system</td>
      <td>1.966515</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>ln382_central_nervous_system</td>
      <td>1.289606</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>efo21_ovary</td>
      <td>0.625725</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>jhesoad1_oesophagus</td>
      <td>1.392272</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>km12_large_intestine</td>
      <td>0.820838</td>
      <td>1</td>
      <td>EIF6</td>
    </tr>
  </tbody>
</table>
</div>



Select only a few cell lines while model building.


```python
len(np.unique(modeling_data.cell_line))
```




    501




```python
np.random.seed(123)
cell_lines = np.random.choice(np.unique(modeling_data.cell_line), 40)
modeling_data = modeling_data[modeling_data.cell_line.isin(cell_lines)]
modeling_data.shape
```




    (3334, 5)




```python
np.unique(modeling_data.gene_symbol)
```




    array(['BRAF', 'COG3', 'COL8A1', 'EGFR', 'EIF6', 'ESPL1', 'GRK5', 'KRAS',
           'PTK2', 'RC3H2', 'RHBDL2', 'SDHB', 'TRIM39'], dtype=object)




```python
# model_testing_genes = ['COG3', 'KRAS', 'COL8A1', 'EIF6']
# modeling_data = modeling_data[modeling_data.gene_symbol.isin(model_testing_genes)]
```


```python
genes = set(modeling_data.gene_symbol.to_list())
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax, gene in zip(axes.flat, genes):
    lfc = modeling_data[modeling_data.gene_symbol == gene].lfc
    sns.distplot(lfc, kde=True, hist=True, ax=ax, color='b')
    
    y_data = ax.lines[0].get_ydata()
    ax.vlines(x=0, ymin=0, ymax=np.max(y_data) * 1.05, linestyles='dashed')
    
    ax.set_title(gene, fontsize=12)
    ax.set_xlabel(None)

fig.tight_layout(pad=1.0)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_18_0.png)



```python
def add_categorical_idx(df, col):
    df[f'{col}_idx'] = df[col].astype('category').cat.codes + 1
    return df

for col in ['barcode_sequence', 'cell_line', 'gene_symbol']:
    modeling_data = add_categorical_idx(modeling_data, col)

modeling_data = modeling_data.reset_index(drop=True)
modeling_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>barcode_sequence</th>
      <th>cell_line</th>
      <th>lfc</th>
      <th>batch</th>
      <th>gene_symbol</th>
      <th>barcode_sequence_idx</th>
      <th>cell_line_idx</th>
      <th>gene_symbol_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>efo21_ovary</td>
      <td>0.625725</td>
      <td>1</td>
      <td>EIF6</td>
      <td>1</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>dbtrg05mg_central_nervous_system</td>
      <td>2.145082</td>
      <td>2</td>
      <td>EIF6</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>bt20_breast</td>
      <td>0.932751</td>
      <td>2</td>
      <td>EIF6</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>sw1783_central_nervous_system</td>
      <td>1.372030</td>
      <td>2</td>
      <td>EIF6</td>
      <td>1</td>
      <td>36</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>kns60_central_nervous_system</td>
      <td>0.803835</td>
      <td>2</td>
      <td>EIF6</td>
      <td>1</td>
      <td>18</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Binary matrix of $[shRNA \times gene]$.


```python
shrna_gene_matrix = modeling_data[['barcode_sequence_idx', 'gene_symbol_idx']] \
    .drop_duplicates() \
    .reset_index(drop=True) \
    .assign(value = lambda df: np.ones(df.shape[0], dtype=int)) \
    .pivot(index='barcode_sequence_idx', columns='gene_symbol_idx', values='value') \
    .fillna(0) \
    .to_numpy() \
    .astype(int)

shrna_gene_matrix
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 1]])




```python
shrna_gene_matrix.shape
```




    (109, 13)



## Model 1. Just an intercept

$$
D \sim N(\mu, \sigma) \\
\mu = \alpha \\
\alpha \sim N(0, 5) \\
\sigma \sim \text{HalfCauchy}(0, 5)
$$

**Model data.**


```python
d2_m1_data = {
    'N': int(modeling_data.shape[0]),
    'y': modeling_data.lfc
}
```

**Compile model.**


```python
d2_m1_file = models_dir / 'd2_m1.cpp'
d2_m1 = StanModel_cache(file=d2_m1_file.as_posix())
```

    Using cached StanModel.



```python
d2_m1_fit = d2_m1.sampling(data=d2_m1_data, iter=2000, chains=2)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)



```python
pystan.check_hmc_diagnostics(d2_m1_fit)
```




    {'n_eff': True,
     'Rhat': True,
     'divergence': True,
     'treedepth': True,
     'energy': True}




```python
az_d2_m1 = az.from_pystan(posterior=d2_m1_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m1)
az.summary(az_d2_m1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha</th>
      <td>-1.290</td>
      <td>0.032</td>
      <td>-1.347</td>
      <td>-1.231</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>2134.0</td>
      <td>2126.0</td>
      <td>2150.0</td>
      <td>1230.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.737</td>
      <td>0.021</td>
      <td>1.695</td>
      <td>1.774</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2039.0</td>
      <td>2035.0</td>
      <td>2044.0</td>
      <td>1304.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m1)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_31_0.png)



```python
az.plot_forest(az_d2_m1, combined=True)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_32_0.png)



```python
az.plot_ppc(az_d2_m1, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_33_0.png)


## Model 2. Varying intercept by shRNA

$$
D_{i|s} \sim N(\mu_{i|s}, \sigma) \\
\mu = \alpha_{i|s} \\
\alpha \sim N(\mu_{\alpha}, \sigma_{\alpha}) \\
\mu_{\alpha} \sim N(0, 2) \\
\sigma_{\alpha} \sim \text{HalfCauchy}(0, 2) \\
\sigma \sim \text{HalfCauchy}(0, 5)
$$

### Generative model for a prior predictive check


```python
N = 1000
S = 100
shrna_barcodes = list(range(1, S+1))
shrna_barcodes_idx = np.repeat(shrna_barcodes, N/S)
```


```python
d2_m2_gen_data = {
    'N': N,
    'S': S,
    'shrna': shrna_barcodes_idx
}
```


```python
d2_m2_gen_file = models_dir / 'd2_m2_generative.cpp'
d2_m2_gen = StanModel_cache(file=d2_m2_gen_file.as_posix())
```

    Using cached StanModel.



```python
d2_m2_gen_fit = d2_m2_gen.sampling(data=d2_m2_gen_data, 
                                   iter=10, warmup=0, chains=1, 
                                   algorithm='Fixed_param')
```


```python
az_d2_m2_gen = az.from_pystan(d2_m2_gen_fit)
```


```python
df = d2_m2_gen_fit.to_dataframe() \
    .drop(['chain', 'draw', 'warmup'], axis=1) \
    .melt(var_name='parameter', value_name='value')
df = df[df.parameter.str.contains('alpha\[')]
sns.distplot(df.value)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_41_0.png)



```python
df = d2_m2_gen_fit.to_dataframe() \
    .drop(['chain', 'draw', 'warmup'], axis=1) \
    .melt(var_name='parameter', value_name='value')
df = df[df.parameter.str.contains('y_pred')]
sns.distplot(df.value)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_42_0.png)



```python
sns.distplot(modeling_data.lfc)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_43_0.png)


**Model data**


```python
d2_m2_data = {
    'N': int(modeling_data.shape[0]),
    'S': np.max(modeling_data.barcode_sequence_idx),
    
    'shrna': modeling_data.barcode_sequence_idx,
    
    'y': modeling_data.lfc,
}
```


```python
d2_m2_data['S']
```




    109



**Compile model.**


```python
d2_m2_file = models_dir / 'd2_m2.cpp'
d2_m2 = StanModel_cache(file=d2_m2_file.as_posix())
```

    Using cached StanModel.



```python
d2_m2_fit = d2_m2.sampling(data=d2_m2_data, iter=1000, chains=2)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)



```python
pystan.check_hmc_diagnostics(d2_m2_fit)
```




    {'n_eff': True,
     'Rhat': True,
     'divergence': True,
     'treedepth': True,
     'energy': True}




```python
az_d2_m2 = az.from_pystan(posterior=d2_m2_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m2)
az.summary(az_d2_m2).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_alpha</th>
      <td>-1.215</td>
      <td>0.125</td>
      <td>-1.459</td>
      <td>-1.001</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1018.0</td>
      <td>993.0</td>
      <td>1012.0</td>
      <td>678.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.276</td>
      <td>0.087</td>
      <td>1.111</td>
      <td>1.430</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>950.0</td>
      <td>939.0</td>
      <td>934.0</td>
      <td>705.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>0.730</td>
      <td>0.202</td>
      <td>0.340</td>
      <td>1.120</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>905.0</td>
      <td>824.0</td>
      <td>907.0</td>
      <td>499.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha[1]</th>
      <td>-0.261</td>
      <td>0.204</td>
      <td>-0.614</td>
      <td>0.142</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>1273.0</td>
      <td>787.0</td>
      <td>1278.0</td>
      <td>595.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>alpha[2]</th>
      <td>-0.779</td>
      <td>0.189</td>
      <td>-1.169</td>
      <td>-0.467</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1260.0</td>
      <td>1213.0</td>
      <td>1254.0</td>
      <td>774.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m2, var_names=['mu_alpha', 'sigma_alpha', 'sigma'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_52_0.png)



```python
az.plot_forest(az_d2_m2, kind='ridgeplot', combined=True, 
               var_names=['mu_alpha', 'sigma_alpha', 'sigma'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_53_0.png)



```python
az.plot_ppc(az_d2_m2, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_54_0.png)



```python
d2_m2_fit.to_dataframe().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chain</th>
      <th>draw</th>
      <th>warmup</th>
      <th>mu_alpha</th>
      <th>sigma_alpha</th>
      <th>alpha[1]</th>
      <th>alpha[2]</th>
      <th>alpha[3]</th>
      <th>alpha[4]</th>
      <th>alpha[5]</th>
      <th>...</th>
      <th>y_pred[3332]</th>
      <th>y_pred[3333]</th>
      <th>y_pred[3334]</th>
      <th>lp__</th>
      <th>accept_stat__</th>
      <th>stepsize__</th>
      <th>treedepth__</th>
      <th>n_leapfrog__</th>
      <th>divergent__</th>
      <th>energy__</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.455545</td>
      <td>1.342933</td>
      <td>0.725803</td>
      <td>-0.069529</td>
      <td>-1.002128</td>
      <td>0.434654</td>
      <td>-2.116150</td>
      <td>...</td>
      <td>-2.176713</td>
      <td>0.699725</td>
      <td>-1.595175</td>
      <td>-2389.397468</td>
      <td>0.723386</td>
      <td>0.490956</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>2447.588094</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1.075712</td>
      <td>1.218258</td>
      <td>1.063995</td>
      <td>-0.213949</td>
      <td>-0.689021</td>
      <td>0.077516</td>
      <td>-1.980579</td>
      <td>...</td>
      <td>-0.258277</td>
      <td>0.272606</td>
      <td>-2.511907</td>
      <td>-2385.244513</td>
      <td>1.000000</td>
      <td>0.490956</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>2433.071477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>-1.373017</td>
      <td>1.251750</td>
      <td>0.071485</td>
      <td>-0.108056</td>
      <td>-1.014538</td>
      <td>0.285502</td>
      <td>-2.715460</td>
      <td>...</td>
      <td>-0.047007</td>
      <td>-0.762857</td>
      <td>0.081047</td>
      <td>-2390.689462</td>
      <td>0.905808</td>
      <td>0.490956</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>2434.283172</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>-1.052398</td>
      <td>1.260361</td>
      <td>0.854451</td>
      <td>-0.470255</td>
      <td>-1.078336</td>
      <td>0.103284</td>
      <td>-2.099655</td>
      <td>...</td>
      <td>-0.927995</td>
      <td>-1.821948</td>
      <td>-0.333319</td>
      <td>-2390.336283</td>
      <td>0.994167</td>
      <td>0.490956</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>2441.297353</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>-1.205699</td>
      <td>1.196109</td>
      <td>0.666829</td>
      <td>0.126224</td>
      <td>-0.626311</td>
      <td>0.434879</td>
      <td>-2.054807</td>
      <td>...</td>
      <td>-1.568530</td>
      <td>-0.767901</td>
      <td>-3.963464</td>
      <td>-2400.591625</td>
      <td>0.617053</td>
      <td>0.490956</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>2453.323424</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3456 columns</p>
</div>



## Model 3. Another varying intercept for target gene

$$
D_{i|s} \sim N(\mu_{i|s}, \sigma) \\
\mu = \alpha_{i|s} + g_{i|l}\\
\alpha_s \sim N(\mu_{\alpha}, \sigma_{\alpha}) \\
g_l \sim N(\mu_g, \sigma_g) \\
\mu_{\alpha} \sim N(0, 2) \quad \sigma_{\alpha} \sim \text{HalfCauchy}(0, 10) \\
\mu_{g} \sim N(0, 2) \quad \sigma_{g} \sim \text{HalfCauchy}(0, 10) \\
\sigma \sim \text{HalfCauchy}(0, 10)
$$


```python
d2_m3_data = {
    'N': int(modeling_data.shape[0]),
    'S': np.max(modeling_data.barcode_sequence_idx),
    'L': np.max(modeling_data.gene_symbol_idx),
    
    'shrna': modeling_data.barcode_sequence_idx,
    'gene': modeling_data.gene_symbol_idx,
    
    'y': modeling_data.lfc,
}
```

**Compile model.**


```python
d2_m3_file = models_dir / 'd2_m3.cpp'
d2_m3 = StanModel_cache(file=d2_m3_file.as_posix())
```

    Using cached StanModel.



```python
d2_m3_control = {'adapt_delta': 0.99, 
                 'max_treedepth': 10}
d2_m3_fit = d2_m3.sampling(data=d2_m3_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m3_control)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)
    WARNING:pystan:7 of 8000 iterations ended with a divergence (0.0875 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:13 of 8000 iterations saturated the maximum tree depth of 10 (0.163 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation



```python
pystan.check_hmc_diagnostics(d2_m3_fit)
```

    WARNING:pystan:Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed
    WARNING:pystan:7 of 8000 iterations ended with a divergence (0.0875 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:13 of 8000 iterations saturated the maximum tree depth of 10 (0.163 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation





    {'n_eff': True,
     'Rhat': False,
     'divergence': False,
     'treedepth': False,
     'energy': True}




```python
az_d2_m3 = az.from_pystan(posterior=d2_m3_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m3)
az.summary(az_d2_m3).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_alpha</th>
      <td>-0.504</td>
      <td>0.990</td>
      <td>-2.236</td>
      <td>1.296</td>
      <td>0.200</td>
      <td>0.143</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>50.0</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.233</td>
      <td>0.089</td>
      <td>1.075</td>
      <td>1.408</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1126.0</td>
      <td>1126.0</td>
      <td>1116.0</td>
      <td>2271.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_g</th>
      <td>-0.788</td>
      <td>0.989</td>
      <td>-2.590</td>
      <td>0.961</td>
      <td>0.201</td>
      <td>0.144</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>56.0</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.413</td>
      <td>0.213</td>
      <td>0.019</td>
      <td>0.764</td>
      <td>0.013</td>
      <td>0.009</td>
      <td>281.0</td>
      <td>281.0</td>
      <td>235.0</td>
      <td>266.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>1.734</td>
      <td>1.074</td>
      <td>-0.294</td>
      <td>3.572</td>
      <td>0.209</td>
      <td>0.150</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>25.0</td>
      <td>70.0</td>
      <td>1.12</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m3, var_names=['g'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_63_0.png)



```python
az.plot_ppc(az_d2_m3, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_64_0.png)



```python
fit3_summary = az.summary(az_d2_m3)
```


```python
fit3_alpha_summary = fit3_summary[fit3_summary.index.str.contains('alpha\[')]
shrna_idx = [re.search(r"\[([A-Za-z0-9_]+)\]", a).group(1) for a in fit3_alpha_summary.index]
shrna_idx = [int(a) + 1 for a in shrna_idx]
fit3_alpha_summary = fit3_alpha_summary \
    .assign(barcode_sequence_idx = shrna_idx) \
    .set_index('barcode_sequence_idx') \
    .join(modeling_data[['barcode_sequence_idx', 'gene_symbol']] \
          .drop_duplicates() \
          .set_index('barcode_sequence_idx'))
fit3_alpha_summary.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>gene_symbol</th>
    </tr>
    <tr>
      <th>barcode_sequence_idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.734</td>
      <td>1.074</td>
      <td>-0.294</td>
      <td>3.572</td>
      <td>0.209</td>
      <td>0.150</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>25.0</td>
      <td>70.0</td>
      <td>1.12</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.355</td>
      <td>1.069</td>
      <td>-1.542</td>
      <td>2.368</td>
      <td>0.201</td>
      <td>0.144</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>76.0</td>
      <td>1.11</td>
      <td>GRK5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.232</td>
      <td>1.076</td>
      <td>-1.697</td>
      <td>2.199</td>
      <td>0.210</td>
      <td>0.150</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>25.0</td>
      <td>69.0</td>
      <td>1.12</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.769</td>
      <td>1.047</td>
      <td>-1.182</td>
      <td>2.648</td>
      <td>0.197</td>
      <td>0.141</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>29.0</td>
      <td>64.0</td>
      <td>1.10</td>
      <td>EGFR</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.070</td>
      <td>1.055</td>
      <td>-3.010</td>
      <td>0.798</td>
      <td>0.207</td>
      <td>0.152</td>
      <td>26.0</td>
      <td>25.0</td>
      <td>26.0</td>
      <td>69.0</td>
      <td>1.11</td>
      <td>COL8A1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.424</td>
      <td>1.032</td>
      <td>-2.309</td>
      <td>1.438</td>
      <td>0.207</td>
      <td>0.148</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>24.0</td>
      <td>66.0</td>
      <td>1.12</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.200</td>
      <td>1.036</td>
      <td>-2.087</td>
      <td>1.686</td>
      <td>0.203</td>
      <td>0.145</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>61.0</td>
      <td>1.11</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.516</td>
      <td>1.036</td>
      <td>-0.311</td>
      <td>3.449</td>
      <td>0.200</td>
      <td>0.143</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>1.11</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.696</td>
      <td>1.037</td>
      <td>-1.187</td>
      <td>2.582</td>
      <td>0.202</td>
      <td>0.145</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>60.0</td>
      <td>1.11</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.456</td>
      <td>1.045</td>
      <td>-2.403</td>
      <td>1.397</td>
      <td>0.202</td>
      <td>0.145</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>56.0</td>
      <td>1.11</td>
      <td>BRAF</td>
    </tr>
  </tbody>
</table>
</div>




```python
fit3_gene_summary = fit3_summary[fit3_summary.index.str.contains('g\[')]
gene_idx = [re.search(r"\[([A-Za-z0-9_]+)\]", a).group(1) for a in fit3_gene_summary.index]
gene_idx = [int(a) + 1 for a in gene_idx]
fit3_gene_summary = fit3_gene_summary \
    .assign(gene_symbol_idx = gene_idx) \
    .set_index('gene_symbol_idx') \
    .join(modeling_data[['gene_symbol_idx', 'gene_symbol']] \
          .drop_duplicates() \
          .set_index('gene_symbol_idx')) \
    .reset_index(drop=False)
fit3_gene_summary.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gene_symbol_idx</th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>gene_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.650</td>
      <td>1.030</td>
      <td>-2.568</td>
      <td>1.168</td>
      <td>0.203</td>
      <td>0.145</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>54.0</td>
      <td>1.11</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.659</td>
      <td>1.034</td>
      <td>-2.633</td>
      <td>1.149</td>
      <td>0.196</td>
      <td>0.140</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>84.0</td>
      <td>1.11</td>
      <td>COG3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-1.189</td>
      <td>1.044</td>
      <td>-3.122</td>
      <td>0.638</td>
      <td>0.207</td>
      <td>0.148</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>25.0</td>
      <td>69.0</td>
      <td>1.12</td>
      <td>COL8A1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.509</td>
      <td>1.018</td>
      <td>-2.390</td>
      <td>1.272</td>
      <td>0.198</td>
      <td>0.141</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>56.0</td>
      <td>1.11</td>
      <td>EGFR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-1.023</td>
      <td>1.057</td>
      <td>-2.932</td>
      <td>0.817</td>
      <td>0.210</td>
      <td>0.150</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>66.0</td>
      <td>1.12</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>-1.072</td>
      <td>1.052</td>
      <td>-2.929</td>
      <td>0.855</td>
      <td>0.210</td>
      <td>0.150</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>71.0</td>
      <td>1.12</td>
      <td>ESPL1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>-0.610</td>
      <td>1.053</td>
      <td>-2.527</td>
      <td>1.289</td>
      <td>0.201</td>
      <td>0.144</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>74.0</td>
      <td>1.11</td>
      <td>GRK5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>-1.084</td>
      <td>1.016</td>
      <td>-2.948</td>
      <td>0.731</td>
      <td>0.206</td>
      <td>0.148</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>64.0</td>
      <td>1.12</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>-0.479</td>
      <td>1.003</td>
      <td>-2.312</td>
      <td>1.270</td>
      <td>0.201</td>
      <td>0.144</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>55.0</td>
      <td>1.12</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>-0.843</td>
      <td>1.042</td>
      <td>-2.673</td>
      <td>1.042</td>
      <td>0.206</td>
      <td>0.147</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>77.0</td>
      <td>1.11</td>
      <td>RC3H2</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in range(fit3_gene_summary.shape[0]):
    plt.plot(np.repeat(fit3_gene_summary.loc[i, 'gene_symbol'], 2), 
             [fit3_gene_summary.loc[i, 'hdi_3%'], fit3_gene_summary.loc[i, 'hdi_97%']],
             color='red', alpha=0.5)


plt.scatter(fit3_gene_summary['gene_symbol'], 
            fit3_gene_summary['mean'],
            s=100, c='r', label='gene')
plt.scatter(fit3_alpha_summary['gene_symbol'], 
            fit3_alpha_summary['mean'], 
            alpha=0.3, s=75, c='b', label='shRNA')

plt.title('shRNA and gene mean values')
plt.xlabel('target gene')
plt.ylabel('estimated effect on LFC')
plt.legend()
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_68_0.png)


## Model 4. Parameters for difference between average gene effect and cell line-specific effect

Note that the varying intercept for shRNA has been renamed from $\alpha$ to $c$.
$\bar g_{l}$ is the average effect of knocking-down gene $l$ while $g_{jl}$ is the cell line $j$-specific effect of knocking-down $l$.

There is now also varying population level standard deviations for each shRNA $\sigma_s$.
This seems to help reduce divergences during sampling by modeling differences in standard deviation for each shRNA.

$$
D_{i|s} \sim N(\mu_{i|s}, \sigma_s) \\
\mu = c_{i|s} + \bar g_{i|l} + g_{i|jl} \\
c_s \sim N(0, \sigma_c) \\
\bar g_l \sim N(\mu_{\bar g}, \sigma_{\bar g}) \\
g_{jl} \sim N(0, \sigma_g) \\
\sigma_c \sim \text{HalfNormal}(0, 3) \\
\mu_{\bar g} \sim N(0, 2) \quad \sigma_{\bar g} \sim \text{HalfNormal}(0, 10) \\
\sigma_g \sim \text{HalfNormal}(0, 5) \\
\sigma_s \sim \text{HalfNormal}(\mu_\sigma, \sigma_\sigma) \\
\mu_\sigma \sim \text{HalfNormal}(0, 2) \quad \sigma_\sigma \sim \text{HalfNormal}(0, 1) \\
$$


```python
d2_m4_data = {
    'N': int(modeling_data.shape[0]),
    'S': np.max(modeling_data.barcode_sequence_idx),
    'L': np.max(modeling_data.gene_symbol_idx),
    'J': np.max(modeling_data.cell_line_idx),
    
    'shrna': modeling_data.barcode_sequence_idx,
    'gene': modeling_data.gene_symbol_idx,
    'cell_line': modeling_data.cell_line_idx,
    
    'y': modeling_data.lfc,
}
```

**Compile model.**


```python
d2_m4_file = models_dir / 'd2_m4.cpp'
d2_m4 = StanModel_cache(file=d2_m4_file.as_posix())
```

    Using cached StanModel.



```python
d2_m4_control = {'adapt_delta': 0.99, 
                 'max_treedepth': 10}
d2_m4_fit = d2_m4.sampling(data=d2_m4_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m4_control)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)
    WARNING:pystan:1 of 8000 iterations ended with a divergence (0.0125 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.



```python
pystan.check_hmc_diagnostics(d2_m4_fit)
```

    WARNING:pystan:1 of 8000 iterations ended with a divergence (0.0125 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.





    {'n_eff': True,
     'Rhat': True,
     'divergence': False,
     'treedepth': True,
     'energy': True}




```python
az_d2_m4 = az.from_pystan(posterior=d2_m4_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m4)
az.summary(az_d2_m4).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sigma_c</th>
      <td>1.236</td>
      <td>0.091</td>
      <td>1.065</td>
      <td>1.405</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5265.0</td>
      <td>5184.0</td>
      <td>5404.0</td>
      <td>5908.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_gbar</th>
      <td>-1.280</td>
      <td>0.191</td>
      <td>-1.641</td>
      <td>-0.934</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1390.0</td>
      <td>1390.0</td>
      <td>1363.0</td>
      <td>2883.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_gbar</th>
      <td>0.422</td>
      <td>0.220</td>
      <td>0.029</td>
      <td>0.803</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>531.0</td>
      <td>531.0</td>
      <td>451.0</td>
      <td>491.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.460</td>
      <td>0.032</td>
      <td>0.402</td>
      <td>0.522</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1826.0</td>
      <td>1826.0</td>
      <td>1825.0</td>
      <td>3296.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>c[0]</th>
      <td>2.257</td>
      <td>0.404</td>
      <td>1.530</td>
      <td>3.038</td>
      <td>0.011</td>
      <td>0.008</td>
      <td>1444.0</td>
      <td>1416.0</td>
      <td>1479.0</td>
      <td>2593.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m4, var_names=['gbar'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_76_0.png)



```python
az.summary(az_d2_m4).tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sigma[104]</th>
      <td>1.042</td>
      <td>0.140</td>
      <td>0.781</td>
      <td>1.306</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>12795.0</td>
      <td>12486.0</td>
      <td>12671.0</td>
      <td>6024.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[105]</th>
      <td>0.982</td>
      <td>0.115</td>
      <td>0.778</td>
      <td>1.204</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>11776.0</td>
      <td>11560.0</td>
      <td>11754.0</td>
      <td>6733.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[106]</th>
      <td>1.208</td>
      <td>0.128</td>
      <td>0.965</td>
      <td>1.448</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>14544.0</td>
      <td>13909.0</td>
      <td>14655.0</td>
      <td>5675.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[107]</th>
      <td>1.053</td>
      <td>0.136</td>
      <td>0.810</td>
      <td>1.315</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>14095.0</td>
      <td>13702.0</td>
      <td>13986.0</td>
      <td>6088.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[108]</th>
      <td>1.083</td>
      <td>0.135</td>
      <td>0.834</td>
      <td>1.340</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>13541.0</td>
      <td>13228.0</td>
      <td>13683.0</td>
      <td>5643.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_forest(az_d2_m4, var_names=['sigma'], combined=True)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_78_0.png)



```python
az.plot_ppc(az_d2_m4, data_pairs={'y':'y_pred'}, num_pp_samples=100)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_79_0.png)



```python
g_jl_post = d2_m4_fit.to_dataframe() \
    .melt(id_vars=['chain', 'draw', 'warmup']) \
    .pipe(lambda d: d[d.variable.str.contains('g\[')])
```


```python
g_jl_post.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chain</th>
      <th>draw</th>
      <th>warmup</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1008000</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.320545</td>
    </tr>
    <tr>
      <th>1008001</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>-0.487421</td>
    </tr>
    <tr>
      <th>1008002</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.703895</td>
    </tr>
    <tr>
      <th>1008003</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>-0.022608</td>
    </tr>
    <tr>
      <th>1008004</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.215098</td>
    </tr>
  </tbody>
</table>
</div>




```python
def extract_g_index(s, i=0, outside_text='g[]'):
    s_mod = [a.strip(outside_text) for a in s]
    s_mod = [a.split(',') for a in s_mod]
    s_mod = [a[i].strip() for a in s_mod]
    return s_mod
```


```python
g_jl_post = g_jl_post \
    .assign(cell_line_idx=lambda x: extract_g_index(x.variable.to_list(), i=0),
            gene_symbol_idx=lambda x: extract_g_index(x.variable.to_list(), i=1)) \
    .astype({'cell_line_idx': int, 'gene_symbol_idx': int}) \
    .groupby(['cell_line_idx', 'gene_symbol_idx']) \
    .mean() \
    .reset_index() \
    .astype({'cell_line_idx': int, 'gene_symbol_idx': int}) \
    .set_index('cell_line_idx') \
    .join(modeling_data[['cell_line', 'cell_line_idx']] \
              .drop_duplicates() \
              .astype({'cell_line_idx': int}) \
              .set_index('cell_line_idx'),
          how='left') \
    .reset_index() \
    .set_index('gene_symbol_idx') \
    .join(modeling_data[['gene_symbol', 'gene_symbol_idx']] \
              .drop_duplicates() \
              .astype({'gene_symbol_idx': int}) \
              .set_index('gene_symbol_idx'),
          how='left') \
    .reset_index() \
    .assign(gene_symbol=lambda x: [f'{a} ({b})' for a,b in zip(x.gene_symbol, x.gene_symbol_idx)],
            cell_line=lambda x: [f'{a} ({b})' for a,b in zip(x.cell_line, x.cell_line_idx)]) \
    .drop(['gene_symbol_idx', 'cell_line_idx'], axis=1) \
    .pivot(index='gene_symbol', columns='cell_line', values='value')
```


```python
from scipy.cluster import hierarchy
from scipy.spatial import distance
```


```python
# Color bar for tissue of origin of cell lines.
cell_line_origin = g_jl_post.columns.to_list()
cell_line_origin = [a.split(' ')[0] for a in cell_line_origin]
cell_line_origin = [a.split('_')[1:] for a in cell_line_origin]
cell_line_origin = [' '.join(a) for a in cell_line_origin]

cell_line_pal = sns.husl_palette(len(np.unique(cell_line_origin)), s=.90)
cell_line_lut = dict(zip(np.unique(cell_line_origin), cell_line_pal))

cell_line_colors = pd.Series(cell_line_origin, index=g_jl_post.columns).map(cell_line_lut)

np.random.seed(123)
row_linkage = hierarchy.linkage(distance.pdist(g_jl_post), method='average')

p = sns.clustermap(g_jl_post, center=0, cmap="YlGnBu", linewidths=0.5, 
                   figsize=(12, 10),
                   cbar_kws={'label': 'mean coeff.'},
                   cbar_pos=[0.06, 0.15, 0.02, 0.2],
                   row_linkage=row_linkage,
                   col_colors=cell_line_colors)
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_85_0.png)



```python
d2_m4_post = d2_m4_fit.to_dataframe()
```


```python
genes_post = d2_m4_post.loc[:, d2_m4_post.columns.str.contains('gbar\[')]

genes = list(np.unique(modeling_data.gene_symbol))
genes.sort()
genes_post.columns = genes

for col in genes_post.columns.to_list():
    sns.distplot(genes_post[[col]], hist=False, label=col, kde_kws={'shade': False, 'alpha': 0.8})

plt.legend()
plt.xlabel('coefficient value')
plt.ylabel('density')
plt.title('Distribution of coefficients for average gene effect')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_87_0.png)



```python
cell_lines_post = d2_m4_post.loc[:, d2_m4_post.columns.str.contains('c\[')]

for col in cell_lines_post.columns.to_list():
    sns.distplot(cell_lines_post[[col]], hist=False, kde_kws={'shade': False, 'alpha': 0.5})

plt.xlabel('coefficient value')
plt.ylabel('density')
plt.title('Distribution of coefficients for average shRNA effect')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_88_0.png)


## Model 5. Multiplicative scaling factor of each shRNA

In the original DEMETER2 paper, the model includes a multiplicative scaling factor for each shRNA, constrained between 0 and 1, along with the addative varying effect for each shRNA.
The factor is multiplied against the gene effect.

In this model, I experiment with this multiplicative factor $\alpha_s$.

**Still need to adjust the model below.**

$$
D_{i|s} \sim N(\mu_{i|s}, \sigma_s) \\
\mu = c_{i|s} + \alpha_s (\bar g_{i|l} + g_{i|jl}) \\
c_s \sim N(0, \sigma_c) \\
\alpha_s \sim \text{Uniform}(0, 1) \\
\bar g_l \sim N(\mu_{\bar g}, \sigma_{\bar g}) \\
g_{jl} \sim N(0, \sigma_g) \\
\sigma_c \sim \text{HalfNormal}(0, 3) \\
\mu_{\bar g} \sim N(0, 2) \quad \sigma_{\bar g} \sim \text{HalfNormal}(0, 10) \\
\sigma_g \sim \text{HalfNormal}(0, 5) \\
\sigma_s \sim \text{HalfNormal}(\mu_\sigma, \sigma_\sigma) \\
\mu_\sigma \sim \text{HalfNormal}(0, 2) \quad \sigma_\sigma \sim \text{HalfNormal}(0, 1) \\
$$


```python
d2_m5_data = {
    'N': int(modeling_data.shape[0]),
    'S': np.max(modeling_data.barcode_sequence_idx),
    'L': np.max(modeling_data.gene_symbol_idx),
    'J': np.max(modeling_data.cell_line_idx),
    
    'shrna': modeling_data.barcode_sequence_idx,
    'gene': modeling_data.gene_symbol_idx,
    'cell_line': modeling_data.cell_line_idx,
    
    'y': modeling_data.lfc,
}
```

**Compile model.**


```python
d2_m5_file = models_dir / 'd2_m5.cpp'
d2_m5 = StanModel_cache(file=d2_m5_file.as_posix())
```

    Using cached StanModel.



```python
d2_m5_control = {'adapt_delta': 0.99, 
                 'max_treedepth': 10}
d2_m5_fit = d2_m5.sampling(data=d2_m5_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m5_control)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)
    WARNING:pystan:10 of 8000 iterations ended with a divergence (0.125 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:3 of 8000 iterations saturated the maximum tree depth of 10 (0.0375 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation



```python
pystan.check_hmc_diagnostics(d2_m5_fit)
```

    WARNING:pystan:10 of 8000 iterations ended with a divergence (0.125 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:3 of 8000 iterations saturated the maximum tree depth of 10 (0.0375 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation





    {'n_eff': True,
     'Rhat': True,
     'divergence': False,
     'treedepth': False,
     'energy': True}




```python
d2_m5_post = d2_m5_fit.to_dataframe()
```


```python
df = d2_m5_post.loc[:, d2_m5_post.columns.str.contains('alpha\[')]
for col in df.columns:
    sns.distplot(df[[col]], hist=False, kde_kws={'alpha': 0.5})
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_96_0.png)



```python
genes_post = d2_m5_post.loc[:, d2_m5_post.columns.str.contains('gbar\[')]

genes = list(np.unique(modeling_data.gene_symbol))
genes.sort()
genes_post.columns = genes

for col in genes_post.columns.to_list():
    sns.distplot(genes_post[[col]], hist=False, label=col, kde_kws={'shade': False, 'alpha': 0.8})

plt.legend()
plt.xlabel('coefficient value')
plt.ylabel('density')
plt.title('Distribution of coefficients for average gene effect')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_97_0.png)



```python
az_d2_m5 = az.from_pystan(posterior=d2_m5_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m5)
az.plot_ppc(az_d2_m5, data_pairs={'y':'y_pred'}, num_pp_samples=100)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_98_0.png)



```python
g_jl_post = d2_m5_fit.to_dataframe() \
    .melt(id_vars=['chain', 'draw', 'warmup']) \
    .pipe(lambda d: d[d.variable.str.contains('g\[')])
```


```python
g_jl_post = g_jl_post \
    .assign(cell_line_idx=lambda x: extract_g_index(x.variable.to_list(), i=0),
            gene_symbol_idx=lambda x: extract_g_index(x.variable.to_list(), i=1)) \
    .astype({'cell_line_idx': int, 'gene_symbol_idx': int}) \
    .groupby(['cell_line_idx', 'gene_symbol_idx']) \
    .mean() \
    .reset_index() \
    .astype({'cell_line_idx': int, 'gene_symbol_idx': int}) \
    .set_index('cell_line_idx') \
    .join(modeling_data[['cell_line', 'cell_line_idx']] \
              .drop_duplicates() \
              .astype({'cell_line_idx': int}) \
              .set_index('cell_line_idx'),
          how='left') \
    .reset_index() \
    .set_index('gene_symbol_idx') \
    .join(modeling_data[['gene_symbol', 'gene_symbol_idx']] \
              .drop_duplicates() \
              .astype({'gene_symbol_idx': int}) \
              .set_index('gene_symbol_idx'),
          how='left') \
    .reset_index() \
    .assign(gene_symbol=lambda x: [f'{a} ({b})' for a,b in zip(x.gene_symbol, x.gene_symbol_idx)],
            cell_line=lambda x: [f'{a} ({b})' for a,b in zip(x.cell_line, x.cell_line_idx)]) \
    .drop(['gene_symbol_idx', 'cell_line_idx'], axis=1) \
    .pivot(index='gene_symbol', columns='cell_line', values='value')
```


```python
# Color bar for tissue of origin of cell lines.
cell_line_origin = g_jl_post.columns.to_list()
cell_line_origin = [a.split(' ')[0] for a in cell_line_origin]
cell_line_origin = [a.split('_')[1:] for a in cell_line_origin]
cell_line_origin = [' '.join(a) for a in cell_line_origin]

cell_line_pal = sns.husl_palette(len(np.unique(cell_line_origin)), s=.90)
cell_line_lut = dict(zip(np.unique(cell_line_origin), cell_line_pal))

cell_line_colors = pd.Series(cell_line_origin, index=g_jl_post.columns).map(cell_line_lut)

np.random.seed(123)
row_linkage = hierarchy.linkage(distance.pdist(g_jl_post), method='average')

p = sns.clustermap(g_jl_post, center=0, cmap="YlGnBu", linewidths=0.5, 
                   figsize=(12, 10),
                   cbar_kws={'label': 'mean coeff.'},
                   cbar_pos=[0.06, 0.15, 0.02, 0.2],
                   row_linkage=row_linkage,
                   col_colors=cell_line_colors)
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_101_0.png)



```python
y_pred = d2_m5_fit.extract(pars='y_pred')['y_pred']
y_pred.shape
```




    (8000, 3334)




```python
m5_pred_mean = np.apply_along_axis(np.mean, 0, y_pred)
m5_pred_hdi = np.apply_along_axis(az.hdi, 0, y_pred, hdi_prob=0.89)
d2_m5_pred = pd.DataFrame({
    'pred_mean': m5_pred_mean, 
    'pred_hdi_low': m5_pred_hdi[0], 
    'pred_hdi_high': m5_pred_hdi[1],
    'obs': d2_m5_data['y'],
    'barcode_sequence_idx': d2_m5_data['shrna'],
    'gene_idx': d2_m5_data['gene'],
    'cell_line_idx': d2_m5_data['cell_line'],
    'barcode_sequence': modeling_data.barcode_sequence,
    'gene_symbol': modeling_data.gene_symbol,
    'cell_line': modeling_data.cell_line,
})
d2_m5_pred
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_mean</th>
      <th>pred_hdi_low</th>
      <th>pred_hdi_high</th>
      <th>obs</th>
      <th>barcode_sequence_idx</th>
      <th>gene_idx</th>
      <th>cell_line_idx</th>
      <th>barcode_sequence</th>
      <th>gene_symbol</th>
      <th>cell_line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.733779</td>
      <td>-0.768984</td>
      <td>2.278009</td>
      <td>0.625725</td>
      <td>1</td>
      <td>5</td>
      <td>11</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>EIF6</td>
      <td>efo21_ovary</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.823323</td>
      <td>-0.654592</td>
      <td>2.390971</td>
      <td>2.145082</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>EIF6</td>
      <td>dbtrg05mg_central_nervous_system</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.942019</td>
      <td>-0.612576</td>
      <td>2.521992</td>
      <td>0.932751</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>EIF6</td>
      <td>bt20_breast</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.856841</td>
      <td>-0.703396</td>
      <td>2.423026</td>
      <td>1.372030</td>
      <td>1</td>
      <td>5</td>
      <td>36</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>EIF6</td>
      <td>sw1783_central_nervous_system</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.838587</td>
      <td>-0.715314</td>
      <td>2.379248</td>
      <td>0.803835</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
      <td>EIF6</td>
      <td>kns60_central_nervous_system</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3329</th>
      <td>-1.837794</td>
      <td>-3.683594</td>
      <td>0.109252</td>
      <td>-3.118520</td>
      <td>109</td>
      <td>13</td>
      <td>32</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
      <td>TRIM39</td>
      <td>shp77_lung</td>
    </tr>
    <tr>
      <th>3330</th>
      <td>-1.113425</td>
      <td>-2.954369</td>
      <td>0.605212</td>
      <td>-1.858803</td>
      <td>109</td>
      <td>13</td>
      <td>33</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
      <td>TRIM39</td>
      <td>sknep1_bone</td>
    </tr>
    <tr>
      <th>3331</th>
      <td>-1.425125</td>
      <td>-3.229695</td>
      <td>0.413542</td>
      <td>-2.398997</td>
      <td>109</td>
      <td>13</td>
      <td>34</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
      <td>TRIM39</td>
      <td>smsctr_soft_tissue</td>
    </tr>
    <tr>
      <th>3332</th>
      <td>-0.936589</td>
      <td>-2.740719</td>
      <td>0.889739</td>
      <td>0.948492</td>
      <td>109</td>
      <td>13</td>
      <td>35</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
      <td>TRIM39</td>
      <td>sudhl4_haematopoietic_and_lymphoid_tissue</td>
    </tr>
    <tr>
      <th>3333</th>
      <td>-1.343153</td>
      <td>-3.158997</td>
      <td>0.433367</td>
      <td>-2.847713</td>
      <td>109</td>
      <td>13</td>
      <td>37</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
      <td>TRIM39</td>
      <td>tuhr14tkb_kidney</td>
    </tr>
  </tbody>
</table>
<p>3334 rows × 10 columns</p>
</div>




```python
genes = list(np.unique(d2_m5_pred.gene_symbol))
genes.sort()

fig, axes = plt.subplots(5, 3, figsize=(12, 12))

for ax, gene in zip(axes.flatten(), genes):
    df = d2_m5_pred[d2_m5_pred.gene_symbol == gene]
    ax.scatter(df.barcode_sequence, df.obs, color='blue', s=50, alpha=0.2)
    ax.scatter(df.barcode_sequence, df.pred_mean, color='red', s=20, alpha=0.5)
    ax.xaxis.set_ticks([])
    ax.set_title(gene)


axes[4, 2].axis('off')
axes[4, 1].axis('off')
fig.tight_layout(pad=1.0)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_104_0.png)



```python
sns.distplot(d2_m5_pred[d2_m5_pred.gene_symbol == 'KRAS'].obs, label='observed')
sns.distplot(d2_m5_pred[d2_m5_pred.gene_symbol == 'KRAS'].pred_mean, label='predicticed')
plt.legend()
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_105_0.png)



```python
g_pred = d2_m5_fit.to_dataframe()
g_pred_cols = g_pred.columns[g_pred.columns.str.contains('g\[')].to_list()
g_pred = g_pred[['chain', 'draw', 'warmup'] + g_pred_cols] \
    .set_index(['chain', 'draw', 'warmup']) \
    .melt() \
    .assign(cell_line_idx=lambda x: [int(a) for a in extract_g_index(x.variable)],
            gene_symbol_idx=lambda x: [int(a) for a in extract_g_index(x.variable, i=1)])

g_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
      <th>cell_line_idx</th>
      <th>gene_symbol_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>g[1,1]</td>
      <td>0.439824</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g[1,1]</td>
      <td>-0.284250</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g[1,1]</td>
      <td>0.572784</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g[1,1]</td>
      <td>-0.132600</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g[1,1]</td>
      <td>0.675018</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
cell_line_gene_idx_map = modeling_data \
    [['cell_line_idx', 'gene_symbol_idx', 'cell_line', 'gene_symbol']] \
    .drop_duplicates() \
    .set_index(['cell_line_idx', 'gene_symbol_idx'])
cell_line_gene_idx_map.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>cell_line</th>
      <th>gene_symbol</th>
    </tr>
    <tr>
      <th>cell_line_idx</th>
      <th>gene_symbol_idx</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <th>5</th>
      <td>efo21_ovary</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>9</th>
      <th>5</th>
      <td>dbtrg05mg_central_nervous_system</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>3</th>
      <th>5</th>
      <td>bt20_breast</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>36</th>
      <th>5</th>
      <td>sw1783_central_nervous_system</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>18</th>
      <th>5</th>
      <td>kns60_central_nervous_system</td>
      <td>EIF6</td>
    </tr>
  </tbody>
</table>
</div>




```python
g_pred = g_pred.set_index(['cell_line_idx', 'gene_symbol_idx']) \
    .join(cell_line_gene_idx_map, how='left') \
    .reset_index()
g_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_line_idx</th>
      <th>gene_symbol_idx</th>
      <th>variable</th>
      <th>value</th>
      <th>cell_line</th>
      <th>gene_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.439824</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>-0.284250</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.572784</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>-0.132600</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.675018</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
  </tbody>
</table>
</div>




```python
kras_muts = pd.read_csv(modeling_data_dir / 'kras_mutants.csv') \
    .assign(cell_line=lambda x: [a.lower() for a in x.cell_line])
kras_muts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_line</th>
      <th>protein_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a427_lung</td>
      <td>p.G12D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a549_lung</td>
      <td>p.G12S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ags_stomach</td>
      <td>p.G12D</td>
    </tr>
    <tr>
      <th>3</th>
      <td>amo1_haematopoietic_and_lymphoid_tissue</td>
      <td>p.A146T</td>
    </tr>
    <tr>
      <th>4</th>
      <td>aspc1_pancreas</td>
      <td>p.G12D</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>sw948_large_intestine</td>
      <td>p.Q61L</td>
    </tr>
    <tr>
      <th>110</th>
      <td>tccpan2_pancreas</td>
      <td>p.G12R</td>
    </tr>
    <tr>
      <th>111</th>
      <td>tov21g_ovary</td>
      <td>p.G13C</td>
    </tr>
    <tr>
      <th>112</th>
      <td>umuc3_urinary_tract</td>
      <td>p.G12C</td>
    </tr>
    <tr>
      <th>113</th>
      <td>yd8_upper_aerodigestive_tract</td>
      <td>p.G138V</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 2 columns</p>
</div>




```python
def violin_by_cell_line(target_gene, mut_gene, mut_cell_lines):
    df = g_pred.pipe(lambda x: x[x.gene_symbol == target_gene])

    cell_line_order = df.groupby('cell_line').mean().sort_values('value').reset_index().cell_line.to_list()
    df = df.set_index('cell_line').loc[cell_line_order].reset_index()
    df = df.assign(mut=lambda x: x.cell_line.isin(mut_cell_lines))

    fig = plt.figure(figsize=(15, 5))
    plt.axhline(y=0, c='k', alpha=0.4, ls='--')
    sns.violinplot('cell_line', 'value', hue='mut', data=df, dodge=False)
    plt.xticks(rotation=60, ha='right')
    plt.title(f'{target_gene} essentiality for {mut_gene} mutant cell lines')
    plt.xlabel(None)
    plt.ylabel('gene effect coeff.')
    plt.legend(title=f'{mut_gene} mut.')
    plt.show()
```


```python
violin_by_cell_line('KRAS', 'KRAS', kras_muts.cell_line)
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_111_0.png)



```python
mutation_data = pd.read_csv(modeling_data_dir / 'ccle_mutations.csv') \
    .assign(cell_line=lambda x: [a.lower() for a in x.tumor_sample_barcode]) \
    .pipe(lambda x: x[x.cell_line.isin(g_pred.cell_line)]) \
    .reset_index(drop=True)
mutation_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tumor_sample_barcode</th>
      <th>hugo_symbol</th>
      <th>chromosome</th>
      <th>start_position</th>
      <th>end_position</th>
      <th>variant_classification</th>
      <th>variant_type</th>
      <th>protein_change</th>
      <th>cell_line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CADOES1_BONE</td>
      <td>NPHP4</td>
      <td>1</td>
      <td>5935092</td>
      <td>5935092</td>
      <td>Silent</td>
      <td>SNP</td>
      <td>p.T962T</td>
      <td>cadoes1_bone</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CADOES1_BONE</td>
      <td>CHD5</td>
      <td>1</td>
      <td>6202224</td>
      <td>6202224</td>
      <td>De_novo_Start_OutOfFrame</td>
      <td>SNP</td>
      <td>NaN</td>
      <td>cadoes1_bone</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CADOES1_BONE</td>
      <td>SLC45A1</td>
      <td>1</td>
      <td>8404071</td>
      <td>8404071</td>
      <td>Nonstop_Mutation</td>
      <td>SNP</td>
      <td>p.*749R</td>
      <td>cadoes1_bone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CADOES1_BONE</td>
      <td>PRAMEF10</td>
      <td>1</td>
      <td>12954514</td>
      <td>12954514</td>
      <td>Missense_Mutation</td>
      <td>SNP</td>
      <td>p.P257S</td>
      <td>cadoes1_bone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CADOES1_BONE</td>
      <td>NBPF1</td>
      <td>1</td>
      <td>16892276</td>
      <td>16892276</td>
      <td>Silent</td>
      <td>SNP</td>
      <td>p.A972A</td>
      <td>cadoes1_bone</td>
    </tr>
  </tbody>
</table>
</div>




```python
braf_muts = mutation_data.pipe(lambda x: x[x.hugo_symbol == 'BRAF'])
violin_by_cell_line('BRAF', 'BRAF', braf_muts.cell_line)
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_113_0.png)



```python
egfr_muts = mutation_data.pipe(lambda x: x[x.hugo_symbol == 'EGFR']).cell_line.to_list()
violin_by_cell_line('BRAF', 'KRAS and EGFR', kras_muts.cell_line.to_list() + egfr_muts)
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_114_0.png)


**TODO: look at effect of the multiplicative scaling factor. Maybe compare results with model 4 where there is no multiplicative factor.**


```python
d2_m4_fit.model_pars
```




    ['sigma_c',
     'mu_gbar',
     'sigma_gbar',
     'sigma_g',
     'c',
     'gbar',
     'g',
     'mu_sigma',
     'sigma_sigma',
     'sigma',
     'y_pred']




```python
d2_m5_fit.model_pars
```




    ['sigma_c',
     'mu_gbar',
     'sigma_gbar',
     'sigma_g',
     'c',
     'alpha',
     'gbar',
     'g',
     'mu_sigma',
     'sigma_sigma',
     'sigma',
     'y_pred']




```python
m4_cl_data = d2_m4_fit.extract(pars='c')
cl_data = {'m4_c': m4_cl_data['c'].mean(axis=0)} 
```


```python
m5_cl_data = d2_m5_fit.extract(pars=['c', 'alpha'])
for key, val in m5_cl_data.items():
    cl_data[f'm5_{key}'] = val.mean(axis=0)
```


```python
cl_df = pd.DataFrame(cl_data)
cl_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>m4_c</th>
      <th>m5_c</th>
      <th>m5_alpha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.256885</td>
      <td>1.401568</td>
      <td>0.228217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.847578</td>
      <td>0.359238</td>
      <td>0.309626</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.731892</td>
      <td>0.613493</td>
      <td>0.479000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.242973</td>
      <td>0.652422</td>
      <td>0.177926</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.547694</td>
      <td>0.102312</td>
      <td>0.772987</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.scatterplot(x='m4_c', y='m5_c', size='m5_alpha', hue='m5_alpha', 
                data=cl_df, sizes=(50, 150), alpha=0.8, legend='brief')
plt.xlabel('model 4 cell line intercept')
plt.ylabel('model 5 cell line intercept')
plt.title('Cell line varying intercepts between models 4 and 5')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_121_0.png)



```python
def extract_prediction_info(fit, name):
    p = fit.extract('y_pred')['y_pred']
    p_hdi = np.apply_along_axis(az.hdi, axis=0, arr=p, hdi_prob=0.89)
    return {
        f'{name}_pred_mean': p.mean(axis=0),
        f'{name}_pred_hdi_dn': p_hdi[0],
        f'{name}_pred_hdi_up': p_hdi[1],
    }
```


```python
model_predictions = pd.DataFrame({
    **extract_prediction_info(d2_m4_fit, "m4"), 
    **extract_prediction_info(d2_m5_fit, "m5"),
    'obs': modeling_data.lfc,
    'cell_line': modeling_data.cell_line,
    'gene_symbol': modeling_data.gene_symbol,
    'barcode_sequence': modeling_data.barcode_sequence,
})
```


```python
model_predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>m4_pred_mean</th>
      <th>m4_pred_hdi_dn</th>
      <th>m4_pred_hdi_up</th>
      <th>m5_pred_mean</th>
      <th>m5_pred_hdi_dn</th>
      <th>m5_pred_hdi_up</th>
      <th>obs</th>
      <th>cell_line</th>
      <th>gene_symbol</th>
      <th>barcode_sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.810189</td>
      <td>-0.775439</td>
      <td>2.326655</td>
      <td>0.733779</td>
      <td>-0.768984</td>
      <td>2.278009</td>
      <td>0.625725</td>
      <td>efo21_ovary</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.969803</td>
      <td>-0.533012</td>
      <td>2.542772</td>
      <td>0.823323</td>
      <td>-0.654592</td>
      <td>2.390971</td>
      <td>2.145082</td>
      <td>dbtrg05mg_central_nervous_system</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.075705</td>
      <td>-0.579965</td>
      <td>2.594612</td>
      <td>0.942019</td>
      <td>-0.612576</td>
      <td>2.521992</td>
      <td>0.932751</td>
      <td>bt20_breast</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.128708</td>
      <td>-0.486943</td>
      <td>2.670705</td>
      <td>0.856841</td>
      <td>-0.703396</td>
      <td>2.423026</td>
      <td>1.372030</td>
      <td>sw1783_central_nervous_system</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.006432</td>
      <td>-0.600970</td>
      <td>2.525803</td>
      <td>0.838587</td>
      <td>-0.715314</td>
      <td>2.379248</td>
      <td>0.803835</td>
      <td>kns60_central_nervous_system</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
  </tbody>
</table>
</div>




```python
genes = np.unique(model_predictions.gene_symbol.astype('string').to_list())
genes.sort()
genes
model_predictions = model_predictions \
    .set_index('gene_symbol') \
    .loc[genes] \
    .reset_index() \
    .reset_index()
```


```python
plt.figure(figsize=(15, 5))
sns.scatterplot(x='index', y='obs', hue='gene_symbol', data=model_predictions)
plt.xlim(-10, len(model_predictions)+10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_126_0.png)



```python
fig, axes = plt.subplots(5, 3, figsize=(10, 15))

for ax, gene in zip(axes.flatten(), genes):
    df = model_predictions[model_predictions.gene_symbol == gene]
    ax.scatter(x=df.m4_pred_mean, y=df.m5_pred_mean, alpha=0.5, s=10, color='#5d877b')
    ax.set_title(gene)

axes[4, 2].axis('off')
axes[4, 1].axis('off')
fig.tight_layout(pad=1.0)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_127_0.png)



```python
def abline(ax, slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--')
```


```python
fig, axes = plt.subplots(len(genes), 2, figsize=(8, 35))
for i, gene in enumerate(genes):
    df = model_predictions[model_predictions.gene_symbol == gene]
    for j, m in enumerate(('m4', 'm5')):
        axes[i, j].scatter(df.loc[:, 'obs'], df.loc[:, f'{m}_pred_mean'], 
                           c='#a86f7a', s=10, alpha=0.7)
        abline(axes[i, j], 1, 0)
        axes[i, j].set_title(f'{gene} ({m})', fontsize=15)
        axes[i, j].set_xlabel('observed', fontsize=12)
        axes[i, j].set_ylabel('predicted', fontsize=12)


fig.tight_layout(pad=1.0)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_129_0.png)



```python

```
