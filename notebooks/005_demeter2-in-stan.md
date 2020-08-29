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
```


```python
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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_6_0.png)



```python
cell_lines = set(modeling_data.cell_line.to_list())
for cell_line in cell_lines:
    lfc = modeling_data[modeling_data.cell_line == cell_line].lfc
    sns.distplot(lfc, kde=True, hist=False, label=None, kde_kws={'alpha': 0.2})

plt.title('LFC distributions')
plt.xlabel('LFC')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_7_0.png)



```python
sns.distplot(modeling_data.lfc)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_8_0.png)



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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_10_0.png)


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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_19_0.png)



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
      <td>0.030</td>
      <td>-1.347</td>
      <td>-1.235</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1958.0</td>
      <td>1952.0</td>
      <td>1925.0</td>
      <td>1230.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.736</td>
      <td>0.021</td>
      <td>1.696</td>
      <td>1.775</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2269.0</td>
      <td>2269.0</td>
      <td>2273.0</td>
      <td>1545.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m1)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_32_0.png)



```python
az.plot_forest(az_d2_m1, combined=True)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_33_0.png)



```python
az.plot_ppc(az_d2_m1, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_34_0.png)


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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_42_0.png)



```python
df = d2_m2_gen_fit.to_dataframe() \
    .drop(['chain', 'draw', 'warmup'], axis=1) \
    .melt(var_name='parameter', value_name='value')
df = df[df.parameter.str.contains('y_pred')]
sns.distplot(df.value)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_43_0.png)



```python
sns.distplot(modeling_data.lfc)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_44_0.png)


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
      <td>-1.222</td>
      <td>0.124</td>
      <td>-1.481</td>
      <td>-1.001</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1058.0</td>
      <td>1058.0</td>
      <td>1082.0</td>
      <td>574.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.276</td>
      <td>0.087</td>
      <td>1.132</td>
      <td>1.461</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1031.0</td>
      <td>1007.0</td>
      <td>1049.0</td>
      <td>643.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>0.714</td>
      <td>0.201</td>
      <td>0.330</td>
      <td>1.062</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1205.0</td>
      <td>1205.0</td>
      <td>1252.0</td>
      <td>551.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>alpha[1]</th>
      <td>-0.250</td>
      <td>0.190</td>
      <td>-0.616</td>
      <td>0.087</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>836.0</td>
      <td>836.0</td>
      <td>847.0</td>
      <td>582.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>alpha[2]</th>
      <td>-0.784</td>
      <td>0.211</td>
      <td>-1.180</td>
      <td>-0.395</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>1100.0</td>
      <td>1047.0</td>
      <td>1084.0</td>
      <td>668.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m2, var_names=['mu_alpha', 'sigma_alpha', 'sigma'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_53_0.png)



```python
az.plot_forest(az_d2_m2, kind='ridgeplot', combined=True, 
               var_names=['mu_alpha', 'sigma_alpha', 'sigma'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_54_0.png)



```python
az.plot_ppc(az_d2_m2, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_55_0.png)



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
      <td>-1.263567</td>
      <td>1.286463</td>
      <td>0.691531</td>
      <td>-0.410589</td>
      <td>-0.486056</td>
      <td>0.140605</td>
      <td>-2.252227</td>
      <td>...</td>
      <td>-1.821982</td>
      <td>-4.610282</td>
      <td>-0.057629</td>
      <td>-2395.220205</td>
      <td>0.834995</td>
      <td>0.418915</td>
      <td>5</td>
      <td>55</td>
      <td>0</td>
      <td>2445.602866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1.294416</td>
      <td>1.323071</td>
      <td>0.697405</td>
      <td>-0.267598</td>
      <td>-1.044003</td>
      <td>0.264281</td>
      <td>-2.362734</td>
      <td>...</td>
      <td>0.730910</td>
      <td>-0.803228</td>
      <td>-1.128669</td>
      <td>-2394.569464</td>
      <td>0.937689</td>
      <td>0.418915</td>
      <td>7</td>
      <td>143</td>
      <td>0</td>
      <td>2451.366998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>-1.374924</td>
      <td>1.356104</td>
      <td>0.803428</td>
      <td>-0.176478</td>
      <td>-1.007182</td>
      <td>0.161903</td>
      <td>-2.297076</td>
      <td>...</td>
      <td>-3.385013</td>
      <td>-1.969810</td>
      <td>0.468591</td>
      <td>-2395.033725</td>
      <td>0.853836</td>
      <td>0.418915</td>
      <td>3</td>
      <td>15</td>
      <td>0</td>
      <td>2456.917527</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>-1.464064</td>
      <td>1.165498</td>
      <td>0.857390</td>
      <td>-0.004899</td>
      <td>-1.059237</td>
      <td>-0.043809</td>
      <td>-2.267916</td>
      <td>...</td>
      <td>0.264705</td>
      <td>0.080555</td>
      <td>0.108367</td>
      <td>-2395.912197</td>
      <td>0.954930</td>
      <td>0.418915</td>
      <td>5</td>
      <td>55</td>
      <td>0</td>
      <td>2449.214887</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>-1.029178</td>
      <td>1.369234</td>
      <td>0.585008</td>
      <td>-0.558714</td>
      <td>-0.510735</td>
      <td>0.471418</td>
      <td>-2.217178</td>
      <td>...</td>
      <td>-2.383169</td>
      <td>-2.772807</td>
      <td>-0.843937</td>
      <td>-2396.046197</td>
      <td>0.936803</td>
      <td>0.418915</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>2450.746997</td>
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
    WARNING:pystan:156 of 8000 iterations ended with a divergence (1.95 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:6 of 8000 iterations saturated the maximum tree depth of 10 (0.075 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation



```python
pystan.check_hmc_diagnostics(d2_m3_fit)
```

    WARNING:pystan:Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed
    WARNING:pystan:156 of 8000 iterations ended with a divergence (1.95 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:6 of 8000 iterations saturated the maximum tree depth of 10 (0.075 %)
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
      <td>-0.773</td>
      <td>1.167</td>
      <td>-2.439</td>
      <td>1.427</td>
      <td>0.270</td>
      <td>0.194</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>33.0</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.242</td>
      <td>0.094</td>
      <td>1.087</td>
      <td>1.430</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>356.0</td>
      <td>342.0</td>
      <td>406.0</td>
      <td>281.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>mu_g</th>
      <td>-0.514</td>
      <td>1.164</td>
      <td>-2.764</td>
      <td>1.165</td>
      <td>0.268</td>
      <td>0.193</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>36.0</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.404</td>
      <td>0.219</td>
      <td>0.023</td>
      <td>0.757</td>
      <td>0.017</td>
      <td>0.012</td>
      <td>166.0</td>
      <td>166.0</td>
      <td>123.0</td>
      <td>71.0</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>1.455</td>
      <td>1.224</td>
      <td>-0.608</td>
      <td>3.800</td>
      <td>0.277</td>
      <td>0.199</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>37.0</td>
      <td>1.14</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m3, var_names=['g'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_64_0.png)



```python
az.plot_ppc(az_d2_m3, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_65_0.png)



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
      <td>1.455</td>
      <td>1.224</td>
      <td>-0.608</td>
      <td>3.800</td>
      <td>0.277</td>
      <td>0.199</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>37.0</td>
      <td>1.14</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.073</td>
      <td>1.229</td>
      <td>-1.856</td>
      <td>2.445</td>
      <td>0.271</td>
      <td>0.195</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>33.0</td>
      <td>1.13</td>
      <td>GRK5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.060</td>
      <td>1.207</td>
      <td>-2.076</td>
      <td>2.311</td>
      <td>0.268</td>
      <td>0.192</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>38.0</td>
      <td>1.13</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.496</td>
      <td>1.229</td>
      <td>-1.501</td>
      <td>2.871</td>
      <td>0.277</td>
      <td>0.199</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>34.0</td>
      <td>1.13</td>
      <td>EGFR</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.351</td>
      <td>1.202</td>
      <td>-3.423</td>
      <td>0.987</td>
      <td>0.267</td>
      <td>0.191</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>38.0</td>
      <td>1.13</td>
      <td>COL8A1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.715</td>
      <td>1.176</td>
      <td>-2.763</td>
      <td>1.527</td>
      <td>0.263</td>
      <td>0.189</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>36.0</td>
      <td>1.14</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.446</td>
      <td>1.223</td>
      <td>-2.388</td>
      <td>1.881</td>
      <td>0.276</td>
      <td>0.198</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>34.0</td>
      <td>1.14</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.247</td>
      <td>1.198</td>
      <td>-0.769</td>
      <td>3.530</td>
      <td>0.267</td>
      <td>0.191</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>36.0</td>
      <td>1.13</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.447</td>
      <td>1.213</td>
      <td>-1.502</td>
      <td>2.785</td>
      <td>0.274</td>
      <td>0.196</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>36.0</td>
      <td>1.13</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.718</td>
      <td>1.213</td>
      <td>-2.667</td>
      <td>1.644</td>
      <td>0.271</td>
      <td>0.195</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>34.0</td>
      <td>1.13</td>
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
      <td>-0.391</td>
      <td>1.199</td>
      <td>-2.725</td>
      <td>1.468</td>
      <td>0.272</td>
      <td>0.196</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>1.13</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.386</td>
      <td>1.227</td>
      <td>-2.671</td>
      <td>1.768</td>
      <td>0.276</td>
      <td>0.198</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>34.0</td>
      <td>1.14</td>
      <td>COG3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.906</td>
      <td>1.190</td>
      <td>-3.291</td>
      <td>1.072</td>
      <td>0.265</td>
      <td>0.190</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>36.0</td>
      <td>1.14</td>
      <td>COL8A1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.241</td>
      <td>1.204</td>
      <td>-2.499</td>
      <td>1.583</td>
      <td>0.278</td>
      <td>0.199</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>32.0</td>
      <td>1.14</td>
      <td>EGFR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-0.733</td>
      <td>1.196</td>
      <td>-3.147</td>
      <td>1.178</td>
      <td>0.268</td>
      <td>0.193</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>37.0</td>
      <td>1.14</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>-0.792</td>
      <td>1.197</td>
      <td>-3.210</td>
      <td>1.147</td>
      <td>0.260</td>
      <td>0.187</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>22.0</td>
      <td>40.0</td>
      <td>1.13</td>
      <td>ESPL1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>-0.333</td>
      <td>1.221</td>
      <td>-2.678</td>
      <td>1.531</td>
      <td>0.273</td>
      <td>0.196</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>23.0</td>
      <td>35.0</td>
      <td>1.13</td>
      <td>GRK5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>-0.798</td>
      <td>1.166</td>
      <td>-3.100</td>
      <td>1.077</td>
      <td>0.266</td>
      <td>0.191</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>34.0</td>
      <td>1.15</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>-0.222</td>
      <td>1.187</td>
      <td>-2.405</td>
      <td>1.640</td>
      <td>0.272</td>
      <td>0.195</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>1.13</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>-0.560</td>
      <td>1.220</td>
      <td>-2.939</td>
      <td>1.463</td>
      <td>0.257</td>
      <td>0.184</td>
      <td>23.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>40.0</td>
      <td>1.12</td>
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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_69_0.png)


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
    WARNING:pystan:64 of 8000 iterations ended with a divergence (0.8 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.



```python
pystan.check_hmc_diagnostics(d2_m4_fit)
```

    WARNING:pystan:64 of 8000 iterations ended with a divergence (0.8 %).
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
      <td>1.237</td>
      <td>0.092</td>
      <td>1.068</td>
      <td>1.410</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5649.0</td>
      <td>5594.0</td>
      <td>5720.0</td>
      <td>5594.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_gbar</th>
      <td>-1.271</td>
      <td>0.181</td>
      <td>-1.627</td>
      <td>-0.944</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>1339.0</td>
      <td>1339.0</td>
      <td>1324.0</td>
      <td>3014.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_gbar</th>
      <td>0.397</td>
      <td>0.215</td>
      <td>0.010</td>
      <td>0.746</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>470.0</td>
      <td>470.0</td>
      <td>355.0</td>
      <td>157.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.460</td>
      <td>0.031</td>
      <td>0.402</td>
      <td>0.519</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>2010.0</td>
      <td>2010.0</td>
      <td>2007.0</td>
      <td>3988.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>c[0]</th>
      <td>2.236</td>
      <td>0.390</td>
      <td>1.531</td>
      <td>2.993</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>1432.0</td>
      <td>1353.0</td>
      <td>1527.0</td>
      <td>2343.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m4, var_names=['gbar'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_77_0.png)



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
      <td>1.038</td>
      <td>0.141</td>
      <td>0.764</td>
      <td>1.294</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5130.0</td>
      <td>914.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[105]</th>
      <td>0.982</td>
      <td>0.115</td>
      <td>0.779</td>
      <td>1.206</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6993.0</td>
      <td>6993.0</td>
      <td>6701.0</td>
      <td>894.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[106]</th>
      <td>1.208</td>
      <td>0.131</td>
      <td>0.964</td>
      <td>1.455</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>10173.0</td>
      <td>9510.0</td>
      <td>10729.0</td>
      <td>6237.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[107]</th>
      <td>1.054</td>
      <td>0.136</td>
      <td>0.798</td>
      <td>1.303</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>14240.0</td>
      <td>13164.0</td>
      <td>14626.0</td>
      <td>5427.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma[108]</th>
      <td>1.084</td>
      <td>0.135</td>
      <td>0.843</td>
      <td>1.343</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>14609.0</td>
      <td>13624.0</td>
      <td>15001.0</td>
      <td>5716.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_forest(az_d2_m4, var_names=['sigma'], combined=True)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_79_0.png)



```python
az.plot_ppc(az_d2_m4, data_pairs={'y':'y_pred'}, num_pp_samples=100)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_80_0.png)



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
      <td>-0.430680</td>
    </tr>
    <tr>
      <th>1008001</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.238186</td>
    </tr>
    <tr>
      <th>1008002</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.170318</td>
    </tr>
    <tr>
      <th>1008003</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.149008</td>
    </tr>
    <tr>
      <th>1008004</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>g[1,1]</td>
      <td>0.331115</td>
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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_86_0.png)



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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_88_0.png)



```python
cell_lines_post = d2_m4_post.loc[:, d2_m4_post.columns.str.contains('c\[')]

for col in cell_lines_post.columns.to_list():
    sns.distplot(cell_lines_post[[col]], hist=False, kde_kws={'shade': False, 'alpha': 0.5})

plt.xlabel('coefficient value')
plt.ylabel('density')
plt.title('Distribution of coefficients for average shRNA effect')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_89_0.png)


## Model 5. Multiplicative scaling factor of each shRNA

In the original DEMETER2 paper, the model includes a multiplicative scaling factor for each shRNA, constrained between 0 and 1, along with the addative varying effect for each shRNA.
The factor is multiplied against the gene effect.

In this model, I experiment with this multiplicative factor $\alpha_s$.

$$
D_{i|s} \sim N(\mu_{i|sjl}, \sigma_s) \\
\mu_{sjl}= c_{s} + \alpha_{s} (\bar g_{l} + g_{jl}) \\
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
    WARNING:pystan:5 of 8000 iterations ended with a divergence (0.0625 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.999 to remove the divergences.
    WARNING:pystan:5 of 8000 iterations saturated the maximum tree depth of 11 (0.0625 %)
    WARNING:pystan:Run again with max_treedepth larger than 11 to avoid saturation



```python
pystan.check_hmc_diagnostics(d2_m5_fit)
```

    WARNING:pystan:2 of 8000 iterations ended with a divergence (0.025 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:2 of 8000 iterations saturated the maximum tree depth of 10 (0.025 %)
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
      <td>0.742871</td>
      <td>-0.748287</td>
      <td>2.329097</td>
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
      <td>0.779841</td>
      <td>-0.867941</td>
      <td>2.339018</td>
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
      <td>0.936184</td>
      <td>-0.649361</td>
      <td>2.477397</td>
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
      <td>0.876974</td>
      <td>-0.743851</td>
      <td>2.407862</td>
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
      <td>0.827171</td>
      <td>-0.709087</td>
      <td>2.426900</td>
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
      <td>-1.836322</td>
      <td>-3.815350</td>
      <td>-0.067366</td>
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
      <td>-1.130218</td>
      <td>-2.953949</td>
      <td>0.672288</td>
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
      <td>-1.417122</td>
      <td>-3.211824</td>
      <td>0.425783</td>
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
      <td>-0.897946</td>
      <td>-2.644816</td>
      <td>0.899824</td>
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
      <td>-1.335326</td>
      <td>-3.045359</td>
      <td>0.599845</td>
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
      <td>-0.072524</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g[1,1]</td>
      <td>0.342332</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g[1,1]</td>
      <td>0.187223</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g[1,1]</td>
      <td>0.253825</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g[1,1]</td>
      <td>0.318944</td>
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
      <td>-0.072524</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.342332</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.187223</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.253825</td>
      <td>2313287_stomach</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>g[1,1]</td>
      <td>0.318944</td>
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
      <td>2.235566</td>
      <td>1.388940</td>
      <td>0.226152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.848993</td>
      <td>0.355141</td>
      <td>0.306266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.703887</td>
      <td>0.636358</td>
      <td>0.487371</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.265645</td>
      <td>0.658361</td>
      <td>0.178976</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.575790</td>
      <td>0.103500</td>
      <td>0.775101</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_vals = np.linspace(np.min(cl_df.m4_c), np.max(cl_df.m4_c))

sns.scatterplot(x='m4_c', y='m5_c', size='m5_alpha', hue='m5_alpha', 
                data=cl_df, sizes=(50, 150), alpha=0.8, legend='brief')
plt.plot(x_vals, x_vals, 'k--', alpha=0.5)
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
      <td>0.798723</td>
      <td>-0.811882</td>
      <td>2.299739</td>
      <td>0.742871</td>
      <td>-0.748287</td>
      <td>2.329097</td>
      <td>0.625725</td>
      <td>efo21_ovary</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.015390</td>
      <td>-0.580458</td>
      <td>2.510070</td>
      <td>0.779841</td>
      <td>-0.867941</td>
      <td>2.339018</td>
      <td>2.145082</td>
      <td>dbtrg05mg_central_nervous_system</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.047552</td>
      <td>-0.572622</td>
      <td>2.584981</td>
      <td>0.936184</td>
      <td>-0.649361</td>
      <td>2.477397</td>
      <td>0.932751</td>
      <td>bt20_breast</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.143413</td>
      <td>-0.416845</td>
      <td>2.732953</td>
      <td>0.876974</td>
      <td>-0.743851</td>
      <td>2.407862</td>
      <td>1.372030</td>
      <td>sw1783_central_nervous_system</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.994660</td>
      <td>-0.538631</td>
      <td>2.604680</td>
      <td>0.827171</td>
      <td>-0.709087</td>
      <td>2.426900</td>
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
plt.xlabel('shRNA index')
plt.ylabel('observed LFC')
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

## Model 6. Multiplicative scaling factor and varying intercept for each cell line

There are two new parameters in this model: $a_j$, a varying intercept per cell line, and $\gamma_j$, a multiplicative scaling factor for each cell line.

$$
D_{i|sjl} \sim N(\mu_{i|sjl}, \sigma_s) \\
\mu_{sjl} = a_j + \gamma_j(c_{s} + \alpha_s (\bar g_{l} + g_{jl})) \\
a_j \sim N(0, \sigma_a) \\
\gamma_j \sim N(1, 2) \\
c_s \sim N(0, \sigma_c) \\
\alpha_s \sim \text{Uniform}(0, 1) \\
\bar g_l \sim N(\mu_{\bar g}, \sigma_{\bar g}) \\
g_{jl} \sim N(0, \sigma_g) \\
\sigma_a \sim \text{HalfNormal}(2, 3) \\
\sigma_c \sim \text{HalfNormal}(0, 3) \\
\mu_{\bar g} \sim N(0, 2) \quad \sigma_{\bar g} \sim \text{HalfNormal}(0, 10) \\
\sigma_g \sim \text{HalfNormal}(0, 5) \\
\sigma_s \sim \text{HalfNormal}(\mu_\sigma, \sigma_\sigma) \\
\mu_\sigma \sim \text{HalfNormal}(0, 2) \quad \sigma_\sigma \sim \text{HalfNormal}(0, 1) \\
$$


```python
d2_m6_data = {
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


```python
d2_m6_file = models_dir / 'd2_m6.cpp'
d2_m6 = StanModel_cache(file=d2_m6_file.as_posix())
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_66c65136081b1616d9b674dc89996d94 NOW.


    No cached model - compiling '../models/d2_m6.cpp'.
    0.78 minutes to compile model



```python
d2_m6_control = {'adapt_delta': 0.99, 
                 'max_treedepth': 10}
d2_m6_fit = d2_m6.sampling(data=d2_m6_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m6_control)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)
    WARNING:pystan:16 of 8000 iterations ended with a divergence (0.2 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:22 of 8000 iterations saturated the maximum tree depth of 10 (0.275 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation
    WARNING:pystan:Chain 1: E-BFMI = 0.145
    WARNING:pystan:Chain 2: E-BFMI = 0.0844
    WARNING:pystan:Chain 3: E-BFMI = 0.154
    WARNING:pystan:Chain 4: E-BFMI = 0.173
    WARNING:pystan:E-BFMI below 0.2 indicates you may need to reparameterize your model



```python
pystan.check_hmc_diagnostics(d2_m6_fit)
```

    WARNING:pystan:n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated
    WARNING:pystan:Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed
    WARNING:pystan:16 of 8000 iterations ended with a divergence (0.2 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:22 of 8000 iterations saturated the maximum tree depth of 10 (0.275 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation
    WARNING:pystan:Chain 1: E-BFMI = 0.145
    WARNING:pystan:Chain 2: E-BFMI = 0.0844
    WARNING:pystan:Chain 3: E-BFMI = 0.154
    WARNING:pystan:Chain 4: E-BFMI = 0.173
    WARNING:pystan:E-BFMI below 0.2 indicates you may need to reparameterize your model





    {'n_eff': False,
     'Rhat': False,
     'divergence': False,
     'treedepth': False,
     'energy': False}




```python
d2_m6_post = d2_m6_fit.to_dataframe()
```


```python
df = d2_m6_post.loc[:, d2_m6_post.columns.str.contains('alpha\[')]
for col in df.columns:
    sns.distplot(df[[col]], hist=False, kde_kws={'alpha': 0.5})
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_137_0.png)



```python
df = d2_m6_post.loc[:, d2_m6_post.columns.str.contains('gamma\[')]
for col in df.columns:
    sns.distplot(df[[col]], hist=False, kde_kws={'alpha': 0.5})
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_138_0.png)



```python
genes_post = d2_m6_post.loc[:, d2_m6_post.columns.str.contains('gbar\[')]

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


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_139_0.png)



```python
az_d2_m6 = az.from_pystan(posterior=d2_m6_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m6)
az.plot_ppc(az_d2_m6, data_pairs={'y':'y_pred'}, num_pp_samples=100)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_140_0.png)


## Model 7. Only a varying intercept for each cell line (no scaling factor)

The previous model had difficult fiting.
This model removes the per-cell-line scaling factor, leaving the parameter for varying effects per cell line.

$$
D_{i|sjl} \sim N(\mu_{i|sjl}, \sigma_s) \\
\mu_{sjl} = a_j + c_{s} + \alpha_s (\bar g_{l} + g_{jl}) \\
a_j \sim N(0, \sigma_a) \\
c_s \sim N(0, \sigma_c) \\
\alpha_s \sim \text{Uniform}(0, 1) \\
\bar g_l \sim N(\mu_{\bar g}, \sigma_{\bar g}) \\
g_{jl} \sim N(0, \sigma_g) \\
\sigma_a \sim \text{HalfNormal}(0, 2) \\
\sigma_c \sim \text{HalfNormal}(0, 3) \\
\mu_{\bar g} \sim N(0, 2) \quad \sigma_{\bar g} \sim \text{HalfNormal}(0, 3) \\
\sigma_g \sim \text{HalfNormal}(0, 3) \\
\sigma_s \sim \text{HalfNormal}(\mu_\sigma, \sigma_\sigma) \\
\mu_\sigma \sim \text{HalfNormal}(0, 2) \quad \sigma_\sigma \sim \text{HalfNormal}(0, 1) \\
$$


```python
d2_m7_data = {
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


```python
d2_m7_file = models_dir / 'd2_m7.cpp'
d2_m7 = StanModel_cache(file=d2_m7_file.as_posix())
```

    Using cached StanModel.



```python
d2_m7_control = {'adapt_delta': 0.99, 
                 'max_treedepth': 10}
d2_m7_fit = d2_m7.sampling(data=d2_m7_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m7_control,
                           n_jobs=5)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)
    WARNING:pystan:1 of 8000 iterations ended with a divergence (0.0125 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:1 of 8000 iterations saturated the maximum tree depth of 10 (0.0125 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation



```python
pystan.check_hmc_diagnostics(d2_m7_fit)
```

    WARNING:pystan:1 of 8000 iterations ended with a divergence (0.0125 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.99 to remove the divergences.
    WARNING:pystan:1 of 8000 iterations saturated the maximum tree depth of 10 (0.0125 %)
    WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation





    {'n_eff': True,
     'Rhat': True,
     'divergence': False,
     'treedepth': False,
     'energy': True}




```python
d2_m7_post = d2_m7_fit.to_dataframe()
```


```python
az_d2_m7 = az.from_pystan(posterior=d2_m7_fit,
                          posterior_predictive='y_pred',
                          observed_data=['y'],
                          posterior_model=d2_m7)
az.plot_ppc(az_d2_m7, data_pairs={'y':'y_pred'}, num_pp_samples=100)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_147_0.png)



```python
d2_m7_a_post = pd.DataFrame(d2_m7_fit.extract('a')['a'])

cell_line_map = modeling_data[['cell_line', 'cell_line_idx']] \
    .drop_duplicates() \
    .sort_values('cell_line_idx') \
    .reset_index(drop=True)

d2_m7_a_post.columns = cell_line_map.cell_line
```


```python
for cell_line in d2_m7_a_post.columns.to_list():
    sns.distplot(d2_m7_a_post[[cell_line]], hist=False, kde_kws={'alpha': 0.5})
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_149_0.png)



```python
d2_m7_pred = pd.DataFrame(d2_m7_fit.extract('y_pred')['y_pred'])
d2_m7_pred.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>3324</th>
      <th>3325</th>
      <th>3326</th>
      <th>3327</th>
      <th>3328</th>
      <th>3329</th>
      <th>3330</th>
      <th>3331</th>
      <th>3332</th>
      <th>3333</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.135660</td>
      <td>0.414827</td>
      <td>0.334429</td>
      <td>1.169825</td>
      <td>0.730670</td>
      <td>2.100070</td>
      <td>-0.097630</td>
      <td>0.825503</td>
      <td>0.811204</td>
      <td>0.886321</td>
      <td>...</td>
      <td>-1.271522</td>
      <td>-1.604833</td>
      <td>-0.681266</td>
      <td>-2.412664</td>
      <td>-0.092858</td>
      <td>-1.769742</td>
      <td>-1.023952</td>
      <td>-1.709334</td>
      <td>-1.446946</td>
      <td>-1.867585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.258568</td>
      <td>1.589077</td>
      <td>0.975810</td>
      <td>2.683896</td>
      <td>1.944490</td>
      <td>2.222514</td>
      <td>-0.369773</td>
      <td>0.907256</td>
      <td>1.329136</td>
      <td>1.167455</td>
      <td>...</td>
      <td>-3.554992</td>
      <td>0.909750</td>
      <td>-1.305267</td>
      <td>-0.659696</td>
      <td>-0.430644</td>
      <td>-1.464651</td>
      <td>-0.405352</td>
      <td>-3.262250</td>
      <td>-0.406125</td>
      <td>-2.403691</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.991447</td>
      <td>0.995235</td>
      <td>0.319739</td>
      <td>0.829170</td>
      <td>0.861540</td>
      <td>1.489871</td>
      <td>-0.840679</td>
      <td>1.842453</td>
      <td>0.537677</td>
      <td>1.225023</td>
      <td>...</td>
      <td>0.159416</td>
      <td>-1.171176</td>
      <td>-1.501274</td>
      <td>-2.026308</td>
      <td>-1.028868</td>
      <td>-1.797697</td>
      <td>-0.161722</td>
      <td>-0.608210</td>
      <td>-1.747198</td>
      <td>-0.662055</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.359276</td>
      <td>1.887486</td>
      <td>2.536779</td>
      <td>2.099641</td>
      <td>0.495947</td>
      <td>0.421383</td>
      <td>0.522812</td>
      <td>0.178429</td>
      <td>0.001630</td>
      <td>2.200154</td>
      <td>...</td>
      <td>-2.485514</td>
      <td>0.115954</td>
      <td>-0.794616</td>
      <td>-1.327708</td>
      <td>-0.756625</td>
      <td>-1.725587</td>
      <td>-4.247362</td>
      <td>-1.169149</td>
      <td>-0.628886</td>
      <td>0.604209</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.403651</td>
      <td>-0.367544</td>
      <td>0.098439</td>
      <td>0.566884</td>
      <td>1.336703</td>
      <td>0.360488</td>
      <td>-1.058102</td>
      <td>1.923034</td>
      <td>0.040789</td>
      <td>2.334603</td>
      <td>...</td>
      <td>-2.767285</td>
      <td>-2.070522</td>
      <td>-2.672827</td>
      <td>-2.081729</td>
      <td>-0.788420</td>
      <td>-2.484335</td>
      <td>-1.050932</td>
      <td>-1.563766</td>
      <td>-1.309522</td>
      <td>-0.289561</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3334 columns</p>
</div>




```python
d2_m7_pred_mean = d2_m7_pred \
    .melt(var_name='data_pt') \
    .groupby('data_pt') \
    .mean()

d2_m7_pred_mean['hdi'] = d2_m7_pred.apply(axis=1, func=lambda x: az.hdi(np.array(x)))

d2_m7_pred_mean = d2_m7_pred_mean \
    .assign(hdi_dn=lambda x: [a[0] for a in x.hdi],
            hdi_up=lambda x: [a[1] for a in x.hdi]) \
    .reset_index()

d2_m7_pred_mean['gene_symbol_idx'] = d2_m7_data['gene']
d2_m7_pred_mean['barcode_sequence_idx'] = d2_m7_data['shrna']

d2_m7_pred_mean = pd.merge(
    d2_m7_pred_mean,
    modeling_data[['gene_symbol', 'gene_symbol_idx']].drop_duplicates(), 
    how='left',
    on='gene_symbol_idx'
)

d2_m7_pred_mean = pd.merge(
    d2_m7_pred_mean,
    modeling_data[['barcode_sequence', 'barcode_sequence_idx']].drop_duplicates(), 
    how='left',
    on='barcode_sequence_idx'
)

d2_m7_pred_mean.head()
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
      <th>data_pt</th>
      <th>value</th>
      <th>hdi</th>
      <th>hdi_dn</th>
      <th>hdi_up</th>
      <th>gene_symbol_idx</th>
      <th>barcode_sequence_idx</th>
      <th>gene_symbol</th>
      <th>barcode_sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.327296</td>
      <td>[-4.596127267026097, 1.7751894749567059]</td>
      <td>-4.596127</td>
      <td>1.775189</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.244436</td>
      <td>[-4.5335408181883485, 1.9609086301908234]</td>
      <td>-4.533541</td>
      <td>1.960909</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.991050</td>
      <td>[-4.41952942019391, 1.9664193474126457]</td>
      <td>-4.419529</td>
      <td>1.966419</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.960443</td>
      <td>[-4.723445979399989, 1.7270904559989635]</td>
      <td>-4.723446</td>
      <td>1.727090</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.039913</td>
      <td>[-4.578403632973594, 1.648208253996473]</td>
      <td>-4.578404</td>
      <td>1.648208</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(5, 3, figsize=(12, 20))

genes = np.unique(d2_m7_pred_mean.sort_values('gene_symbol').gene_symbol.to_list())

for ax, gene in zip(axes.flatten(), genes):
    
    g_data = d2_m7_pred_mean[d2_m7_pred_mean.gene_symbol == gene] \
        .reset_index(drop=True) \
        .reset_index()
    
    for i in range(len(g_data)):
        ax.plot((i, i), (g_data.hdi_dn[i], d2_m7_pred_mean.hdi_up[i]), 
                c='#c9c9c9', alpha=0.5)
    
    sns.scatterplot(x='index', y='value', hue='barcode_sequence', data=g_data,
                    legend=False, ax=ax)
    
    ax.set_title(gene, fontsize=12)

axes[4, 2].axis('off')
axes[4, 1].axis('off')
fig.tight_layout(pad=1.0)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_152_0.png)



```python
d2_m7_pred_mean
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
      <th>data_pt</th>
      <th>value</th>
      <th>hdi</th>
      <th>hdi_dn</th>
      <th>hdi_up</th>
      <th>gene_symbol_idx</th>
      <th>barcode_sequence_idx</th>
      <th>gene_symbol</th>
      <th>barcode_sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.327296</td>
      <td>[-4.596127267026097, 1.7751894749567059]</td>
      <td>-4.596127</td>
      <td>1.775189</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.244436</td>
      <td>[-4.5335408181883485, 1.9609086301908234]</td>
      <td>-4.533541</td>
      <td>1.960909</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.991050</td>
      <td>[-4.41952942019391, 1.9664193474126457]</td>
      <td>-4.419529</td>
      <td>1.966419</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.960443</td>
      <td>[-4.723445979399989, 1.7270904559989635]</td>
      <td>-4.723446</td>
      <td>1.727090</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.039913</td>
      <td>[-4.578403632973594, 1.648208253996473]</td>
      <td>-4.578404</td>
      <td>1.648208</td>
      <td>5</td>
      <td>1</td>
      <td>EIF6</td>
      <td>ACAGAAGAAATTCTGGCAGAT</td>
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
    </tr>
    <tr>
      <th>3329</th>
      <td>3329</td>
      <td>-1.649936</td>
      <td>[-4.585495669287962, 1.7745792535842833]</td>
      <td>-4.585496</td>
      <td>1.774579</td>
      <td>13</td>
      <td>109</td>
      <td>TRIM39</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
    </tr>
    <tr>
      <th>3330</th>
      <td>3330</td>
      <td>-1.139632</td>
      <td>[-4.292058452422176, 2.0046199948082952]</td>
      <td>-4.292058</td>
      <td>2.004620</td>
      <td>13</td>
      <td>109</td>
      <td>TRIM39</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
    </tr>
    <tr>
      <th>3331</th>
      <td>3331</td>
      <td>-1.293152</td>
      <td>[-4.616257206899434, 1.939295593995174]</td>
      <td>-4.616257</td>
      <td>1.939296</td>
      <td>13</td>
      <td>109</td>
      <td>TRIM39</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
    </tr>
    <tr>
      <th>3332</th>
      <td>3332</td>
      <td>-1.042668</td>
      <td>[-4.399500114545466, 1.695579964952048]</td>
      <td>-4.399500</td>
      <td>1.695580</td>
      <td>13</td>
      <td>109</td>
      <td>TRIM39</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
    </tr>
    <tr>
      <th>3333</th>
      <td>3333</td>
      <td>-1.509146</td>
      <td>[-4.35022691859504, 2.0816350404236923]</td>
      <td>-4.350227</td>
      <td>2.081635</td>
      <td>13</td>
      <td>109</td>
      <td>TRIM39</td>
      <td>TGCTCTCATGGGTCTAGATAT</td>
    </tr>
  </tbody>
</table>
<p>3334 rows × 9 columns</p>
</div>




```python
plt.scatter(x=d2_m7_pred_mean.loc[:, 'value'], y=d2_m7_data['y'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_154_0.png)



```python

```
