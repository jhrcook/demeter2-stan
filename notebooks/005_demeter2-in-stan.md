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
    sns.distplot(lfc, kde=True, hist=False, ax=ax, kde_kws={'shade': True}, color='b')
    
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
model_testing_genes = ['COG3', 'KRAS', 'COL8A1', 'EIF6']
modeling_data = modeling_data[modeling_data.gene_symbol.isin(model_testing_genes)]
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
      <td>3</td>
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
      <td>3</td>
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
      <td>3</td>
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
      <td>3</td>
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
      <td>3</td>
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




    array([[0, 0, 1, 0],
           [0, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [1, 0, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [1, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 1, 0]])




```python
shrna_gene_matrix.shape
```




    (26, 4)



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
start = timer()
d2_m1_file = models_dir / 'd2_m1.cpp'
d2_m1 = pystan.StanModel(file=d2_m1_file.as_posix())
end = timer()
print(f'{(end - start) / 60:.2f} minutes to compile model')
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_2a3da92f7ba3e62805394cf29daed746 NOW.


    0.74 minutes to compile model



```python
d2_m1_fit = d2_m1.sampling(data=d2_m1_data, iter=2000, chains=2)
```


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
      <th>hpd_3%</th>
      <th>hpd_97%</th>
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
      <td>-1.855</td>
      <td>0.055</td>
      <td>-1.957</td>
      <td>-1.751</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2097.0</td>
      <td>2092.0</td>
      <td>2084.0</td>
      <td>1201.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.713</td>
      <td>0.039</td>
      <td>1.641</td>
      <td>1.790</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1533.0</td>
      <td>1533.0</td>
      <td>1540.0</td>
      <td>1179.0</td>
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
start = timer()
d2_m2_gen_file = models_dir / 'd2_m2_generative.cpp'
d2_m2_gen = pystan.StanModel(file=d2_m2_gen_file.as_posix())
end = timer()
print(f'{(end - start) / 60:.2f} minutes to compile model')
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_112516ecd402d13dafa0de6e631a9e45 NOW.


    0.67 minutes to compile model



```python
d2_m2_gen_fit = d2_m2_gen.sampling(data=d2_m2_gen_data, iter=10, chains=1, algorithm='Fixed_param')
```

    WARNING:pystan:`warmup=0` forced with `algorithm="Fixed_param"`.



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




    26



**Compile model.**

The current problem is that the various values for `alpha` of different shRNA are the same because `sigma_alpha` is so small.
Maybe try with just two shRNA, but I need to figure out why the value for `sigma_alpha` is shrinking so fast.


```python
start = timer()
d2_m2_file = models_dir / 'd2_m2.cpp'
d2_m2 = pystan.StanModel(file=d2_m2_file.as_posix())
end = timer()
print(f'{(end - start) / 60:.2f} minutes to compile model')
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_27c69fe8813455b868fbd0406f6b51bd NOW.


    0.72 minutes to compile model



```python
d2_m2_fit = d2_m2.sampling(data=d2_m2_data, iter=1000, chains=2)
```


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
      <th>hpd_3%</th>
      <th>hpd_97%</th>
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
      <td>-1.826</td>
      <td>0.248</td>
      <td>-2.291</td>
      <td>-1.370</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1945.0</td>
      <td>1945.0</td>
      <td>1929.0</td>
      <td>800.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.249</td>
      <td>0.194</td>
      <td>0.927</td>
      <td>1.631</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1577.0</td>
      <td>1394.0</td>
      <td>1849.0</td>
      <td>795.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>0.703</td>
      <td>0.209</td>
      <td>0.295</td>
      <td>1.072</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>2354.0</td>
      <td>2056.0</td>
      <td>2387.0</td>
      <td>738.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>alpha[1]</th>
      <td>-0.800</td>
      <td>0.220</td>
      <td>-1.259</td>
      <td>-0.430</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>2638.0</td>
      <td>2025.0</td>
      <td>2585.0</td>
      <td>666.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>alpha[2]</th>
      <td>-2.268</td>
      <td>0.201</td>
      <td>-2.637</td>
      <td>-1.899</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>1741.0</td>
      <td>1688.0</td>
      <td>1718.0</td>
      <td>862.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(az_d2_m2, var_names=['alpha'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_52_0.png)



```python
az.plot_forest(az_d2_m2, kind='ridgeplot', combined=True, var_names=['alpha'])
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
      <th>y_pred[960]</th>
      <th>y_pred[961]</th>
      <th>y_pred[962]</th>
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
      <td>-1.717343</td>
      <td>1.153475</td>
      <td>0.768823</td>
      <td>-0.517442</td>
      <td>-1.839851</td>
      <td>-1.309714</td>
      <td>-2.768498</td>
      <td>...</td>
      <td>-3.806916</td>
      <td>-5.609710</td>
      <td>-3.315077</td>
      <td>-723.551342</td>
      <td>0.808957</td>
      <td>0.554071</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>739.722806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1.667439</td>
      <td>1.467565</td>
      <td>0.805954</td>
      <td>-1.191569</td>
      <td>-2.040256</td>
      <td>-1.101814</td>
      <td>-2.732943</td>
      <td>...</td>
      <td>-4.549657</td>
      <td>-4.526206</td>
      <td>-3.209196</td>
      <td>-727.934890</td>
      <td>0.815801</td>
      <td>0.554071</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>742.966630</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>-1.649163</td>
      <td>1.431593</td>
      <td>0.872164</td>
      <td>-1.353188</td>
      <td>-1.990979</td>
      <td>-1.180292</td>
      <td>-2.622750</td>
      <td>...</td>
      <td>-4.853599</td>
      <td>-3.584805</td>
      <td>-6.296632</td>
      <td>-730.323014</td>
      <td>0.980871</td>
      <td>0.554071</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>742.993687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>-2.053052</td>
      <td>1.165102</td>
      <td>0.577352</td>
      <td>-0.302050</td>
      <td>-2.542299</td>
      <td>-1.870215</td>
      <td>-2.097786</td>
      <td>...</td>
      <td>-4.024895</td>
      <td>-3.423712</td>
      <td>-3.636841</td>
      <td>-730.812147</td>
      <td>0.960364</td>
      <td>0.554071</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>750.384967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>-1.603363</td>
      <td>1.230839</td>
      <td>1.026905</td>
      <td>-1.301742</td>
      <td>-1.983990</td>
      <td>-1.168913</td>
      <td>-2.630462</td>
      <td>...</td>
      <td>-3.843858</td>
      <td>-4.776930</td>
      <td>-2.566063</td>
      <td>-727.492163</td>
      <td>1.000000</td>
      <td>0.554071</td>
      <td>3</td>
      <td>7</td>
      <td>0</td>
      <td>744.349184</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1001 columns</p>
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
start = timer()
d2_m3_file = models_dir / 'd2_m3.cpp'
d2_m3 = pystan.StanModel(file=d2_m3_file.as_posix())
end = timer()
print(f'{(end - start) / 60:.2f} minutes to compile model')
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_85af818ae9105b1cf51943262fb26a36 NOW.


    0.75 minutes to compile model



```python
d2_m3_control = {'adapt_delta': 0.999, 
                 'max_treedepth': 20}
d2_m3_fit = d2_m3.sampling(data=d2_m3_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m3_control)
```

    WARNING:pystan:6 of 8000 iterations ended with a divergence (0.075 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.999 to remove the divergences.



```python
pystan.check_hmc_diagnostics(d2_m3_fit)
```

    WARNING:pystan:6 of 8000 iterations ended with a divergence (0.075 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.999 to remove the divergences.





    {'n_eff': True,
     'Rhat': True,
     'divergence': False,
     'treedepth': True,
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
      <th>hpd_3%</th>
      <th>hpd_97%</th>
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
      <td>-0.940</td>
      <td>1.322</td>
      <td>-3.404</td>
      <td>1.478</td>
      <td>0.093</td>
      <td>0.066</td>
      <td>201.0</td>
      <td>201.0</td>
      <td>199.0</td>
      <td>345.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.198</td>
      <td>0.186</td>
      <td>0.886</td>
      <td>1.558</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1098.0</td>
      <td>1098.0</td>
      <td>1002.0</td>
      <td>1730.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_g</th>
      <td>-0.838</td>
      <td>1.315</td>
      <td>-3.236</td>
      <td>1.647</td>
      <td>0.088</td>
      <td>0.063</td>
      <td>221.0</td>
      <td>221.0</td>
      <td>220.0</td>
      <td>383.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.853</td>
      <td>1.050</td>
      <td>0.014</td>
      <td>2.259</td>
      <td>0.038</td>
      <td>0.027</td>
      <td>773.0</td>
      <td>773.0</td>
      <td>342.0</td>
      <td>332.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>1.649</td>
      <td>1.368</td>
      <td>-0.849</td>
      <td>4.215</td>
      <td>0.096</td>
      <td>0.068</td>
      <td>205.0</td>
      <td>205.0</td>
      <td>204.0</td>
      <td>382.0</td>
      <td>1.02</td>
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
      <th>hpd_3%</th>
      <th>hpd_97%</th>
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
      <td>1.649</td>
      <td>1.368</td>
      <td>-0.849</td>
      <td>4.215</td>
      <td>0.096</td>
      <td>0.068</td>
      <td>205.0</td>
      <td>205.0</td>
      <td>204.0</td>
      <td>382.0</td>
      <td>1.02</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.157</td>
      <td>1.367</td>
      <td>-2.392</td>
      <td>2.688</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>206.0</td>
      <td>206.0</td>
      <td>205.0</td>
      <td>369.0</td>
      <td>1.02</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.037</td>
      <td>1.404</td>
      <td>-3.631</td>
      <td>1.576</td>
      <td>0.093</td>
      <td>0.066</td>
      <td>230.0</td>
      <td>227.0</td>
      <td>229.0</td>
      <td>362.0</td>
      <td>1.02</td>
      <td>COL8A1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.593</td>
      <td>1.358</td>
      <td>-3.099</td>
      <td>1.927</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>205.0</td>
      <td>205.0</td>
      <td>204.0</td>
      <td>358.0</td>
      <td>1.02</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.437</td>
      <td>1.359</td>
      <td>-3.945</td>
      <td>1.081</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>205.0</td>
      <td>205.0</td>
      <td>205.0</td>
      <td>347.0</td>
      <td>1.02</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.542</td>
      <td>1.389</td>
      <td>-1.987</td>
      <td>3.220</td>
      <td>0.097</td>
      <td>0.069</td>
      <td>204.0</td>
      <td>204.0</td>
      <td>203.0</td>
      <td>331.0</td>
      <td>1.02</td>
      <td>COG3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.695</td>
      <td>1.391</td>
      <td>-3.292</td>
      <td>1.852</td>
      <td>0.097</td>
      <td>0.069</td>
      <td>205.0</td>
      <td>205.0</td>
      <td>204.0</td>
      <td>323.0</td>
      <td>1.02</td>
      <td>COG3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.708</td>
      <td>1.357</td>
      <td>-4.227</td>
      <td>0.846</td>
      <td>0.095</td>
      <td>0.068</td>
      <td>202.0</td>
      <td>202.0</td>
      <td>201.0</td>
      <td>343.0</td>
      <td>1.02</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.701</td>
      <td>1.358</td>
      <td>-1.861</td>
      <td>3.227</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>206.0</td>
      <td>206.0</td>
      <td>205.0</td>
      <td>351.0</td>
      <td>1.02</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.278</td>
      <td>1.359</td>
      <td>-2.908</td>
      <td>2.143</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>204.0</td>
      <td>204.0</td>
      <td>203.0</td>
      <td>351.0</td>
      <td>1.03</td>
      <td>KRAS</td>
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
      <th>hpd_3%</th>
      <th>hpd_97%</th>
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
      <td>-0.453</td>
      <td>1.376</td>
      <td>-3.146</td>
      <td>1.990</td>
      <td>0.097</td>
      <td>0.069</td>
      <td>201.0</td>
      <td>201.0</td>
      <td>200.0</td>
      <td>315.0</td>
      <td>1.02</td>
      <td>COG3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-1.236</td>
      <td>1.399</td>
      <td>-3.987</td>
      <td>1.198</td>
      <td>0.093</td>
      <td>0.066</td>
      <td>227.0</td>
      <td>227.0</td>
      <td>225.0</td>
      <td>365.0</td>
      <td>1.02</td>
      <td>COL8A1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.963</td>
      <td>1.355</td>
      <td>-3.530</td>
      <td>1.495</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>203.0</td>
      <td>203.0</td>
      <td>201.0</td>
      <td>350.0</td>
      <td>1.02</td>
      <td>EIF6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.922</td>
      <td>1.345</td>
      <td>-3.294</td>
      <td>1.665</td>
      <td>0.095</td>
      <td>0.067</td>
      <td>200.0</td>
      <td>200.0</td>
      <td>199.0</td>
      <td>341.0</td>
      <td>1.03</td>
      <td>KRAS</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in range(fit3_gene_summary.shape[0]):
    plt.plot(np.repeat(fit3_gene_summary.loc[i, 'gene_symbol'], 2), 
             [fit3_gene_summary.loc[i, 'hpd_3%'], fit3_gene_summary.loc[i, 'hpd_97%']],
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

$$
D_{i|s} \sim N(\mu_{i|s}, \sigma) \\
\mu = c_{i|s} + \bar g_{i|l} - g_{i|jl} \\
c_s \sim N(0, \sigma_c) \\
\bar g_l \sim N(\mu_{\bar g}, \sigma_{\bar g}) \\
g_{jl} \sim N(0, \sigma_g) \\
\sigma_c \sim \text{HalfCauchy}(0, 3) \\
\mu_{\bar g} \sim N(0, 2) \quad \sigma_{\bar g} \sim \text{HalfCauchy}(0, 10) \\
\sigma_g \sim \text{HalfCauchy}(0, 5) \\
\sigma \sim \text{HalfCauchy}(0, 10)
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
start = timer()
d2_m4_file = models_dir / 'd2_m4.cpp'
d2_m4 = pystan.StanModel(file=d2_m4_file.as_posix())
end = timer()
print(f'{(end - start) / 60:.2f} minutes to compile model')
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_fab05e14f81a486812ff3fbe526c78c0 NOW.


    0.74 minutes to compile model



```python
d2_m4_control = {'adapt_delta': 0.999, 
                 'max_treedepth': 20}
d2_m4_fit = d2_m4.sampling(data=d2_m4_data, 
                           iter=3000, warmup=1000, chains=4, 
                           control=d2_m4_control)
```

    WARNING:pystan:Maximum (flat) parameter count (1000) exceeded: skipping diagnostic tests for n_eff and Rhat.
    To run all diagnostics call pystan.check_hmc_diagnostics(fit)
    WARNING:pystan:35 of 8000 iterations ended with a divergence (0.438 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.999 to remove the divergences.



```python
pystan.check_hmc_diagnostics(d2_m4_fit)
```

    WARNING:pystan:35 of 8000 iterations ended with a divergence (0.438 %).
    WARNING:pystan:Try running with adapt_delta larger than 0.999 to remove the divergences.





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
      <th>hpd_3%</th>
      <th>hpd_97%</th>
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
      <td>1.202</td>
      <td>0.192</td>
      <td>0.860</td>
      <td>1.562</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>8617.0</td>
      <td>7747.0</td>
      <td>9899.0</td>
      <td>5735.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_gbar</th>
      <td>-1.708</td>
      <td>0.573</td>
      <td>-2.777</td>
      <td>-0.654</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>3012.0</td>
      <td>3012.0</td>
      <td>3135.0</td>
      <td>3148.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_gbar</th>
      <td>0.831</td>
      <td>0.841</td>
      <td>0.010</td>
      <td>2.220</td>
      <td>0.021</td>
      <td>0.015</td>
      <td>1610.0</td>
      <td>1610.0</td>
      <td>1187.0</td>
      <td>1619.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.534</td>
      <td>0.058</td>
      <td>0.420</td>
      <td>0.636</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3501.0</td>
      <td>3501.0</td>
      <td>3470.0</td>
      <td>4471.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c[0]</th>
      <td>2.612</td>
      <td>0.463</td>
      <td>1.736</td>
      <td>3.486</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>2157.0</td>
      <td>2157.0</td>
      <td>2164.0</td>
      <td>3143.0</td>
      <td>1.0</td>
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
az.plot_ppc(az_d2_m4, data_pairs={'y':'y_pred'}, num_pp_samples=100)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_77_0.png)



```python
az.plot_forest(az_d2_m4, var_names='g')
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_78_0.png)


**To-Do:** 

- show $g_{jl}$ for each gene
- show distribution of intercepts for shRNA $c_s$
- show distribution of values for cell lines


```python

```
