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


    0.81 minutes to compile model



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
      <td>-1.857</td>
      <td>0.055</td>
      <td>-1.954</td>
      <td>-1.748</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1552.0</td>
      <td>1542.0</td>
      <td>1559.0</td>
      <td>1251.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.712</td>
      <td>0.039</td>
      <td>1.644</td>
      <td>1.790</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1593.0</td>
      <td>1593.0</td>
      <td>1599.0</td>
      <td>1284.0</td>
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


    0.86 minutes to compile model



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


    0.74 minutes to compile model



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
      <td>-1.815</td>
      <td>0.263</td>
      <td>-2.280</td>
      <td>-1.302</td>
      <td>0.008</td>
      <td>0.005</td>
      <td>1199.0</td>
      <td>1180.0</td>
      <td>1204.0</td>
      <td>578.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>1.254</td>
      <td>0.189</td>
      <td>0.943</td>
      <td>1.612</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1382.0</td>
      <td>1264.0</td>
      <td>1544.0</td>
      <td>816.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>alpha[0]</th>
      <td>0.698</td>
      <td>0.200</td>
      <td>0.297</td>
      <td>1.058</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1877.0</td>
      <td>1563.0</td>
      <td>1886.0</td>
      <td>703.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>alpha[1]</th>
      <td>-0.793</td>
      <td>0.212</td>
      <td>-1.126</td>
      <td>-0.364</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>1896.0</td>
      <td>1868.0</td>
      <td>1897.0</td>
      <td>848.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>alpha[2]</th>
      <td>-2.265</td>
      <td>0.203</td>
      <td>-2.651</td>
      <td>-1.907</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>2034.0</td>
      <td>1875.0</td>
      <td>2044.0</td>
      <td>648.0</td>
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
az.plot_forest(az_d2_m2, combined=True, var_names=['alpha'])
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_53_0.png)



```python
az.plot_ppc(az_d2_m2, data_pairs={'y':'y_pred'}, num_pp_samples=50)
plt.show()
```


![png](005_demeter2-in-stan_files/005_demeter2-in-stan_54_0.png)



```python
d2_m2_fit
```

    WARNING:pystan:Truncated summary with the 'fit.__repr__' method. For the full summary use 'print(fit)'





    
    Warning: Shown data is truncated to 100 parameters
    For the full summary use 'print(fit)'
    
    Inference for Stan model: anon_model_27c69fe8813455b868fbd0406f6b51bd.
    2 chains, each with iter=1000; warmup=500; thin=1; 
    post-warmup draws per chain=500, total post-warmup draws=1000.
    
                  mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
    mu_alpha     -1.82  7.6e-3   0.26  -2.34  -1.98  -1.82  -1.65   -1.3   1186    1.0
    sigma_alpha   1.25  5.1e-3   0.19   0.95   1.12   1.23   1.36   1.67   1363    1.0
    alpha[1]       0.7  4.6e-3    0.2    0.3   0.57   0.69   0.83   1.11   1877    1.0
    alpha[2]     -0.79  4.5e-3   0.21  -1.19  -0.94  -0.79  -0.65  -0.38   2198    1.0
    alpha[3]     -2.26  4.6e-3    0.2  -2.65   -2.4  -2.26  -2.13  -1.87   1960    1.0
    alpha[4]     -1.51  5.0e-3   0.21  -1.92  -1.66  -1.52  -1.38  -1.08   1692    1.0
    alpha[5]     -2.37  4.8e-3   0.22   -2.8  -2.51  -2.37  -2.22  -1.96   2060    1.0
    alpha[6]      0.07  4.5e-3   0.21  -0.34  -0.07   0.08   0.22   0.49   2253    1.0
    alpha[7]     -1.16  5.8e-3   0.21  -1.59   -1.3  -1.15  -1.02  -0.74   1300    1.0
    alpha[8]     -2.63  4.8e-3   0.21  -3.05  -2.76  -2.63  -2.49  -2.19   1965    1.0
    alpha[9]     -0.21  4.8e-3   0.21  -0.61  -0.36  -0.21  -0.06    0.2   1932    1.0
    alpha[10]     -1.2  4.3e-3   0.21  -1.61  -1.33  -1.19  -1.06  -0.77   2351    1.0
    alpha[11]     0.16  5.1e-3   0.21  -0.2610.0e-3   0.17   0.31   0.59   1731    1.0
    alpha[12]    -2.01  5.1e-3    0.2  -2.41  -2.15  -2.01  -1.87  -1.64   1564    1.0
    alpha[13]    -2.67  4.5e-3   0.19  -3.07   -2.8  -2.67  -2.55   -2.3   1830    1.0
    alpha[14]     -2.8  5.9e-3    0.2  -3.19  -2.94  -2.81  -2.67  -2.42   1166    1.0
    alpha[15]    -1.13  4.4e-3   0.22  -1.54  -1.28  -1.13  -0.98   -0.7   2381    1.0
    alpha[16]    -3.08  5.1e-3   0.21  -3.49  -3.22  -3.07  -2.93  -2.64   1792    1.0
    alpha[17]    -2.25  4.2e-3    0.2  -2.66  -2.38  -2.25  -2.11  -1.86   2332    1.0
    alpha[18]    -1.74  4.3e-3   0.21  -2.15   -1.9  -1.75  -1.59  -1.33   2438    1.0
    alpha[19]    -1.85  4.4e-3    0.2  -2.23  -1.98  -1.84  -1.71  -1.45   2071    1.0
    alpha[20]    -1.73  4.1e-3    0.2  -2.09  -1.87  -1.73  -1.59  -1.34   2269    1.0
    alpha[21]    -3.02  4.0e-3    0.2   -3.4  -3.16  -3.03  -2.89  -2.64   2338    1.0
    alpha[22]    -2.18  4.4e-3    0.2  -2.57  -2.31  -2.18  -2.04   -1.8   2013    1.0
    alpha[23]    -3.19  5.3e-3    0.2  -3.58  -3.32  -3.18  -3.06  -2.79   1400    1.0
    alpha[24]     -2.3  4.5e-3    0.2  -2.68  -2.43   -2.3  -2.17  -1.92   1899    1.0
    alpha[25]     -2.9  4.7e-3    0.2  -3.27  -3.04   -2.9  -2.77  -2.49   1824    1.0
    alpha[26]    -4.16  4.9e-3    0.2  -4.54  -4.29  -4.16  -4.03  -3.73   1702    1.0
    sigma         1.26  6.3e-4   0.03   1.21   1.24   1.26   1.28   1.32   2170    1.0
    y_pred[1]     0.74    0.04   1.28  -1.64  -0.14   0.66   1.65   3.23   1029    1.0
    y_pred[2]     0.73    0.04   1.33  -1.83  -0.15   0.68   1.66   3.39    965    1.0
    y_pred[3]     0.71    0.04   1.27  -1.68   -0.2   0.76   1.58   3.15    941    1.0
    y_pred[4]     0.64    0.04   1.25  -1.93   -0.2   0.63   1.45   3.12    919    1.0
    y_pred[5]     0.64    0.04   1.32  -1.94  -0.25   0.64   1.57   3.25    923    1.0
    y_pred[6]     0.61    0.04    1.3  -1.95  -0.23   0.61   1.51   3.03   1008    1.0
    y_pred[7]     0.73    0.04   1.31  -1.91  -0.14   0.79    1.6   3.26    997    1.0
    y_pred[8]     0.72    0.04   1.26  -1.77  -0.14   0.73    1.6   3.02   1103    1.0
    y_pred[9]     0.74    0.04   1.25  -1.69  -0.09   0.73    1.6   3.22   1144    1.0
    y_pred[10]    0.69    0.04   1.27  -1.81  -0.17   0.69   1.53   3.32   1093    1.0
    y_pred[11]    0.68    0.04   1.26  -1.84  -0.14   0.68   1.54   3.06    946    1.0
    y_pred[12]    0.75    0.04   1.29  -1.85  -0.12   0.76    1.7   3.38    969    1.0
    y_pred[13]    0.73    0.04   1.28  -1.83  -0.05   0.75   1.55   3.34   1112    1.0
    y_pred[14]    0.73    0.04   1.27   -1.8  -0.06    0.7   1.49   3.18   1026    1.0
    y_pred[15]    0.73    0.04   1.26  -1.65  -0.17   0.73   1.55   3.16   1115    1.0
    y_pred[16]    0.69    0.04   1.27  -1.79  -0.14    0.7   1.56   3.18   1016    1.0
    y_pred[17]    0.72    0.04   1.28  -1.77   -0.1   0.68    1.6   3.12   1041    1.0
    y_pred[18]    0.75    0.04   1.25  -1.79  -0.06   0.75   1.56   3.21   1104    1.0
    y_pred[19]    0.67    0.04    1.3  -1.84  -0.13   0.66   1.59   3.13    965    1.0
    y_pred[20]    0.76    0.04   1.25  -1.71   -0.1   0.79   1.56   3.26   1066   1.01
    y_pred[21]    0.68    0.04   1.25  -1.78  -0.16   0.68   1.44   3.14   1105    1.0
    y_pred[22]    0.64    0.04   1.24  -1.77  -0.19    0.6   1.43   3.11    841    1.0
    y_pred[23]    0.62    0.04   1.33   -1.9  -0.36   0.65   1.56   3.23    948    1.0
    y_pred[24]    0.72    0.04   1.31  -1.88  -0.23   0.75   1.53    3.2   1057    1.0
    y_pred[25]    0.62    0.04   1.29  -1.91  -0.25   0.64   1.48   3.13    861    1.0
    y_pred[26]    0.66    0.04   1.32  -2.07  -0.22   0.65   1.55    3.3    952    1.0
    y_pred[27]    0.77    0.04   1.29  -1.83  -0.09    0.8   1.66   3.23    918    1.0
    y_pred[28]    0.72    0.04   1.25  -1.59  -0.15    0.7   1.57   3.15    912    1.0
    y_pred[29]    0.68    0.04   1.28  -1.81  -0.17   0.72   1.51   3.21   1114    1.0
    y_pred[30]    0.62    0.04   1.22   -1.8  -0.19   0.57   1.46   2.98    993    1.0
    y_pred[31]    0.67    0.04   1.29  -1.82  -0.21   0.66   1.56   3.19    973    1.0
    y_pred[32]    0.67    0.04   1.27  -1.84  -0.19    0.7   1.55   3.14   1029    1.0
    y_pred[33]    0.69    0.04   1.28   -1.7   -0.2   0.66    1.5   3.38   1012    1.0
    y_pred[34]     0.7    0.04   1.24  -1.79  -0.08   0.74   1.52   3.05   1006    1.0
    y_pred[35]    0.69    0.04    1.3  -1.85  -0.19   0.72   1.62   3.09    969    1.0
    y_pred[36]    0.63    0.04   1.27  -1.87  -0.19   0.62   1.51   3.07   1040    1.0
    y_pred[37]    0.73    0.04   1.28  -1.69  -0.15   0.73   1.63   3.21    881    1.0
    y_pred[38]   -0.74    0.04   1.24  -3.19  -1.52  -0.78   0.12   1.67   1172    1.0
    y_pred[39]   -0.81    0.04    1.3  -3.29  -1.66  -0.82   0.02   1.75   1068    1.0
    y_pred[40]   -0.72    0.04   1.25  -3.17  -1.55  -0.69   0.12   1.67    871    1.0
    y_pred[41]   -0.77    0.04   1.26  -3.27  -1.61   -0.8   0.07   1.67    852    1.0
    y_pred[42]    -0.8    0.04    1.3  -3.36  -1.64  -0.81   0.02   1.91   1054    1.0
    y_pred[43]   -0.84    0.04   1.29  -3.35  -1.75  -0.81   0.06    1.7    964    1.0
    y_pred[44]   -0.77    0.04   1.32  -3.35  -1.72  -0.78   0.15   1.81    878    1.0
    y_pred[45]   -0.72    0.04   1.27  -3.16  -1.64   -0.7   0.15   1.81    849    1.0
    y_pred[46]   -0.76    0.04    1.3   -3.5  -1.58  -0.72   0.09   1.64   1026    1.0
    y_pred[47]   -0.77    0.05   1.29  -3.09  -1.66  -0.81   0.12   1.72    701    1.0
    y_pred[48]   -0.86    0.04   1.25  -3.33  -1.68  -0.92-1.8e-3   1.62    958    1.0
    y_pred[49]   -0.75    0.04   1.29  -3.22   -1.6  -0.75   0.12   1.83    978    1.0
    y_pred[50]   -0.85    0.05   1.26  -3.47  -1.66  -0.84  -0.02   1.59    749    1.0
    y_pred[51]   -0.73    0.04   1.24  -3.17  -1.56  -0.78   0.06   1.78   1025    1.0
    y_pred[52]   -0.85    0.04   1.26  -3.26  -1.73  -0.87-1.0e-3   1.56   1063    1.0
    y_pred[53]    -0.8    0.04   1.27  -3.28  -1.64  -0.75   0.02   1.66    958    1.0
    y_pred[54]   -0.79    0.04   1.24  -3.31  -1.63  -0.77   0.01   1.67   1079    1.0
    y_pred[55]   -0.87    0.04   1.32  -3.33  -1.79  -0.88 3.1e-3   1.78    942    1.0
    y_pred[56]   -0.84    0.04   1.29  -3.23  -1.75  -0.85   0.02   1.76   1017    1.0
    y_pred[57]   -0.84    0.04   1.25  -3.16  -1.69  -0.86 5.1e-3   1.68   1022    1.0
    y_pred[58]   -0.78    0.04   1.26  -3.21  -1.65   -0.8    0.1   1.65   1037    1.0
    y_pred[59]   -0.79    0.04   1.26  -3.29  -1.61  -0.81   0.03   1.72    782    1.0
    y_pred[60]   -0.82    0.04   1.27   -3.3  -1.67  -0.79  -0.01   1.62   1004    1.0
    y_pred[61]   -0.79    0.04   1.23  -3.19  -1.65  -0.77-8.8e-3   1.67   1014    1.0
    y_pred[62]   -0.85    0.04   1.25  -3.27  -1.72   -0.8-9.2e-3   1.52   1022    1.0
    y_pred[63]   -0.86    0.04   1.29  -3.35  -1.71  -0.89   0.02   1.67    921    1.0
    y_pred[64]   -0.79    0.04   1.27  -3.34  -1.68  -0.77   0.05   1.86    847    1.0
    y_pred[65]   -0.84    0.04   1.33   -3.3   -1.8  -0.86   0.11   1.83   1041    1.0
    y_pred[66]   -0.83    0.04    1.3  -3.43  -1.68  -0.84   0.04   1.74   1019    1.0
    y_pred[67]    -0.8    0.04   1.27  -3.26  -1.65  -0.83   0.07   1.69    895    1.0
    y_pred[68]    -0.8    0.04   1.32  -3.35  -1.68  -0.85   0.12   1.84   1055    1.0
    y_pred[69]   -0.82    0.04   1.27  -3.39  -1.68   -0.8   0.02   1.78   1024    1.0
    y_pred[70]   -0.84    0.04    1.3   -3.4  -1.77  -0.88   0.07   1.71    915    1.0
    lp__        -722.6    0.21   3.75 -730.6 -724.9 -722.3 -719.8 -716.2    315   1.01
    
    Samples were drawn using NUTS at Sun Aug 23 09:18:11 2020.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).




```python

```
