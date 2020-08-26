# Preparation of CCLE data


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import janitor

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (8.0, 5.0)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 15

data_dir = Path('../data')
modeling_data_dir = Path('../modeling_data')
```


```python
mutation_df = pd.read_csv(data_dir / 'CCLE_mutation_data.csv').clean_names()
mutation_df.head()
```

    /home/jc604/.conda/envs/demeter2-stan/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3146: DtypeWarning: Columns (25,26,27,31) have mixed types.Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





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
      <th>hugo_symbol</th>
      <th>entrez_gene_id</th>
      <th>ncbi_build</th>
      <th>chromosome</th>
      <th>start_position</th>
      <th>end_position</th>
      <th>strand</th>
      <th>variant_classification</th>
      <th>variant_type</th>
      <th>reference_allele</th>
      <th>...</th>
      <th>iscosmichotspot</th>
      <th>cosmichscnt</th>
      <th>exac_af</th>
      <th>wes_ac</th>
      <th>sangerwes_ac</th>
      <th>sangerrecalibwes_ac</th>
      <th>rnaseq_ac</th>
      <th>hc_ac</th>
      <th>rd_ac</th>
      <th>wgs_ac</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGRN</td>
      <td>375790</td>
      <td>37</td>
      <td>1</td>
      <td>979072</td>
      <td>979072</td>
      <td>+</td>
      <td>Silent</td>
      <td>SNP</td>
      <td>A</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>NaN</td>
      <td>27:24</td>
      <td>9:10</td>
      <td>9:12</td>
      <td>104:20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15:13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ATAD3A</td>
      <td>55210</td>
      <td>37</td>
      <td>1</td>
      <td>1459233</td>
      <td>1459233</td>
      <td>+</td>
      <td>Silent</td>
      <td>SNP</td>
      <td>A</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>0.000008</td>
      <td>29:49</td>
      <td>33:40</td>
      <td>30:38</td>
      <td>315:308</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17:31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NADK</td>
      <td>65220</td>
      <td>37</td>
      <td>1</td>
      <td>1685635</td>
      <td>1685635</td>
      <td>+</td>
      <td>Missense_Mutation</td>
      <td>SNP</td>
      <td>G</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>NaN</td>
      <td>25:39</td>
      <td>16:19</td>
      <td>17:20</td>
      <td>176:266</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14:23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PLCH2</td>
      <td>9651</td>
      <td>37</td>
      <td>1</td>
      <td>2436128</td>
      <td>2436128</td>
      <td>+</td>
      <td>Missense_Mutation</td>
      <td>SNP</td>
      <td>G</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>NaN</td>
      <td>9:20</td>
      <td>19:22</td>
      <td>20:20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23:15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LRRC47</td>
      <td>57470</td>
      <td>37</td>
      <td>1</td>
      <td>3703695</td>
      <td>3703695</td>
      <td>+</td>
      <td>Silent</td>
      <td>SNP</td>
      <td>G</td>
      <td>...</td>
      <td>False</td>
      <td>0</td>
      <td>0.000033</td>
      <td>19:21</td>
      <td>7:19</td>
      <td>8:17</td>
      <td>87:104</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11:16</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
mutation_columns = [
    'tumor_sample_barcode', 'hugo_symbol', 'chromosome', 'start_position', 
    'end_position', 'variant_classification', 'variant_type', 'protein_change'
]
mutation_df = mutation_df[mutation_columns]
```


```python
hotspot_codons = [12, 13, 59, 61, 146]
hotspot_codons_re = '12|13|59|61|146'

kras_mutations = mutation_df \
    .pipe(lambda x: x[x.hugo_symbol == 'KRAS']) \
    .pipe(lambda x: x[x.variant_classification == 'Missense_Mutation']) \
    .pipe(lambda x: x[x.protein_change.str.contains(hotspot_codons_re)]) \
    [['tumor_sample_barcode', 'protein_change']] \
    .drop_duplicates() \
    .rename({'tumor_sample_barcode': 'cell_line'}, axis=1) \
    .groupby('cell_line') \
    .aggregate(lambda x: ';'.join(x.protein_change))
kras_mutations
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
      <th>protein_change</th>
    </tr>
    <tr>
      <th>cell_line</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A427_LUNG</th>
      <td>p.G12D</td>
    </tr>
    <tr>
      <th>A549_LUNG</th>
      <td>p.G12S</td>
    </tr>
    <tr>
      <th>AGS_STOMACH</th>
      <td>p.G12D</td>
    </tr>
    <tr>
      <th>AMO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</th>
      <td>p.A146T</td>
    </tr>
    <tr>
      <th>ASPC1_PANCREAS</th>
      <td>p.G12D</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>SW948_LARGE_INTESTINE</th>
      <td>p.Q61L</td>
    </tr>
    <tr>
      <th>TCCPAN2_PANCREAS</th>
      <td>p.G12R</td>
    </tr>
    <tr>
      <th>TOV21G_OVARY</th>
      <td>p.G13C</td>
    </tr>
    <tr>
      <th>UMUC3_URINARY_TRACT</th>
      <td>p.G12C</td>
    </tr>
    <tr>
      <th>YD8_UPPER_AERODIGESTIVE_TRACT</th>
      <td>p.G138V</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 1 columns</p>
</div>




```python
kras_mutations[kras_mutations.protein_change.str.contains(';')]
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
      <th>protein_change</th>
    </tr>
    <tr>
      <th>cell_line</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NCIH2291_LUNG</th>
      <td>p.G12V;p.G12C</td>
    </tr>
  </tbody>
</table>
</div>




```python
mutation_df.to_csv(modeling_data_dir / 'ccle_mutations.csv', index=False)
kras_mutations.to_csv(modeling_data_dir / 'kras_mutants.csv', index=True)
```


```python

```
