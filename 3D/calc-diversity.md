---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python tags=["parameters"]
pca_input_file = "job_2023-10-04_test_v000/data-pca.npz"
active_input_files = [
    f"job_2023-10-04_test_v000/active_train_save_{i}.npz"
    for i in range(2)
]
work_dir = '.'
```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry, valmap
from toolz.curried import map as map_
import ot
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity
from toolz.curried import itemmap, groupby, get, second, valmap, first
from pymks.fmks.func import sequence
import os
```

```python
import sklearn; sklearn.__version__
```

```python
def entropy(data):
    """Caclulate the entropy for a set of samples
    
    Args:
      data: an (NxM) array of data where N is the samples and M are the features
      
    Returns:
      a single value of entropy for all the samples
    """
   
    return -np.mean(
        #KernelDensity(kernel='gaussian', bandwidth='scott').fit(data).score_samples(data)
        KernelDensity(kernel='gaussian').fit(data).score_samples(data)
    )

def swap_(list_):
    """Swap a list of dictionaries with same keys to be a dictionary of lists
    """
    f = lambda k: (k, list(pluck(k, list_)))
    return pipe(
        list_[0].keys(),
        map_(f),
        list,
        dict
    )

def swap(list_):
    def f(item):
        k0, k1 = item[0].split('_')
        return ((k0, int(k1)), item[1])

    def sort_(arrs):
        return pipe(
            arrs,
            lambda x: sorted(x, key=sequence(get(0), second)),
            pluck(1),
            list
        )

    def rekey_i(data_i):
        return pipe(
            data_i.items(),
            map_(f),
            groupby(lambda item: item[0][0]),
            valmap(sort_)
        )
    
    return pipe(
        list_,
        map_(rekey_i),
        list,
        swap_
    )


@curry
def aggregate(data_active_):
    by_index = lambda index: list(map_(entropy, data_active_[index]))
    fmean = lambda x: dict(mean=x.mean(axis=0), std=x.std(axis=0))
    return pipe(
        len(data_active_),
        range,
        map_(by_index),
        list,
        np.array,
        fmean
    )


```

```python
data_pca = np.load(pca_input_file)['x_data_pca']
data_active = [np.load(file_, allow_pickle=True) for file_ in active_input_files]
```

```python
data_agg = valmap(aggregate, swap(data_active))
```

```python
entropy_all = entropy(data_pca)
```

```python
for k, v in data_agg.items():
    np.savez(os.path.join(work_dir, k + '-diversity.npz'), **v)
np.savez(os.path.join(work_dir, 'entropy.npz'), entropy_all=entropy_all)
```
