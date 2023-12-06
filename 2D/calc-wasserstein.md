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
pca_input_file = "data-pca.npz"
active_input_files = [
    "job_2023-09-08_test-merge_v000/active_train_save_0.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_1.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_2.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_3.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_4.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_5.npz"
]
work_dir = "."
```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry, valmap
from toolz.curried import map as map_
import ot
import matplotlib.pyplot as plt
import matplotlib
import os
```

```python
@curry
def wasserstein(arr1, arr2):
    """Calculate the Wasserstein distance between two arrays
    
    Args:
      arr1: an (n x m) array
      arr1: an (l x m) array
      
    Second index must be equal length
    """
    f = lambda x: np.ones((x.shape[0]),) / x.shape[0]
    g = lambda x: x / x.max()
    return ot.emd2(
        f(arr1),
        f(arr2),
        g(ot.dist(arr1, arr2))
    )


def swap(list_):
    """Swap a list of dictionaries with same keys to be a dictionary of lists
    """
    f = lambda k: (k, list(pluck(k, list_)))
    return pipe(
        list_[0].keys(),
        map_(f),
        list,
        dict
    )


@curry
def aggregate(data_pca, data_active_):
    by_index = lambda index: list(map_(wasserstein(data_pca), data_active_[index]))
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
data_agg = valmap(aggregate(data_pca), swap(data_active))
```

```python
for k, v in data_agg.items():
    np.savez(os.path.join(work_dir, k + '-wasserstein.npz'), **v)

```

