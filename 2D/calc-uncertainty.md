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
uncertainty_input_files = [
    f"job_2023-10-03_test-uncertainty_v000/active_uncertainty_{i}.npz"
    for i in range(1)
]
work_dir = "."
```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry, valmap, identity
from toolz.curried import map as map_
import ot
import matplotlib.pyplot as plt
import matplotlib
import os
```

```python
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
def aggregate(data_active_):
    fmean = lambda x: dict(mean=x.mean(axis=0), std=x.std(axis=0))
    return pipe(
        data_active_,
        np.array,
        fmean
    )
```

```python

data_uncertainty = [np.load(file_, allow_pickle=True) for file_ in uncertainty_input_files]
```

```python
data_agg = valmap(aggregate, swap(data_uncertainty))
```

```python
for k, v in data_agg.items():
    np.savez(os.path.join(work_dir, k + '-uncertainty.npz'), **v)
```
