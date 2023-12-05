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
input_files = ['active_data.h5']
work_dir = "."
```

```python
import numpy as np
from toolz.curried import merge_with
import matplotlib.pyplot as plt
import matplotlib
import os
```

```python
data_list = [np.load(input_file) for input_file in input_files]
```

```python
def merge_func(x):
    return dict(
        mean=np.mean(x, axis=0),
        std=np.std(x, axis=0),
        scores=np.array(x)
    )

output = merge_with(merge_func, *data_list)
print('hello')
```

```python
print('got here')
for k, v in output.items():
    print('k:', k)
    print('v:', v.keys())
    np.savez(os.path.join(work_dir, k + '-curve.npz'), **v)
```

