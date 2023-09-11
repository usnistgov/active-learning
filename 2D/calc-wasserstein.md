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

```python
pca_input_file = "data-pca.npz"
active_input_files = [
    "job_2023-09-08_test-merge_v000/active_train_save_0.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_1.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_2.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_3.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_4.npz",
    "job_2023-09-08_test-merge_v000/active_train_save_5.npz"
]
plot_file = "representation.png"
```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry
from toolz.curried import map as map_
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



```

```python
data_pca = np.load(input_file)['x_data_pca']
data_active = [np.load(file_, allow_pickle=True) for file_ in active_input_files]
```

```python
data_active_ = swap(data_active)
```

```python
list(map_(wasserstein(data_pca), data_active_['igs'][0]))
```

```python
len(data_active_['gsx'])
```

```python
data = np.load(input_file)
x_data_pca = data['x_data_pca']
y_data = data['y_data']
```

```python
x_data_pca.shape
```

```python
active_data = np.load(active_data_file, allow_pickle=True)
```

```python
active_data['gsx'].shape
```

# Action items from Hao meet

 - install pot package

 - get this notebook working with sample data
 
 - https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html

 - https://github.com/hliu56/Active-Learning-Using-various-representations/blob/main/%5Bupload%5D-AL_Martern_MAE_POT.ipynb
 
 - push this branch to github and let Hao know
 
 - For next meeting on Wednesday get conda build working for 2D

```python
import ot
```

```python
active_data['gsx'][0].shape
```

```python
x_data_pca.shape
```

```python
result = ot.dist(x_data_pca, active_data['gsx'][0])
```

```python
result.shape
```

```python
n = 50  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

# loss matrix
M = ot.dist(xs, xt)
```

```python
def optimal_transport(df1, df2):
    
    xx0 = df1
    xx1 = df2
   
    a0, a1 = np.ones((xx0.shape[0]),) / xx0.shape[0], np.ones((xx1.shape[0]),) / xx1.shape[0]

    M0=ot.dist(xx0,xx1)
    M0/=M0.max()
    d_emd0 = ot.emd2(a0, a1, M0)

    return d_emd0
```

```python
optimal_transport(x_data_pca, active_data['gsx'][0])
```

```python
active_data_file = "job_2023-09-08_test-merge_v000/active_train_save_{index}.npz"
```

```python
active_data_ = [
    np.load(active_data_file.format(index=i), allow_pickle=True)
    for i in range(20)
]
```

```python
active_data_[0]
```

```python
from toolz.curried import merge_with

def merge_func(x):
    return dict(
        mean=np.mean(x, axis=0),
        std=np.std(x, axis=0),
        scores=np.array(x)
    )

outputs = dict()

for k in active_data_[0].keys():
    outputs_ = []
    for j in range(100):
        loads = [active_data_[i][k][j] for i in range(20)]
        output = merge_with(merge_func, *loads)
        outputs_.append(output)
```

```python
active_data_[0]['gsx'].shape
```

```python
values = dict()
dim = 3
for k in active_data.keys():
    values[k] = [
        optimal_transport(x_data_pca[:, :dim], y[:, :dim])
        for y in active_data[k]
    ]
```

```python
import matplotlib.pyplot as plt

for k in values.keys():
    if k == 'igs':
        lt='dashed'
    else:
        lt='solid'
    plt.plot(values[k], label=k, ls=lt)
plt.legend()
```

```python

```
