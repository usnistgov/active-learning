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
pca_input_file =  'job_2023-09-25_wasserstein_v000/data-pca.npz'
active_input_files = [
    'job_2023-09-25_wasserstein_v000/active_train_save_0.npz'
]
plot_file = "wasserstein.png"
```

```python

```

```python

```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry, valmap
from toolz.curried import map as map_
import ot
import matplotlib.pyplot as plt
import matplotlib
from toolz.curried import itemmap, groupby, get, second, valmap, first
from pymks.fmks.func import sequence
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
    
    return ot.sinkhorn2(
        f(arr1),
        f(arr2),
        g(ot.dist(arr1, arr2)),
        1.0
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
def plot_wasserstein(scores, opt=None, opt_error=None, error_freq=20, ylog=False):
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='w'
    plt.figure(figsize=(10, 8))
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14) 
    ax = plt.gca()
    matplotlib.rc('font', **dict(size=16))
    names = dict(
        uncertainty=('Uncertainty', 'solid'),
        random=("Random", 'dotted'),
        gsx=("GSX", 'dashed'),
        gsy=("GSY", 'dashdot'),
        igs=("IGS", (5, (10, 3)))
    )

    offset = 10
    for k, v in scores.items():
        y = v['mean']
        x = np.arange(len(y))
        
        if ylog:
            p = ax.semilogy(x, y, label=names[k][0], lw=3, linestyle=names[k][1])
        else:
            p = ax.plot(x, y, label=names[k][0], lw=3, linestyle=names[k][1])
        
        e = v['std']
        xe, ye, ee = x[offset::error_freq], y[offset::error_freq], e[offset::error_freq]
        ax.errorbar(xe, ye, yerr=ee, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3)
        offset += 5
        
    if opt is not None:
        xx = [0, 50, 100, 150, 200]
        yy = [opt] * len(xx)
        ee = [opt_error] * len(xx)
        
        if ylog:
            p = ax.semilogy(xx, yy, 'k--', label='Optimal')
        else:
            p = ax.plot(xx, yy, 'k--', label='Optimal')
        
        ax.errorbar(xx, yy, yerr=ee, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3)

    plt.legend(fontsize=16)
    plt.xlabel('N (queries)', fontsize=16)
    plt.ylabel(r'Wasserstein Distance', fontsize=16)
   
    return plt, ax
```

```python
plot_wasserstein(data_agg)
plt.title('Wasserstein Distances for 2D Composite')
plt.savefig(plot_file, dpi=200)
```

```python

```
