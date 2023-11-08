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
#active_input_files = [
#    "job_2023-09-08_test-merge_v000/active_train_save_0.npz",
#    "job_2023-09-08_test-merge_v000/active_train_save_1.npz",
#    "job_2023-09-08_test-merge_v000/active_train_save_2.npz",
#    "job_2023-09-08_test-merge_v000/active_train_save_3.npz",
#    "job_2023-09-08_test-merge_v000/active_train_save_4.npz",
#    "job_2023-09-08_test-merge_v000/active_train_save_5.npz"
#]
plot_file = "uncertainty.png"
```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry, valmap, identity
from toolz.curried import map as map_
import ot
import matplotlib.pyplot as plt
import matplotlib
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
def plot_diversity(scores, opt=None, error_freq=20, ylog=False):
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
        
        if ylog:
            p = ax.semilogy(xx, yy, 'k--', label='Optimal')
        else:
            p = ax.plot(xx, yy, 'k--', label='Optimal')
        
    plt.legend(fontsize=16)
    plt.xlabel('N (queries)', fontsize=16)
    plt.ylabel(r'Uncertainty', fontsize=16)
   
    return plt, ax
```

```python

plot_diversity(data_agg, ylog=False, error_freq=100)
plt.title('Uncertainty for 3D Composite')
plt.savefig(plot_file, dpi=200)
```

```python

```
