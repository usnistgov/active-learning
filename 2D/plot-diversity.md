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
plot_file = "diversity.png"
work_dir = "."
```

```python
import numpy as np
from toolz.curried import pipe, pluck, curry, valmap
from toolz.curried import map as map_
import ot
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity
import os
```

```python
import sklearn; sklearn.__version__
```

```python
def plot_diversity(scores, opt=None, error_freq=80, ylog=False):
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='w'
    plt.figure(figsize=(10, 8))
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14) 
    ax = plt.gca()
    matplotlib.rc('font', **dict(size=16))
    names = dict(
        uncertainty=('Uncertainty sampling', 'solid'),
        random=("Random sampling", 'dotted'),
        gsx=("GSx", 'dashed'),
        gsy=("GSy", 'dashdot'),
        igs=("iGS", (5, (10, 3)))
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
        xx = [0, 200, 400, 600, 800]
        yy = [opt] * len(xx)
        
        if ylog:
            p = ax.semilogy(xx, yy, 'k--', label='Optimal')
        else:
            p = ax.plot(xx, yy, 'k--', label='Optimal')
        
    plt.legend(fontsize=16)
    plt.xlabel('Number of samples', fontsize=16)
    plt.ylabel(r'Entropy', fontsize=16)
   
    return plt, ax
```

```python
data_agg = dict()
keys = sorted(['gsx', 'igs', 'random', 'uncertainty', 'gsy'])
for k in keys:
    data_agg[k] = np.load(os.path.join(work_dir, k + '-diversity.npz'))
entropy_all = np.load(os.path.join(work_dir, 'entropy.npz'))['entropy_all']
```

```python

plot_diversity(data_agg, opt=entropy_all)
plt.title('(c)')
plt.savefig(plot_file, dpi=200)
```
