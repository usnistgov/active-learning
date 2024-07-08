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
plot_file = "wasserstein.png"
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
def plot_wasserstein(scores, opt=None, opt_error=None, error_freq=20, ylog=False, fontsize=20):
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='w'
    plt.figure(figsize=(10, 8))
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize) 
    ax = plt.gca()
    matplotlib.rc('font', **dict(size=fontsize))
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
        ax.tick_params(size=10, width=2)
        
    if opt is not None:
        xx = [0, 50, 100, 150, 200]
        yy = [opt] * len(xx)
        ee = [opt_error] * len(xx)
        
        if ylog:
            p = ax.semilogy(xx, yy, 'k--', label='Optimal')
        else:
            p = ax.plot(xx, yy, 'k--', label='Optimal')
        
        ax.errorbar(xx, yy, yerr=ee, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3, fontsize=22)
        ax.tick_params(size=10, width=2)

    #plt.legend(fontsize=16)
    plt.xlabel('Number of samples', fontsize=fontsize)
    plt.ylabel(r'Wasserstein distance', fontsize=fontsize)
   
    return plt, ax
```

```python
data_agg = dict()
keys = sorted(['gsx', 'igs', 'random', 'uncertainty', 'gsy'])
for k in keys:
    data_agg[k] = np.load(os.path.join(work_dir, k + '-wasserstein.npz'))
```

```python
plot_wasserstein(data_agg, error_freq=80)
plt.title('(b)')
plt.savefig(plot_file, dpi=400)
```
