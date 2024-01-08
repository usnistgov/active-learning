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
output_file = 'plot.png'
overall_input_file = "overall-accuracy.npz"
scoring = 'mse'
ylog = False
work_dir = "."
input_file = 'data/data_pca-500-51-51.npz'
```

```python
import numpy as np
from toolz.curried import merge_with
import matplotlib.pyplot as plt
import matplotlib
import os
```


```python
def plot_scores(scores, opt=None, opt_error=None, error_freq=20, scoring='mse', ylog=False, scale=1.0):

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
            p = ax.semilogy(x, y * scale, label=names[k][0], lw=3, linestyle=names[k][1])
        else:
            p = ax.plot(x, y * scale, label=names[k][0], lw=3, linestyle=names[k][1])
        
        e = v['std']
        xe, ye, ee = x[offset::error_freq], y[offset::error_freq], e[offset::error_freq]
        ax.errorbar(xe, ye * scale, yerr=ee * scale, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3)
        offset += 5
        
    if opt is not None:
        xx = [0, 400, 800, 1200, 1600]
        yy = [opt * scale] * len(xx)
        ee = [opt_error * scale] * len(xx)
        
        if ylog:
            p = ax.semilogy(xx, yy, 'k--', label='Optimal')
        else:
            p = ax.plot(xx, yy, 'k--', label='Optimal')
            
        ax.errorbar(xx, yy, yerr=ee, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3)

    plt.legend(fontsize=16)
    plt.xlabel('Number of samples', fontsize=16)
    ylabel = dict(mse=r'MSE', mae=r'MAE', r2=r'$R^2$')[scoring]
    plt.ylabel(ylabel, fontsize=16)
    
    ylim = dict(r2=(0.4, 1), mae=None, mse=None)[scoring]
    if ylim is not None:
        plt.ylim(*ylim)
    
    return plt, ax
```

```python
output = dict()
keys = sorted(['gsx', 'igs', 'random', 'uncertainty', 'gsy'])
for k in keys:
    output[k] = np.load(os.path.join(work_dir, k + '-curve.npz'))

```

```python
overall_scores = np.load(overall_input_file)['test_scores']
opt = np.mean(overall_scores)
err = np.std(overall_scores)
```

```python
data = np.load(input_file)
y_data = data['y_data']
scale = 2 / np.std(y_data)
```


```python
plt, ax = plot_scores(output, error_freq=100, opt=opt, opt_error=err, scoring=scoring, ylog=ylog, scale=scale)
plt.title('(a)')
plt.savefig(output_file, dpi=200)
```

```python

```

```python

```
