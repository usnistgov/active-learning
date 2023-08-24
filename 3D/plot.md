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
output_file = 'plot.png'
overall_input_file = "overall-accuracy.npz"
scoring = 'mse'
```

```python
import hdfdict
import numpy as np
from toolz.curried import merge_with
import matplotlib.pyplot as plt
import matplotlib
```

```python
data_list = [hdfdict.load(input_file) for input_file in input_files]
```

```python
def merge_func(x):
    return dict(
        mean=np.mean(x, axis=0),
        std=np.std(x, axis=0),
        scores=np.array(x)
    )

output = merge_with(merge_func, *data_list)

```

```python
def plot_scores(scores, opt=None, opt_error=None, error_freq=20, scoring='mse'):

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
        p = ax.plot(x, y, label=names[k][0], lw=3, linestyle=names[k][1])
        e = v['std']
        xe, ye, ee = x[offset::error_freq], y[offset::error_freq], e[offset::error_freq]
        ax.errorbar(xe, ye, yerr=ee, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3)
        offset += 5
        
    if opt is not None:
        xx = [0, 100, 200, 300, 400]
        yy = [opt] * 5
        ee = [opt_error] * 5
        p = ax.plot(xx, yy, 'k--', label='Optimal')
        ax.errorbar(xx, yy, yerr=ee, alpha=0.5, ls='none', ecolor=p[-1].get_color(), elinewidth=3, capsize=4, capthick=3)

    plt.legend(fontsize=16)
    plt.xlabel('N (queries)', fontsize=16)
    ylabel = r'MSE' if (scoring == 'mse') else r'$R^2$'
    plt.ylabel(ylabel, fontsize=16)
    if scoring == 'r2':
        plt.ylim(0.4, 1)
    
    return plt, ax
```

```python
overall_scores = np.load(overall_input_file)['test_scores']
opt = np.mean(overall_scores)
err = np.std(overall_scores)
```

```python
plt, ax = plot_scores(output, error_freq=40, opt=opt, opt_error=err, scoring=scoring)
plt.title('Active Learning Curves for 3D Composite')
plt.savefig(output_file, dpi=200)
```

```python

```

```python

```
