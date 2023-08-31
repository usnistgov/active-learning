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

# Test Model

```python
import numpy as np

from active import split_on_ids
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from pymks import (
    PrimitiveTransformer,
    TwoPointCorrelation,
    GenericTransformer,
)

from dask_ml.decomposition import IncrementalPCA
from dask_ml.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

```python
def make_gp_model_matern():
    kernel = Matern(length_scale=1.0, nu=0.5)
#    kernel = 0.5 * RBF(length_scale=1) + WhiteKernel(noise_level=1)
    regressor = GaussianProcessRegressor(kernel=kernel)
    return regressor

def make_linear_model():
    return Pipeline(steps=(
        ('poly', PolynomialFeatures(degree=3)),
        ('regressor', LinearRegression()),
    ))

```

```python
def plot_parity(y_test, y_predict, label='Testing Data'):
    pred_data = np.array([y_test, y_predict])
    line = np.min(pred_data), np.max(pred_data)
    plt.plot(pred_data[0], pred_data[1], 'o', label=label)
    plt.plot(line, line, '-', linewidth=3, color='k')
    plt.title('Goodness of Fit', fontsize=20)
    plt.xlabel('Actual', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.legend(loc=2, fontsize=15)
    return plt
```

```python
def train_test_split_(x_data, y_data, prop, random_state=None):
    ids = np.random.choice(len(x_data), int(prop * len(x_data)), replace=False)
    x_0, x_1 = split_on_ids(x_data, ids)
    y_0, y_1 = split_on_ids(y_data, ids)
    return x_0, x_1, y_0, y_1
```

```python
def split(x_data, y_data, train_sizes=(0.9, 0.09), random_state=None):
    x_pool, x_, y_pool, y_ = train_test_split_(
        x_data,
        y_data,
        train_sizes[0],
        random_state=random_state
    )
    x_test, x_calibrate, y_test, y_calibrate = train_test_split_(
        x_,
        y_,
        train_sizes[1] / (1 - train_sizes[0]),
        random_state=random_state
    ) 
    return x_pool, x_test, x_calibrate, y_pool, y_test, y_calibrate
```

```python

```

```python
data = np.load('data-pca.npz')
```

```python
x_data_pca = data['x_data_pca'][:]
y_data = data['y_data']
```

```python
x_pool, x_test, x_train, y_pool, y_test, y_train = split(x_data_pca, y_data, (0.6, 0.2))
```

```python
x_train.shape
```

```python
x_test.shape
```

```python
#model = make_linear_model()
model = make_gp_model_matern()
```

```python
model.fit(x_train, y_train)
```

```python
y_train_predict = model.predict(x_train)
```

```python
y_test_predict = model.predict(x_test)
```

```python
plot_parity(y_train, y_train_predict)
```

```python
plot_parity(y_test, y_test_predict)
```

```python
print(r2_score(y_test, y_test_predict))
#print(y_test.shape)
#print(y_test)
print(model.score(x_test, y_test))
#sklearn.metrics.r2_score()
```

```python

```

```python
scores = []
for _ in range(40):
    x_pool, x_test, x_train, y_pool, y_test, y_train = split(x_data_pca, y_data, (0.0, 0.2))
    model = make_gp_model_matern()
    model.fit(x_train, y_train)
    print(x_train.shape)
    scores += [model.score(x_test, y_test)]
    
print(scores)
```

```python
np.mean(scores)
```

```python
np.std(scores)
```

```python

```
