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

# 2D PCA Analysis

```python tags=["parameters"]
input_file = 'data-gen/data-500.npz'
output_file = 'data-pca.npz'
```

```python
import numpy as np
import dask.array as da
from sklearn.pipeline import Pipeline

from pymks import (
    PrimitiveTransformer,
    TwoPointCorrelation,
    GenericTransformer,
)

#from dask_ml.decomposition import IncrementalPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
```

```python
def pca_steps():
    return (
        ("discritize",PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0)),
        ("correlations",TwoPointCorrelation(periodic_boundary=True, cutoff=25, correlations=[(0, 0)])),
        ('flatten', GenericTransformer(lambda x: x.reshape(x.shape[0], -1))),
        ('pca', KernelPCA(n_components=15)),
    )

def make_pca_model():
    return Pipeline(steps=pca_steps())
```

```python
data = np.load(input_file)
x_data = data['x_data']
y_data = data['y_data'].reshape(-1)
print(x_data.shape)
print(y_data.shape)
```

```python
x_data_da = da.from_array(x_data, chunks=(50, 51, 51))
model = make_pca_model()
x_data_pca = model.fit_transform(x_data_da)
```

```python
print(x_data_pca.shape)
print(y_data.shape)
```

```python
np.savez(output_file, x_data_pca=x_data_pca, y_data=y_data)
```

```python

```

```python

```

```python

```
