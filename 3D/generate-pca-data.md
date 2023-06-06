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

# Generate PCA Data

```python
import numpy as np
from pymks import plot_microstructures, GenericTransformer, PrimitiveTransformer, TwoPointCorrelation
import dask.array as da
from sklearn.pipeline import Pipeline
from dask_ml.decomposition import IncrementalPCA
```

```python
def pca_steps():
    return (
        ("reshape", GenericTransformer(
            lambda x: x.reshape(x.shape[0], 51, 51,51)
        )),    
        ("discritize",PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0)),
        ("correlations",TwoPointCorrelation(periodic_boundary=True, cutoff=20, correlations=[(0, 0)])),
        ('flatten', GenericTransformer(lambda x: x.reshape(x.shape[0], -1))),
        ('pca', IncrementalPCA(n_components=15)),
    )

def make_pca_model():
    return Pipeline(steps=pca_steps())
```

```python
data = np.load('data_shuffled.npz')
```

```python
x_data = data['x_data']
y_data = data['y_data'].reshape(-1)
```

```python
print(x_data.shape)
print(y_data.shape)
```

```python
plot_microstructures(*x_data[:10].reshape(10, 51, 51, 51)[:, :, :, 0], cmap='gray', colorbar=False)
```

```python
x_data_da = da.from_array(x_data, chunks=(100, -1))
```

```python
model = make_pca_model()
```

```python
x_data_pca = model.fit_transform(x_data_da).compute()
```

```python
np.savez('data_pca_test.npz', x_data_pca=x_data_pca, y_data=y_data)
```
