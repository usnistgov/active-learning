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
input_file = 'data/data_pca-500-51-51.npz'
scoring = 'r2'
n_iterations = 40
output_file = 'overall-accuracy.npz'
nu = 0.5
```

```python
from active import split, make_gp_model_matern
import numpy as np
```

```python
data = np.load(input_file)
x_data_pca = data['x_data_pca']
y_data = data['y_data']
```

```python
test_scores = []
train_scores = []

for i in range(n_iterations):
    print(i)
    x_pool, x_test, x_train, y_pool, y_test, y_train = split(x_data_pca, y_data, (0.8, 0.2))
    model = make_gp_model_matern(scoring, nu=nu)
    model.fit(x_pool, y_pool)
    train_score = model.score(x_pool, y_pool)
    test_score = model.score(x_test, y_test)
    test_scores += [test_score]
    train_scores += [train_score]
```

```python
np.savez(output_file, test_scores=test_scores)
```
