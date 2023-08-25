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

# Active learning

```python tags=["parameters"]
input_file = 'data/data_pca-500-51-51.npz'
n_query = 100
n_iterations = 5
output_file = 'data/output_500-51-51.h5'
```

```python
import numpy as np

from active import split_on_ids
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from toolz.curried import curry, merge_with
from active import next_sample_gsx, next_sample_igs
from tqdm.notebook import trange, tqdm
import hdfdict
import matplotlib
from dask_ml.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from toolz.curried import map as fmap

import hdfdict
import h5py
```

```python
def make_gp_model_matern():
    kernel = Matern(length_scale=1.0, nu=0.5)
    regressor = GaussianProcessRegressor(kernel=kernel)
    return regressor
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
def query_uncertainty(model, x_pool, y_pool, batch_size):
    stds = model.predict(x_pool, return_std=True)[1]
    ids = np.argsort(stds)[::-1][:batch_size]
    return rework_pool(x_pool, y_pool, ids)

def query_random(model, x_pool, y_pool, batch_size):
    ids = np.random.choice(len(x_pool), batch_size, replace=False)
    return rework_pool(x_pool, y_pool, ids)


def evaluate_model(x_pool, x_test, x_train, y_pool, y_test, y_train, model, query_func, batch_size):
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    x_, x_pool, y_, y_pool = query_func(model, x_pool, y_pool, batch_size)
    x_train = np.vstack([x_train, x_])
    y_train = np.append(y_train, y_)
    return model, x_pool, x_train, y_pool, y_train, test_score, train_score
```

```python

def query_helper(model, x_pool, y_pool, init_scores, update_scores, next_func):
    if not hasattr(model, 'query_data'):
        model.query_data = [], init_scores()
    labeled_samples, scores = model.query_data
    scores = update_scores(model, scores)
    next_id = next_func(labeled_samples, scores)
    model.query_data = (labeled_samples + [next_id], scores)
    x_, _, y_, _ = rework_pool(x_pool, y_pool, [next_id])
    return x_, x_pool, y_, y_pool


def gsx_query(model, x_pool, y_pool, batch_size):
    return query_helper(
        model,
        x_pool,
        y_pool,
        lambda: x_pool,
        lambda m, s: s,
        next_sample_gsx
    )

def gsy_query(model, x_pool, y_pool, batch_size):
    return query_helper(
        model,
        x_pool,
        y_pool,
        lambda: None,
        lambda m, s: m.predict(x_pool).reshape(-1, 1),
        next_sample_gsx
    )


def igs_query(model, x_pool, y_pool, batch_size):
    return query_helper(
        model,
        x_pool,
        y_pool,
        lambda: (x_pool, None),
        lambda m, s: (s[0], m.predict(x_pool).reshape(-1, 1)),
        next_sample_igs
    )
```

```python
def merge_func(x):
    return dict(
        mean=np.mean(x, axis=0),
        std=np.std(x, axis=0),
        scores=np.array(x)
    )

def run_multiple(x_data_pca, y_data, train_sizes, learners, n_query, n_iter, batch_size):
    _ = fmap(lambda _: run_all(x_data_pca, y_data, train_sizes, learners, n_query, batch_size), trange(n_iter, position=0, desc='outer loop'))
    all_data = list(_)
    return merge_with(merge_func, *all_data)
```

```python
def run_all(x_data_pca, y_data, train_sizes, learners, n_query, batch_size):
    data = split(x_data_pca, y_data, train_sizes)
    test_scores = dict()
    for k in tqdm(learners, position=1, desc="learner loop"):
        test_scores[k] = run(data, learners[k][0], learners[k][1], n_query, batch_size=batch_size)[1]
    return test_scores
```

```python
def run(data, query_func, model_func, n_iter, train_sizes=(0.87, 0.004), batch_size=1):
    x_pool, x_test, x_train, y_pool, y_test, y_train = data
    print('x_train.shape', x_train.shape)
    print('x_test.shape:', x_test.shape)
    model = model_func()
    train_scores = []
    test_scores = []
    
#    for k in tqdm.tqdm(learners):
#        test_scores[k] = run(data, learners[k][0], learners[k][1], n_query)[1]
    
    for _ in trange(n_iter, position=2, desc='iter loop'):
#        model_ = model
#        model = model_func()
#        if hasattr(model_, "query_data"):
#            model.query_data = model_.query_data
        model, x_pool, x_train, y_pool, y_train, test_score, train_score  = evaluate_model(
            x_pool, x_test, x_train, y_pool, y_test, y_train,
            model, 
            query_func,
            batch_size
        )
        
        train_scores += [train_score] * batch_size
        test_scores += [test_score] * batch_size
       
    return train_scores, test_scores
    
```

```python
def rework_pool(x_pool, y_pool, ids):
    x_, x_pool = split_on_ids(x_pool, ids)
    y_, y_pool = split_on_ids(y_pool, ids)
    return x_, x_pool, y_, y_pool
```

# Load data and run the active learning

```python
data = np.load(input_file)
```

```python
x_data_pca = data['x_data_pca']
y_data = data['y_data']
```

```python
learners_gp = dict(
    uncertainty=(query_uncertainty, make_gp_model_matern),
    random=(query_random, make_gp_model_matern),
    gsx=(gsx_query, make_gp_model_matern),
    gsy=(gsy_query, make_gp_model_matern),
    igs=(igs_query, make_gp_model_matern)
)

output = run_multiple(x_data_pca, y_data, (0.79, 0.2), learners_gp, n_query, n_iterations, 1)
```

```python
# from https://github.com/SiggiGue/hdfdict/issues/6
f = h5py.File(output_file, 'w')
hdfdict.dump(output, output_file)
f.close()
```
