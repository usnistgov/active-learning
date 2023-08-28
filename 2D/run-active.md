---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python tags=["parameters"]
n_query = 100
input_file = 'data/data_pca-500-51-51.npz'
output_file = 'data/active_data.h5'
scoring = 'mse'
```

```python
from tqdm.notebook import trange, tqdm
import numpy as np
from active import make_gp_model_matern, split, split_on_ids, next_sample_gsx, next_sample_igs
import h5py
import hdfdict
```

```python
def run_all(x_data_pca, y_data, train_sizes, learners, n_query, scoring):
    data = split(x_data_pca, y_data, train_sizes)
    test_scores = dict()
    for k in tqdm(learners, position=1, desc="learner loop"):
        test_scores[k] = run(data, learners[k][0], learners[k][1], n_query, scoring)[1]
    return test_scores
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


def gsx_query(model, x_pool, y_pool):
    return query_helper(
        model,
        x_pool,
        y_pool,
        lambda: x_pool,
        lambda m, s: s,
        next_sample_gsx
    )

def gsy_query(model, x_pool, y_pool):
    return query_helper(
        model,
        x_pool,
        y_pool,
        lambda: None,
        lambda m, s: m.predict(x_pool).reshape(-1, 1),
        next_sample_gsx
    )


def igs_query(model, x_pool, y_pool):
    return query_helper(
        model,
        x_pool,
        y_pool,
        lambda: (x_pool, None),
        lambda m, s: (s[0], m.predict(x_pool).reshape(-1, 1)),
        next_sample_igs
    )


def query_uncertainty(model, x_pool, y_pool):
    stds = model.predict(x_pool, return_std=True)[1]
    ids = np.argsort(stds)[::-1][:1]
    return rework_pool(x_pool, y_pool, ids)

def query_random(model, x_pool, y_pool):
    ids = np.random.choice(len(x_pool), 1, replace=False)
    return rework_pool(x_pool, y_pool, ids)


def run(data, query_func, model_func, n_iter, scoring, train_sizes=(0.87, 0.004)):
    x_pool, x_test, x_train, y_pool, y_test, y_train = data
    
    model = model_func(scoring)
    train_scores = []
    test_scores = []
    
    for _ in trange(n_iter, position=2, desc='iter loop'):
        model, x_pool, x_train, y_pool, y_train, test_score, train_score  = evaluate_model(
            x_pool, x_test, x_train, y_pool, y_test, y_train,
            model, 
            query_func
        )
        
        train_scores += [train_score]
        test_scores += [test_score]
       
    return train_scores, test_scores

def evaluate_model(x_pool, x_test, x_train, y_pool, y_test, y_train, model, query_func):
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    x_, x_pool, y_, y_pool = query_func(model, x_pool, y_pool)
    x_train = np.vstack([x_train, x_])
    y_train = np.append(y_train, y_)
    return model, x_pool, x_train, y_pool, y_train, test_score, train_score

def rework_pool(x_pool, y_pool, ids):
    x_, x_pool = split_on_ids(x_pool, ids)
    y_, y_pool = split_on_ids(y_pool, ids)
    return x_, x_pool, y_, y_pool
```

```python
learners_gp = dict(
    uncertainty=(query_uncertainty, make_gp_model_matern),
    random=(query_random, make_gp_model_matern),
    gsx=(gsx_query, make_gp_model_matern),
    gsy=(gsy_query, make_gp_model_matern),
    igs=(igs_query, make_gp_model_matern)
)
```

```python
data = np.load(input_file)
x_data_pca = data['x_data_pca']
y_data = data['y_data']
```

```python
data = run_all(x_data_pca, y_data, (0.795, 0.2), learners_gp, n_query, scoring)
```

```python
# from https://github.com/SiggiGue/hdfdict/issues/6
f = h5py.File(output_file, 'w')
hdfdict.dump(data, output_file)
f.close()

```

```python

```