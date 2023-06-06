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

# Active Learning Notebook

Use the large data set with active learning.


## Notes 04-27-2022

Did some debugging. The linear model works right now. The GP model is not working currently so have asked Berkay if there are parameters that give some good values. Random runs throught the `multiple_rounds` function, but the accuracy is extremely high at the very start 0.94. This is with 30 training samples. I'm wondering if the data is somehow sorted so that the first 1000 samples that I'm currently using for testing are similar and hence the high accuracy even with only 30 sample for training. Next step is to pre-shuffle the data and see if this makes the early stages of the random have a lower accuracy. 

TODO:

 - [x] pre-shuffle the data (as you recently did in the pace visualization repository).
 - [x] try and get the GPR model working
 - [ ] examine efficiency
   - [ ] do the fit for the PCA only with the training data (not the pooled data)
 - [ ] batch jobs. don't relearn so frequently
 - [ ] parallel
 
 Integerate into ModAL
 
 - [ ] Figure out how to cache the greedy sampling distances and labeled and unlabeled samples
   - [ ] should that data be returned from the query with an updated model. 
   - is there an expectation that the model is updated after the query???
 


## Notes 07-28-2022

We now have a reasonably accurate GPR model. It's accuracy against a test set is 0.84. That isn't marvelous, but acceptable. However, it still doesn't work well with active learning. My current understanding is that this is to do with the lack of variation in the standard deviation between all the samples. Essentially, it takes a lot of samples to generate an accurate model regardless of the samples used to calibrate the model. I'm not sure what this means, but maximum uncertainty active learning doesn't work at all with this data set. It's likely that none of the other methods will do any better.

 - What do we call this sort of data. Data that requires a large number of samples to calibrate regardless of the samples chosen and how optimal the samples are.

### Debugging

One of the confusing issues with the debugging is that the system was selecting the same sample over and over again. The issue is that the way ModAL is set up, it was necessary to keep the pool of data consistent. This is due to how the greedy alogrithms are set up. Anyway, I did the uncertainty sampling outside of the ModAL system. I'll have to rework the way we're using ModAL.

### Next step

Show image of test and train accuracy as the number of samples are increased with different batch numbers for GPR model.


## Notes 2022-08-07

After chatting with Berkay it seems sensible to do the PCA part of the workflow up front.

## Notes 2022-08-22

 - Doing the PCA up front works. There is nothing wrong with this from the perspective of active learning as we are not using the y-data at all.
 - What's next
   - [x] Produce a good image with only random and uncertainty for one run
   - [x] Get the GS? working without modal
   - [x] try with more PCAs
   - [x] Produce an image with all the different algorithms present for at least a single run
   - [x] multiple runs
   - [ ] using the linear model as well
   - [ ] Get things working with modal, but without active.py
   - [ ] Get things working with active.py
   
## Notes 2022-08-23

Action items from meeting with Olga

 - Add figures to paper
 - Understand why IGS is so much better for GP model
 - Check that all models converge for GP model
 - Determine how many iterations are required to get converged curves
 - perhaps try different PCA strategy to see how that impacts the results
 
 ## Notes for 2022-12-06
 
 - Ideas:
   - Try with linear model to demo that it works with that as well
   - Use ModAL and get it included
     - integrate GS techniques
     - return a new x_pool without the updated sample
     - parallel?
   - Create clean notebook with the examples after getting ModAL working
   - Plot with 95% margins
   - Check out second paper as well
   
 - Action items for meeting:
   - [ ] 20% testing data
   - [ ] 20 repetitions per curve
   - [ ] Add figures to paper with captions and some text
   - [ ] Create training figure as well
   - [ ] Determine why IGS works:
     - [ ] probability distributions
     - [ ] PDF of distances between y data and x data
     - [ ] Wasserstein distance
     - [ ] Kernel distance estimator - distance between 2 distributions
     - [ ] diversity and representation

```python
import numpy as np
import dask.array as da
from dask_ml.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dask_ml.decomposition import IncrementalPCA
from dask_ml.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
#from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from pymks import (
    generate_multiphase,
    plot_microstructures,
    PrimitiveTransformer,
    TwoPointCorrelation,
    GenericTransformer,
    solve_fe
)

from toolz.curried import curry, pipe, valmap, itemmap, iterate, do, merge_with
from toolz.curried import map as map_
from modAL.models import ActiveLearner, CommitteeRegressor, BayesianOptimizer
from modAL.disagreement import max_std_sampling
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI
from tqdm.notebook import trange, tqdm
import types
from pymks.fmks.func import sequence
from active import gsx_query, gsy_query, igs_query, multiple_rounds, three_way_split, flatten, split_on_ids

import dask.array as da

from itertools import cycle
import hdfdict
import h5py
```

## Load the data

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

## Functions to generate models

Here we use the GPR model as it returns a probability that's required by the `ActiveLearner` class.

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
```

```python
def make_gp_model():
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e6)) + WhiteKernel(noise_level=0.05)
    regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    return regressor
    
def make_gp_model_matern():
    kernel = Matern(length_scale=1.0)
    regressor = GaussianProcessRegressor(kernel=kernel)
    return regressor

def make_gp_model_old():
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
    regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    return Pipeline(steps=(
        ('poly', PolynomialFeatures(degree=3)),
        ('regressor', regressor),
    ))

    #    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=0.1)
#    regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=99)
#    return Pipeline(steps=pca_steps() + (
#        ('regressor', regressor),
#    ))

def make_linear_model():
    return Pipeline((
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', LinearRegression()),
    ))

def make_pca_model():
    return Pipeline(steps=pca_steps())
```

## The Oracle

The oracle function is an FE simulation on the 2D grid.

```python
@curry 
def oracle_func(x_data, y_data, query_instance):
    idx, query_value = query_instance
    return query_value.reshape(1, -1), np.array([y_data[idx]]).reshape(1)
```

## Helper Functions

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

## Set up the active learners

One is a GPR using the maximum std and the other is random

```python
from active import make_igs, make_gsx, make_gsy, make_bayes, make_uncertainty, make_ensemble, make_random, make_active

#distance_transformer = lambda x: pca_model().fit_transform(x)

#def query_uncertainty(model, x_pool):
#    if not hasattr(model, 'batch_ids') or len(model.batch_ids) == 0:
#        uncertainties = model.predict(x_pool, return_std=True)[1]
#        args = np.argsort(uncertainties)[::-1]
#        model.batch_ids = args[:5]
#    next_item = (model.batch_ids[0], x_pool[model.batch_ids[0]])
#    model.batch_ids = model.batch_ids[1:]
#    print('next id', next_item[0])
#    return next_item

query_uncertainty = lambda model, x_: pipe(
    model.predict(x_, return_std=True)[1],
    np.argmax,
    lambda i: (i, x_[i]), 
    do(lambda x: print('id:', x[0]))
)

def make_learners(x_train, y_train):
    return dict(
        uncertainty=make_active(query_uncertainty)(make_gp_model_matern, x_train, y_train),
        random=make_random(make_gp_model_matern, x_train, y_train),
#        ensemble=make_ensemble(x_train, y_train),
#        bayes=make_bayes(make_gp_model, x_train, y_train),
#        gsx=make_gsx(distance_transformer)(make_linear_model, x_train, y_train),
#        gsy=make_gsy(make_linear_model, x_train, y_train),
#        igs=make_igs(distance_transformer)(make_linear_model, x_train, y_train)
    )

#random_learner = make_random(make_gp_model, x_train, y_train)
```

## Check the data

```python
plot_microstructures(*x_data[:10].reshape(10, 51, 51, 51)[:, :, :, 0], cmap='gray', colorbar=False)
```

# Make PCA Data

```python
x_data_da = da.from_array(x_data, chunks=(100, -1))
```

```python
model = make_pca_model()
x_data_pca = model.fit_transform(x_data_da).compute()
```

```python
del x_data
del x_data_da
```

```python
print(x_data_pca.shape)
print(y_data.shape)
```

```python
np.savez('data_pca.npz', x_data_pca=x_data_pca, y_data=y_data)
```

## Load the PCA Data

```python
data = np.load('data_pca.npz')
```

```python
x_data_pca = data['x_data_pca']
y_data = data['y_data']
```

# Train test split and PCA

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

## Test the model

```python
x_pool, x_test, x_train, y_pool, y_test, y_train = split(
    x_data_pca[:, :3], y_data, train_sizes=(0.87, 0.003)
)
```

```python
x_pool.shape
```

```python
x_test.shape
```

```python
x_train.shape
```

```python
#model = make_gp_model_matern()
model = make_linear_model()
```

```python
model.fit(x_train, y_train)
```

```python
%%timeit
y_pool_predict = model.predict(x_pool)
```

```python
%%timeit
y_pool_predict = model.predict(x_pool)
```

```python
#y_pool_predict, pool_std = model.predict(x_pool, return_std=True)
y_pool_predict = model.predict(x_pool)
```

```python
y_test_predict = model.predict(x_test)
```

```python
plot_parity(y_pool, y_pool_predict)
```

```python
plot_parity(y_test, y_test_predict)
```

```python
from sklearn.metrics import r2_score

print(r2_score(y_test, y_test_predict))
#print(y_test.shape)
#print(y_test)
print(model.score(x_test, y_test))
```

## Debug

```python
def rework_pool(x_pool, y_pool, ids):
    x_, x_pool = split_on_ids(x_pool, ids)
    y_, y_pool = split_on_ids(y_pool, ids)
    return x_, x_pool, y_, y_pool
```

```python
from active import next_sample_gsx, next_sample_igs

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
def run(data, query_func, model_func, n_iter, train_sizes=(0.87, 0.004), batch_size=1):
    x_pool, x_test, x_train, y_pool, y_test, y_train = data
    
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
import matplotlib

def plot_scores(scores, opt=None, opt_error=None, error_freq=20):

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
    plt.ylabel(r'$R^2$', fontsize=16);
    plt.ylim(0.4, 1)
    
    return plt, ax
```

## Multiple runs

```python
from tqdm.notebook import trange, tqdm

def run_all(x_data_pca, y_data, train_sizes, learners, n_query, batch_size):
    data = split(x_data_pca, y_data, train_sizes)
    test_scores = dict()
    for k in tqdm(learners, position=1, desc="learner loop"):
        test_scores[k] = run(data, learners[k][0], learners[k][1], n_query, batch_size=batch_size)[1]
    return test_scores
```

```python
from toolz.curried import map as fmap

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
learners_gp = dict(
    uncertainty=(query_uncertainty, make_gp_model_matern),
    random=(query_random, make_gp_model_matern),
    gsx=(gsx_query, make_gp_model_matern),
    gsy=(gsy_query, make_gp_model_matern),
    igs=(igs_query, make_gp_model_matern)
)

output = run_multiple(x_data_pca, y_data, (0.795, 0.2), learners_gp, 400, 10, 1)
```

```python
# from https://github.com/SiggiGue/hdfdict/issues/6

save_file = 'output_.h5'
f = h5py.File(save_file, 'w')
hdfdict.dump(output, save_file)
f.close()
```

```python
output_ = hdfdict.load('output_.h5')
```

```python
plt, ax = plot_scores(output_, error_freq=40, opt=0.94435, opt_error=0.0059322)
plt.title('Active Learning Curves for 3D Composite')
plt.savefig('plot.png', dpi=200)
```

# Determine accuracy using all the data

```python
test_scores = []
train_scores = []

for i in range(20):
    print(i)
    x_pool, x_test, x_train, y_pool, y_test, y_train = split(x_data_pca, y_data, (0.8, 0.2))
    model = make_gp_model_matern()
    model.fit(x_pool, y_pool)
    train_score = model.score(x_pool, y_pool)
    test_score = model.score(x_test, y_test)
    test_scores += [test_score]
    train_scores += [train_score]
```

```python
print(test_scores)
```

```python
print(train_scores)
```

```python
print(np.mean(test_scores))
print(np.std(test_scores))

```

# Misc

```python
plot_scores(output)

```

```python
learners_gp = dict(
#    uncertainty=(query_uncertainty, make_gp_model_matern),
    random=(query_random, make_gp_model_matern),
#    gsx=(gsx_query, make_gp_model_matern),
#    gsy=(gsy_query, make_gp_model_matern),
#    igs=(igs_query, make_gp_model_matern)
)

output_random = run_multiple(x_data_pca, y_data, (0.87, 0.004), learners_gp, 1, 1, 1)
```

```python
from toolz.curried import merge

output = merge([output_random, output_igs])
```

```python
plot_scores(output)
```

```python
plot_scores(output)
```

```python
plot_scores(output)
```

```python
plot_scores(output)
```

```python
plot_scores(output)
```

```python
plot_scores(output)
```

```python
plot_scores(output)
```

```python
output200 = output
```

```python
plot_scores(output200) # 200 iterations
```

# Test the ultimate possible accuracy

```python
learners_gp = dict(
#    uncertainty=(query_uncertainty, make_gp_model_matern),
    random=(query_random, make_gp_model_matern),
#    gsx=(gsx_query, make_gp_model_matern),
#    gsy=(gsy_query, make_gp_model_matern),
#    igs=(igs_query, make_gp_model_matern)
)

output = run_multiple(x_data_pca, y_data, (0.895, 0.1), learners_gp, 15, 5, 500)
```

```python
plot_scores(output)
```

```python
print(output['random']['mean'][-1])
```

```python
learners_gp = dict(
#    uncertainty=(query_uncertainty, make_gp_model_matern),
#    random=(query_random, make_gp_model_matern),
#    gsx=(gsx_query, make_gp_model_matern),
#    gsy=(gsy_query, make_gp_model_matern),
    igs=(igs_query, make_gp_model_matern)
)

output = run_multiple(x_data_pca, y_data, (0.895, 0.1), learners_gp, 500, 5, 1)
```

```python
ax = plot_scores(output)
ax.plot([0, 500], [0.9458, 0.9458], 'k--')
```

## Linear Model

```python
learners = dict(
    random=(query_random, make_linear_model),
    gsx=(gsx_query, make_linear_model),
    gsy=(gsy_query, make_linear_model),
    igs=(igs_query, make_linear_model)
)
```

```python
output = run_multiple(x_data_pca[:, :3], y_data, (0.87, 0.002), learners, 50, 100)
```

```python
output
```

```python
plot_scores(output, error_freq=10)
```

## Run the learners (linear model)

```python
scores = multiple_rounds(x_data_pca, y_data, 1, 200, make_learners, oracle_func, (0.85, 0.14, 0.01), 99)
```

```python
scores
```

```python
scores
```

```python
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
ax = plt.gca()

for k, v in scores.items():
    y = v[0]
    e = v[1]
    x = np.arange(len(y))
    ax.plot(x, y, label=k, linestyle=next(linecycler))
    if k in ['std', 'bayes', 'gsx', 'gsy', 'igs']:
        ax.fill_between(x, y - e, y + e, alpha=0.1)
plt.legend()
plt.xlabel('iterations')
plt.ylabel('R^2');
plt.ylim(0, 1)
```

## Run the learners (GP model)

```python
scores = multiple_rounds(x_data[:n_use], y_data[:n_use], 1, 20, make_learners, oracle_func, props, 99)
```

```python
scores

```


## The results

```python
plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
ax = plt.gca()

for k, v in scores.items():
    y = v[0]
    e = v[1]
    x = np.arange(len(y))
    ax.plot(x, y, label=k)
    if k in ['std', 'bayes', 'gsx', 'gsy', 'igs']:
        ax.fill_between(x, y - e, y + e, alpha=0.1)
plt.legend()
plt.xlabel('iterations')
plt.ylabel('R^2');
plt.ylim(0, 1)
```

```python
plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
ax = plt.gca()

for k, v in scores.items():
    y = v[0]
    e = v[1]
    x = np.arange(len(y))
    ax.plot(x, y, label=k)
    if k in ['std', 'bayes', 'gsx', 'gsy', 'igs']:
        ax.fill_between(x, y - e, y + e, alpha=0.1)
plt.legend()
plt.xlabel('iterations')
plt.ylabel('R^2');
plt.ylim(0.9, 1)
```

## Check what the accuracy actually looks like

```python
y_pred_std = learner_accuracy['std'][1].predict(x_test)
y_pred_random = learner_accuracy['random'][1].predict(x_test)
```

```python
plot_parity(y_test, y_pred_random, label='random')
```

```python
plot_parity(y_test, y_pred_std, label='std')
```

```python

```
