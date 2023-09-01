from toolz.curried import curry, pipe
import numpy as np
from pymks.fmks.func import sequence
from toolz.curried import curry, pipe, valmap, itemmap, iterate, do, merge_with
from toolz.curried import map as map_
from dask_ml.model_selection import train_test_split
import tqdm
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
import types


def split_on_ids(arr, ids):
    mask = np.zeros(len(arr), dtype=bool)
    mask[ids] = True
    return arr[mask], arr[~mask]

def calc_distances(d0, d1):
    return np.linalg.norm(d0[:, None, :] - d1[None, :, :], ord=2, axis=-1)

def calc_distances_nk(labeled_samples, scores):
    scores_labeled, scores_unlabeled = split_on_ids(scores, labeled_samples)
    if len(scores_labeled) == 0:
        scores_labeled = np.mean(scores_unlabeled, axis=0)[None]
    return calc_distances(scores_unlabeled, scores_labeled)

def next_sample(distances_nk, labeled_samples, scores):
    distances_n = np.min(distances_nk, axis=1)
    _, unlabeled_ids = split_on_ids(np.arange(len(scores)), labeled_samples)
    return unlabeled_ids[np.argmax(distances_n)]

def next_sample_gsx(labeled_samples, scores):
    distances_nk = calc_distances_nk(labeled_samples, scores)
    return next_sample(distances_nk, labeled_samples, scores)

def next_sample_igs(labeled_samples, scores):
    x_scores, y_scores = scores
    distances_nk_x = calc_distances_nk(labeled_samples, x_scores)
    distances_nk_y = calc_distances_nk(labeled_samples, y_scores)
    return next_sample(distances_nk_x * distances_nk_y, labeled_samples, x_scores)

def query_helper(model, x_pool, init_scores, update_scores, next_func):
    if not hasattr(model, 'query_data'):
        model.query_data = [], init_scores()
    labeled_samples, scores = model.query_data
    scores = update_scores(model, scores)
    next_id = next_func(labeled_samples, scores)
    model.query_data = (labeled_samples + [next_id], scores)
    return next_id, x_pool[next_id]

@curry
def gsx_query(distance_transformer, model, x_pool):
    return query_helper(
        model,
        x_pool,
        lambda: distance_transformer(x_pool),
        lambda m, s: s,
        next_sample_gsx
    )

def gsy_query(model, x_pool):
    return query_helper(
        model,
        x_pool,
        lambda: None,
        lambda m, s: m.predict(x_pool).reshape(-1, 1),
        next_sample_gsx
    )

@curry
def igs_query(distance_transformer, model, x_pool):
    return query_helper(
        model,
        x_pool,
        lambda: (distance_transformer(x_pool), None),
        lambda m, s: (s[0], m.predict(x_pool).reshape(-1, 1)),
        next_sample_igs
    )


@curry
def evaluate_item(oracle, x_pool, x_test, y_test, iterations, item):
    name, learner = item
    return name, evaluate_learner(oracle, x_pool, x_test, y_test, iterations, learner)

@curry
def one_round(x_data, y_data, iterations, seed, make_learners, oracle_func, props):
    x_pool, x_test, x_train, y_pool, y_test, y_train, indices_pool, indices_test, indices_train = three_way_split(x_data, y_data, props, seed)
    print(x_pool.shape)
    print(x_test.shape)
    print(x_train.shape)
    oracle = oracle_func(x_pool, y_pool)
    eval_item = evaluate_item(oracle, x_pool, x_test, y_test, iterations)
    return itemmap(eval_item, make_learners(x_train, y_train))

def multiple_rounds(x_data, y_data, rounds, iterations, make_learners, oracle_func, props, seed):
    process_scores = sequence(
        lambda _: one_round(x_data, y_data, iterations, seed, make_learners, oracle_func, props),
        valmap(lambda x: x.scores)
    )

    return pipe(
        [None] * rounds,
        map_(process_scores),
        list,
        lambda x: merge_with(np.vstack)(*x),
        valmap(lambda x: (np.mean(x, axis=0), np.std(x, axis=0)))
    )

def flatten(x_data):
    return x_data.reshape(x_data.shape[0], -1)

def three_way_split(x_data, y_data, props, random_state):
    indices = np.arange(len(x_data))
    x_0, x_, y_0, y_, indices_0, indices_ = train_test_split(x_data, y_data, indices, train_size=props[0], random_state=random_state)
    x_1, x_2, y_1, y_2, indices_1, indices_2 = train_test_split(x_, y_, indices_, train_size=props[1] / (1 - props[0]), random_state=random_state)
    return flatten(x_0), flatten(x_1), flatten(x_2), y_0, y_1, y_2, indices_0, indices_1, indices_2


@curry
def iterate_times(func, times, value):
    iter_ = iterate(func, value)
    for _ in tqdm.tqdm(range(times)):
        next(iter_)
    return next(iter_)


@curry
def update_learner(oracle_func, x_pool, x_test, y_test, learner):
    query = sequence(
        learner.query,
        oracle_func,
    )
    return pipe(
        learner,
        do(lambda x: x.teach(*query(x_pool))),
        do(lambda x: x.scores.append(x.score(x_test, y_test)))
    )

@curry
def evaluate_learner(oracle_func, x_pool, x_test, y_test, n_query, learner):
    learner.scores = [learner.score(x_test, y_test)]
    return iterate_times(
        update_learner(oracle_func, x_pool, x_test, y_test),
        n_query,
        learner
    )

@curry
def make_learner(klass, query_func, model_func, x_train, y_train):
    return klass(
        estimator=model_func(),
        query_strategy=query_func,
        X_training=x_train,
        y_training=y_train,
    )





query_uncertainty = lambda model, x_: pipe(
    model.predict(x_, return_std=True)[1],
    np.argmax,
    lambda i: (i, x_[i])
)

query_random = lambda model, x_: pipe(
    np.random.randint(0, len(x_)),
    lambda i: (i, x_[i])
)

def make_gsx(distance_transformer):
    return make_active(gsx_query(distance_transformer))


def make_igs(distance_transformer):
    return make_active(igs_query(distance_transformer))


def make_ensemble(x_train, y_train, model_func):
    ensemble_learner = CommitteeRegressor(
        learner_list=[
            ActiveLearner(
                estimator=model_func(),
                X_training=x_train_,
                y_training=y_train_
            )
            for x_train_, y_train_ in zip(np.array_split(x_train, 5), np.array_split(y_train, 5))
        ],
        query_strategy=max_std_sampling
    )

    ## required because CommitteeRegressor does not have a score function

    def score(self, x_true, y_true):
        y_pred = self.predict(x_true)
        return r2_score(y_true, y_pred)

    ensemble_learner.score = types.MethodType(score, ensemble_learner)

    return ensemble_learner


def train_test_split_(x_data, y_data, prop, random_state=None):
    ids = np.random.choice(len(x_data), int(prop * len(x_data)), replace=False)
    x_0, x_1 = split_on_ids(x_data, ids)
    y_0, y_1 = split_on_ids(y_data, ids)
    return x_0, x_1, y_0, y_1


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


def make_gp_model_matern(scoring, nu=0.5):
    kernel = Matern(length_scale=1.0, nu=nu)
    regressor = GaussianProcessRegressor(kernel=kernel)
    score_func = dict(mse=mean_squared_error, mae=mean_absolute_error, r2=None)[scoring]
    if score_func is not None:
        scorer = make_scorer(score_func)
        regressor.score = types.MethodType(scorer, regressor)
    return regressor
