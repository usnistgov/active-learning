import dask.array as da
import numpy as np
from numpy.random import default_rng
from toolz import pipe

from sklearn.utils import shuffle

path_ = './'

x_data = np.array(da.from_zarr(path_ + "x_data.zarr" , chunks=(100, -1)))
y_data = np.load(path_ + "y_data_large.npy")

x_shuffle, y_shuffle = shuffle(x_data, y_data)

np.savez_compressed('data_shuffled.npz', x_data=x_shuffle, y_data=y_shuffle)
