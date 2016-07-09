import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cPickle as pkl
from sklearn.datasets import make_moons, make_blobs
from sklearn.decomposition import PCA

from flip_gradients import flip_gradient
from utils import *

Xs, ys = make_blobs(300, centers=[[0, 0], [0, 1]], cluster_std=0.2)
Xt, yt = make_blobs(300, centers=[[1, -1], [1, 0]], cluster_std=0.2)

plt.scatter(Xs[:,0], Xs[:,1], c=ys, alpha=0.4)
plt.scatter(Xt[:,0], Xt[:,1], c=yt, cmap='cool', alpha=0.4)
plt.show()
#import time
#time.sleep(10)