import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import bernoulli
import math
import scipy.stats
import argparse


# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('--alpha', type=float, default=0.5, help='First input integer')
parser.add_argument('--B', type=float, default=0.3, help='Second input integer')

args = parser.parse_args()

print(args.alpha)

np.random.seed(0)

TRAIN_SET_SIZE = 1000
d=2
cov1 = np.eye(d)
cov2 = np.eye(d)
cov2[0,0] = 8.
cov2[0,1] = 0.1
cov2[1,0] = 0.1
cov2[1,1] = 0.0024

def mixture_gauss(d, N, w_star = np.array([1,0]),  frac=0.25):
    total = int(N * (frac + 1))
    vecs = np.zeros((total, d))
    for i in range(total):
        if np.random.uniform() > 0.5:
            vecs[i, :] = np.random.multivariate_normal([0]*d, cov1)
        else:
            vecs[i, :] = np.random.multivariate_normal([0]*d, cov2)

    # Define split sizes
    N_train = int(0.8 * N)

    # Create datasets
    x_train = vecs[:N_train, :]
    x_test = vecs[N_train:, :]
    
    # Compute dot product with w_star to get labels
    y_train = (np.dot(vecs[:N_train, :], w_star) > 0).astype(int) * 2 - 1
    y_test = (np.dot(vecs[N_train:, :], w_star) > 0).astype(int) * 2 - 1

    return x_train, x_val, x_test, y_train, y_val, y_test




def add_noise(features, labels, alpha, B, w_star=np.array([1, 0])):
    """
    Add noise to the labels according to the Tsybakov noise condition.
    """
    # Compute the decision boundary
    h_w_x = np.dot(features, w_star)

    # Compute the probability of flipping each label
    p_flip = 0.5 - np.minimum(1/2,B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    # Flip the labels with probability p_flip
    flip = (np.random.rand(len(labels)) < p_flip)
    noisy_labels = labels.copy()
    noisy_labels[flip] = -noisy_labels[flip]

    return noisy_labels


x_train, x_test, y_train_orig, y_test_orig = mixture_gauss(d, TRAIN_SET_SIZE)
y_train = add_noise(x_train, y_train_orig, args.alpha, args.B)
y_test = add_noise(x_test, y_test_orig, args.alpha, args.B)
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train_orig.npy', y_train_orig)
np.save('y_test_orig.npy', y_test_orig)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

x1_values = x_train[:, 0]
x2_values = x_train[:, 1]

# Separate positive and negative instances
positive_indices = y_train == 1
negative_indices = y_train == -1

# plt.scatter(x1_values[positive_indices], x2_values[positive_indices], color='blue', label='+1')
# plt.scatter(x1_values[negative_indices], x2_values[negative_indices], color='red', label='-1')
# plt.legend()
# plt.show()

