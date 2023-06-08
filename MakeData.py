import sys
import numpy as np
import matplotlib.pyplot as plt
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import bernoulli
import math
import argparse


# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('--integer_input1', type=float, default=0.5, help='First input integer')
parser.add_argument('--integer_input2', type=int, default=1, help='Second input integer')

args = parser.parse_args()

print(args.integer_input1)

np.random.seed(0)

NUM_TRIALS = 50
etas = np.linspace(0.05,0.45,9)
epss = [0.05]
NUM_ITERS = 2000
NUM_FLIPS = 1
TRAIN_SET_SIZE = 100000


def mixture_gauss(d,N,frac=0.25):
    total = int(N*(frac+1))
    cov1 = np.eye(d)
    cov2 = np.eye(d)
    cov2[0,0] = 8.
    cov2[0,1] = 0.1
    cov2[1,0] = 0.1
    cov2[1,1] = 0.0024
    vecs = np.zeros((total,d))
    for i in range(total):
        if np.random.uniform() > 0.5:
            vecs[i,:] = np.random.multivariate_normal([0]*d,cov1)
        else:
            vecs[i,:] = np.random.multivariate_normal([0]*d,cov2)
    x_train = vecs[:N,:]
    x_test = vecs[N:,:]
    y_train = (vecs[:N,0]>0).astype(int)*2 - 1
    y_test = (vecs[N:,0]>0).astype(int)*2 - 1
    return x_train, x_test, y_train, y_test

def L(x,y,lam,w):
    prods = -np.matmul(x,w)*y
    return np.average(0.5*prods + (0.5 - lam)*np.abs(prods))


def corrupt(eta, xs, ys):
    def flip(p, x):
        if np.random.random() < p:
            return -x
        else:
            return x

    noisy_ys = np.array(ys)
    for i in range(len(ys)):
        if xs[i, 1] > 0.3:
            noisy_ys[i] = flip(eta, ys[i])
    return noisy_ys


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


d = 2
x_train, x_test, y_train_orig, y_test_orig = mixture_gauss(d, TRAIN_SET_SIZE)
y_train = add_noise(x_train, y_train_orig, args.integer_input1, args.integer_input2)
y_test = add_noise(x_test, y_test_orig, args.integer_input1, args.integer_input2)

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










