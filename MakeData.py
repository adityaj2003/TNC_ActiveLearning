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

def uniform_ball_distribution_from_gaussian(d, N, w_star=np.array([1, 0]), frac=0.25):
    total = int(N * (frac + 1))

    # Generate 3D Gaussian random numbers
    random_points = np.random.randn(total, 3)

    # Normalize each point to have a magnitude of 1 (this projects it onto the surface of a unit sphere)
    normalized_points = random_points / np.linalg.norm(random_points, axis=1)[:, None]

    # Project onto the unit ball in R^2
    features = normalized_points[:, :2]

    N_train = int(0.8 * N)

    x_train = features[:N_train, :]
    x_test = features[N_train:, :]

    y_train = (np.dot(features[:N_train, :], w_star) > 0).astype(int) * 2 - 1
    y_test = (np.dot(features[N_train:, :], w_star) > 0).astype(int) * 2 - 1

    return x_train, x_test, y_train, y_test



def mixture_gauss(d, N, w_star = np.array([1,0]),  frac=0.25):
    total = int(N * (frac + 1))
    vecs = np.zeros((total, d))
    for i in range(total):
        if np.random.uniform() > 0.5:
            vecs[i, :] = np.random.multivariate_normal([0]*d, cov1)
        else:
            vecs[i, :] = np.random.multivariate_normal([0]*d, cov2)

    N_train = int(0.8 * N)

    x_train = vecs[:N_train, :]
    x_test = vecs[N_train:, :]
    
    # Compute dot product with w_star to get labels
    y_train = (np.dot(vecs[:N_train, :], w_star) > 0).astype(int) * 2 - 1
    y_test = (np.dot(vecs[N_train:, :], w_star) > 0).astype(int) * 2 - 1

    return x_train, x_test, y_train, y_test




def add_noise(features, labels, alpha, B, w_star=np.array([1, 0])):
    """
    Add noise to the labels according to the Tsybakov noise condition.
    """
    h_w_x = np.dot(features, w_star)

    p_flip = 0.5 - np.minimum(1/2,B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    flip = (np.random.rand(len(labels)) < p_flip)
    noisy_labels = labels.copy()
    noisy_labels[flip] = -noisy_labels[flip]

    return noisy_labels


def single_gauss(d, cov1, cov2):
    vecs = np.zeros((1, d))
    if np.random.uniform() > 0.5:
        vecs[0, :] = np.random.multivariate_normal([0]*d, cov1)
    else:
        vecs[0, :] = np.random.multivariate_normal([0]*d, cov2)

    x_train = vecs
    y_train = (vecs[0, 0] > 0).astype(int) * 2 - 1
    return x_train[0], y_train


def single_point_uniform_ball():
    try:
        random_point = np.random.randn(2)
        norm = np.linalg.norm(random_point)
        
        if norm == 0:
            raise ValueError("Generated point has zero norm. Regenerating...")

        normalized_point = random_point / norm
        feature = normalized_point[:2]
        label = (feature[0] > 0).astype(int) * 2 - 1
        return feature, label

    except ValueError as e:
        print(e)
        return single_point_uniform_ball()




def add_noise_single(feature, label, alpha, B, w_star=np.array([1, 0])):
    h_w_x = np.dot(feature, w_star)
    p_flip = 0.5 - min(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    flip = (np.random.rand() < p_flip)
    noisy_label = label
    if flip:
        noisy_label = -noisy_label

    return noisy_label

def determine_area(x, w_star, w, alpha):
    """Determine which area the instance x belongs to."""
    dot_star = np.dot(x, w_star)
    dot_w = np.dot(x, w)
    
    if dot_star > 0 and dot_w <= 0:
        return "A"
    elif dot_star > 0 and dot_w > 0:
        return "B" if np.arccos(np.dot(w_star, w)) <= alpha else "C"
    elif dot_star <= 0 and dot_w <= 0:
        return "D"
    return None

def add_described_noise(feature, label, w_star, alpha):
    # Rotate w_star by angle alpha to get w
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    w = np.dot(rot_matrix, w_star/np.linalg.norm(w_star))

    noisy_label = label
    area = determine_area(feature, w_star, w, alpha)
    if area in ["A", "B"] and np.random.rand() < 0.2:
        noisy_label = -noisy_label
    
    return noisy_label


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--alpha', type=float, default=0.5, help='First input integer')
#     parser.add_argument('--B', type=float, default=0.3, help='Second input integer')

#     args = parser.parse_args()

#     print(args.alpha)

#     np.random.seed(0)

#     TRAIN_SET_SIZE = 100000
#     d=2
#     cov1 = np.eye(d)
#     cov2 = np.eye(d)
#     cov2[0,0] = 8.
#     cov2[0,1] = 0.1
#     cov2[1,0] = 0.1
#     cov2[1,1] = 0.0024
#     w_star = np.array([1,0])

#     x_train, x_test, y_train_orig, y_test_orig = mixture_gauss(d, TRAIN_SET_SIZE, w_star = w_star)

#     alpha = 0.05
#     h_w_x = np.dot(x_train, w_star)

#     # Compute the probability of flipping each label
#     eta = 0.5 - np.minimum(1/2,args.B * (np.abs(h_w_x)**((1-alpha)/alpha)))

#     w_star_norm = w_star / np.linalg.norm(w_star)
#     rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
#     w = np.dot(rot_matrix, w_star_norm)

#     y_train = np.array([add_noise_1bit_compressed_sensing(y, x, w, w_star, alpha, eta[0]) for y, x in zip(y_train_orig, x_train)])
#     y_test =  np.array([add_noise_1bit_compressed_sensing(y, x, w, w_star, alpha, eta[0]) for y, x in zip(y_test_orig, x_test)])
#     np.save('x_train.npy', x_train)
#     np.save('x_test.npy', x_test)
#     np.save('y_train_orig.npy', y_train_orig)
#     np.save('y_test_orig.npy', y_test_orig)
#     np.save('y_train.npy', y_train)
#     np.save('y_test.npy', y_test)

#     x1_values = x_train[:, 0]
#     x2_values = x_train[:, 1]

#     # Separate positive and negative instances
#     positive_indices = y_train == 1
#     negative_indices = y_train == -1

#     # plt.scatter(x1_values[positive_indices], x2_values[positive_indices], color='blue', label='+1')
#     # plt.scatter(x1_values[negative_indices], x2_values[negative_indices], color='red', label='-1')
#     # plt.legend()
#     # plt.show()





####This is the main function for the noise in the 1bit paper in a separate piece of code (because it is a bit different and we want to keep it separate)


if __name__ == '__main__':
    # Set fixed values for alpha and eta
    alpha = 1.0
    eta = 0.2

    # Seed for reproducibility
    np.random.seed(0)

    # Define constants
    TRAIN_SET_SIZE = 100000
    d = 2
    cov1 = np.eye(d)
    cov2 = np.eye(d)
    cov2[0, 0] = 8.
    cov2[0, 1] = 0.1
    cov2[1, 0] = 0.1
    cov2[1, 1] = 0.0024
    w_star = np.array([1, 0])
    x_train, x_test, y_train_orig, y_test_orig = mixture_gauss(d, TRAIN_SET_SIZE, w_star=w_star)
    y_train = [add_described_noise(x, y, w_star, alpha) for x, y in zip(x_train, y_train_orig)]
    y_test = [add_described_noise(x, y, w_star, alpha) for x, y in zip(x_test, y_test_orig)]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.save('y_train_orig.npy', y_train_orig)
    np.save('y_test_orig.npy', y_test_orig)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
