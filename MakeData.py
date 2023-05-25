import sys
from sklearn.svm import SVC
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

np.random.seed(0)

NUM_TRIALS = 50
etas = np.linspace(0.05,0.45,9)
epss = [0.05]
NUM_ITERS = 2000
NUM_FLIPS = 1

def leakyrelu(lam,z):
    if z > 0:
        return (1 - lam)*z
    else:
        return lam * z

def acc(x,y,w):
    return np.sum(np.matmul(x,w)*y > 0)/float(len(y))

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
    y_train = (vecs[:N,1]>0).astype(int)*2 - 1
    y_test = (vecs[N:,1]>0).astype(int)*2 - 1
    return x_train, x_test, y_train, y_test

def L(x,y,lam,w):
    prods = -np.matmul(x,w)*y
    return np.average(0.5*prods + (0.5 - lam)*np.abs(prods))

algos = ['rf','lreg','our','rcn']
algo_ids = {algos[i]:i for i in range(len(algos))}

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



x_train, x_test, y_train, y_test = mixture_gauss(2, 1000)



x1_values = x_train[:, 0]
x2_values = x_train[:, 1]

# Separate positive and negative instances
positive_indices = y_train == 1
negative_indices = y_train == -1

# plt.scatter(x1_values[positive_indices], x2_values[positive_indices], color='blue', label='+1')
# plt.scatter(x1_values[negative_indices], x2_values[negative_indices], color='red', label='-1')
# plt.legend()
# plt.show()






from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches


def plot_random_forest(x_train, y_train, x_test, y_test):
    B_values = [0.1, 0.3, 1, 3, 9]
    alpha_values = [1 / 2, 1]
    NUM_TRIALS = 1/0

    fig, axs = plt.subplots(len(alpha_values))

    for alpha_index, alpha in enumerate(alpha_values):
        accuracy_values_rf = []
        accuracy_values_lr = []
        std_dev_rf = []
        std_dev_lr = []

        for B in B_values:
            acc_rf = []
            acc_lr = []
            for num_trial in range(NUM_TRIALS):
                # Add Tsybakov noise to the training labels
                noisy_y_train = add_noise(x_train, y_train, alpha, B)

                # Train and evaluate Random Forest
                rf_model = RandomForestClassifier(max_depth=20, random_state=0)
                rf_model.fit(x_train, noisy_y_train)
                y_test_pred_rf = rf_model.predict(x_test)
                accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
                acc_rf.append(accuracy_rf)

                # Train and evaluate Logistic Regression
                lr_model = LogisticRegression(random_state=0)
                lr_model.fit(x_train, noisy_y_train)
                y_test_pred_lr = lr_model.predict(x_test)
                accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
                acc_lr.append(accuracy_lr)

            accuracy_values_rf.append(np.mean(acc_rf))
            accuracy_values_lr.append(np.mean(acc_lr))
            std_dev_rf.append(np.std(acc_rf))
            std_dev_lr.append(np.std(acc_lr))

        axs[alpha_index].errorbar(B_values, accuracy_values_rf, yerr=std_dev_rf, label='Random Forest', fmt='-o')
        axs[alpha_index].errorbar(B_values, accuracy_values_lr, yerr=std_dev_lr, label='Logistic Regression', fmt='-o')

        axs[alpha_index].set_xlabel('B')
        axs[alpha_index].set_ylabel('Accuracy')
        axs[alpha_index].set_title(f'Alpha = {alpha}')
        axs[alpha_index].legend()

    plt.tight_layout()
    plt.show()


plot_random_forest(x_train, y_train, x_test, y_test)