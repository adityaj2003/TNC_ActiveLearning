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
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# parser.add_argument('integer_input1', type=int, default=1, help='First input integer')
# parser.add_argument('integer_input2', type=int, default=2, help='Second input integer')

args = parser.parse_args()

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
x_train, x_test, y_train_orig, y_test_orig = mixture_gauss(d, 10000)
y_train = add_noise(x_train, y_train_orig, 0.5, 1)
y_test = add_noise(x_test, y_test_orig, 0.5, 1)

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

def plot_TNC_baselines(x_train, y_train, x_test, y_test):
    B_values = [0.1, 0.3, 1, 3, 9]
    alpha_values = [1 / 2, 1]
    NUM_TRIALS = 50
    default_marker_size = 6  # this is the default size in matplotlib for 'o' marker
    half_size = default_marker_size / 2  # setting size to half

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
                noisy_y_train = add_noise(x_train, y_train, alpha, B)

                # Train and evaluate Random Forest
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
                rf_model.fit(x_train, noisy_y_train)
                y_test_pred_rf = rf_model.predict(x_test)
                noisy_y_test = add_noise(x_test, y_test, alpha, B)
                accuracy_rf = accuracy_score(noisy_y_test, y_test_pred_rf)
                acc_rf.append(accuracy_rf)

                # Train and evaluate Logistic Regression
                lr_model = LogisticRegression(C= 50 / 1000, max_iter = 200,
                             penalty='l2', solver='liblinear',
                             fit_intercept=False,
                             tol=0.1)
                lr_model.fit(x_train, noisy_y_train)
                y_test_pred_lr = lr_model.predict(x_test)
                accuracy_lr = accuracy_score(noisy_y_test, y_test_pred_lr)
                acc_lr.append(accuracy_lr)

            accuracy_values_rf.append(np.mean(acc_rf))
            accuracy_values_lr.append(np.mean(acc_lr))
            std_dev_rf.append(np.std(acc_rf))
            std_dev_lr.append(np.std(acc_lr))

        axs[alpha_index].errorbar(B_values, accuracy_values_rf, yerr=std_dev_rf, label='Random Forest', fmt='-o', markersize=half_size)
        axs[alpha_index].errorbar(B_values, accuracy_values_lr, yerr=std_dev_lr, label='Logistic Regression', fmt='-o', markersize=half_size)

        axs[alpha_index].set_xlabel('B')
        axs[alpha_index].set_ylabel('Accuracy')
        axs[alpha_index].set_title(f'Alpha = {alpha}')
        axs[alpha_index].legend()

    plt.tight_layout()
    plt.show()

# plot_TNC_baselines(x_train, y_train_orig, x_test, y_test_orig)


epsilon = 0.1
delta = 0.1
A = 2
alpha = 0.5

# Constants for Big O and Big Theta
O_constant = 50
Theta_constant = 10


theta = O_constant * ((1 / np.log2(1 / epsilon))**2) * (epsilon/2)
print("theta:", theta)
sigma = Theta_constant * (1/A)**((1-alpha)/(3*alpha-1)) * theta**((2*alpha)/(3*alpha-1))
print("sigma:", sigma)
rho = Theta_constant * (1/A)**((2*(1-alpha))/(3*alpha-1)) * theta**((2*(1-alpha))/(3*alpha-1))
print("rho:", rho)
S = np.log(6 / delta)
index = 0

def sample_x_from_DX(index):
    x = x_train[index, :]
    return x

def query_labeling_oracle(index):
    return y_train[index]

def phi_prime_of_sigma_t(w, x):
    # print("e power", math.e**x)
    i = np.dot(w, x) / np.linalg.norm(w)
    # print("w/l1NormOfW:", i)
    return abs(np.exp(-i/sigma)/(sigma*(1+np.exp(-i/sigma))**2))

def ACTIVE_FO(w):
    #Implement a streaming algo where I draw the next sample every time!!!!!!
    index = np.random.randint(x_train.shape[0])
    x = sample_x_from_DX(index)
    q_wx = sigma * phi_prime_of_sigma_t(w, x)  # query probability
    # print(q_wx)
    Z = bernoulli.rvs(q_wx)

    if Z == 1:
        y = query_labeling_oracle(index)

        w_norm = np.linalg.norm(w)
        h_wxy = -1/sigma * y * (x/w_norm**2 - np.dot(w, x) * w / w_norm**3)

        #Assert that h_wxy and w are orthogonal!!!!
        return h_wxy
    else:
        return np.zeros_like(w)


def ACTIVE_PSGD(N, beta):
    # initialize w1 randomly on the unit l2-ball in R^d
    w1 = np.random.normal(size=d)
    w1 = w1 / np.linalg.norm(w1)
    print("N", N)
    w = [None]*(N+1)
    w[0] = w1
    for i in range(1, N+1):
        gi = ACTIVE_FO(w[i-1])
        vi = w[i-1] - beta*gi
        w[i] = vi / np.linalg.norm(vi)
    R = np.random.randint(N)
    return w[R]

N = d / (sigma**2 * rho**4)
beta = rho**2 * sigma**2 / d


ws = ACTIVE_PSGD(math.ceil(N), beta)

print(ws)

predictions = np.dot(x_test, ws)

for i in range(0,len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = -1
    else:
        predictions[i] = 1

# Calculate accuracy
test_accuracy = np.mean(predictions == y_test)

lr_model = LogisticRegression(C= 50 / 1000, max_iter = 200,
                             penalty='l2', solver='liblinear',
                             fit_intercept=False,
                             tol=0.1)
lr_model.fit(x_train, y_train)
y_test_pred_lr = lr_model.predict(x_test)

lr_accuracy = accuracy_score(y_test, y_test_pred_lr)

print("LR Accuracy:", lr_accuracy)
print("ACTIVE_PSGD:", test_accuracy)

