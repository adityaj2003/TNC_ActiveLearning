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
TRAIN_SET_SIZE = 100000
d = 2

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train_orig = np.load('y_train_orig.npy')
y_test_orig = np.load('y_test_orig.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')


epsilon = 0.1
delta = 0.1
A = 3
alpha = 0.5
label_used_array = [False]*TRAIN_SET_SIZE
num_labels_accessed = 0
index = 0
# Constants for Big O and Big Theta
O_constant = 50
Theta_constant = 10


theta = O_constant * ((1 / np.log2(1 / epsilon))**2) * (epsilon/2)
sigma = Theta_constant * (1/A)**((1-alpha)/(3*alpha-1)) * theta**((2*alpha)/(3*alpha-1))
rho = Theta_constant * (1/A)**((2*(1-alpha))/(3*alpha-1)) * theta**((2*(1-alpha))/(3*alpha-1))
S = np.log(6 / delta)


def sample_x_from_DX(index):
    x = x_train[index, :]
    return x

def query_labeling_oracle(index):
    return y_train[index]

def phi_prime_of_sigma_t(w, x):
    i = np.dot(w, x) / np.linalg.norm(w, 1)
    return abs(np.exp(-i/sigma)/(sigma*(1+np.exp(-i/sigma))**2))

def ACTIVE_FO(w):
    global index
    global TRAIN_SET_SIZE
    #Implement a streaming algo where I draw the next sample every time!!!!!!
    if (label_used_array[index]):
        index = (index + 1) % TRAIN_SET_SIZE
        return np.zeros_like(w)
    x = sample_x_from_DX(index)
    q_wx = sigma * phi_prime_of_sigma_t(w, x)  # query probability
    Z = bernoulli.rvs(q_wx)

    if Z == 1:
        global num_labels_accessed
        num_labels_accessed += 1
        y = query_labeling_oracle(index)
        w_norm = np.linalg.norm(w)
        h_wxy = -1/sigma * y * (x/w_norm**2 - np.dot(w, x) * w / w_norm**3)
        label_used_array[index] = True
        index = (index+1)%TRAIN_SET_SIZE
        #Assert that h_wxy and w are orthogonal!!!!
        assert np.isclose(np.dot(h_wxy, w), 0)
        return h_wxy
    else:
        index = (index+1)%TRAIN_SET_SIZE
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
    print("R:", R)
    return w[R]

N = d / (sigma**2 * rho**4)
beta = rho**2 * sigma**2 / d


def TNC_learning(epsilon, delta):
    A = 3
    alpha = 0.5
    label_used_array = [False] * TRAIN_SET_SIZE
    num_labels_accessed = 0
    index = 0
    # Constants for Big O and Big Theta
    O_constant = 50
    Theta_constant = 50
    theta = O_constant * ((1 / np.log2(1 / epsilon)) ** 2) * (epsilon / 2)
    sigma = Theta_constant * (1 / A) ** ((1 - alpha) / (3 * alpha - 1)) * theta ** ((2 * alpha) / (3 * alpha - 1))
    rho = Theta_constant * (1 / A) ** ((2 * (1 - alpha)) / (3 * alpha - 1)) * theta ** (
                (2 * (1 - alpha)) / (3 * alpha - 1))
    S = np.log(6 / delta)
    w_s = []
    for s in range(S):
        N = ceil(d / (sigma ** 2 * rho ** 4))
        beta = rho ** 2 * sigma ** 2 / d
        w = ACTIVE_PSGD(N, beta)
        w_s.append(w)

    g_s = []
    for s in range(S):
        g = ACTIVE_FO(w_s[s])
        g_s.append(g)
    g_bar = np.mean(g_s, axis=0)
    s_star = np.argmin(np.linalg.norm(g_bar))
    w_tilde = w_s[s_star]





#
# ws = ACTIVE_PSGD(math.ceil(N), beta)
#
# print(ws)
#
# predictions = np.dot(x_test, ws)
#
# for i in range(0,len(predictions)):
#     if predictions[i] < 0.0:
#         predictions[i] = -1
#     else:
#         predictions[i] = 1
#
# # Calculate accuracy
# test_accuracy = np.mean(predictions == y_test)
#
# lr_model = LogisticRegression(C= 50 / 1000, max_iter = 200,
#                              penalty='l2', solver='liblinear',
#                              fit_intercept=False,
#                              tol=0.1)
# lr_model.fit(x_train, y_train)
# y_test_pred_lr = lr_model.predict(x_test)
#
# lr_accuracy = accuracy_score(y_test, y_test_pred_lr)
#
# print("LR Accuracy:", lr_accuracy)
# print("ACTIVE_PSGD:", test_accuracy)
# print("Labels Accessed:", num_labels_accessed)
# print("% Labels Used:", num_labels_accessed/N)

N_values = np.linspace(1000, 100000, num=100)  # change num to adjust the number of points

# Initialize an array to hold the corresponding accuracies
accuracies = []
num_labels_used = []
# Calculate the accuracy for each N value
for N in N_values:
    label_used_array = [False]*TRAIN_SET_SIZE
    index = 0
    num_labels_accessed = 0
    ws = ACTIVE_PSGD(math.ceil(N), beta)
    predictions = np.dot(x_test, ws)
    for i in range(0, len(predictions)):
        if predictions[i] < 0.0:
            predictions[i] = -1
        else:
            predictions[i] = 1

    # Calculate accuracy and append it to the accuracies array
    test_accuracy = accuracy_score(predictions, y_test)
    print(test_accuracy)
    if (test_accuracy < 0.5):
        print(ws)
    print("Labels Accessed:", num_labels_accessed)
    num_labels_used.append(num_labels_accessed)
    accuracies.append(test_accuracy)

num_labels_used = np.array(num_labels_used)
accuracies = np.array(accuracies)

# you can sort the num_labels_used and get the indices of the sorted array
sorted_indices = np.argsort(num_labels_used)

# use these indices to sort both the num_labels_used and accuracies
sorted_num_labels_used = num_labels_used[sorted_indices]
sorted_accuracies = accuracies[sorted_indices]

# now you can plot
plt.figure(figsize=(10, 5))
plt.plot(sorted_num_labels_used, sorted_accuracies, label="Test Accuracy")
plt.xlabel("Number of Labels used")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Number of Labels used vs Accuracy")
plt.text(0.01, 0.95, f'Theta constant: {Theta_constant}', transform=plt.gca().transAxes)
plt.text(0.01, 0.90, f'O constant: {O_constant}', transform=plt.gca().transAxes)
plt.show()
