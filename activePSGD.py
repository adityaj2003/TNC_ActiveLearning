import argparse
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
import traceback


parser = argparse.ArgumentParser()

parser.add_argument('--alpha', type=float, default=0.5, help='First input integer')
parser.add_argument('--B', type=float, default=0.3, help='Second input integer')

args = parser.parse_args()



x_train = np.load('x_train.npy')
x_val = np.load('x_val.npy')
x_test = np.load('x_test.npy')
y_train_orig = np.load('y_train_orig.npy')
y_test_orig = np.load('y_test_orig.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')
y_val_orig = np.load('y_val_orig.npy')

TRAIN_SET_SIZE = len(x_train)
d = 2
epsilon = 0.1
delta = 0.1
A = 1
alpha = 0.5
label_used_array = [False]*TRAIN_SET_SIZE
O_constant = 50
Theta_constant = 10

theta = O_constant * ((1 / np.log2(1 / epsilon))**2) * (epsilon/2)
sigma = Theta_constant * (1/A)**((1-alpha)/(3*alpha-1)) * theta**((2*alpha)/(3*alpha-1))
rho = Theta_constant * (1/A)**((2*(1-alpha))/(3*alpha-1)) * theta**((2*(1-alpha))/(3*alpha-1))
S = np.log(6 / delta)

print("sigma: ", sigma)

def single_gauss(d):
    vecs = np.zeros((1, d))
    if np.random.uniform() > 0.5:
        vecs[0, :] = np.random.multivariate_normal([0]*d, cov1)
    else:
        vecs[0, :] = np.random.multivariate_normal([0]*d, cov2)

    x_train = vecs
    y_train = (vecs[0, 1] > 0).astype(int) * 2 - 1

    return x_train, y_train

def phi_prime_of_sigma_t(w, x):
    i = np.dot(w, x) / np.linalg.norm(w)
    return abs(np.exp(-i/sigma)/(sigma*(1+np.exp(-i/sigma))**2))



def ACTIVE_FO(w):
    x,y = single_gauss(d)
    q_wx = sigma * phi_prime_of_sigma_t(w, x)
    Z = bernoulli.rvs(q_wx)
    if Z == 1:
        num_labels_accessed += 1
        w_norm = np.linalg.norm(w)
        h_wxy = -1/sigma * y * (x/w_norm**2 - np.dot(w, x) * w / w_norm**3)
        assert np.isclose(np.dot(h_wxy, w), 0), 1
        return h_wxy
    else:
        return np.zeros_like(w), 0

iterate_accuracies_noisy = []
iterate_accuracies = []
iterate_labels_used = []

def ACTIVE_PSGD(N, beta):
    w1 = np.random.normal(size=d)
    w1 = w1 / np.linalg.norm(w1)
    print("w1", w1)
    w = [None]*(N+1)
    w[0] = w1
    for i in range(1, N+1):
        gi = ACTIVE_FO(w[i-1])
        vi = w[i-1] - beta*gi
        w[i] = vi / np.linalg.norm(vi)
        predictions = np.dot(x_test, w[i])
        for j in range(0,len(predictions)):
            if predictions[j] < 0.0:
                predictions[j] = -1
            else:
                predictions[j] = 1
        iterate_accuracies_noisy.append(accuracy_score(y_test, predictions))
        iterate_accuracies.append(accuracy_score(y_test_orig,predictions))
        iterate_labels_used.append(num_labels_accessed)
    R = np.random.randint(N)
    print("wLast",w[-1])
    return w[R]

def TNC_Learning_New(epsilon, delta):
    w1 = np.random.normal(size=d)
    w1 = w1 / np.linalg.norm(w1)
    w = [None]*(1000000)
    w[0] = w1
    print("w1", w1)
    i = 1
    global num_labels_accessed
    num_labels_accessed = 0
    while li < 1000:
        gi, li = ACTIVE_FO(w[i-1])
        vi = w[i-1] - beta*gi
        w[i] = vi / np.linalg.norm(vi)
        predictions = np.dot(x_test, w[i])
        for j in range(0,len(predictions)):
            if predictions[j] < 0.0:
                predictions[j] = -1
            else:
                predictions[j] = 1
        iterate_accuracies_noisy.append(accuracy_score(y_test, predictions))
        iterate_accuracies.append(accuracy_score(y_test_orig,predictions))
        iterate_labels_used.append(num_labels_accessed)
        i += li

#Store





N = d / (sigma**2 * rho**4)
beta = rho**2 * sigma**2 / d
print("beta: ", beta)

# def TNC_learning(epsilon, delta,N):
#     num_labels_accessed = 0
#     index = 0
#     # Constants for Big O and Big Theta
#     O_constant = 50
#     Theta_constant = 50
#     theta = O_constant * ((1 / np.log2(1 / epsilon)) ** 2) * (epsilon / 2)
#     sigma = Theta_constant * (1 / A) ** ((1 - alpha) / (3 * alpha - 1)) * theta ** ((2 * alpha) / (3 * alpha - 1))
#     rho = Theta_constant * (1 / A) ** ((2 * (1 - alpha)) / (3 * alpha - 1)) * theta ** (
#                 (2 * (1 - alpha)) / (3 * alpha - 1))
#     S = np.log(6 / delta)
#     w_s = []
#     for s in range(math.ceil(S)):
#         beta = rho ** 2 * sigma ** 2 / d
#         w = ACTIVE_PSGD(N, beta)
#         w_s.append(w)

#     M1 = 100 

#     g_s = []
#     for s in range(math.ceil(S)):
#         g_s_i = []
#         for _ in range(M1):
#             g_s_i.append(ACTIVE_FO(w_s[s]))
#         g_s.append(np.mean(g_s_i, axis=0)) 

#     s_star = np.argmin(np.linalg.norm(g_s, axis=1))
#     w_tilde = w_s[s_star]

#     errors = []
#     for w in [w_tilde, -w_tilde]:
#         error = 0
#         for i in range(len(x_val)):
#             x_i = x_val[i]
#             y_i = y_val[i]
#             if np.sign(np.dot(w, x_i)) != y_i:
#                 error += 1
#         errors.append(error / len(x_val))

#     w_hat = [w_tilde, -w_tilde][np.argmin(errors)]

#     return w_hat



lr_model = LogisticRegression(C= 50 / 1000, max_iter = 200,
                             penalty='l2', solver='liblinear',
                             fit_intercept=False,
                             tol=0.1)
lr_model.fit(x_train, y_train)
y_test_pred_lr = lr_model.predict(x_test)

lr_accuracy = accuracy_score(y_test, y_test_pred_lr)


# N_values = np.linspace(100, 2000, num=100) 
num_sigmas = 8
fig, axs = plt.subplots(num_sigmas, num_sigmas, figsize=(30,30))
num_labels_used = []
sigmas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
betas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]

for i, sigmaIterate in enumerate(sigmas):
    accuracies = []
    for l, betaIterate in enumerate(betas):
        try:
            sigma = sigmaIterate
            beta = betaIterate
            iterate_accuracies = []
            iterate_labels_used = []
            TNC_Learning_New(epsilon, delta)
            #Make TNC_LEARNING_NEW return iterate_labels_used and iterate_accuracies instead of declaring them as global variables
            axs[l,i].plot(iterate_labels_used, iterate_accuracies, label="Test Accuracy")
            axs[l,i].set_title(f"Sigma = {sigma}, Beta = {beta}")
            axs[l,i].set_xlabel("Beta")
            axs[l,i].set_ylabel("Accuracy")
        except:
            traceback.print_exc()

plt.tight_layout()
plt.show()



# num_labels_used = np.array(num_labels_used)
# accuracies = np.array(accuracies)


# plt.figure(figsize=(10, 5))
# plt.plot(num_labels_used, accuracies, label="Test Accuracy")
# plt.xlabel("Number of Labels used")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Number of Labels used vs Accuracy")
# plt.text(0.01, 0.95, f'Theta constant: {Theta_constant}', transform=plt.gca().transAxes)
# plt.text(0.01, 0.90, f'O constant: {O_constant}', transform=plt.gca().transAxes)
# plt.show()


def bayes_optimal_classifier(x,alpha,B,w_star=np.array([1, 0])):
    prediction = None
    if (x[0] > 0):
        prediction = 1
    else:
        prediction = -1
    h_w_x = np.dot(x, w_star)
    p_flip = 0.5 - np.minimum(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    if (p_flip > 0.5):
        prediction = -prediction
    return prediction
    
def plot_iterate_accuracies_active_psgd(alpha_value, B_value):
    y_pred = np.array([bayes_optimal_classifier(x, alpha_value, B_value) for x in x_test])

    bayes_optimal_accuracy = accuracy_score(y_test, y_pred)

    ws = ACTIVE_PSGD(math.ceil(10000), beta)
    plt.figure(figsize=(10,6))

    plt.plot(iterate_labels_used, iterate_accuracies_noisy, label='Noisy Accuracies', linestyle='-')
    plt.plot(iterate_labels_used, iterate_accuracies, label='True Accuracies', linestyle=':')
    plt.axhline(y=bayes_optimal_accuracy, color='r', linestyle='--', 
                label=f'Bayes Optimal Classifier (alpha={alpha_value}, B={B_value})')
    plt.xlabel('Labels Used')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Labels Used')
    plt.legend()
    plt.show()


