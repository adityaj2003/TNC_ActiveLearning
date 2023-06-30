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
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import bernoulli
import math
import traceback


parser = argparse.ArgumentParser()

parser.add_argument('--alpha', type=float, default=0.5, help='First input integer')
parser.add_argument('--B', type=float, default=0.3, help='Second input integer')
parser.add_argument('--sigma', type=float, default=0.5, help='First input integer')
parser.add_argument('--beta', type=float, default=0.3, help='Second input integer')
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

print("x_train 1 val", x_train[0])

TRAIN_SET_SIZE = len(x_train)
d = 2
epsilon = 0.1
delta = 0.1
A = 1
alpha = 0.5
O_constant = 50
num_labels_accessed = 0
Theta_constant = 10

theta = O_constant * ((1 / np.log2(1 / epsilon))**2) * (epsilon/2)
sigma = Theta_constant * (1/A)**((1-alpha)/(3*alpha-1)) * theta**((2*alpha)/(3*alpha-1))
rho = Theta_constant * (1/A)**((2*(1-alpha))/(3*alpha-1)) * theta**((2*(1-alpha))/(3*alpha-1))
S = np.log(6 / delta)
print("sigma", sigma)

def single_gauss(d):
    vecs = np.zeros((1, d))
    if np.random.uniform() > 0.5:
        vecs[0, :] = np.random.multivariate_normal([0]*d, cov1)
    else:
        vecs[0, :] = np.random.multivariate_normal([0]*d, cov2)

    x_train = vecs
    y_train = (vecs[0, 1] > 0).astype(int) * 2 - 1
    return x_train[0], y_train

d=2
cov1 = np.eye(d)
cov2 = np.eye(d)
cov2[0,0] = 8.
cov2[0,1] = 0.1
cov2[1,0] = 0.1
cov2[1,1] = 0.0024

def add_noise_single(feature, label, alpha, B, w_star=np.array([1, 0])):
    h_w_x = np.dot(feature, w_star)
    # Compute the probability of flipping the label
    p_flip = 0.5 - min(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    flip = (np.random.rand() < p_flip)
    noisy_label = label
    if flip:
        noisy_label = -noisy_label

    return noisy_label

def phi_prime_of_sigma_t(w, x):
    i = np.dot(w, x) / np.linalg.norm(w)
    if i < 0.0:
        i = -i
    return abs(np.exp(-i/sigma)/(sigma*(1+np.exp(-i/sigma))**2))



def ACTIVE_FO(w):
    x,y = single_gauss(d)
    y = add_noise_single(x, y, args.alpha, args.B)
    q_wx = sigma * phi_prime_of_sigma_t(w, x)
    Z = bernoulli.rvs(q_wx)
    if Z == 1:
        w_norm = np.linalg.norm(w)
        h_wxy = -1/sigma * y * (x/w_norm**2 - np.dot(w, x) * w / w_norm**3)
        assert np.isclose(np.dot(h_wxy, w), 0)
        return h_wxy, 1
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
    global num_labels_accessed
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
    print("w1", w1)
    w = [None]*(1000000)
    w[0] = w1
    i = 1
    while i < 10000:
        (gi, li) = ACTIVE_FO(w[i-1])
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
        iterate_labels_used.append(i)
        i += li


# N = d / (sigma**2 * rho**4)
beta = rho**2 * sigma**2 / d
print("beta", beta)
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

def bayes_optimal_classifier(x,alpha,B,w_star=np.array([1, 0])):
    prediction = None
    if (np.dot(x,w_star) > 0.0):
        prediction = 1
    else:
        prediction = -1
    h_w_x = np.dot(x, w_star)
    p_flip = 0.5 - np.minimum(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    if (p_flip > 0.5):
        prediction = -prediction
    return prediction


bayes_optimal_accuracies = [bayes_optimal_classifier(x,args.alpha,args.B) for x in x_test]
bayes_optimal_accuracy = accuracy_score(y_test, bayes_optimal_accuracies)
plt.figure(figsize=(10,6))
sigma = args.sigma
beta = args.beta
try:
    iterate_accuracies = []
    iterate_labels_used = []
    TNC_Learning_New(epsilon, delta)
    plt.plot(iterate_labels_used, iterate_accuracies_noisy, label='Noisy Accuracies', linestyle='-')
    plt.plot(iterate_labels_used, iterate_accuracies, label='True Accuracies', linestyle=':')
    plt.axhline(y=bayes_optimal_accuracy, color='r', linestyle='--', 
                label=f'Bayes Optimal Classifier (alpha={args.alpha}, B={args.B})')
    plt.title(f"Sigma = {sigma}, Beta = {beta}")
    plt.xlabel("Labels Accessed")
    plt.ylabel("Accuracy")
except:
    traceback.print_exc()

plt.tight_layout()
plt.savefig(f"plot_sigma_{sigma}_beta_{beta}.png")
plt.close()  # Close the plot



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


