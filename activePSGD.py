import argparse
import sys
import time
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
import matplotlib.image as mpimg
from MakeData import add_noise_single, single_gauss, mixture_gauss, add_noise
import math
import traceback
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=0.5, help='First input integer')
    parser.add_argument('--B', type=float, default=0.3, help='Second input integer')
    parser.add_argument('--sigma', type=float, default=0.001, help='First input integer')
    parser.add_argument('--beta', type=float, default=0.003, help='Second input integer')
    args = parser.parse_args()
    print(args.sigma)

x_test = np.load('x_test.npy')
y_test_orig = np.load('y_test_orig.npy')
y_test = np.load('y_test.npy')



TRAIN_SET_SIZE = 100000
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


cov1 = np.eye(d)
cov2 = np.eye(d)
cov2[0,0] = 8.
cov2[0,1] = 0.1
cov2[1,0] = 0.1
cov2[1,1] = 0.0024


def phi_prime_of_sigma_t(w, x):
    i = np.dot(w, x) / np.linalg.norm(w)
    if i < 0.0:
        i = -i
    return abs(np.exp(-i/sigma)/(sigma*(1+np.exp(-i/sigma))**2))



def ACTIVE_FO(w):
    x,y = single_gauss(d, cov1, cov2)
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

execution_times = []
def TNC_Learning_New(epsilon, delta):
    iterate_accuracies_noisy = []
    iterate_accuracies = []
    iterate_labels_used = []
    w1 = np.random.normal(size=d)
    w1 = w1 / np.linalg.norm(w1)
    w = [None]*(1000)
    w[0] = w1
    i = 1
    start_time = time.time()
    while i < 1000:
        gi, li = ACTIVE_FO(w[i-1])
        vi = w[i-1] - beta*gi
        w[i] = vi / np.linalg.norm(vi)
        predictions = np.dot(x_test, np.mean(w[i//2:i+1], axis=0))
        for j in range(0,len(predictions)):
            if predictions[j] < 0.0:
                predictions[j] = -1
            else:
                predictions[j] = 1
        if li == 1:
            iterate_accuracies_noisy.append(accuracy_score(y_test, predictions))
            iterate_accuracies.append(accuracy_score(y_test_orig,predictions))
            iterate_labels_used.append(i)
            elapsed_time = time.time() - start_time
            execution_times.append(elapsed_time)
            start_time = time.time()
        i += li
        if (len(execution_times) == 1000):
            plt.plot(execution_times, marker='o', linestyle='-')
            plt.xlabel('Iteration')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Execution Time for Each Iteration')
            plt.grid(True)
            plt.savefig("execution_times_iterate.png")

    return iterate_accuracies_noisy, iterate_accuracies, iterate_labels_used


# N = d / (sigma**2 * rho**4)
beta = rho**2 * sigma**2 / d
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
    if (np.dot(x,w_star) > 0):
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
num_trials = 5
all_accuracies_noisy = []
all_accuracies = []

for trial in range(num_trials):
    print(f"Starting trial {trial+1}")
    
    try:
        iterate_accuracies_noisy, iterate_accuracies, iterate_labels_used = TNC_Learning_New(epsilon, delta)
        all_accuracies_noisy.append(iterate_accuracies_noisy)
        all_accuracies.append(iterate_accuracies)
    except:
        traceback.print_exc()


avg_accuracies_noisy = np.mean(all_accuracies_noisy, axis=0)
std_accuracies_noisy = np.std(all_accuracies_noisy, axis=0)

avg_accuracies = np.mean(all_accuracies, axis=0)
std_accuracies = np.std(all_accuracies, axis=0)

def tensorboard_plot(avg_accuracies_noisy, std_accuracies_noisy, avg_accuracies, std_accuracies, sigma, beta):
    writer = SummaryWriter()
    layout = {
    "ABCDE": {
        f"Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}": ["Multiline", [f"Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean", f"Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_minus_std", f"Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_plus_std"]],
        f"True_Label_Accuracy_sigma_{sigma}_beta_{beta}": ["Multiline", [f"True_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean", f"True_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_minus_std", f"True_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_plus_std"]],
    },
    }
    writer.add_custom_scalars(layout)

    for i, mean in enumerate(avg_accuracies_noisy):
        writer.add_scalar(f'Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean', mean, i)
        writer.add_scalar(f'Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_minus_std', mean - std_accuracies_noisy[i], i)
        writer.add_scalar(f'Noisy_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_plus_std', mean + std_accuracies_noisy[i], i)

    for i, mean in enumerate(avg_accuracies): 
        writer.add_scalar(f'True_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean', mean, i)
        writer.add_scalar(f'True_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_minus_std', mean - std_accuracies[i], i)
        writer.add_scalar(f'True_Label_Accuracy_sigma_{sigma}_beta_{beta}/Mean_plus_std', mean + std_accuracies[i], i)
        
    writer.flush()
    writer.close()


tensorboard_plot(avg_accuracies_noisy, std_accuracies_noisy, avg_accuracies, std_accuracies, sigma, beta)

scores_mean_LR = np.load('LR_scores_mean.npy')
scores_std_LR = np.load('LR_scores_std.npy')

scores_mean_RF = np.load('RF_scores_mean.npy')
scores_std_RF = np.load('RF_scores_std.npy')


fig = plt.figure(figsize=(10,8))

ax3 = fig.add_subplot(1,1,1)

ax3.errorbar(range(1, 1000), scores_mean_LR, yerr=scores_std_LR, fmt='-', label='LR Learning Curve', color='green', linewidth=0.5)
ax3.errorbar(range(1, 1000), scores_mean_RF, yerr=scores_std_RF, fmt='-', label='RF Learning Curve', color='purple', linewidth=0.5)
ax3.axhline(y=bayes_optimal_accuracy, color='r', linestyle='--', 
                label=f'Bayes Optimal Classifier')
l_noisy = ax3.errorbar(iterate_labels_used, avg_accuracies_noisy, linestyle='-', marker='.', label='Noisy Accuracies', color='blue', markersize=2)
ax3.fill_between(iterate_labels_used, 
                 avg_accuracies_noisy - std_accuracies_noisy, 
                 avg_accuracies_noisy + std_accuracies_noisy, 
                 color='blue', alpha=0.4)

l_true = ax3.errorbar(iterate_labels_used, avg_accuracies,  linestyle=':', marker='.',  label='True Accuracies', color='orange', markersize=2)
ax3.fill_between(iterate_labels_used, 
                 avg_accuracies - std_accuracies, 
                 avg_accuracies + std_accuracies, 
                 color='orange', alpha=0.4)

ax3.axhline(y=bayes_optimal_accuracy, color='r', linestyle='--', 
            label=f'Bayes Optimal Classifier (alpha={args.alpha}, B={args.B})')

ax3.set_title(f"Sigma = {sigma}, Beta = {beta}")
ax3.legend(loc='lower right')

ax3.set_xlabel("Labels Accessed")
ax3.set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig(f"plot_sigma_{sigma}_beta_{beta}.png")
plt.close()
