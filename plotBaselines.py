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

    return x_train, x_test, y_train, y_test




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


TRAIN_SET_SIZE = 1000
d=2
cov1 = np.eye(d)
cov2 = np.eye(d)
cov2[0,0] = 8.
cov2[0,1] = 0.1
cov2[1,0] = 0.1
cov2[1,1] = 0.0024

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train_orig = np.load('y_train_orig.npy')
y_test_orig = np.load('y_test_orig.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')


w_star = np.array([1, 0])
def plot_TNC_baselines():
    B_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    alpha_values = [1 / 2, 1]
    NUM_TRIALS = 50
    default_marker_size = 6
    half_size = default_marker_size / 2

    fig, axs = plt.subplots(len(alpha_values))
    for alpha_index, alpha in enumerate(alpha_values):
        accuracy_values_rf = []
        accuracy_values_lr = []
        std_dev_rf = []
        std_dev_lr = []
        accuracy_values_rf_true = []
        accuracy_values_lr_true = []
        for B in B_values:
            acc_rf = []
            acc_lr = []
            acc_rf_true = []
            acc_lr_true = []
            for num_trial in range(NUM_TRIALS):
                x_train, x_test, y_train_orig, y_test_orig = mixture_gauss(d, TRAIN_SET_SIZE)
                noisy_y_train = add_noise(x_train, y_train_orig, alpha, B)
                noisy_y_test = add_noise(x_test, y_test_orig, alpha, B)
                # Random Forest
                rf_model = RandomForestClassifier(max_depth=5, random_state=0)
                rf_model.fit(x_train, noisy_y_train)
                y_test_pred_rf = rf_model.predict(x_test)
                accuracy_rf = accuracy_score(noisy_y_test, y_test_pred_rf)
                acc_rf.append(accuracy_rf)

                accuracy_rf_true = accuracy_score(y_test_orig, y_test_pred_rf)
                acc_rf_true.append(accuracy_rf_true)

                # Logistic Regression
                lr_model = LogisticRegression(C=50 / 1000, max_iter=200, penalty='l2', solver='liblinear',
                                              fit_intercept=False, tol=0.1)
                lr_model.fit(x_train, noisy_y_train)
                y_test_pred_lr = lr_model.predict(x_test)
                accuracy_lr = accuracy_score(noisy_y_test, y_test_pred_lr)
                acc_lr.append(accuracy_lr)

                accuracy_lr_true = accuracy_score(y_test_orig, y_test_pred_lr)
                acc_lr_true.append(accuracy_lr_true)

            accuracy_values_rf.append(np.mean(acc_rf))
            accuracy_values_lr.append(np.mean(acc_lr))
            std_dev_rf.append(np.std(acc_rf))
            std_dev_lr.append(np.std(acc_lr))

            accuracy_values_rf_true.append(np.mean(acc_rf_true))
            accuracy_values_lr_true.append(np.mean(acc_lr_true))
        #Add alpha and B values along with Gaussian w* and Noise w* values on the plot. 
        axs[alpha_index].errorbar(B_values, accuracy_values_rf, yerr=std_dev_rf, label='RF', fmt='-o',
                                  markersize=half_size)
        axs[alpha_index].errorbar(B_values, accuracy_values_lr, yerr=std_dev_lr, label='LR', fmt='-o',
                                  markersize=half_size)
        axs[alpha_index].plot(B_values, accuracy_values_rf_true, 'r--', label='RF True')
        axs[alpha_index].plot(B_values, accuracy_values_lr_true, 'b--', label='LR True')
        for i, txt in enumerate(accuracy_values_rf):
            axs[alpha_index].annotate("{:.2f}".format(txt), (B_values[i], accuracy_values_rf[i]))
        for i, txt in enumerate(accuracy_values_lr):
            axs[alpha_index].annotate("{:.2f}".format(txt), (B_values[i], accuracy_values_lr[i]))

        axs[alpha_index].set_xlabel('B')
        axs[alpha_index].set_ylabel('Accuracy')
        axs[alpha_index].set_title(f'Alpha = {alpha}, w_star={w_star}')
        axs[alpha_index].legend()

    plt.tight_layout()
    plt.show()


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


bayes_optimal_accuracies = [bayes_optimal_classifier(x, 0.5, 0.3) for x in x_test]
bayes_optimal_accuracy = accuracy_score(y_test, bayes_optimal_accuracies)
# plot_TNC_baselines()

def plot_LR_learning_curve(x_train, y_train, x_test, y_test):
    scores = []
    for _ in range(5):  # Run each experiment 5 times
        score = []
        for i in range(1, 1000):
            try:
                clf = LogisticRegression(C= 100, max_iter=200, penalty='l2', solver='liblinear',
                                         fit_intercept=False, tol=0.1)
                clf.fit(x_train[:i], y_train[:i])
                y_pred = clf.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                score.append(acc)
            except ValueError:
                score.append(None)
                continue
        scores.append(score)

    # Calculate mean and standard deviation of scores
    scores_mean = np.mean(scores, axis=0)
    scores_std = np.std(scores, axis=0)

    # Plot the learning curve with error bars
    plt.errorbar(range(1, 1000), scores_mean, yerr=scores_std, fmt='-o')
    plt.axhline(y=bayes_optimal_accuracy, color='r', linestyle='--', 
                label=f'Bayes Optimal Classifier (alpha={0.5}, B={0.3})')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Logistic Regression Learning Curve')
    plt.savefig('LR_learning_curve.png')
    plt.close()


def plot_RF_learning_curve(x_train, y_train, x_test, y_test):
    scores = []
    for _ in range(5):  # Run each experiment 5 times
        score = []
        for i in range(1, 1000):
            try:
                clf = RandomForestClassifier(max_depth=5, random_state=0)
                clf.fit(x_train[:i], y_train[:i])
                y_pred = clf.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                score.append(acc)
            except ValueError:
                score.append(None)
                continue
        scores.append(score)

    # Calculate mean and standard deviation of scores
    scores_mean = np.mean(scores, axis=0)
    scores_std = np.std(scores, axis=0)

    # Plot the learning curve with error bars
    plt.errorbar(range(1, 1000), scores_mean, yerr=scores_std, fmt='-o')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Random Forest Learning Curve')
    plt.savefig('RF_learning_curve.png')
    plt.close()



plot_LR_learning_curve(x_train, y_train, x_test, y_test)

plot_RF_learning_curve(x_train, y_train, x_test, y_test)


def plot_multiple_LR_learning_curves(x_train, y_train, x_test, y_test):
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.ravel() 

    for idx, C in enumerate(C_values):
        scores = []
        for _ in range(5):  # Run each experiment 5 times
            score = []
            for i in range(1, 1000):
                try:
                    clf = LogisticRegression(C=C, max_iter=200, penalty='l2', solver='liblinear',
                                             fit_intercept=False, tol=0.1)
                    clf.fit(x_train[:i], y_train[:i])
                    y_pred = clf.predict(x_test)
                    acc = accuracy_score(y_test, y_pred)
                    score.append(acc)
                except ValueError:
                    score.append(None)
                    continue
            scores.append(score)

        # Calculate mean and standard deviation of scores
        scores_mean = np.mean(scores, axis=0)
        scores_std = np.std(scores, axis=0)

        # Plotting on each subplot with error bars
        axs[idx].errorbar(range(1, 1000), scores_mean, yerr=scores_std, fmt='-o')
        axs[idx].set_title(f'Learning Curve for C={C}')
        axs[idx].set_xlabel('Number of Training Instances')
        axs[idx].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig("Varying_C_LR_Learning_Curve.png")
    plt.close()

def plot_multiple_RF_learning_curves(x_train, y_train, x_test, y_test):
    max_depth_values = [1, 3, 10, 30, 100]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust to accommodate 5 subplots
    axs = axs.ravel()  # Flatten the array for easy indexing

    for idx, max_depth in enumerate(max_depth_values):
        scores = []
        for _ in range(5):  # Run each experiment 5 times
            score = []
            for i in range(1, 1000):
                try:
                    clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
                    clf.fit(x_train[:i], y_train[:i])
                    y_pred = clf.predict(x_test)
                    acc = accuracy_score(y_test, y_pred)
                    score.append(acc)
                except ValueError:
                    score.append(None)
                    continue
            scores.append(score)

        # Calculate mean and standard deviation of scores
        scores_mean = np.mean(scores, axis=0)
        scores_std = np.std(scores, axis=0)

        # Plotting on each subplot with error bars
        axs[idx].errorbar(range(1, 1000), scores_mean, yerr=scores_std, fmt='-o')
        axs[idx].set_title(f'Learning Curve for max_depth={max_depth}')
        axs[idx].set_xlabel('Number of Training Instances')
        axs[idx].set_ylabel('Accuracy')

    # Remove extra subplots (since we only need 5)
    for idx in range(len(max_depth_values), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.savefig("Varying_max_depth_RF_Learning_Curve.png")
    plt.close()


# plot_multiple_RF_learning_curves(x_train, y_train, x_test, y_test)
# plot_multiple_LR_learning_curves(x_train, y_train, x_test, y_test)