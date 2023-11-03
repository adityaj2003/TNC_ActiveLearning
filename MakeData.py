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
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.stats
import argparse


class DataGenerator:
    def __init__(self, d, N, w_star=np.array([1, 0]), frac=0.25):
        self.d = d
        self.N = N
        self.w_star = w_star
        self.frac = frac
        self.temp = 0
        
    def uniform_ball_distribution_from_gaussian(self):
        total = int(self.N * (self.frac + 1))
        random_points = np.random.randn(total, 3)
        normalized_points = random_points / np.linalg.norm(random_points, axis=1)[:, None]
        features = normalized_points[:, :2]

        N_train = int(0.8 * self.N)

        x_train = features[:N_train, :]
        x_test = features[N_train:, :]

        y_train = (np.dot(features[:N_train, :], self.w_star) > 0).astype(int) * 2 - 1
        y_test = (np.dot(features[N_train:, :], self.w_star) > 0).astype(int) * 2 - 1

        return x_train, x_test, y_train, y_test

    def mixture_gauss(self, cov1, cov2):
        total = int(self.N * (self.frac + 1))
        vecs = np.zeros((total, self.d))
        for i in range(total):
            if np.random.uniform() > 0.5:
                vecs[i, :] = np.random.multivariate_normal([0]*self.d, cov1)
            else:
                vecs[i, :] = np.random.multivariate_normal([0]*self.d, cov2)

        N_train = int(0.8 * self.N)

        x_train = vecs[:N_train, :]
        x_test = vecs[N_train:, :]

        y_train = (np.dot(vecs[:N_train, :], self.w_star) > 0).astype(int) * 2 - 1
        y_test = (np.dot(vecs[N_train:, :], self.w_star) > 0).astype(int) * 2 - 1

        return x_train, x_test, y_train, y_test
        
    def add_noise(self, features, labels, alpha, B):
        h_w_x = np.dot(features, self.w_star)
        p_flip = 0.5 - np.minimum(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
        flip = (np.random.rand(len(labels)) < p_flip)
        noisy_labels = labels.copy()
        noisy_labels[flip] = -noisy_labels[flip]
        return noisy_labels

    def single_gauss(self, cov1, cov2):
        vecs = np.zeros((1, self.d))
        if np.random.uniform() > 0.5:
            vecs[0, :] = np.random.multivariate_normal([0]*self.d, cov1)
        else:
            vecs[0, :] = np.random.multivariate_normal([0]*self.d, cov2)

        x_train = vecs
        y_train = (vecs[0, 0] > 0).astype(int) * 2 - 1
        return x_train[0], y_train

    def single_point_uniform_ball(self):
        random_point = np.random.randn(3)
        normalized_point = random_point / np.linalg.norm(random_point)
        feature = normalized_point[:2]
        label = (np.dot(feature, self.w_star) > 0).astype(int) * 2 - 1
        return feature, label





    def add_noise_single(self, feature, label, alpha, B):
        h_w_x = np.dot(feature, self.w_star)
        p_flip = 0.5 - min(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
        flip = (np.random.rand() < p_flip)
        noisy_label = label
        if flip:
            noisy_label = -noisy_label
        return noisy_label

    def determine_area(self, x, w, alpha):
        angle_x_w = np.arccos(np.abs(np.dot(x, w)) / (np.linalg.norm(x) * np.linalg.norm(w)))
        angle_x_w_star = np.arccos(np.abs(np.dot(x, self.w_star)) / (np.linalg.norm(x) * np.linalg.norm(self.w_star)))
    
        
        angle_x_h_w = np.pi/2 - angle_x_w
        angle_x_h_w_star = np.pi/2 - angle_x_w_star
        
        prediction_w = np.dot(x, w) > 0
        prediction_w_star = np.dot(x, self.w_star) > 0
        
        if angle_x_h_w_star <= alpha and prediction_w != prediction_w_star:
            return "A"
        elif angle_x_h_w < angle_x_h_w_star:
            return "B"
        else:
            return "D"




    def add_described_noise(self, feature, label, alpha, eta):
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        w = np.dot(rot_matrix, self.w_star/np.linalg.norm(self.w_star))

        noisy_label = label
        area = self.determine_area(feature, w, alpha)
        if area in ["A", "B"] and np.random.rand() < eta:
            noisy_label = -noisy_label
        
        return noisy_label
    
    def draw_data(self, method='uniform_ball_distribution_from_gaussian', **kwargs):
        if method == 'uniform_ball_distribution_from_gaussian':
            return self.uniform_ball_distribution_from_gaussian()
        elif method == 'mixture_gauss':
            return self.mixture_gauss(**kwargs)
        elif method == 'add_noise':
            features, labels = kwargs.get('features'), kwargs.get('labels')
            alpha, B = kwargs.get('alpha'), kwargs.get('B')
            return self.add_noise(features, labels, alpha, B)
        elif method == 'single_gauss':
            cov1, cov2 = kwargs.get('cov1'), kwargs.get('cov2')
            return self.single_gauss(cov1, cov2)
        elif method == 'single_point_uniform_ball':
            return self.single_point_uniform_ball()
        elif method == 'add_noise_single':
            feature, label = kwargs.get('feature'), kwargs.get('label')
            alpha, B = kwargs.get('alpha'), kwargs.get('B')
            return self.add_noise_single(feature, label, alpha, B)
        elif method == 'add_described_noise':
            feature, label, alpha = kwargs.get('feature'), kwargs.get('label'), kwargs.get('alpha')
            return self.add_described_noise(feature, label, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")


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

def bayes_optimal_classifier(x,alpha,B,w_star=np.array([1, 0])):
    prediction = None
    if (np.dot(x,w_star) > 0):
        prediction = 1
    else:
        prediction = -1
    # h_w_x = np.dot(x, w_star)
    # p_flip = 0.5 - np.minimum(1/2, B * (np.abs(h_w_x)**((1-alpha)/alpha)))
    # if (p_flip > 0.5):
    #     prediction = -prediction
    return prediction


if __name__ == '__main__':

    alpha = 1.081
    eta = 0.3

    np.random.seed(0)

    TRAIN_SET_SIZE = 100000
    d = 2
    cov1 = np.eye(d)
    cov2 = np.eye(d)
    cov2[0, 0] = 8.
    cov2[0, 1] = 0.1
    cov2[1, 0] = 0.1
    cov2[1, 1] = 0.0024
    w_star = np.array([1, 0])

    data_gen = DataGenerator(d, TRAIN_SET_SIZE, w_star=w_star)
    x_train, x_test, y_train_orig, y_test_orig = data_gen.draw_data(
        method='uniform_ball_distribution_from_gaussian', cov1=cov1, cov2=cov2)
    y_train = [data_gen.add_described_noise(x, y, alpha, eta) for x, y in zip(x_train, y_train_orig)]
    y_test = [data_gen.add_described_noise(x, y, alpha, eta) for x, y in zip(x_test, y_test_orig)]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x, y = x_test[:, 0], x_test[:, 1]
    colors = ['blue' if label == 1 else 'red' for label in y_test]

    plt.scatter(x, y, color=colors, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.axis('equal')
    plt.show()
    
    # Save the data to disk
    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.save('y_train_orig.npy', y_train_orig)
    np.save('y_test_orig.npy', y_test_orig)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    # alphas = np.linspace(0.05, np.pi, 10)
    # etas = np.arange(0.05, 0.55, 0.05)

    # accuracy_differences = np.zeros((len(alphas), len(etas)))

    # for i, alpha in enumerate(alphas):
    #     for j, eta in enumerate(etas):
    #         accuracies_diff = 0
    #         data_gen = DataGenerator(d, 100000, w_star=np.array([1, 0]))
    #         x_train, x_test, y_train_orig, y_test_orig = data_gen.draw_data(
    #                 method='uniform_ball_distribution_from_gaussian', cov1=cov1, cov2=cov2)
    #         y_train = [data_gen.add_described_noise(x, y, alpha, eta) for x, y in zip(x_train, y_train_orig)]
    #         y_test = [data_gen.add_described_noise(x, y, alpha, eta) for x, y in zip(x_test, y_test_orig)]
    #         y_train = np.array(y_train)
    #         y_test = np.array(y_test)
    #         for run in range(5):
                
    #             clf = LogisticRegression(C=100, max_iter=200, penalty='l2', solver='liblinear',
    #                                     fit_intercept=False, tol=0.1)
    #             clf.fit(x_train[run*1000:run*1000+1000], y_train[run*1000:run*1000+1000])
                
    #             lr_predictions = clf.predict(x_test)
    #             lr_accuracy = accuracy_score(y_test, lr_predictions)
                
    #             bayes_optimal_accuracies = [bayes_optimal_classifier(x, alpha, 0.3) for x in x_test]
    #             bayes_optimal_accuracy = accuracy_score(y_test, bayes_optimal_accuracies)
                
    #             accuracies_diff += (bayes_optimal_accuracy - lr_accuracy) * 100
            
    #         accuracy_differences[i, j] = accuracies_diff / 5
    # sorted_indices = np.argsort(accuracy_differences, axis=None)[::-1]
    # for idx in sorted_indices:
    #     i, j = np.unravel_index(idx, accuracy_differences.shape)
    #     print(f'Alpha: {alphas[i]:.3f}, Eta: {etas[j]:.3f}, Accuracy Difference: {accuracy_differences[i, j]:.3f}\n')
