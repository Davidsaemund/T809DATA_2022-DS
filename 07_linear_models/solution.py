# Author: Davíð Sæmundsson
# Date: 5.10.2022
# Project: Linear models
# Acknowledgements: Got help with Eyjólfur in class on Wednesday and then
# discussing assignment with Björgvin Freyr
# Some code used from example on piazza 

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    phi = np.ones([features.shape[0], mu.shape[0]])
    for n in range(phi.shape[0]):
        for m in range(phi.shape[1]):
            phi[n, m] = multivariate_normal(mu[m, :], np.identity(
                mu.shape[1])*sigma).pdf(features[n, :])

    return phi


def _plot_mvn(
    phi: np.ndarray,
):
    for i in range(phi.shape[1]):
        plt.plot(phi[:, i], label="function "+str(i+1))
    plt.legend(loc="lower left")
    plt.show()
    return 0


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    return np.matmul(np.matmul(np.linalg.inv(np.eye(fi.shape[1])*lamda + np.matmul(fi.T, fi)), fi.T), targets)


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    return np.matmul(w.T, mvn_basis(features, mu, sigma).T).T


# X, t = load_regression_iris()
# N, D = X.shape

# M, sigma = 10, 10
# mu = np.zeros((M, D))
# for i in range(D):
#     mmin = np.min(X[i, :])
#     mmax = np.max(X[i, :])
#     mu[:, i] = np.linspace(mmin, mmax, M)
# # print(mu)

# fi = mvn_basis(X, mu, sigma)
# #print(fi)
# #print(fi.shape[0], fi.shape[1])
# #print(t)
# #_plot_mvn(fi)

# lamda = 0.001
# wml = max_likelihood_linreg(fi, t, lamda)
# #print(wml)
# prediction = linear_model(X, mu, sigma, wml)
# #print(prediction)
# #print(t)
# plt.plot(prediction, label='Prediction values')
# plt.plot(t, label='Target values')
# plt.legend(loc='upper left')
# plt.show()
