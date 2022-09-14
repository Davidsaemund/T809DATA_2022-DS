# Author: Davíð Sæmundsson
# Date: 08.09.2022
# Project: Assignment 4, Classification Based on Probability 
# Acknowledgements: Worked on together with Björgvin Freyr
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    mean = []
    for i in range(targets.shape[0]):
        if targets[i] == selected_class:
            mean.append(features[i,:])
    mean_array = np.array(mean)

    return np.mean(mean_array,axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    cov = []
    for i in range(targets.shape[0]):
        if targets[i] == selected_class:
            cov.append(features[i,:])
    cov_array = np.array(cov)

    return np.cov(cov_array, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    #print(multivariate_normal(mean=class_mean, cov=class_cov).pdf(feature))
    return multivariate_normal.pdf(feature,class_mean,class_covar)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        new_list = []
        for c in range(len(classes)):
            new_list.append(likelihood_of_class(test_features[i], means[c], covs[c]))
        likelihoods.append(new_list)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    predict_1 = []
    for i in range(likelihoods.shape[0]):
        predict_1.append(np.argmax(likelihoods[i,:]))
    return np.array(predict_1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    priori = prior(train_targets,classes)
    print(priori)
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        new_list = []
        for c in range(len(classes)):
            new_list.append(likelihood_of_class(test_features[i], means[c], covs[c])*priori[c])
        likelihoods.append(new_list)
    return np.array(likelihoods)


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    samples_count = len(targets)
    class_probs = []
    for c in classes:
        class_count = 0
        for t in targets:
            if t == c:
                class_count += 1
        class_probs.append( class_count / samples_count )
    #print(class_probs)
    return class_probs


def confusion_mat(predictions: np.ndarray, classes: list, test_targets: np.ndarray) :
    confusion_matrix = []
    for c in classes:
        confusion_matrix.append(len(classes)*[0])
    for i in range(len(test_targets)):
        confusion_matrix[predictions[i]][test_targets[i]] += 1
    return confusion_matrix

    
def accuracy(predictions: np.ndarray, test_targets: np.ndarray):
    sum_corrects = 0
    for i in range(test_targets.shape[0]):
        if predictions[i] == test_targets[i]:
            sum_corrects += 1
    return sum_corrects/test_targets.shape[0]
#Test case S1.1
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)
# test_1_1 = mean_of_class(train_features, train_targets, 0)
# print(test_1_1)

#Test case S1.2
# test_1_2 = covar_of_class(train_features, train_targets, 0)
# print(test_1_2)

# Test case S1.3
# class_mean = mean_of_class(train_features, train_targets, 0)
# class_cov = covar_of_class(train_features, train_targets, 0)
# test_1_3 = likelihood_of_class(test_features[0, :], class_mean, class_cov)
# print(test_1_3)

#Test case S1.4
#test_1_4 = maximum_likelihood(train_features, train_targets, test_features, classes)
#print(test_1_4)

#Test case S1.5
# likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
# test_1_5 = predict(likelihoods)
# #print(test_1_5)
# con_m_1 = confusion_mat(test_1_5,classes,test_targets)
# print(con_m_1)
# acc_1 = accuracy(test_1_5,test_targets)
# print(acc_1)

# for i in range(test_targets.shape[0]):
#     if test_1_5[i] == test_targets[i]:
#         print("1",end="")
#     else:
#         print("0",end="")

#Test case S2.1
# aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
# #print(aposteriori)
# test_2_1 = predict(aposteriori)
# print(test_2_1)
# con_m_2 = confusion_mat(test_2_1,classes, test_targets)
# print(con_m_2)
# acc_2 = accuracy(test_2_1,test_targets)
# print(acc_2)


# for i in range(test_targets.shape[0]):
#     if test_2_1[i] == test_targets[i]:
#         print("1",end="")
#     else:
#         print("0",end="")