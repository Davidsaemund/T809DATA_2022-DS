# Author: Davíð Sæmundsson
# Date: 20.09.2022
# Project: Assignment 5, Backdrop
# Acknowledgements: Worked on together with Björgvin Freyr
#

from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x <= -100:
        return 0.0
    return (1/(1+np.exp(-x)))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return (sigmoid(x)*(1-sigmoid(x)))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weigthed_sum = 0
    for i in range(x.shape[0]):
        weigthed_sum += x[i]*w[i]
    return weigthed_sum, sigmoid(weigthed_sum)


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    a1 = []
    a2 = []
    z1_list = []
    y = []
    bias = np.array([1])
    z0 = np.concatenate((bias,x))
    #print(W1)
    for i in range(M):
        #print(W1[i,:])
        temp_a1, temp_z1 = perceptron(z0,W1[:,i])
        a1.append(temp_a1)
        z1_list.append(temp_z1)
    z1 = np.concatenate((bias,z1_list))
    for j in range(K):
        # print(W2[:,j])
        # print(z1)
        temp_a2, temp_y = perceptron(z1,W2[:,j])
        a2.append(temp_a2)
        y.append(temp_y)
    
    
    
    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    # 1.
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    # 2.
    delta_k = []
    for k in range(K):
        delta_k.append(y[k]-target_y[k])
    #print(a1,a2)
    #3.
    delta_j = []
    for j in range(M+1):
        sum = 0
        for k in range(K):
            #print(j,k)
            sum += W2[j,k]*delta_k[k]
        if j != 0:
            delta_j.append(d_sigmoid(a1[j-1])*sum)
    #print(delta_j)
    # 4.
    dE1 = np.zeros((W1.shape[0], W1.shape[1])) 
    dE2 = np.zeros((W2.shape[0], W2.shape[1])) 
    # 5.
    
    for i in range(dE1.shape[0]):
        for j in range(dE1.shape[1]):
            dE1[i,j] = delta_j[j]*z0[i]

    for j in range(dE2.shape[0]):
        for k in range(dE2.shape[1]):
            dE2[j,k] = delta_k[k]*z1[j]

    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    misclassification_rate = []
    guesses = []
    E_total = []
    
    #print(X_train,t_train)
    for i in range(iterations):
        dE1_total = np.zeros((W1.shape[0], W1.shape[1])) 
        dE2_total = np.zeros((W2.shape[0], W2.shape[1]))
        last_guess = []
        mis_rate = 0
        E_each = 0
        for j in range(X_train.shape[0]):
            x = X_train[j,:]
            t_test = np.zeros(K)
            t_test[t_train[j]] = 1
            y, dE1, dE2 = backprop(x, t_test, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2
            for k in range(K):
                E_each += t_test[k]*np.log(y[k]) + (1-t_test[k])*np.log(1-y[k])
            
            last_guess.append(np.argmax(y))
            if(np.argmax(y) != t_train[j]):
                mis_rate += 1
        W1 = W1- (eta * (dE1_total / X_train.shape[0]))
        W2 = W2- (eta * (dE2_total / X_train.shape[0]))
        E_total.append(-E_each/X_train.shape[0])
        misclassification_rate.append(mis_rate/X_train.shape[0])
        guesses.append(last_guess)
    return W1, W2, E_total, misclassification_rate, guesses[-1] 


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    guess = []
    for j in range(X.shape[0]):
        x = X[j,:]
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        guess.append(np.argmax(y))
    return np.array(guess)

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
#Test for 1.1 
# sigma1 = sigmoid(-101)
# dsigma1 = d_sigmoid(0.2)
# print(sigma1)
# print(dsigma1)

# #Test for 1.2 
# perp_1 = perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))
# perp_2 = perceptron(np.array([0.2,0.4]),np.array([0.1,0.4]))

# print(perp_1,"\n",perp_2)

#Test for 1.3 FFNN
# np.random.seed(23)
# features, targets, classes = load_iris()
# (train_features, train_targets), (test_features, test_targets) = \
#     split_train_test(features, targets)

# # initialize the random generator to get repeatable results
# #np.random.seed(1234)

# # Take one point:
# x = train_features[0, :]
# K = 3 # number of classes
# M = 6
# D = 4
# # Initialize two random weight matrices
# W1 = 2 * np.random.rand(D + 1, M) - 1
# W2 = 2 * np.random.rand(M + 1, K) - 1
# y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

# print("y",y,"\nz0",z0,"\nz1",z1,"\na1",a1,"\na2",a2)
# z1 = []
# a1 = [-0.78715749, -0.73890218, 1.6671074 , -0.37155955, 3.37644349,3.17797518, -5.04807589, 0.26887954, 2.36005949, 9.93383443]
# for i in range(len(a1)): 
#     z1.append(sigmoid(a1[i]))
# print(z1)

# TEST FOR 1.4
# initialize random generator to get predictable results
# np.random.seed(23)
# features, targets, classes = load_iris()
# (train_features, train_targets), (test_features, test_targets) = \
#     split_train_test(features, targets)
# acc_list = []
# # initialize the random generator to get repeatable results
# #np.random.seed(1234)
# for m in range(1,25):
#     # Take one point:
#     x = train_features[0, :]
#     K = 3 # number of classes
#     M = m
#     D = train_features.shape[1]
#     # Initialize two random weight matrices
#     W1 = 2 * np.random.rand(D + 1, M) - 1
#     W2 = 2 * np.random.rand(M + 1, K) - 1
#     #y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
#     #target_y = np.zeros(K)
#     #target_y[targets[0]] = 1.0
#     #y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
#     #print("y",y,"\ndE1",dE1,"\ndE2",dE2)

#     W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
#     #print("W1tr",W1tr,"\nW2tr",W2tr,"\nEtotal",Etotal,"\nMis_rate",misclassification_rate,"\nlast_gu",last_guesses)

#     guess = test_nn(test_features,M,K,W1tr,W2tr)
#     #print(guess)
#     #print(test_targets)

#     acc = accuracy(guess,test_targets)
#     print(acc)

#     confusion_matrix = confusion_mat(guess,classes,test_targets)
#     print(confusion_matrix)
#     acc_list.append(acc*100.0)
# plt.plot(np.array(acc_list))
# plt.ylabel("Accuracy")
# plt.xlabel("M")
# plt.show()

# # plt.plot(Etotal)
# # plt.ylabel("E total")
# # plt.xlabel("Iteration")
# # plt.show()
# # plt.plot(misclassification_rate)
# # plt.ylabel("Misclassification_rate")
# # plt.xlabel("Iteration")
# # plt.show()
