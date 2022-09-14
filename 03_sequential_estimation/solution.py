# Author: Davíð Sæmundsson
# Date: 08.09.2022
# Project: Assignment 3, Sequential Estimation 
# Acknowledgements: Worked on together with Björgvin Freyr
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov = np.eye(k,k)
    cov = np.square(var)*cov
    return np.random.multivariate_normal(mean,cov,n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return (mu + ((x-mu)/n)) 


def _plot_sequence_estimate():
    
    data_x = gen_data(100,3,np.array([0,1,-1]),np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range((data_x.shape[0])):
        #print(estimates[i],data_x[i],data_x.shape[0])
        estimates.append(update_sequence_mean(estimates[i], data_x[i], len(estimates)))
    #print(estimates)
    plt.plot([e[0] for e in estimates[1:]], label='First dimension')
    plt.plot([e[1] for e in estimates[1:]], label='Second dimension')
    plt.plot([e[2] for e in estimates[1:]], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    ...


def _plot_mean_square_error():
    ...


# Naive solution to the independent question.

# def gen_changing_data(
#     n: int,
#     k: int,
#     start_mean: np.ndarray,
#     end_mean: np.ndarray,
#     var: float
# ) -> np.ndarray:
#     # remove this if you don't go for the independent section
#     ...


# def _plot_changing_sequence_estimate():
#     # remove this if you don't go for the independent section
#     ...


# Test cases
# np.random.seed(1234)
# test_1_1 = gen_data(2, 3, np.array([0, 1, -1]), 1.3)
# print(test_1_1)
# np.random.seed(1234)
# test_1_1_2 = gen_data(5, 1, np.array([0.5]), 0.5)
# print(test_1_1_2)

#Test cases 1.2

# var = np.sqrt(1)
# data_x = gen_data(300,3,np.array([0,1,-1]),var)
# scatter_3d_data(data_x)
# bar_per_axis(data_x)
# print(data_x[0,:])

#Test case 1.4
# var = np.sqrt(3)
# data_x = gen_data(300,3,np.array([0,1,-1]),var)
# mean = np.mean(data_x[0,:],0)
# new_x = gen_data(1,3, np.array([0,0,0]),1)
# test_1_4 = update_sequence_mean(mean, new_x, data_x.shape[0])
# #print(mean)
# #print(new_x)
# print(test_1_4)

# _plot_sequence_estimate()

