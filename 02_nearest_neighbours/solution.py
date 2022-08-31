# Author: Davíð Sæmundsson
# Date: 25.08.2022
# Project: 02_Nearest neighbours knn
# Acknowledgements: Björgvin Freyr Jónsson helped me with figuring out some coding problems
# #

from cProfile import label
from dis import dis
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

def remove_one(points: np.ndarray, i: int):
    '''
    Removes the i-th from points and returns
    the new array
    '''
    return np.concatenate((points[0:i], points[i+1:]))

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum = 0
    for j in range(len(x)):
        sum += np.square(x[j]-y[j])
        
    return np.sqrt(sum)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x,points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    
    distances = euclidian_distances(x,points)
    distances = np.argsort(distances)
    k_nearest = []
    
    for i in range(k):
        k_nearest.append(distances[i])
    return k_nearest

def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    new_list = []
    for c in classes:
        new_list.append(np.count_nonzero(targets == c))
    new_list_1 = np.array(new_list)
    return np.argmax(new_list_1)

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    targets = []
    k_nearest_list = k_nearest(x,points,k)
    
    for k in k_nearest_list:
        targets.append(point_targets[k])
   
    targets_np = np.array(targets)
    guess_target = vote(targets_np,classes)
   
    return guess_target


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    prediction_list = []
    for i in range(len(points)):
        prediction_list.append(knn((points[i]),remove_one(points,i),remove_one(point_targets,i),classes,k))
    return prediction_list

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions_list = knn_predict(points,point_targets,classes,k)
    sum_corrects = 0
    for i in range(len(point_targets)):
        if predictions_list[i] == point_targets[i]:
            sum_corrects += 1
    return sum_corrects/len(point_targets)




def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions_list = knn_predict(points,point_targets,classes,k)
    confusion_matrix = []
    for c in classes:
        confusion_matrix.append(len(classes)*[0])
    
    for i in range(len(point_targets)):
        confusion_matrix[predictions_list[i]][point_targets[i]] += 1
    return confusion_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    accuracy = []
    for k in range(1, len(point_targets)):
        accuracy.append(knn_accuracy(points,point_targets,classes,k))
    best_acc = 0
    for i in range(len(accuracy)):
        if best_acc < accuracy[i]:
            best_acc = accuracy[i]
            best_k = i + 1 
    return best_k 


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    predictions = knn_predict(points,point_targets,classes,k)
    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        if predictions[i] == point_targets[i]: 
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='green',
            linewidths=2)
        else: 
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red',
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # new_list = []
    # for c in classes:
    #     new_list.append(np.count_nonzero(targets == c))
    # new_list_1 = np.array(new_list)
    # return np.argmax(new_list_1)
    new_list = []
    for c in classes:
        sum = 0
        for j in range(targets.shape[0]):
            if targets[j] == c:
                sum += 1/distances[j]
            
        new_list.append(sum)

    new_list_1 = np.array(new_list)
    
    return np.argmax(new_list_1)



def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    targets = []
    distances = []
    k_nearest_list = k_nearest(x,points,k)
    
    for k in k_nearest_list:
        targets.append(point_targets[k])
        distances.append(points[k])

    
    targets_np = np.array(targets)
    distances_np =np.array(distances)
    
    distances = euclidian_distances(x,distances_np)
    
    guess_target = weighted_vote(targets_np,distances,classes)
   
    return guess_target


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    prediction_list = []
    for i in range(len(points)):
        prediction_list.append(wknn((points[i]),remove_one(points,i),remove_one(point_targets,i),classes,k))
    return prediction_list

def wknn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions_list = wknn_predict(points,point_targets,classes,k)
    sum_corrects = 0
    for i in range(len(point_targets)):
        if predictions_list[i] == point_targets[i]:
            sum_corrects += 1
    return sum_corrects/len(point_targets)

def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    wknn_acc = []
    knn_acc = []
    k_list = []
    for k in range(points.shape[0]-1):
        wknn_acc.append(wknn_accuracy(points,targets,classes,k+1))
        knn_acc.append(knn_accuracy(points,targets,classes,k+1))
        k_list.append(k+1)
    print(k_list)
    print(wknn_acc)
    print(knn_acc)
    plt.plot(k_list,knn_acc,label="knn accuracy")
    plt.plot(k_list,wknn_acc,label="wknn accuracy")
    plt.legend()
    plt.show()
    
    #***B.Theoretical****
    # The difference in accuracy when k increases is understandable since the more k points you
    # use the more likely it is that they have different target but when you add weight to 
    # the distance then it's more likely that the points that you are adding are further away
    # and therefor not as likely to be in the same class as the points that are closer to the 
    # point we are looking from 


# d, t, classes = load_iris()
# x, points = d[0,:], d[1:, :]
# x_target, point_targets = t[0], t[1:]

# test = euclidian_distance(x, points[0])
# print(test)
# test_1 = euclidian_distance(x, points[50])
# print(test_1)

# test_2 = k_nearest(x, points, 1)
# print(test_2)
# test_3 = k_nearest(x, points, 10)
# print(test_3)

# test_4 = vote(np.array([1,1,1,1,0,0,0,0,0,0,0,0]), np.array([0,1]))
# print(test_4)

# test_5 = knn(x, points, point_targets, classes, 1)
# print(test_5)

# test_6 = knn(x, points, point_targets, classes, 5)
# print(test_6)
# #
# test_7 = knn(x, points, point_targets, classes, 149)
# print(test_7)


# d, t, classes = load_iris()
# (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
# predictions = knn_predict(d_test, t_test, classes, 5)
# print(predictions)

# test_1 = knn_accuracy(d_test, t_test, classes, 10)
# test_2 = knn_accuracy(d_test, t_test, classes, 5)
# print(test_1)
# print(test_2)
# test_10 = knn_confusion_matrix(d_test, t_test, classes, 10)
# test_11 = knn_confusion_matrix(d_test, t_test, classes, 20)
# print(test_10)
# print(test_11)
# test_12 = best_k(d_train, t_train, classes)
# print(test_12)

# test_13 = wknn_predict(d_test, t_test, classes, 5)
# print(test_13)
# compare_knns(d_test,t_test,classes)

