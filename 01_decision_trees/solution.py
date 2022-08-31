# Author: Davíð Sæmundsson
# Date: 20.08.2022  
# Project: Assigment 1, Sprint 1
# Acknowledgements: 
# prior function taken from lecture, Eyjólfur Ingi and start of split data and the class
# Björgvin Freyr Jónsson helped me with figuring out some coding problems


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


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


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:,split_feature_index] < theta,:]
    targets_1 = targets[features[:,split_feature_index] < theta]

    features_2 = features[features[:,split_feature_index] >= theta,:]
    targets_2 = targets[features[:,split_feature_index] >= theta]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return 0.5 * (1-sum(np.square(prior(targets,classes))))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return ((len(t1)*g1)/n) + ((len(t2)*g2)/n) 


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1, t_2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        values = np.array(features[:,i])
        thetas = np.linspace(values.min(), values.max(), num_tries+2)[1:-1]
        
        # iterate thresholds
        for theta in thetas:
            gini_impu = total_gini_impurity(features,targets,classes,i,theta)
            if gini_impu < best_gini:
                best_gini = gini_impu
                best_dim = i
                best_theta = theta
        # print(min(gini_impu_list))
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit( self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score( self.test_features, self.test_targets)
        

    def plot(self):
        plot_tree(self.tree)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        accuracy_1 = []
        x_axis = []
        for i in range(len(self.train_targets)):
            new_features = self.train_features[:i+1]
            new_targets = self.train_targets[:i+1]
            self.tree.fit(new_features, new_targets)
            accuracy_1.append(self.accuracy())
            x_axis.append(i)
        plt.plot(x_axis,accuracy_1)
        plt.show()
            

    def guess(self):
        return self.tree.predict( self.test_features)
        
    def confusion_matrix(self):
        predict_array = self.guess()
        confusion_matrix = []
        for c in self.classes:
            confusion_matrix.append(len(self.classes)*[0])
        
        for i in range(len(self.test_targets)):
            confusion_matrix[predict_array[i]][self.test_targets[i]] += 1
        return confusion_matrix    

        


#Main program

#Test case for prior function
# x = prior([0, 2, 3, 3], [0, 1, 2, 3])
# print(x)


# features, targets, classes = load_iris()
# test = total_gini_impurity(features, targets, classes, 2, 1.65)
# test = total_gini_impurity(features, targets, classes, 2, 1.9516129032258065)
# print(test)

# test1 = brute_best_split(features, targets, classes, 30)
# print(test1)

# features, targets, classes = load_iris()
# dt = IrisTreeTrainer(features, targets, classes=classes)
# dt.train()
# print(f'The accuracy is: {dt.accuracy()}')
# dt.plot()
# print(f'I guessed: {dt.guess()}')
# print(f'The true targets are: {dt.test_targets}')
# print(dt.confusion_matrix())

# features, targets, classes = load_iris()
# dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
# dt.plot_progress()
