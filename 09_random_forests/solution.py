# Author: Davíð Sæmundsson
# Date: 16.10.2022
# Project: 09 Random Forests
# Acknowledgements: Project discussed with Björgvin Freyr
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from collections import OrderedDict


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train,self.t_train)
        self.predict = self.classifier.predict(self.X_test)
        self.feature_import = []
        self.feature_idx = []
    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        return confusion_matrix(self.t_test,self.predict)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test,self.predict)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        return precision_score(self.t_test,self.predict)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test,self.predict)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return np.average(cross_val_score(self.classifier,self.X_test,self.predict,cv=10))

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        
        self.feature_import = self.classifier.feature_importances_
        self.feature_import = np.array(self.feature_import)
        #print(self.feature_import)
        feature_names = [f"{i}" for i in range(self.feature_import.shape[0])]
        plt.bar(feature_names,self.feature_import)
        plt.show()
        self.feature_idx = np.flip(np.argsort(self.feature_import))
        return self.feature_idx


def _plot_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175
    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def _plot_extreme_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                bootstrap=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                bootstrap=True,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175
    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels
    
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


# classifier_type = DecisionTreeClassifier()
# cc = CancerClassifier(classifier_type)
# print("CM: ",cc.confusion_matrix())
# print("Acc: ",cc.accuracy())
# print("Prec: ",cc.precision())
# print("ReCall: ", cc.recall())
# print("CVA: ", cc.cross_validation_accuracy())
# x_bar = []
# acc_feature_sqrt = []
# for i in range(10,160+1):
#     x_bar.append(i)
#     classifier_type = RandomForestClassifier(n_estimators=i,max_features="sqrt",random_state=109)
#     cc = CancerClassifier(classifier_type)
#     acc_feature_sqrt.append(cc.accuracy())

# acc_feature_log2 = []
# for i in range(10,160+1):
#     classifier_type = RandomForestClassifier(n_estimators=i,max_features="log2",random_state=109)
#     cc = CancerClassifier(classifier_type)
#     acc_feature_log2.append(cc.accuracy())

# acc_feature_none = []
# for i in range(10,160+1):
#     classifier_type = RandomForestClassifier(n_estimators=i,max_features=None,random_state=109)
#     cc = CancerClassifier(classifier_type)
#     acc_feature_none.append(cc.accuracy())

# plt.plot(x_bar,acc_feature_none, label="Max_features = None ")
# plt.plot(x_bar,acc_feature_log2, label="Max_features = log2 ")
# plt.plot(x_bar,acc_feature_sqrt, label="Max_features = sqrt ")
# plt.legend(loc="lower left")
# plt.xlabel('n_estimators')
# plt.ylabel('Accuracy')
# plt.show()

# classifier_type = RandomForestClassifier(n_estimators=140,max_features='sqrt',random_state=109)
# cc = CancerClassifier(classifier_type)
# print("CM: ",cc.confusion_matrix())
# print("Acc: ",cc.accuracy())
# print("Prec: ",cc.precision())
# print("ReCall: ", cc.recall())
# print("CVA: ", cc.cross_validation_accuracy())
# print("Feat_idx", cc.feature_importance())

#_plot_oob_error()

# classifier_type = ExtraTreesClassifier(n_estimators=100,max_features='sqrt',random_state=109)
# cc = CancerClassifier(classifier_type)
# print("CM: ",cc.confusion_matrix())
# print("Acc: ",cc.accuracy())
# print("Prec: ",cc.precision())
# print("ReCall: ", cc.recall())
# print("CVA: ", cc.cross_validation_accuracy())
# print("Feat_idx", cc.feature_importance())

#_plot_extreme_oob_error()