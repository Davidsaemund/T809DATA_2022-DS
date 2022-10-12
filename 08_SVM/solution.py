# Author: Davíð Sæmundson
# Date: 05.09.2022
# Project: Support Vector Machines
# Acknowledgements: Results compared with Björgvin Freyr
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel():
    X, t = make_blobs(n_samples=40, centers=2)
    clf = svm.SVC(C=1000,kernel='linear')
    clf.fit(X,t)
    plot_svm_margin(clf,X,t)
    plt.show()



def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    
    plt.subplot(1,num_plots,index)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')
    

def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000,kernel='rbf',gamma='scale')
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 1)

    clf = svm.SVC(C=1000,kernel='rbf',gamma=0.2)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 2)

    clf = svm.SVC(C=1000,kernel='rbf',gamma=2)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 3)

    plt.show()
    

def _compare_C():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    C_in = [1000,0.5,0.3,0.05,0.0001]

    for i in range(len(C_in)):
        clf = svm.SVC(C=C_in[i],kernel='linear')
        clf.fit(X,t)
        plt.subplot(1,len(C_in),i+1)
        plot_svm_margin(clf,X,t)
    plt.show()

def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_train,t_train)
    y = svc.predict(X_test)
    print(y)
    acc_score = accuracy_score(t_test,y)
    print(acc_score)
    prec_score = precision_score(t_test,y)
    print(prec_score)
    reca_score = recall_score(t_test,y)
    print(reca_score)

    return [acc_score,prec_score,reca_score]

#_plot_linear_kernel()
#_compare_C()
(X_train, t_train), (X_test, t_test) = load_cancer()
svc = svm.SVC(C=1000, kernel='poly')
test_1 = train_test_SVM(svc, X_train, t_train, X_test, t_test)
print(test_1)
