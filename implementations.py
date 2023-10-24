"""
Repository of all functions of the project 1
"""
import numpy as np

from utils import *
from helpers import *

import matplotlib.pyplot as plt

def least_squares(y, tx):
    """
    Calculate the least squares solution and the corresponding loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights.
        loss: scalar.

    """
    
    leftHand = (tx.T).dot(y)
    rightHand = (tx.T).dot(tx)
    
    w = np.linalg.solve(rightHand, leftHand)

    #Compute loss as in the course example
    e = y-tx.dot(w)
    loss = 1/2* np.mean(e**2)
    
    return w, loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights.
        loss: scalar.
    """

    # Define parameters to store w and loss
    w = initial_w
    gradient = np.asarray(compute_gradient(y, tx, w))
    for n_iter in range(max_iters):
        # Computer loss using mse and the gradient
        w = w - gamma*gradient
        gradient = np.asarray(compute_gradient(y, tx, w))

    loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """
    The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: (default=1) a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights.
        loss: scalar.
    """

    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):

        # Define initial gradient and iterate over the the batch
        gradient = np.zeros(2)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient+=compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*(1/abs(batch_size))*gradient
            
    loss = compute_loss_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression algorithm.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights.
        loss: scalar.

    """
    
    # We take the direct formula from the course
    # Note that the inverse is guaranteed to exist
    lambda_1 = lambda_*2*tx.shape[0]
    w = np.linalg.inv(np.add((tx.T).dot(tx), lambda_1*np.identity(tx.shape[1]))).dot(tx.T).dot(y)

    error = y - tx.dot(w)
    loss = 0.5 * np.mean(error**2)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    The logistic gradient descent algorithm.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        max_iters: int
        gamma: float

    Returns:
        w: optimal weights.
        loss: scalar.

    """
    w = initial_w
    gradient = compute_gradient_llh(y, tx, w)
    for n_iter in range(max_iters):
        w = w - gamma*gradient
        gradient = compute_gradient_llh(y, tx, w)
    loss = compute_loss_llh(y, tx, w)
    return w, loss

def l1_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Logistic regression with L1 regularization.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: float, regularization parameter
        initial_w: shape=(D, 1)
        max_iters: int
        gamma: float

    Returns:
        w: optimal weights.
        loss: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_llh(y, tx, w)
        w = w - gamma * gradient
        w = np.sign(w) * np.maximum(0, np.abs(w) - gamma * lambda_)

    loss = compute_loss_llh(y, tx, w) + lambda_ * np.sum(np.abs(w))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    The logistic gradient descent with penalty algorithm (L2 regularization).

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: int
        w:  shape=(D, 1)
        max_iters: int
        gamma: float

    Returns:
        w: optimal weights.
        loss: scalar.
    """
    
    w = initial_w
    gradient = compute_gradient_llh(y, tx, w) + 2*lambda_*w
    for n_iter in range(max_iters):
        w = w - gamma * gradient
        gradient = compute_gradient_llh(y, tx, w) + 2*lambda_*w

    loss = compute_loss_llh(y, tx, w)
    return w, loss

def aicrowd_submission(y_pred, path):
    """
    Create a csv file to submit to aicrowd
    :param y_pred: predictions to submit. Must be of shape (109379, )
    The ids of the predictions must go from 328135 to 437513. they are generated in this function
    """
    indices = np.arange(328135, 437514)
    if y_pred.shape[0] != 109379:
        raise ValueError("Error: y_pred must be of shape (109379, )")
    create_csv_submission(indices, y_pred, path + "submission.csv")

def confusion_matrix(y_pred, y_true):
    """
    Calculate the confusion matrix
    
    Args:
        y_pred (np.array): predicted labels
        y_true (np.array): true labels
        
    Returns:
        confusion_matrix (np.array): confusion matrix
    """
    
    # True positives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    # False positives
    fp = np.sum((y_pred == 1) & (y_true == -1))
    
    # False negatives
    fn = np.sum((y_pred == -1) & (y_true == 1))
    
    # True negatives
    tn = np.sum((y_pred == -1) & (y_true == -1))
    
    return np.array([[tn, fp], [fn, tp]])

def score(y_pred, y_true):
    """
        Calculate the F1 score and the precision
    
    Args:
        y_pred (np.array): predicted labels
        y_true (np.array): true labels
        
    Returns:
        f1_score (float): F1 score
        precision (float): precision
    """
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_pred, y_true)

    # Calculate the F1 score
    f1_score = 2*cm[1,1]/(2*cm[1,1] + cm[0,1] + cm[1,0])

    # Calculate the precision
    precision = cm[1,1]/(cm[1,1] + cm[0,1])

    return f1_score, precision

def plot_results(y_pred, y_true):
    """
    Plot the results of the predictions
    :param y_pred: predicted labels
    :param y_true: true labels
    """
    f1_score, precision = score(y_pred, y_true)
    print("F1 score: ", f1_score)
    print("Precision: ", precision)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.title("Confusion matrix")
    plt.imshow(confusion_matrix(y_true, y_pred), cmap=plt.cm.plasma)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([-1, 1], ['No Stroke', 'Stroke'])
    plt.yticks([-1, 1], ['No Stroke', 'Stroke'])
    plt.show()

def predict(w,X):
    """
    Predict the labels of the data X using the weights of the model
    :param w: weights of the model
    :param X: data
    :return: predicted labels
    """
    z = np.dot(X,w)
    y = sigmoid(z)
    y_pred = np.where(y>=0.5,1,-1)
    return y_pred


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common