"""
Repository of all functions of the project 1
"""
import numpy as np

from utils import *
from helpers import *


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
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
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
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
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

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    The logistic gradient descent with penalty algorithm.

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