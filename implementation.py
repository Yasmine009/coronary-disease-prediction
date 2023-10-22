"""
Repository of all functions of the project 1
"""
import numpy as np

from utils import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        # Computer loss using mse and the gradient
        loss = compute_loss_mse(y, tx, w)
        gradient = np.asarray(compute_gradient(y, tx, w))
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws, losses


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: (default=1) a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        loss = compute_loss_mse(y, tx, w)
        
        # Define initial gradient and iterate over the the batch
        gradient = np.zeros(2)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient+=compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*(1/abs(batch_size))*gradient
            
            # store w and loss
            ws.append(w)
            losses.append(loss)
        
        # Do we return only the last W vector of the iteration (same thing for the ? 
        
    return ws, losses

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    
    # We take the direct formula from the course
    # Note that the inverse is guaranteed to exist
    lambda_1 = lambda_*2*tx.shape[0]
    w = np.linalg.inv(np.add((tx.T).dot(tx), lambda_1*np.identity(tx.shape[1]))).dot(tx.T).dot(y)
    return w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    The logistic gradient descent algorithm.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        losses: np.array
        w: np.array

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        
        loss = compute_loss_llh(y, tx, w)
        gradient = compute_gradient_llh(y, tx, w)
        w = w - gamma*gradient
        
        ws.append(w)
        losses.append(loss)
    
    return ws, losses

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    The logistic gradient descent with penalty algorithm.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        losses: np.array
        w: np.array

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        
        loss = compute_loss_llh(y, tx, w) + lambda_*np.squeeze(w.T.dot(w))
        gradient = compute_gradient_llh(y, tx, w) + 2*lambda_*w
        
        w = w - gamma * gradient
        
        ws.append(w)
        losses.append(loss)
    
    return ws, losses

