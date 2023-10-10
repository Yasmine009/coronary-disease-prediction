# -*- coding: utf-8 -*-
"""
Stochastic Gradient Descent
"""
from helpers import batch_iter
from costs import compute_loss_mse


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y-(tx.dot(w))
    d_w0 = (-1)/np.shape(y)[0]*np.sum(e)
    d_w1 = (-1)/np.shape(y)[0]*e.dot(tx[:, 1])
    
    return np.array([d_wo, d_w1])


def mean_squared_error_sgd(y, tx, initial_w, batch_size=1, max_iters, gamma):
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
