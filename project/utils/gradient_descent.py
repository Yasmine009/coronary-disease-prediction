# -*- coding: utf-8 -*-
"""
Gradient Descent
"""
from costs import compute_loss_mse

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y-(tx.dot(w))
    d_w0 = (-1)/np.shape(y)[0]*np.sum(e)
    d_w1 = (-1)/np.shape(y)[0]*e.dot(tx[:, 1])
    
    return np.array([d_wo, d_w1])


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
        
        # Do we return only the last W vector of the iteration ? 
    
    return ws, losses
