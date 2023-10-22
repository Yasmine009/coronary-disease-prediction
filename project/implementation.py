"""
Repository of all functions of the project 1
"""
import numpy as np
import csv


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    sig = 1/(1 + np.exp(-t))
    return sig
"""
Cost functions
"""
def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return (1/2*len(y))*np.sum((y-tx.dot(w))**2)


def compute_loss_mae(y, tx, w):
    """Calculate the loss using MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean(np.abs(y-tx.dot(w)))


def compute_loss_rmse(y, tx, w):
    """Calculate the loss using RMSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    
    return np.sqrt(2*compute_loss_mse(y, tx, w))


# changer dimensions de y et w
def calculate_loss_llh(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    n = len(y)
    s = sigmoid(tx.dot(w))
    loss = - (y.T.dot(np.log(s)) + (1-y).T.dot(np.log(1 - s)))
        
    return 1/n * np.squeeze(loss)


"""
Gradient Descent
"""
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - (tx.dot(w))
    N = y.shape[0]
    gradient = -(1/N) * tx.T.dot(e)
    return gradient

# changer dimensions
def calculate_gradient_llh(y, tx, w):
    """Compute the gradient of negative log likelihood loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    """
    N = y.shape[0]
    grad = (1/N) * tx.T.dot(sigmoid(tx@w)-y)
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # Compute loss using mse 
        loss = compute_loss_mse(y, tx, w)
        # Compute the gradient
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws, losses

"""
Stochastic Gradient Descent
"""
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - (tx.dot(w))
    N = y.shape[0]
    stoch_gradient = -(1/N) * tx.T.dot(e)
    return stoch_gradient


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: (default=1) a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size):
            loss = compute_loss_mse(minibatch_y, minibatch_tx, w)
            stoch_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w=w)
            w = w - gamma * stoch_gradient

            # store w and loss
            ws.append(w)
            losses.append(loss)

    return ws, losses

"""
Least Squares
"""
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    # Do not use use the invert of the (X.T).dot(X) matrix as it might not be invertible
    
    leftHand = (tx.T).dot(y)
    rightHand = (tx.T).dot(tx)
    w = np.linalg.solve(rightHand, leftHand)

    # Compute loss as in the course example
    N = len(y)
    e = y - np.dot(tx, w)
    mse = (1 / (2 * N)) * np.dot(e.T, e)
    
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    
    # We take the direct formula from the course
    # Note that the inverse is guaranteed to exist
    D = tx.shape[1]
    N = tx.shape[0]
    lambda_prime = lambda_ * 2 * N
    w = np.linalg.inv(tx.T @ tx + (lambda_prime*np.identity(D))).dot(tx.T).dot(y)

    return w


# changer dimensions
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


    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        
        loss = calculate_loss_llh(y, tx, w)
        gradient = calculate_gradient_llh(y, tx, w)
        w = w - gamma*gradient
        
        ws.append(w)
        losses.append(loss)
    
    return ws, losses


# changer dimensions
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

    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        
        loss = calculate_loss(y, tx, w) + lambda_*np.squeeze(w.T.dot(w))
        gradient = calculate_gradient(y, tx, w) + 2*lambda_*w
        
        w = w - gamma * gradient
        
        ws.append(w)
        losses.append(loss)
    
    return ws, losses



# remove helpers functions

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
