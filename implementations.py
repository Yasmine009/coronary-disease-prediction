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

# reg_logistic_regression but we add a check to see whether the loss is decreasing
def reg_logistic_regression_v2(y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8):
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
    loss_prev = compute_loss_llh(y, tx, w)
    gradient = compute_gradient_llh(y, tx, w) + 2*lambda_*w
    for n_iter in range(max_iters):
        w = w - gamma * gradient
        gradient = compute_gradient_llh(y, tx, w) + 2*lambda_*w
        loss = compute_loss_llh(y, tx, w)
        if (n_iter > 0) & ((loss_prev - loss) < threshold):
            break
        loss_prev = loss
            
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
    

def grid_search(lambda_values, gamma_values, y_training, x_training, y_validation, x_validation, max_iters=1e5, threshold=1e-8, show_plot=True):
    """
    Perform a grid search to find the best hyperparameters for the logistic regression model with L2 regularization.
    The criteria is the F1 score.

    :param lambda_values: list of values for the regularization parameter
    :param gamma_values: list of values for the learning rate
    :param y_training: labels of the training set
    :param x_training: data of the training set
    :param y_validation: labels of the validation set
    :param x_validation: data of the validation set
    :param max_iters: maximum number of iterations for the gradient descent
    :param threshold: threshold for the stopping criterion
    :param show_plot: whether to show the plot of the results

    :return: best_lambda, best_gamma, best_f1score
    """

    # Initialize variables to store the best hyperparameters and corresponding results
    best_lambda = None
    best_gamma = None
    best_f1score = 0  # Initialize with a low value

    # Initialize a dictionary to store f1scores for different combinations
    f1scores = {}

    # Loop over all combinations of hyperparameters
    for lambda_ in lambda_values:
        for gamma in gamma_values:
            # Initialize weights
            w_initial = np.zeros(x_training.shape[1])
            
            # Train the model with current hyperparameters
            w, loss = reg_logistic_regression_v2(y_training, x_training, lambda_, w_initial, int(max_iters), gamma, threshold=threshold)
            
            # Calculate the accuracy for the training set and the validation set
            y_validation_pred = predict(w, x_validation)

            f1score, precision = score(y_validation_pred, y_validation)
            
            # Store the f1score for this combination
            f1scores[(lambda_, gamma)] = f1score

            # Check if the current model has a higher f1score than the best model so far
            if f1score > best_f1score:
                best_f1score = f1score
                best_lambda = lambda_
                best_gamma = gamma

    if show_plot:
        # Create a color gradient plot of f1scores
        f1score_matrix = np.zeros((len(lambda_values), len(gamma_values)))
        for i, lambda_ in enumerate(lambda_values):
            for j, gamma in enumerate(gamma_values):
                f1score_matrix[i, j] = f1scores[(lambda_, gamma)]

        plt.figure(figsize=(8, 6))
        c = plt.pcolormesh(gamma_values, lambda_values, f1score_matrix, cmap='viridis')
        plt.colorbar(c, label='F1 score')
        plt.xlabel('Gamma')
        plt.ylabel('Lambda')
        plt.title('Grid Search for Hyperparameters')
        plt.show()

    return best_lambda, best_gamma, best_f1score

def cross_validation(x, y, k_fold,  lambda_, initial_w = 0, max_iters = 100000, gamma = 0, threshold=1e-8):
    """
    Cross validation for the logistic regression model with L2 regularization.
    The criteria is the F1 score.

    :param x: data
    :param y: labels
    :param k_fold: number of folds
    :param lambda_: regularization parameter
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations for the gradient descent
    :param gamma: learning rate
    :param threshold: threshold for the stopping criterion

    :return: w, loss
    """
    # Split the data into k_fold parts
    x_split = np.array_split(x, k_fold)
    y_split = np.array_split(y, k_fold)

    # Initialize variables to store the best hyperparameters and corresponding results
    best_f1score = 0  # Initialize with a low value

    # Loop over all combinations of hyperparameters
    for i in range(k_fold):
        # Initialize weights
        if initial_w == 0:
            w_0 = np.zeros(x.shape[1])
        else:
            w_0 = initial_w

        # Get the training and validation sets
        x_training = np.concatenate([x_split[j] for j in range(k_fold) if j != i])
        y_training = np.concatenate([y_split[j] for j in range(k_fold) if j != i])
        
        x_validation = x_split[i]
        y_validation = y_split[i]

        # Train the model with current hyperparameters
        w, loss = reg_logistic_regression_v2(y_training, x_training, lambda_, w_0, int(max_iters), gamma, threshold)

        # Calculate the accuracy for the training set and the validation set
        y_validation_pred = predict(w, x_validation)

        f1score, precision = score(y_validation_pred, y_validation)

        # Check if the current model has a higher f1score than the best model so far
        if f1score > best_f1score:
            best_f1score = f1score
            best_w = w
            best_loss = loss

    return best_w, best_loss