"""
Utils function for computing regressions and other calculations
"""
import numpy as np
import matplotlib.pyplot as plt

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
    return 1/(1+np.exp(-t))

"""
Cost functions
"""
def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y-(tx.dot(w))
    return (1 / 2 * np.mean(e**2))


def compute_loss_mae(y, tx, w):
    """Calculate the loss using MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y-(tx.dot(w))
    return np.mean(np.abs(e))


def compute_loss_rmse(y, tx, w):
    """Calculate the loss using RMSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    
    return np.sqrt(2*compute_loss_mse(y, tx, w))

def compute_loss_llh(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss_llh(y, tx, w), 8)
    1.52429481
    """

    n = len(y)
    s = sigmoid(tx.dot(w))
    loss = - (y.T.dot(np.log(s)) + (1-y).T.dot(np.log(1 - s)))
        
    return 1/n * np.squeeze(loss)


"""
Gradient computation functions
"""
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y- tx.dot(w)
    gradient = -tx.T.dot(e) / len(e)
    
    return gradient

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e) / len(e)
    
    return gradient


def compute_gradient_llh(y, tx, w):
    """Compute the gradient of negative log likelihood loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    n = len(y)
    s = sigmoid(tx.dot(w))
    return tx.T.dot(s-y)*(1/n)

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

def predict_labels_logistic_regression(initial_weights, weights, data):
    """
    Generate class predictions given weights, and a test data matrix
    
    Args:
        weights (np.array): weights of the model
        data (np.array): data to predict
        
    Returns:
        y_pred (np.array): predicted labels
    """
    
    # Predict using sigmoid function
    y_pred = sigmoid(np.dot(data, weights)+initial_weights)

    # Replace 0 by -1
    y_pred[np.where(y_pred == 0)] = -1

    return y_pred

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

                     
            
def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    
    # Create a permutation of indices
    permuted_indices = np.random.permutation(len(x))
    
    # Split the indices based on the given ratio
    split_index = int(np.floor(len(x) * ratio))
    
    train_indices = permuted_indices[:split_index]
    test_indices = permuted_indices[split_index:]
    
    # Extract data based on the split indices
    x_tr = x[train_indices]
    x_te = x[test_indices]
    
    y_tr = y[train_indices]
    y_te = y[test_indices]
    
    return x_tr, x_te, y_tr, y_te


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
    plt.xticks([-1, 1])
    plt.yticks([-1, 1])
    plt.show()

def predict(w,X, r=0.789):
    """
    Predict the labels of the data X using the weights of the model
    :param w: weights of the model
    :param X: data
    :param r: prediction threshold
    :return: predicted labels
    """
    z = np.dot(X,w)
    y = sigmoid(z)
    y_pred = np.where(y>=r,1,-1)
    return y_pred

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
    best_w = np.zeros(x.shape[1])
    best_loss = np.max

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

# KNN class
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
    