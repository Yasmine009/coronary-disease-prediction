import numpy as np

from implementations import *
from utils import *
from helpers import *
from preprocess import *

# Load data from the folder data/dataset (as extracted from the )
x_train, x_test, y_train, train_ids, test_ids = load_csv_data('data/dataset')

best_param_nan = 97

# Filter and process training data
f_x, cols_excluded = filter_cols(x_train, best_param_nan)
headers = load_csv_headers("data/dataset/x_train.csv")
features_removed = unused_features_indices(headers)
all_cols_excluded = np.unique(np.concatenate((np.array(features_removed), cols_excluded)))
T_x = np.delete(f_x, all_cols_excluded, axis=1)
T_x = build_log(T_x)
T_x = standardize(T_x)

# Filter and process testing data
f_x_test, junk = filter_cols(x_test, best_param_nan)  # junk because we do not need the cols_excluded, since we want to exclude the same columns as in the train set
T_x_test = np.delete(f_x_test, all_cols_excluded, axis=1)
T_x_test = build_log(T_x_test)
T_x_test = standardize(T_x_test)

# Split the data into train and test sets
x_training_full, x_validation_full, y_training_full, y_validation_full = split_data(T_x, y_train, 0.8)
x_training = x_training_full
x_validation = x_validation_full

# For y keep only labels column for training (drop indexes)
y_training = y_training_full[:, 1]
y_validation = y_validation_full[:, 1]

# Perform classification using logistic regression & classification
# Define the best parameters of the algorithm.
best_max_iters = 13
best_gamma = 0.48
best_prediction_threshold = 0.789

# Initialization and regression
w_initial = np.zeros(x_training.shape[1])
w, loss = logistic_regression(y_training, x_training, w_initial, best_max_iters, best_gamma)

# Submit to aicrowd, .csv file is at the root at the folder
aicrowd_submission(predict(w, T_x_test, best_prediction_threshold), "")

