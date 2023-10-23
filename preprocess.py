import numpy as np
import matplotlib.pyplot as plt 

"""
Repository for functions useful for preprocessing data
"""


def remove_feature(x, cols_excluded):
    """
    Remove features given our model.
    
    Args:
        x (np.array): data
        cols_excluded (np.array): indcides of the array of features to remove in x
    Returns:
        f_x: filtered matrix
    """

    f_x = np.delete(x, cols_excluded, axis=1)
    return f_x

def show_NaN_cols(x, x_label):
    """
    Plot the percentage of columns of x that contains a specific percentage of NaN values.

    Args:
        x (np.array): input data to analyze  
        x_label: string that describes the dataset x
    """
    
    percentiles = list(range(0, 101, 5))

    # Calculating the percentage of columns with a NaN percentage between x% and (x+5)%
    percentage_columns_with_percentile_nans = [
        (np.isnan(x).sum(axis=0)[
            (np.isnan(x).sum(axis=0) >= x.shape[0]*p/100) &
            (np.isnan(x).sum(axis=0) < x.shape[0]*(p+5)/100)
        ].shape[0] / x.shape[1]) * 100 for p in percentiles[:-1]  # Exclude the last percentile (100%)
    ]

    # Labels for the x-axis
    percentile_labels = [f'{p}-{p+5}%' for p in percentiles[:-1]]

    # Displaying the graph
    plt.figure(figsize=(15, 6))
    plt.bar(percentile_labels, percentage_columns_with_percentile_nans)

    plt.title('Percentage of columns with NaN values in intervals of 5% in the {}'.format(x_label))
    plt.xlabel('Percentage Range of NaN')
    plt.ylabel('Percentage of columns')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def filter_cols(x, threshold):
    """
    Add the mean of the columns with NaN and return indices of the columns to exclude in the regression including
    those where the standard deviation is equal to 0.
        
    Args:
        x (np.array): data
        threshold (int): % of NaN values that a column of x should contain 
        
    Returns:
        f_x: data with NaN values replaced by zeroes
        rem_cols (np.array): list of indices of columns that contains more than threshold% of NaN values and that
            doesn't contain any 0 standard deviation
    """
    
    f_x = x
    error_matrix = np.isnan(x)
    
    #Identify indices of the columns with more than threshold percentage of NaN values
    nan_ = (np.sum(error_matrix, axis=0) / x.shape[0]) * 100
    rem_cols_NaN = np.where(nan_ > threshold)[0]
    
    #Replace NaN values with the mean of it's columns
    column_means = np.nanmean(f_x, axis=0)
    f_x[error_matrix] = np.take(column_means, np.where(error_matrix)[1])
    
    #Identify columns with 0 standard deviation
    std_x = np.std(f_x, axis=0)
    rem_cols_std = np.where(std_x == 0)[0]
    
    #Final array with indices of the columns to exclude in the classification
    rem_cols = np.unique(np.concatenate((rem_cols_NaN, rem_cols_std)))
    
    return f_x, rem_cols

def standardize(x):
    """
    Standardize the data set x
    """

    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x