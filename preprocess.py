import numpy as np
import matplotlib.pyplot as plt 

"""
Repository for functions useful for preprocessing data
"""

calculated_variables = ['ACTIN11_',
'ACTIN21_',
'DROCDY3_',
'FC60_',
'FRUTDA1_',
'FTJUDA1_',
'GRENDAY_',
'MAXVO2_',
'METVL11_',
'METVL21_',
'ORNGDAY_',
'PA1MIN_',
'PA1VIGM_',
'PADUR1_',
'PADUR2_',
'PAFREQ1_',
'PAFREQ2_',
'PAINACT2',
'PAMIN11_',
'PAMIN21_',
'PAMISS1_',
'PAVIG11_',
'PAVIG21_',
'VEGEDA1_',
'_AGE80',
'_AGE65YR',
'_AGEG5YR',
'_AGE_G',
'_AIDTST3',
'_ASTHMS1',
'_BMI5',
'_BMI5CAT',
'_CASTHM1',
'_CHLDCNT',
'_CHOLCHK',
'_DRDXAR1',
'_DRNKWEK',
'_DUALCOR',
'_DUALUSE',
'_EDUCAG',
'_FLSHOT6',
'_FRT16',
'_FRTLT1',
'_FRTRESP',
'_FRUITEX',
'_FRUTSUM',
'_HCVU651',
'_HISPANC',
'_INCOMG',
'_LLCPWT',
'_LMTACT1',
'_LMTSCL1',
'_LMTWRK1',
'_LTASTH1',
'_MICHD',
'_MINAC11',
'_MINAC21',
'_MISFRTN',
'_MISVEGN',
'_MRACE1',
'_PA30021',
'_PA150R2',
'_PA300R2',
'_PACAT1',
'_PAINDX1',
'_PAREC1',
'_PASTAE1',
'_PASTRNG',
'_PNEUMO2',
'_PRACE1',
'_RACE',
'_RACEG21',
'_RACEGR3',
'_RACE_G1',
'_RFBING5',
'_RFBMI5',
'_RFCHOL',
'_RFDRHV5',
'_RFHLTH',
'_RFHYPE5',
'_RFSEAT2',
'_RFSEAT3',
'_RFSMOK3',
'_SMOKER3',
'_TOTINDA',
'_VEG23',
'_VEGESUM',
'_VEGETEX',
'_VEGLT1',
'_VEGRESP']

# Based on https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf these describe PID's, c
# cellphone or state identification numbers. They are not relevant for the estimation.
unnecessary_variables = [
'Id', '_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'SEQNO', '_PSU', 
'DISPCODE',  'CTELENUM', 'PVTRESD1', 'COLGHOUS', 'STATERES', 'CELLFON3', 'CTELNUM1', 'CELLFON2', 'PVTRESD2',
'CCLGHOUS', 'CSTATE', 'LANDLINE', 'HHADULT', 'CADULT', 'NUMWOMEN', 'NUMMEN' , 'NUMADULT', 'LADULT',          
]

def unused_features_indices(headers):
    """
    Remove unused features 
    """
    # features name as dict
    headers_dict = dict(zip(headers, range(len(headers))))
    unused_features = [headers_dict[key] for key in unnecessary_variables]
    return unused_features

def filter_features(features_name):
    return

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