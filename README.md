### Coronary disease prediction
This project is a part of the EPFL Machine Learning Course CS-433. 

## Project Description
Given the annual Behavioral Risk Factor Surveillance System (BRFSS) 2015 data (https://www.cdc.gov/brfss/annual_data/annual_2015.html), we are given a vector of features collecting the health-related data of a person, and asked to predict whether this situation leads to a coronary heart disease (MICHD) or not. To do this, we use the binary classification techniques we have discussed in the lectures.

### Contributors
* Anders Hominal : anders.hominal@gmail.com
* Nicolas Filimonov : nicolas.filimonov@epfl.ch
* Yasmine Chaker : yasmine.chaker@epfl.ch

### Files
* `preprocessing.py` : functions used to preprocess our dataset
* `utils.py` : file containing the function used for computation and data visualization (plots, KNN, loss calculations, etc)
* `feature_selection.ipynb`: a jupyter notebook where we detail each step of the feature selection
* `helpers.py`: helpers functions mostly used for reading/saveing and parsing files
* `implementations.py` : file containing graded functions
* `run.py`: file to reproduce our best result on AICrowd

### Download and extract data
To download the dataset used, go on: `https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/dataset_files?unique_download_uri=301883&challenge_id=66`. Extract the zip `dataset.zip` in a folder named `data` such that the path from the root of the project folder to the data files `x_train.csv`, `x_test.csv` will be: `data/dataset/x_train.csv`.

### Requirements
* python (3.9) or higher
* numpy (1.23.1)
* matplotlib (3.5.2)

### Best model prediction
To reproduce the best result (and best submission) shown on `https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/leaderboards` simply input the following command in your terminal (be careful of your python version):

* python run.py

The best results will then be saved in a csv file under the name `submission.csv`. This file will be saved at the root of the project folder.
