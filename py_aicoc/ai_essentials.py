# AUTOGENERATED! DO NOT EDIT! File to edit: Tabular/Lab1-LendingClub-AiImmersion.ipynb (unless otherwise specified).

__all__ = ['loan_df', 'loan_df_orig', 'loan_df', 'loan_df1', 'loan_df2', 'loan_df3', 'loan_df4', 'loan_df', 'loan_df',
           'loan_df', 'my_analysis', 'mode', 'x_cols', 'X', 'y', 'clf', 'X_test', 'Y_test_predict', 'cnf_matrix',
           'class_names', 'rf_feature_importance', 'sort_rf_feature_importance']

# Cell

# pick up lc_utils_2020.py file
import sys
sys.path.append("../Tabular")

# Cell
print("Importing Data")
# Code functions that are needed to run this lab
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import math

import pandas as pd
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import glob

# custom library for some helper functions
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import myenv as myenv
from lc_utils_2020 import *

# Cell
loan_df = load_sample_data('ornl')
loan_df_orig = loan_df.copy()
loan_df.head(30)

# Cell
print("Descriptive Statistics")
# This function provide the number of rows/cols
# Information on the types of data
# and a report of descriptive statistics

# quick_overview_1d(loan_df)
categorical_overview,numerical_overview = quick_overview_1d_v2(loan_df)

# Cell
print("Creating Loan Default column")
# function to create loan status ....
loan_df = create_loan_default(loan_df)

# Cell
print("Handling Nulls and NaNs")
# The following cleaning of the data makes use of the steps shown below.....

# loan_df1 = drop_sparse_numeric_columns(loan_df, threshold=0.03)
loan_df1 = drop_sparse_columns(loan_df,pct_missing_threshold=0.6)
loan_df2 = impute_columns(loan_df1)
loan_df3 = handle_employee_length(loan_df2)
loan_df4 = handle_revol_util(loan_df3)
loan_df = loan_df4
columns_with_nans(loan_df4)


# Cell
print("Data Preparation Handling Time Objects")
loan_df = create_time_features(loan_df)
loan_df.head(3)

# Cell
print("Transforming Data into binary indicator columns")
# Transform categorical data into binary indicator columns
# code hint, uses pd.get_dummies

loan_df = one_hot_encode_keep_cols(loan_df)
loan_df.head() # once complete, see how many new columns you have!

# Cell
print("Train Test Set Creation")
# Instantiate lendingclub_ml object that will hold our test, and contain methods used for testing.
# Implementation done like this to ease the burden on users for keeping track of train/test sets for different
# models we are going to build.

my_analysis = lendingclub_ml(loan_df)

# Cell

# Create a train / test split of your data set.  Paramter is test set size percentage
# Returns data in the form of dataframes

my_analysis.create_train_test(test_size=0.33)

# Cell
print("Setting Baseline")
# Set our baseline
my_analysis.train_df['default'].describe()

# Cell

# modes
# pca           : principal components only
# raw           : all the data non reduced
# raw_no_grades : all the data non reduced except the grade info provided by lending club

mode = 'raw' # ae , raw, raw_no_grades
x_cols =[]
if(mode == 'pca') :
            x_cols = [x for x in my_analysis.train_df.columns if 'PC' in x]
elif(mode == 'raw') :
            x_cols = [x for x in my_analysis.train_df.columns if 'PC' not in x]
            x_cols.remove('default')
elif(mode == 'raw_no_grades') :
            x_cols = [x for x in my_analysis.train_df.columns if 'PC' not in x]
            import re
            x_cols = [x for x in x_cols if not re.match('^[ABCDEFG]',x)]
            x_cols.remove('default')

#print(x_cols)


# Cell
print("Random Forest Example")
# Build a dataframe with selected columns ...
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = my_analysis.train_df[x_cols]
y = my_analysis.Y_train

# Cell

## Simple Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
clf = RandomForestClassifier(max_depth=5,n_estimators=300, random_state=0)

clf.fit(X,y)



# Cell
print("Confusion Matrix")
X_test = my_analysis.test_df[x_cols]
Y_test_predict = np.where(clf.predict(X_test) > 0.5, 1, 0 )

cnf_matrix =confusion_matrix(my_analysis.Y_test, Y_test_predict)
class_names =  ['Default','Paid']
plot_confusion_matrix(cnf_matrix, class_names)

# Cell
print("RF Feature Importance")
# Random Forest Explainability
#print(len(clf.feature_importances_))
#print(len(X.columns))
rf_feature_importance = dict(zip(X.columns,clf.feature_importances_))


sort_rf_feature_importance = sorted(rf_feature_importance.items(), key=lambda x: x[1], reverse=True)

for i in sort_rf_feature_importance:
    print(i[0], i[1])