# lc_utils.py

# Code functions that are needed to run this lab
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import math

import pandas as pd
#from pandas import scatter_matrix
from pandas.plotting import scatter_matrix


#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import glob

# custom library for some helper functions 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
import seaborn as sns
from itertools import compress
import itertools
import operator

from abc import ABCMeta, abstractmethod

#CLASS_ENVIRONMENT = ["wsl-1231" : "" , 
#print("CLASS_ENVIRONMENT = {}".format(CLASS_ENVIRONMENT))
# utility print function
def nprint(mystring) :
    print("**df_utils:{}** : {}".format(sys._getframe(1).f_code.co_name,mystring))



class MLDF :
    """
    Abstract class for standard handling of ML dataframes.  This is the stuff i always do no matter what

    will have standard implementations, and then specific custom DF stuff

    """
    def __init__(self,df):
        self.df = df
        self.df_meta = None


    def quick_overview_1d(self) :
        '''
        Prints useful statistics about the dataframe ...
        '''
        self.df_meta = pd.DataFrame(self.df.dtypes, columns=["dtype"])
        nprint("There are " + str(len(self.df)) + " observations in the dataset.")
        nprint("There are " + str(len(self.df.columns)) + " variables in the dataset.")
    
        nprint("\n\n")
        nprint("\n****************** Histogram of data types  *****************************\n")
        nprint("use df.dtypes ...")
        print(self.df_meta['dtype'].value_counts())
    
        # Cardinality Report
        nprint("\n\n")
        nprint("\n****************** Generating Cardinality Report (all types) *****************************\n")
        #tmpdf = self.df.select_dtypes(include=['object'])
        # Cardinality report
        card_count = []
        card_idx = []
        for c in self.df.columns :
            #print("{} {} ".format(c, len(self.df[c].value_counts())))
            card_count.append(len(self.df[c].value_counts()))
            card_idx.append(c)
        card_df = pd.DataFrame(card_count, index =card_idx, 
                                              columns =['cardinality']) 
        self.df_meta = self.df_meta.join(card_df, how="outer")
        
        nprint("\n****************** Generating NaNs Report *****************************\n")
        nan_df = pd.DataFrame(self.df.isna().sum(), columns=['nan_count'])
        self.df_meta = self.df_meta.join(nan_df, how="outer")
        # Add pct missing
        self.df_meta['pct_missing'] = 100 * (self.df_meta['nan_count'] / len(self.df))



        nprint("\n******************Generating Descriptive Statistics (numerical columns only) *****************************\n")
        print(" running df.describe() ....")
        desc_df = self.df.describe().transpose()
        self.df_meta = self.df_meta.join(desc_df, how="outer")

        self.df_meta = self.df_meta.sort_values(by=['dtype'],ascending=False)

        return self.df_meta
    
    def detailed_overview(df) :
        print(df.info())

    def quick_overview_2d(self, cols) :
        nprint("There are " + str(len(self.df)) + " observations in the dataset.")
        nprint("There are " + str(len(self.df.columns)) + " variables in the dataset.")
    
        tmpdf = self.df[cols]
        corr_df = tmpdf.corr()
        # plot the heatmap
        sns.set_style(style = 'white')
        # Add diverging colormap from red to blue
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(corr_df, cmap=cmap,
            xticklabels=corr_df.columns,
            yticklabels=corr_df.columns,vmin=-1.0,vmax=1.0)

    # def random_sample(self,pct ) :
    #     '''
    #     Shrink down based on pct ....
    #     '''
    #    self.df  =

    # Here are the custom methods required !
    @abstractmethod
    def do_something(self):
        pass