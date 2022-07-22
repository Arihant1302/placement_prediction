from logging import exception
from tkinter import EXCEPTION
from typing_extensions import Self
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder

class Presprocessor:

    '''     Here we will clean the data and get it ready for
            furthert operations .

            '''
    def __init__(self,data):
        self.data = data
    def is_null_present(self,data):
        '''
                Lets check if there are any null values in the dataset
    
                MethodName : is_null_present
                Operation  : Checks if there are any any mising values.
                Returns    : Returns true if there are null values present also returns the columns which has null values.
                onFailure  : Raise exception.
                
                '''
        self.data = data
        self.cwmv = []
        self.cols = data.columns
        self.null_present = False
        try:
            self.null_counts = data.isnull().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                        self.null_present=True
                        self.cwmv.append(self.cols[i])
            return self.null_present,self.cwmv            
        except exception as e:
            raise Exception()
    def impute_missing_values(self,data,cwmv):
        '''  
            MethodName : impute_missing_values
            Operation  : Replaces the missing values in the column with the mode values of the column.
            Returns    : The dataset with no missing values
            onFailure  : Raise Exception
            
            '''
        try : 
            self.data = data
            self.col_with_missing_val = cwmv
            for col in self.col_with_missing_val:
                data[col]=data[col].fillna(data[col].mode()[0])
            return data
        except exception as e:
            raise Exception()
    def cat_var_present(self,data):

            '''
                MethodName : cat_var_present
                Operation  : Checks if there are any categorical columns.
                Returns    : Returns true if there are categorical features present also returns a list of the features.
                onFailure  : Raise exception

            '''
            self.ctv = []
            self.cols = data.columns
            self.cat_var = False
            self.a = list(data.select_dtypes(include=['object']).columns)
            self.cat_var_count = len(self.a) 
            try:  
                if self.cat_var_count>0:
                    self.cat_var=True
                    if(self.cat_var):
                        for i in range(len(self.a)):
                            self.ctv.append(self.a[i])
                return self.cat_var,self.ctv
            except exception as e:
                    raise EXCEPTION()

    def enc(self,data):
        '''
            MethodName : encoder
            Operation  : Converts the categorical features into binary
            Returns    : A dataframe with no categorical features
            onFailure : raise exception    
        '''
        try:
            self.le = preprocessing.LabelEncoder()
            for i in self.ctv:
                data[i]=self.le.fit_transform(data[i])
                return data
        except exception as e:
                raise EXCEPTION()