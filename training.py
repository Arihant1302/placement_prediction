from tkinter import EXCEPTION
import preprocess
import tuner
import pandas as pd
from logging import exception


class trainmodel:
    data = pd.read_csv("D:\Ml Projects\placementprediction\data\data.csv")
    
    def __init__(self,data):
        self.data = data
    
    def training(self,data):
        '''
                    MethodName  : training
                    Function    : completes the whole process of trining the model.
                    Returns     : the model that best fits the data
                    onFailure   : Raise exception
        '''
        self.data = data
        try:
            self.preprocessor = preprocess.Presprocessor(data)
            self.null_val_present,self.cols_with_nul_value = self.preprocessor.is_null_present(data)
            if self.null_val_present:
                self.data = self.preprocessor.impute_missing_values(data,self.cols_with_nul_value)
                print(" null values removed")
                print("No. null value : {}".format(self.data.isnull().sum()))
            else:
                return self.data
            self.cat_var_present,self.col_with_cat_values = self.preprocessor.cat_var_present(data)
            if self.cat_var_present : 
                self.data = self.preprocessor.enc(data,self.col_with_cat_values)
                print("catvalues printed")
                self.count = len(list(data.select_dtypes(include=['object']).columns))
                print("cat val checking count: {}".format(self.count))
            else:
                return self.data
            self.model_finder = tuner.models(self.data)
            self.final_model = self.model_finder.best_model_finder(self.data)
            print("succesfully built model {}".format(self.final_model))
            return self.final_model
        except Exception as e:
                        print("This is the error {} :".format(e))
