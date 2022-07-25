
'''
        Lets Import the rquired modules
'''


from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
import pandas as pd
from flask import Flask,render_template,Response,request

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
        self.cols = self.data.columns
        self.null_present = False
        try:
            self.null_counts = self.data.isnull().sum()
            if self.null_counts>0:
                self.data = self.data.fillna((self.data[col].mode()[0])
)
        except Exception as e:
                       print("This is the error {}".format(e))

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
                self.data[col]=self.data[col].fillna(self.data[col].mode()[0])
            return self.data
        except Exception as e:
                       print("This is the error {}".format(e))

    def split_transform(self,data):
        '''  
            MethodName : split_transform
            Operation  : split the df into x and y also transforms the y feature (categorical) into binary feature.
            Returns    : x/independent and y /dependent features
            onFailure  : Raise Exception
            
        '''
        try:
            self.x = self.data[["ssc_p","hsc_p","degree_p","etest_p"]]
            print("the X",self.x)
            self.y = self.data["status"]
            print("the y",self.y)
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(self.y)
            print(self.data)
            return self.x,self.y
        except Exception as e:
            print("This is the exception : {}".format(e))

    def train_test(self,x,y):
        '''  
            MethodName : train_test
            Operation  : split the df into train and test 
            Returns    : x_train,x_test,y_train,y_test
            onFailure  : Raise Exception
            
        '''
        try:
            print(self.data)
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,random_state=40,shuffle=True)
            print(self.data)
            return self.x_train,self.x_test,self.y_train,self.y_test 
        except Exception as e:
            print("This is the exception : {}".format(e))
    
    from logging import exception


class models(Presprocessor):
    def __init__(self,data):
        self.data=data

    def final_processing(self,data):
        '''
                    MethodName : final_processing
                    Function   : completes all the preprocesing tht is to be done on the dataset.
                    Returns    : train_x,test_x,train_y,test_y
                    onFailure  : Raise Exception
        '''
        try:
            self.preprocessor = preprocess.Presprocessor(self.data)
            self.null_val_present,self.cols_with_nul_value = self.preprocessor.is_null_present(self.data)
            if self.null_val_present:
                self.data = self.preprocessor.impute_missing_values(self.data,self.cols_with_nul_value)
                print(" null values removed")
                print("No. of null value : {}".format(self.data.isnull().sum()))
                print(self.data)
            else:
                return self.data
            self.x,self.y = self.preprocessor.split_transform(self.data)
            self.train_x,self.test_x,self.train_y,self.test_y=self.preprocessor.train_test(self.x,self.y)
            return self.train_x,self.test_x,self.train_y,self.test_y
        except Exception as e:
            print("This is the Exception : {} ".format(e))

    def best_model_finder(self,a,b,c,d):
        '''
                    MethodName : best_model_finder
                    Function   : Split the dataframe and trains the dataframe on XGB and Random forest to find the best model to be used
                    Returns    : Accuracy score and of the best model
                    onFailure  : Raise Exception
        '''
        try:
            self.rf = RandomForestClassifier()
            self.train = self.rf.fit(self.train_x,self.train_y)
            self.rf_pred = self.rf.predict(self.test_x)
            self.rf_score = accuracy_score(self.test_y,self.rf_pred)
            self.xgb = XGBClassifier()
            self.xgb_model = self.xgb.fit(self.train_x,self.train_y)
            self.xgb_pred = self.xgb.predict(self.test_x)
            self.xgb_score = accuracy_score(self.test_y,self.xgb_pred)
            print(self.data)
            if(self.rf_score < self.xgb_score):
                with open('model_pkl', 'wb') as files:
                    self.model = pickle.dump(self.xgb, files)
                return self.model
            else:
                with open('model_pkl', 'wb') as files:
                    self.model = pickle.dump(self.rf, files)
                return self.model
        except Exception as e:
            print("This is the error : {} ".format(e))
    
    def final_model(self,a,b,c,d):
        '''
                    MethodName : final_model
                    Function   : does the final prediction
                    Returns    : returns the prediction
                    onFailure  : Raise Exception
        '''
        try:
            self.res = pickle.load(open('model_pkl','rb'))
            self.pred = self.res.predict([[a,b,c,d]])
            return self.pred
        except Exception as e:
            print("This is the Exception : {}".format(e))



        df = pd.read_csv("D:\Ml Projects\placementprediction\data\data.csv")
        model1 = tuner.models(df)
        trainx,testx,trainy,testy = model1.final_processing(df)
        model1.best_model_finder(trainx,testx,trainy,testy)
        abs = model1.final_model(60,50,60,90)
        return abs
    


