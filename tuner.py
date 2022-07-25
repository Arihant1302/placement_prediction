from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
import preprocess



class models:

    def __init__(self,df):
        self.df = df
  
    def driver(self,df):
        '''
                    MethodName : final_processing
                    Function   : completes all the preprocesing tht is to be done on the dataset.
                    Returns    : train_x,test_x,train_y,test_y
                    onFailure  : Raise Exception
        '''
        try:
            self.preprocessor = preprocess.Presprocessor(self.df)
            self.df = self.preprocessor.is_null(self.df)
            self.df = self.preprocessor.encoding_variables(self.df)
            self.train_x,self.test_x,self.train_y,self.test_y=self.preprocessor.split_xy(self.df)
            return self.train_x,self.test_x,self.train_y,self.test_y
        except Exception as e:
            print("This is the Exception : {} ".format(e))

    def best_model_finder(self,train_x,test_x,train_y,test_y):

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