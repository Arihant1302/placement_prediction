from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
import preprocess



def driver(df):
        '''
                    MethodName : final_processing
                    Function   : completes all the preprocesing tht is to be done on the dataset.
                    Returns    : train_x,test_x,train_y,test_y
                    onFailure  : Raise Exception
        '''
        try:
            df = preprocess.is_null(df)
            df = preprocess.encoding_variables(df)
            train_x,test_x,train_y,test_y=preprocess.split_xy(df)
            return train_x,test_x,train_y,test_y
        except Exception as e:
            print("This is the Exception2 : {} ".format(e))

def best_model_finder(train_x,test_x,train_y,test_y):

        '''
                    MethodName : best_model_finder
                    Function   : Split the dataframe and trains the dataframe on XGB and Random forest to find the best model to be used
                    Returns    : Accuracy score and of the best model
                    onFailure  : Raise Exception
        '''
        try:
            rf = RandomForestClassifier()
            train = rf.fit(train_x,train_y)
            rf_pred = rf.predict(test_x)
            rf_score = accuracy_score(test_y,rf_pred)
            xgb = XGBClassifier()
            xgb_model = xgb.fit(train_x,train_y)
            xgb_pred = xgb.predict(test_x)
            xgb_score = accuracy_score(test_y,xgb_pred)
            if(rf_score < xgb_score):
                with open('model_pkl', 'wb') as files:
                    model = pickle.dump(xgb, files)
                return model
            else:
                with open('model_pkl', 'wb') as files:
                    model = pickle.dump(rf, files)
                return model
        except Exception as e:
            print("This is the error : {} ".format(e))
    
def final_model(a,b,c,d):
        '''
                    MethodName : final_model
                    Function   : does the final prediction
                    Returns    : returns the prediction
                    onFailure  : Raise Exception
        '''
        try:
            res = pickle.load(open('model_pkl','rb'))
            pred = res.predict([[a,b,c,d]])
            return pred
        except Exception as e:
            print("This is the Exception3 : {}".format(e))