from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
#import preprocess

'''
        Lets Import the rquired modules
'''


from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder 

class Final:

    '''     Here we will clean the data and get it ready for
            furthert operations .

            '''
    cwmv = []
    ctv = []

    def __init__(self,df):
        self.df = df

    def is_null(self,data):
        '''
                Lets check if there are any null values in the dataset
    
                MethodName : is_null
                Operation  : recognize that there are mising values in the column and change thoose values.
                Returns    : Returns a data fram with no missing values.
                onFailure  : Raise exception.
                
                '''
        try:
            self.cols = self.df.columns
            self.null_present = False
            self.null_counts = self.df.isnull().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                        self.null_present=True
                        self.cwmv.append(self.cols[i])
                self.col_with_missing_values = self.cwmv
                for col in self.col_with_missing_values:
                    self.df[col]=self.df[col].fillna(self.df[col].mode()[0])
            return self.df     
        except Exception as e:
                       print("This is the error {}".format(e))
        
    self.df = self.df

    def encoding_variables(self,df):
        '''
                MethodName : encoding_variables
                Operation  : transform cataegorical features into non-categorical.
                Return     : a df with no categorical features.
                OnFailure  : Raise Exception.

        '''
        self.cat_var = False
        self.a = list(self.df.select_dtypes(include=['object']).columns)
        self.cat_var_count = len(self.a) 
        if self.cat_var_count>0:
            self.cat_var=True
        if(self.cat_var):
            for i in range(len(self.a)):
                self.ctv.append(self.a[i])
        self.le = preprocessing.LabelEncoder()
        for i in self.ctv:
            self.df[i]=self.le.fit_transform(df[i])
        return self.df                       

    self.df = self.df

    def split_xy(self,df):
        '''  
            MethodName : split_transform
            Operation  : split the df into x and y also transforms the y feature (categorical) into binary feature.
            Returns    : x/independent and y /dependent features
            onFailure  : Raise Exception
            
        '''
        try:
            self.x = self.df.drop(columns=["status","sl_no","gender","hsc_s","hsc_b","ssc_b","degree_t","workex","mba_p","salary"])
            self.y = self.df["status"]
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,random_state=40,shuffle=True)
            return self.x_train,self.x_test,self.y_train,self.y_test 
        except Exception as e:
            print("This is the exception : {}".format(e))
    
    self.df = self.df


    def driver(self,df):
        '''
                    MethodName : final_processing
                    Function   : completes all the preprocesing tht is to be done on the dataset.
                    Returns    : train_x,test_x,train_y,test_y
                    onFailure  : Raise Exception
        '''
        try:
            self.train_x,self.test_x,self.train_y,self.test_y=self.split_xy(self.df)
            return self.train_x,self.test_x,self.train_y,self.test_y
        except Exception as e:
            print("This is the Exception : {} ".format(e))

    self.df = self.df
            

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

    self.df = self.df
            
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