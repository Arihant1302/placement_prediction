
'''
        Lets Import the rquired modules
'''


from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder 

class Presprocessor:

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
    
                MethodName : is_null_present
                Operation  : Checks if there are any any mising values.
                Returns    : Returns true if there are null values present also returns the columns which has null values.
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

    '''def impute_missing_values(self,data,cwmv):
            MethodName : impute_missing_values
            Operation  : Replaces the missing values in the column with the mode values of the column.
            Returns    : The dataset with no missing values
            onFailure  : Raise Exception
            
            
        try : 
            self.data = data
            self.col_with_missing_val = cwmv
            for col in self.col_with_missing_val:
                self.data[col]=self.data[col].fillna(self.data[col].mode()[0])
            return self.data
        except Exception as e:
                       print("This is the error {}".format(e))'''

    def encoding_variables(self,df):
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

    '''  def train_test(self,x,y):
            MethodName : train_test
            Operation  : split the df into train and test 
            Returns    : x_train,x_test,y_train,y_test
            onFailure  : Raise Exception            
        
        try:
            print(self.data)
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,random_state=40,shuffle=True)
            print(self.data)
            return self.x_train,self.x_test,self.y_train,self.y_test 
        except Exception as e:
            print("This is the exception : {}".format(e))
    '''
    