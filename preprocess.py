
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
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                        self.null_present=True
                        self.cwmv.append(self.cols[i])
            return self.null_present,self.cwmv            
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
    
    