
'''
        Lets Import the rquired modules
'''


from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder 

cwmv = []
ctv = []

def is_null(df):
        '''
                Lets check if there are any null values in the dataset
    
                MethodName : is_null
                Operation  : recognize that there are mising values in the column and change thoose values.
                Returns    : Returns a data fram with no missing values.
                onFailure  : Raise exception.
                
                '''
        try:
            cols = df.columns
            null_present = False
            null_counts = df.isnull().sum()
            for i in range(len(null_counts)):
                if null_counts[i]>0:
                        null_present=True
                        cwmv.append(cols[i])
                col_with_missing_values = cwmv
                for col in col_with_missing_values:
                    df[col]=df[col].fillna(df[col].mode()[0])
            return df     
        except Exception as e:
                       print("This is the error {}".format(e))

def encoding_variables(df):
        '''
                MethodName : encoding_variables
                Operation  : transform cataegorical features into non-categorical.
                Return     : a df with no categorical features.
                OnFailure  : Raise Exception.

        '''
        cat_var = False
        a = list(df.select_dtypes(include=['object']).columns)
        cat_var_count = len(a) 
        if cat_var_count>0:
            cat_var=True
        if(cat_var):
            for i in range(len(a)):
                ctv.append(a[i])
        le = preprocessing.LabelEncoder()
        for i in ctv:
            df[i]=le.fit_transform(df[i])
        return df                       

def split_xy(df):
        '''  
            MethodName : split_transform
            Operation  : split the df into x and y also transforms the y feature (categorical) into binary feature.
            Returns    : x/independent and y /dependent features
            onFailure  : Raise Exception
            
        '''
        try:
            x = df.drop(columns=["status","sl_no","gender","hsc_s","hsc_b","ssc_b","degree_t","workex","mba_p","salary"])
            y = df["status"]
            x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=40,shuffle=True)
            return x_train,x_test,y_train,y_test 
        except Exception as e:
            print("This is the exception1 : {}".format(e))
