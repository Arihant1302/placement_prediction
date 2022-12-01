from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class data_preprocessing_:

    def is_null_sum(df):
        '''
                    Lets check if there are any null values in the dataset
        
                    MethodName : is_null_sum
                    Operation  : recognize that there are mising values in the column and replace thoose values with the mean of the column.
                    Returns    : Returns a dataframe with no missing values.
                    onFailure  : Raise exception.
            '''
        try:
            
            cwmv = []
            cols = df.columns
            null_present = False
            null_counts_of_columns = df.isnull().sum()
            for i in range(len(null_counts_of_columns)):
                if null_counts_of_columns[i]>0:
                        null_present=True
                        cwmv.append(cols[i])
            col_with_missing_values = cwmv
            for col in col_with_missing_values:
                df[col]=df[col].fillna(int(df[col].mean()))
            return df
        except Exception as e:
            print("Exception occured")

        
    def encoding_variables(df):
        '''  
                    MethodName : encoding_variables
                    Operation  : Transforms the categorical features to binary feature.
                    Returns    : df with no categorical features
                    onFailure  : Raise Exception
                    
            '''
        try:
            
            ctv = []
            cat_var = False
            cols = df.columns
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
        except Exception as e:
            print(e)
                
        
    def features_with_high_corr(df):
        '''

                MethodName : features_with_high_corr
                Operation  : checks the correlation between different features and returns an array of features having high corelation
                Returns    : An array of highly corelated features
                onFailure  : Raise Exception
                
            '''
        try:
            df2 = df.iloc[:,1:]
            corr = df2.corr()    
            corr.style.background_gradient(cmap="PuBu")
            arr1 = corr["status"]
            cols_with_high_corr = arr1[(arr1 > 0.1) & (arr1<1)].index.values[0::]
            return cols_with_high_corr
        except Exception as e:
            print(e)
        
    
    def splitter(df,l):
        '''

                MethoName : splitter        
                Operation : splits the df into training and testing data
                Returns   : x_train , x_test , y_train , y_test
                OnFailure : Raise Exception

        '''
        try:
            x = df[l]
            y = df["status"]
            x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=40,shuffle=True)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            print(e)

