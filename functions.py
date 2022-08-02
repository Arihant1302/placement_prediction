from sklearn import preprocessing

def is_null_sum(df):
    cwmv = []
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
    return df,cwmv

def impute_missing_values(df,cwmv):
    data = df
    col_with_missing_values = cwmv
    for col in col_with_missing_values:
        data[col]=data[col].fillna(data[col].mode()[0])
    return data


def encoding_variables(df):
    ctv = []
    cols = df.columns
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
    return df,ctv
            

def encoder(df,ctv):
    le = preprocessing.LabelEncoder()
    for i in ctv:
        df[i]=le.fit_transform(df[i])
    return df