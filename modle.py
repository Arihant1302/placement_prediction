import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sympy import AssumptionsContext
import functions

df = pd.read_csv("data\data.csv")


x = df.drop(columns=["status","sl_no","gender","hsc_s","hsc_b","ssc_b","degree_t","workex","mba_p","salary","specialisation"])

y=df.status

df,cwmv = functions.is_null_sum(df)
df = functions.impute_missing_values(df,cwmv)
df,ctv = functions.encoding_variables(df)
functions.encoder(df,ctv)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=40,shuffle=True)
rf = RandomForestClassifier()
final_model=rf.fit(x_train,y_train)


def placement(a,b,c,d):
    ress = final_model.predict([[a,b,c,d]])
    final = print(ress)
    return final
