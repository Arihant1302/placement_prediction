import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
from prep import data_preprocessing_

print(sys.path)
class best_model_:

    def the_best_model():
        '''
                MethodName : the_best_model
                Operation  : return the model which has the highest accurcy in prediction
                Returns    : A pkl file of the best model
                onFailure  : Raise Exception
        '''
        try:
            df = pd.read_csv("D:\Ml Projects\placementprediction\data\data.csv")
            print(df)

            df = data_preprocessing_.is_null_sum(df)
            df = data_preprocessing_.encoding_variables(df)

            cols_to_keep = data_preprocessing_.features_with_high_corr(df)
            model_pipeline = []
            model_pipeline.append(LogisticRegression())
            model_pipeline.append(DecisionTreeClassifier())
            model_pipeline.append(KNeighborsClassifier())
            model_pipeline.append(RandomForestClassifier())

            model_list=['Logistic Regression','Decision Tree','KNN','RF']
            acc_list = []

            x_train,x_test,y_train,y_test = data_preprocessing_.splitter(df,cols_to_keep)

            for model in model_pipeline:
                model.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                acc_list.append(metrics.accuracy_score(y_test,y_pred))


            result_df = pd.DataFrame({'Model':model_list,'Accuracy':acc_list})

            best_model = result_df[(result_df.Accuracy > 0.85)].Model.values[0::]

            models_dict = {"Logistic Regression":model_pipeline[0],"Decision Tree":model_pipeline[1],"KNN":model_pipeline[2],"RF":model_pipeline[3]}

            final_model = pickle.dump(models_dict[best_model[0]], open('model.pkl', 'wb'))

            return final_model
        except Exception as e:
            print(e)

        

    

    def placement_predictor(a,b,c,d,e):
        '''
            MethodName : placement_predictor
            Operation  : Predic whether the candidate gets placed/not
            Returns    : Placed/Not Placed text
            OnFailure  : Raise Exception

        '''
        try:
            last_model = pickle.load(open(best_model_.the_best_model(),'rb'))
            prediction = last_model.predict([[a,b,c,d,e]])
            return prediction
        except Exception as e:
            print(e)

