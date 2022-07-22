from tkinter import EXCEPTION
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score

class rf:
    def __init__(self,data):
        self.data=data
    def split(self,data):
        '''
                    MethodName  : split
                    Function    : splits the whole dataset into training and testing 
                    Returns     : returns x_train,y_train,x_test,y_test
                    onFalure    : raise Exception
        '''
        try:
            self.x = data.drop(columns=["status","sl_no","gender","hsc_s","hsc_b","ssc_b","degree_t","workex","mba_p","salary"])
            self.y = data.status
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,random_state=40,shuffle=True)
            return self.x_train,self.x_test,self.y_train,self.y_test
        except exception as e:
            raise EXCEPTION()
    def rf_model_traininig(self,x_train,y_train):
        '''
                    MethodName : model_training
                    Function   : Trains the random forest clssifier model with the dataset
                    Returns    : Accuracy score for the tained model
                    onFailure  : Raise Exception
        '''
        rf = RandomForestClassifier()
        self.train = rf.fit(self.x_train,self.y_train)
        self.rfscore = accuracy_score(self.train)
        return self.rfscore
    
