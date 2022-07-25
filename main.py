import pickle
import pandas as pd
from flask import Flask,render_template,Response,request
#from flask_cors import CORS,cross_origin
import tuner


"""app = Flask(__name__)
CORS(app)

@app.route('/') # To render Homepage
def home_page():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if (request.method=='POST'):
         operation =request.form['values']
         """
df = pd.read_csv("D:\Ml Projects\placementprediction\data\data.csv")
model1 = tuner.models(df)
trainx,testx,trainy,testy  =  model1.driver(df)
model1.best_model_finder(trainx,testx,trainy,testy)
model1.final_model(60,50,60,90)
