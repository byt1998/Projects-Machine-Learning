import pandas as pd
from flask import Flask,request,jsonify
import pickle
from tensorflow import keras
import numpy as np

# inisiasi 
app = Flask(__name__)

# open model
def open_model(model_path):
    '''
    helper function for loading model
    '''
    with open(model_path,'rb') as f:
        model=pickle.load(f)
    return model


model_pipe=open_model('pipe.pkl')
model = keras.models.load_model('model.h5')

# fungsi untuk inference hasil
def inference_churn(data,model):  
    label=['No','Yes']
    res=model.predict(data)
    y_pred = np.where(res > 0.5, 1, 0)
    y_pred= int(y_pred)
    return y_pred,label[y_pred]

# halaman home
@app.route("/")
def homepage():
    return "<h1> Deployment Model BackEnd!</h1>"

# halaman inference churn
@app.route("/churnprediction",methods=['POST'])
def churn_predict():
    content=request.json
    newdata=[content['gender'],content['SeniorCitizen'],content['Partner'],content['Dependents'],content['tenure'],
            content['PhoneService'],content['MultipleLines'],content['InternetService'],content['OnlineSecurity'],content['OnlineBackup'],
            content['DeviceProtection'],content['TechSupport'],content['StreamingTV'],content['StreamingMovies'],content['Contract'],
            content['PaperlessBilling'],content['PaymentMethod'],content['MonthlyCharges'],content['TotalCharges']]
    columns = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity',
               'OnlineBackup','DeviceProtection','TechSupport' ,'StreamingTV' ,'StreamingMovies' ,'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']
    data=pd.DataFrame([newdata],columns=columns)
    new_data_fix=model_pipe.transform(data)
    res_idx,res_label = inference_churn(new_data_fix,model)
    result={'label_idx': res_idx,'label_name':str(res_label)}
    #result=jsonify(label_idx=str(res_idx),label_names=res_label)
    response=jsonify(success=True,result=result)
    return response, 200

# run app di local
# jika deploy di heroku, comment baris dibawah ini
# app.run(debug=True)
