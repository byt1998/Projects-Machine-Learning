from flask import Flask, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import pandas as pd
import tensorflow as tf 
from tensorflow import keras

app = Flask(__name__)

def open_model(model_path):
    with open(model_path, 'rb') as f:
        model=joblib.load(f)
    return model

scaler = open_model('fe.pkl')
model_func = tf.keras.models.load_model('model_func.h5')

def scalee_delay(data):
    """
    input : list with length : 4 --> [1, 2, 3, 4]
    output : predicted class (idx, label)
    """
    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
    data = pd.DataFrame([data], columns=columns)
    data_scaled = scaler.transform(data)
    return data_scaled


def inference_delay(data):
    """
    input : list with length : 4 --> [1, 2, 3, 4]
    output : predicted class (idx, label)
    """
    LABEL = ["Not Churn", "Churn"]
    res = model_func.predict(scalee_delay(data))
    return res[0], LABEL[res[0]]


@app.route("/")
def welcome():
    return "<h3>Welcome to my Predict Churn</h3>"

@app.route("/predict", methods=["POST"])
def churn_predict():

    content = request.json
    newdata = [
        content['gender'], 
        content['SeniorCitizen'],
        content['Partner'],
        content['Dependents'],
        content['tenure'],
        content['PhoneService'],
        content['MultipleLines'],
        content['InternetService'],
        content['OnlineSecurity'],
        content['OnlineBackup'],
        content['DeviceProtection'],
        content['TechSupport'],
        content['StreamingTV'],
        content['StreamingMovies'],
        content['Contract'],
        content['PaperlessBilling'],
        content['PaymentMethod'],
        content['MonthlyCharges'],
        content['TotalCharges']
               ]
    res_idx, res_label = inference_delay(newdata)
    result = {
        'label_idx': str(res_idx),
        'label_name': res_label
    }
    response = jsonify(success=True,
                       result=result)
    return response, 200

app.run(debug=True)