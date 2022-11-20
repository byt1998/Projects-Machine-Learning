from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

label = ["Not Failure", "Failure"]
datacol = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque',
       'tool_wear', 'Type_H', 'Type_L', 'Type_M']
with open("best_pipe_rf.pkl", "rb") as file_1:
    best_pipe_rf = pickle.load(file_1)
with open("best_pipe_svm.pkl", "rb") as file_2:
    best_pipe_svm = pickle.load(file_2)
with open("best_pipe_xgb.pkl", "rb") as file_3:
    best_pipe_xgb = pickle.load(file_3)
# with open("model_smote.pkl", "rb") as file_4:
#     model_smote = pickle.load(file_4)

@app.route("/")
def welcome():
    return "<h3>Welcome to my Predictive Maintenance</h3>"

@app.route("/predict", methods=["GET", "POST"])
def predict_maintenance():
    if request.method == "POST":
        content = request.json
        try:
            new_var = {'air_temperature': content['air_temperature'],
                        'process_temperature': content['process_temperature'],
                         'rotational_speed': content['rotational_speed'],
                         'torque' : content['torque'],
                         'tool_wear': content['tool_wear'],
                         'Type_H': content['Type_H'],
                         'Type_L': content['Type_L'],
                         'Type_M': content['Type_M']}
            new_data = pd.DataFrame([new_var])
            res = best_pipe_xgb.predict(new_data)
            result = {'class':str(res[0]),
                    'class_name':label[res[0]]}
            response = jsonify(success=True,
                                result=result)
            return response, 200  
        except Exception as e:
            response = jsonify(success=False,
                                message=str(e))
            return response, 400

    
    # return dari method get
    return "<p>Silahkan gunakan method POST untuk mode <em>inference model</em></p>"

# app.run(debug=True)