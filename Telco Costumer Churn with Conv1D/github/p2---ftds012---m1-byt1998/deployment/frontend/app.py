import streamlit as st
import pandas as pd
import pickle
import requests
import json
from PIL import Image
from streamlit_option_menu import option_menu

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
img = Image.open("FOTO 4x6 Merah #2.jpg")
st.set_page_config(
    page_title="2nd Milestone - Enggar Kristian",
    page_icon= img,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.github.com/byt1998',
        'Report a bug': "https://www.google.com/bug",
        'About': "# Milestone 02. Enggar's Streamlit!"
    }
)

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Predict Churn"],
        icons=["bar-chart-line"],
        menu_icon="cast",
        default_index=0,
    )


with open ('fe.pkl', 'rb') as f:
    prc = pickle.load(f)

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

st.header('Table Costumer Telco Prediction Churn')
if st.checkbox('Show Data'):
    st.write(data)
else:
    st.write('Check to see Data!')
    st.subheader('Attribute information')
    st.write('gender: Whether the customer is a male or a female')
    st.write('SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)')
    st.write('Partner: Whether the customer has a partner or not (Yes, No)')
    st.write('Dependents: Whether the customer has dependents or not (Yes, No)')
    st.write('tenure: Number of months the customer has stayed with the company')
    st.write('PhoneService: Whether the customer has a phone service or not (Yes, No)')
    st.write('MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)')
    st.write('InternetService: Customer’s internet service provider (DSL, Fiber optic, No)')
    st.write('OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)')
    st.write('OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)')
    st.write('DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)')
    st.write('TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)')
    st.write('StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service')
    st.write('StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)')
    st.write('Contract: The contract term of the customer (Month-to-month, One year, Two year)')
    st.write('PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)')
    st.write('PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))')
    st.write('MonthlyCharges: The amount charged to the customer monthly')
    st.write('TotalCharges: The total amount charged to the customer')
    st.write('Churn: Whether the customer churned or not (Yes or No)')


st.markdown(
        '<hr><h3 style="text-align: center">Customer Information</h3>',
        unsafe_allow_html=True,
    )

gender = st.selectbox('gender', ['Male', 'Female'])

seniorCitizen = st.selectbox('seniorCitizen', [1, 0])

Partner = st.selectbox('Partner', ['No', 'Yes'])

Dependents = st.selectbox('Dependents', ['No', 'Yes'])

tenure = st.number_input('tenure (month)', min_value=1, max_value=100)

PhoneService = st.selectbox('PhoneService Subscription', ['No', 'Yes'])

MultipleLines = st.selectbox('MultipleLines Subscription', ['No', 'Yes'])

InternetService = st.selectbox('InternetService Subscription', ['No', 'DSL', 'Fiber optic'])

OnlineSecurity = st.selectbox('OnlineSecurity Subscription', ["No", "Yes"])

OnlineBackup = st.selectbox('OnlineBackup Subscription', ['No', 'Yes'])

DeviceProtection = st.selectbox('DeviceProtection Subscription', ['No', 'Yes'])

TechSupport = st.selectbox('TechSupport Subscription', ['No', 'Yes'])

StreamingTV = st.selectbox('StreamingTV Subscription', ['No', 'Yes'])

StreamingMovies = st.selectbox('StreamingMovies Subscription', ['No', 'Yes'])

Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

PaperlessBilling = st.selectbox('PaperlessBilling', ['Yes', 'No'])

PaymentMethod = st.selectbox('PaymentMethod',['Electronic check','Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

MonthlyCharges = st.number_input('MonthlyCharges (Dollar)', min_value=0, max_value=150)

TotalCharges = st.number_input('TotalCharges (Dollar)', min_value=0, max_value=10000)

data = [
        gender,
        seniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
    ]

col5, col6, col7 = st.columns([1.75, 1, 1])
with col6:
    predict = st.button("Predict")

# Make list columns
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges','TotalCharges']

# Change list to dataframe
data_df = pd.DataFrame([data], columns = columns)

# Scaling and Encoding with columntransformer
data_df_prc = prc.transform(data_df)

# Predict proba from ANN
data_df_prc_list = data_df_prc.tolist()

input_data_json = json.dumps({
    'signature_name':'serving_default',
    'instances':data_df_prc_list
})

if predict:
        URL = "http://tf-serving-backend.herokuapp.com/v1/models/churn:predict"
        r = requests.post(URL, data=input_data_json)
        result = r.json()
        
        # if predict proba more than 0.5 will change to Churn, and less than 0.5 Not Churn.
        if result['predictions'][0][0] > 0.5:
            st.markdown(
                '<h2 style="text-align: center"> Churn </h2>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<h2 style="text-align: center"> Not Churn </h2>',
                unsafe_allow_html=True,
            )

# if predict:
#     with st.expander("See result"):
#         # URL = "http://127.0.0.1:5000/happiness_index_prediction" # URL lokal
#         URL = "http://arnaz-deployment-backend.herokuapp.com/happiness_index_prediction" # URL Heroku
#         r = requests.post(URL, json=data)
#         res = r.json()
#         if r.status_code == 200:
#             result=res['result']
#             name_ = name if name else "There"
#             hi = f"<h1 style='text-align: center;'>{name_.title()}</h1>"
#             Score = f"<h2 style='text-align: center;'>Happiness Index: <br> <span style='color: #519259'>{result}</span></h2>"
#             st.markdown(hi, unsafe_allow_html=True)
#             st.markdown(Score, unsafe_allow_html=True)
                        
#         else:
#             st.write("Prediction failed")
#             st.write(res["error"])