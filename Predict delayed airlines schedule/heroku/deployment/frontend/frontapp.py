import streamlit as st
import requests
from PIL import Image
from streamlit_option_menu import option_menu


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
        options=["Predictive"],
        icons=["bar-chart-line"],
        menu_icon="cast",
        default_index=0,
    )

st.title("Predictive Maintenance Application")
air_temperature = st.number_input("Air Temperature [K]")
process_temperature = st.number_input("Process Temperature [K]")
rotational_speed = st.number_input("Rotational Speed [rpm]")
torque = st.number_input("Torque [Nm]")
tool_wear = st.number_input("Tool Wear [min]")
Type_H = st.selectbox("Quality High", [0, 1])
Type_L = st.selectbox("Quality Low", [0, 1])
Type_M = st.selectbox("Quality Medium", [0, 1])

# inference
data = {'air_temperature':air_temperature,
        'process_temperature':process_temperature,
        'rotational_speed': rotational_speed,
        'torque':torque,
        'tool_wear':tool_wear,
        'Type_H':Type_H,
        'Type_L':Type_L,
        'Type_M':Type_M}

URL = "http://127.0.0.1:5000/predict"

# komunikasi
r = requests.post(URL, json=data)
res = r.json()
if r.status_code == 200:
    st.title(res['result']['class_name'])
elif r.status_code == 400:
    st.title("ERROR BOSS")
    st.write(res['message'])