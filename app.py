import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('best_fit_model.pkl')

# Features the model was trained on
features = ['Brand', 'MRP', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

st.set_page_config(page_title="E-Commerce Smartphone Price Predictor", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Home", "Prediction"])

if page == "Home":
    st.title("📱 E-Commerce Smartphone Price Predictor")
    st.image("https://cdn.pixabay.com/photo/2017/01/06/19/15/smartphone-1957741_1280.jpg", use_container_width=True)
    st.markdown("""
    ### Overview
    This project predicts smartphone selling prices based on features such as **Brand, RAM, ROM, Battery, Display Size, and Cameras**.
    
    #### Features Used:
    - Brand
    - MRP
    - RAM (GB)
    - ROM (GB)
    - Display Size (inches)
    - Battery (mAh)
    - Front Camera (MP)
    - Back Camera (MP)
    
    ---
    """)

elif page == "Prediction":
    st.title("📊 Price Prediction")

    input_features = {}
    for feat in features:
        if feat == 'Brand':
            input_features[feat] = st.selectbox("Select Brand", ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Vivo', 'Oppo'])
        else:
            input_features[feat] = st.number_input(f"Enter {feat}", value=0.0)

    if st.button('Predict Price'):
        df = pd.DataFrame([input_features], columns=features)
        pred = model.predict(df)
        st.success(f"💰 Predicted Selling Price: ₹{pred[0]:,.2f}")
