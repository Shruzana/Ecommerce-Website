import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_fit_model.pkl")

# Features for prediction
features = ['Brand', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

# App Title
st.set_page_config(page_title="E-Commerce Smartphone Price Predictor", layout="wide")

# Sidebar Navigation
menu = ["Home", "Prediction", "About"]
choice = st.sidebar.selectbox("Navigation", menu)

# HOME PAGE
if choice == "Home":
    st.title("📱 E-Commerce Smartphone Price Predictor")
    st.image("https://cdn.pixabay.com/photo/2014/04/03/10/32/mobile-phone-311797_1280.png", use_container_width=True)

    st.markdown("""
    ### Project Overview  
    This app predicts the **selling price** of smartphones based on specifications like RAM, ROM, Display Size, Camera, and more.  
    The prediction is powered by a machine learning model trained on real e-commerce data from Amazon and Flipkart.  

    **Features used for prediction**:
    - Brand
    - MRP (Maximum Retail Price)
    - RAM (GB)
    - ROM / Storage (GB)
    - Display Size (inches)
    - Battery (mAh)
    - Front Camera (MP)
    - Back Camera (MP)
    """)

# PREDICTION PAGE
elif choice == "Prediction":
    st.title("🔮 Smartphone Price Prediction")

    input_features = {}

    # Dropdown for Brand
    brand_list = ['Samsung', 'Apple', 'Xiaomi', 'Realme', 'Oppo', 'Vivo', 'OnePlus', 'Other']
    input_features['Brand'] = st.selectbox("Select Brand", brand_list)

    # Numeric inputs
    for feat in features[1:]:  # Skip Brand as it's already taken
        input_features[feat] = st.number_input(f"Enter {feat}", min_value=0.0, step=0.1)

    if st.button("Predict Price"):
        df = pd.DataFrame([input_features])
        pred = model.predict(df)
        st.success(f"💰 Predicted Selling Price: ₹{pred[0]:,.2f}")

# ABOUT PAGE
elif choice == "About":
    st.title("ℹ️ About this Project")
    st.write("""
    - **Developer:** Your Name  
    - **Model:** Trained using Lasso Regression  
    - **Dataset:** Scraped from Amazon & Flipkart mobile listings  
    - **Goal:** Help users and sellers estimate the selling price of smartphones based on specifications.  
    """)

