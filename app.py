import streamlit as st
import pandas as pd
import joblib

# Load your saved model & scaler
model = joblib.load("Pickles/lasso.pkl")  # Replace with your actual model file
scaler = joblib.load("Pickles/sc.pkl")   # Replace if scaling is used

# Available brands
brands = ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Realme', 'Oppo', 'Vivo', 'Motorola']

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

if page == "Home":
    st.title("📱 Smartphone Price Prediction Project")
    st.image("project_image.jpg", caption="E-Commerce Price Prediction", use_column_width=True)  # Change image path
    st.header("Overview")
    st.write("""
    This project predicts the selling price of smartphones based on various features.
    It uses a machine learning model trained on e-commerce data (Amazon, Flipkart).
    """)
    st.subheader("Features Used in Prediction")
    st.markdown("""
    - **Brand**: Company of the smartphone  
    - **MRP**: Maximum Retail Price (₹)  
    - **RAM**: Memory in GB  
    - **ROM**: Storage in GB  
    - **Display Size**: Screen size in inches  
    - **Battery**: Capacity in mAh  
    - **Front Camera**: Megapixels  
    - **Back Camera**: Megapixels  
    """)

elif page == "Prediction":
    st.title("📊 Predict Smartphone Selling Price")

    # Collect user inputs
    brand = st.selectbox("Brand", brands)
    mrp = st.number_input("MRP (₹)", min_value=1000, step=500)
    ram = st.number_input("RAM (GB)", min_value=1, step=1)
    rom = st.number_input("ROM (GB)", min_value=8, step=8)
    display_size = st.number_input("Display Size (inches)", min_value=4.0, step=0.1)
    battery = st.number_input("Battery (mAh)", min_value=1000, step=500)
    front_cam = st.number_input("Front Camera (MP)", min_value=1, step=1)
    back_cam = st.number_input("Back Camera (MP)", min_value=5, step=1)

    # Convert brand to numeric encoding (example: one-hot or label encoding)
    brand_mapping = {b: i for i, b in enumerate(brands)}
    brand_encoded = brand_mapping[brand]

    # Prepare input DataFrame
    input_data = pd.DataFrame([[brand_encoded, mrp, ram, rom, display_size, battery, front_cam, back_cam]],
                              columns=['Brand', 'MRP', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)'])

    # Scale input if necessary
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        pred_price = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted Selling Price: ₹{pred_price:,.2f}")
    coef_dict = dict(zip(features, model.coef_))
    st.write('Model Coefficients:')
    st.json(coef_dict)
    st.write(f"Intercept: {model.intercept_:.2f}")

