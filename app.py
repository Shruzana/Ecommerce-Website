import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load("best_fit_model.pkl")  # Trained model
scaler = joblib.load("scaler.pkl")  # Scaler used during training

# Features list (must be same as in train.py)
FEATURES = ['Brand', 'MRP', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

st.title("📱 E-Commerce Smartphone Price Predictor")

# User inputs
brand = st.selectbox("Select Brand", ["Samsung", "Apple", "Xiaomi", "OnePlus", "Realme", "Oppo", "Vivo", "Other"])
mrp = st.number_input("Enter MRP", min_value=0.0, step=1.0)
ram = st.number_input("Enter RAM (GB)", min_value=0.0, step=1.0)
rom = st.number_input("Enter ROM (GB)", min_value=0.0, step=1.0)
display_size = st.number_input("Enter Display Size (inches)", min_value=0.0, step=0.1)
battery = st.number_input("Enter Battery Capacity (mAh)", min_value=0.0, step=100.0)
front_cam = st.number_input("Enter Front Camera (MP)", min_value=0.0, step=1.0)
back_cam = st.number_input("Enter Back Camera (MP)", min_value=0.0, step=1.0)

if st.button("Predict Price"):
    # Convert brand to numeric encoding (same encoding as training)
    brand_mapping = {
        "Samsung": 0, "Apple": 1, "Xiaomi": 2, "OnePlus": 3,
        "Realme": 4, "Oppo": 5, "Vivo": 6, "Other": 7
    }
    brand_val = brand_mapping.get(brand, 7)

    # Create dataframe with exact same columns
    input_data = pd.DataFrame([[brand_val, mrp, ram, rom, display_size, battery, front_cam, back_cam]],
                              columns=FEATURES)

    # Apply same scaler
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Predicted Selling Price: ₹{prediction:,.2f}")
