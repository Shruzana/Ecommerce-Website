import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Pickles/knn.pkl")   # Change path if needed
scaler = joblib.load("Pickles/sc.pkl")   # If you used scaling in training

st.title("📱 E-Commerce Smartphone Price Predictor")

# Take inputs
mrp = st.number_input("Enter MRP", min_value=0.0, step=0.01)
discount_price = st.number_input("Enter Discount_Price", min_value=0.0, step=0.01)
discount = st.number_input("Enter Discount (%)", min_value=0.0, step=0.01)
ram = st.number_input("Enter RAM (GB)", min_value=0.0, step=0.01)
rom = st.number_input("Enter ROM (GB)", min_value=0.0, step=0.01)
display_size = st.number_input("Enter Display Size (inches)", min_value=0.0, step=0.01)
battery = st.number_input("Enter Battery (mAh)", min_value=0.0, step=0.01)
front_cam = st.number_input("Enter Front Camera (MP)", min_value=0.0, step=0.01)
back_cam = st.number_input("Enter Back Camera (MP)", min_value=0.0, step=0.01)

# Prepare input data exactly as in training
input_data = pd.DataFrame(
    [[mrp, discount_price, discount, ram, rom, display_size, battery, front_cam, back_cam]],
    columns=['MRP', 'Discount_Price', 'Discount', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']
)

# Apply scaling (if used in training)
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"📌 Predicted Value: {prediction[0]:.2f}")
