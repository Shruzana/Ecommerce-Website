import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("best_fit_model.pkl")

# Expected features (as per your trained model)
features = ['Brand', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction"])

# HOME PAGE
if page == "🏠 Home":
    st.title("📱 E-Commerce Smartphone Price Predictor")
    st.image("https://cdn.pixabay.com/photo/2017/01/06/19/15/smartphone-1957740_960_720.jpg", use_container_width=True)

    st.markdown("""
    ## 📌 Project Overview
    Predict smartphone selling price based on brand, specifications, and camera quality.

    **Features:**
    - Brand
    - RAM
    - ROM
    - Display Size
    - Battery
    - Front Camera (MP)
    - Back Camera (MP)
    """)

    st.info("💡 Go to the **Prediction** tab from the sidebar to try it!")

# PREDICTION PAGE
elif page == "📊 Prediction":
    st.title("📊 Predict Smartphone Selling Price")

    # Input dictionary
    input_features = {}

    # Brand dropdown
    brands = ['Samsung', 'Apple', 'Redmi', 'OnePlus', 'Realme', 'Vivo', 'Oppo', 'Motorola', 'Poco', 'Others']
    input_features['Brand'] = st.selectbox("Select Brand", brands)

    # Numeric inputs
    input_features['RAM'] = st.number_input("Enter RAM (GB)", min_value=0.0)
    input_features['ROM'] = st.number_input("Enter ROM (GB)", min_value=0.0)
    input_features['Display_Size'] = st.number_input("Enter Display Size (inches)", min_value=0.0)
    input_features['Battery'] = st.number_input("Enter Battery Capacity (mAh)", min_value=0.0)
    input_features['Front_Cam(MP)'] = st.number_input("Enter Front Camera (MP)", min_value=0.0)
    input_features['Back_Cam(MP)'] = st.number_input("Enter Back Camera (MP)", min_value=0.0)

    # Predict Button
    if st.button("🚀 Predict Price"):
        df = pd.DataFrame([input_features])

        # One-hot encode Brand if needed
        if 'Brand' in features:
            df = pd.get_dummies(df, columns=['Brand'])
            # Ensure all expected brand columns exist (in case user selects a brand not in training data)
            for col in [c for c in features if c.startswith('Brand_')]:
                if col not in df:
                    df[col] = 0

        # Reorder columns to match model training
        df = df.reindex(columns=features, fill_value=0)

        # Prediction
        prediction = model.predict(df)[0]
        st.success(f"💰 Predicted Selling Price: ₹{prediction:,.2f}")
