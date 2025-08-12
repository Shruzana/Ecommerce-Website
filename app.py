import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('best_fit_model.pkl')

# Features for prediction
features = ['Brand', 'MRP', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction"])

# ================= HOME PAGE =================
if page == "🏠 Home":
    st.title("📱 E-Commerce Smartphone Price Predictor")
    st.image("https://cdn.pixabay.com/photo/2017/01/06/19/15/smartphone-1957740_960_720.jpg", use_container_width=True)

    st.markdown("""
    ## 📌 Project Overview
    This project predicts the selling price of smartphones using features like **Brand**, **RAM**, **ROM**, **Display Size**, and **Camera Quality**.  
    The model was trained on an e-commerce dataset to help sellers estimate competitive prices.

    ### 🔍 Features Used:
    - **Brand**: Company of the smartphone (e.g., Samsung, Apple, Redmi, OnePlus, etc.)
    - **MRP**: Maximum Retail Price.
    - **RAM**: Memory in GB.
    - **ROM**: Storage capacity in GB.
    - **Display Size**: Screen size in inches.
    - **Battery**: Battery capacity in mAh.
    - **Front Camera (MP)**: Front camera resolution.
    - **Back Camera (MP)**: Rear camera resolution.
    """)

    st.info("💡 Use the **Prediction** tab in the sidebar to test the model!")

# ================= PREDICTION PAGE =================
elif page == "📊 Prediction":
    st.title("📊 Predict Smartphone Selling Price")

    # Input fields
    input_features = {}

    # Brand dropdown
    brands = ['Samsung', 'Apple', 'Redmi', 'OnePlus', 'Realme', 'Vivo', 'Oppo', 'Motorola', 'Poco', 'Others']
    input_features['Brand'] = st.selectbox("Select Brand", brands)

    # Numeric inputs
    for feat in features[1:]:  # Skip 'Brand'
        input_features[feat] = st.number_input(f"Enter {feat}", value=0.0)

    if st.button("🚀 Predict Price"):
        df = pd.DataFrame([input_features])
        pred = model.predict(df)
        st.success(f"💰 Predicted Selling Price: ₹{pred[0]:,.2f}")

    if st.button("📈 Show Model Coefficients"):
        if hasattr(model, "coef_"):
            coef_dict = dict(zip(features, model.coef_))
            st.write("Model Coefficients:")
            st.json(coef_dict)
            st.write(f"Intercept: {model.intercept_:.2f}")
        else:
            st.warning("This model does not have coefficients (e.g., tree-based models).")


