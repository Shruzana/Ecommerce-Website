import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("best_fit_model.pkl")

# Expected features (as per your trained model)
features = ['Brand', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "ℹ️ Overview", "📊 Prediction"])

# ========================
# HOME PAGE
# ========================
if page == "🏠 Home":
    st.title("📱 Products Discount Data Analysis & Estimation")
    st.image(
        "https://cdn.pixabay.com/photo/2021/01/08/09/24/smartphone-5899905_1280.jpg",
        use_container_width=True
    )

    st.markdown("""
    ## Welcome to the Smartphone Discount Prediction
    This app helps you **predict the discount price** of smartphones
    based on their brand, RAM, storage, display size, battery, and camera details.

    **💡 Why use this app?**
    - Helps e-commerce sellers plan competitive discounts.
    - Assists customers in estimating the best deal.
    - Useful for data analysis & price trend insights.

    Navigate to the **Prediction** tab from the sidebar to try it yourself!
    """)

# ========================
# OVERVIEW PAGE
# ========================
elif page == "ℹ️ Overview":
    st.title("📖 Project Overview")
    st.markdown("""
    ### 📌 Objective
    Predict the **Discount Price** of smartphones using machine learning.

    ### 📊 Dataset
    The model was trained on data scraped from:
    - **Amazon** 📦
    - **Flipkart** 🛒

    ### 📐 Features Used
    - **Brand** 🏷️
    - **RAM** (GB) 💾
    - **ROM** (GB) 📂
    - **Display Size** (inches) 📱
    - **Battery** (mAh) 🔋
    - **Front Camera (MP)** 🤳
    - **Back Camera (MP)** 📷

    ### ⚙️ How It Works
    1. Enter smartphone specifications.
    2. The app processes the input to match training data format.
    3. The model predicts the **discount price**.

    ### 📈 Use Cases
    - Price strategy planning for e-commerce platforms.
    - Budget estimation for buyers.
    - Competitive market analysis.
    """)

# ========================
# PREDICTION PAGE
# ========================
elif page == "📊 Prediction":
    st.title("📊 Predict Smartphone Discount Price")

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
    if st.button("🚀 Predict Discount Price"):
        df = pd.DataFrame([input_features])

        # Encoding Brand if required
        if 'Brand' in features:
            df = pd.get_dummies(df, columns=['Brand'])
            for col in [c for c in features if c.startswith('Brand_')]:
                if col not in df:
                    df[col] = 0

        # Reorder columns to match training features
        df = df.reindex(columns=features, fill_value=0)

        # Prediction
        prediction = model.predict(df)[0]
        st.success(f"💰 Predicted Discount Price: ₹{prediction:,.2f}")



