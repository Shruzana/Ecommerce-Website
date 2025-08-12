import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('best_fit_model.pkl')
features = ['MRP', 'Discount_Price', 'Discount', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']

st.title('E-Commerce Smartphone Price Predictor')

# Interactive user inputs
input_features = {}
for feat in features:
    input_features[feat] = st.number_input(f"Enter {feat}", value=0.0)

if st.button('Predict Price'):
    df = pd.DataFrame([input_features])
    pred = model.predict(df)
    st.success(f'Predicted Selling Price: ₹{pred[0]:.2f}')

# Show model coefficients (terms)
if st.button('Show Model Terms (Coefficients)'):
    coef_dict = dict(zip(features, model.coef_))
    st.write('Model Coefficients:')
    st.json(coef_dict)
    st.write(f"Intercept: {model.intercept_:.2f}")

