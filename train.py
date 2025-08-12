
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


data = pd.read_csv(r"C:\Users\DELL\Downloads\ML_STream_lit\ML_STream_lit\Data\D_Ecommerce.csv")


features = ['Brand', 'Brand_Model', 'RAM', 'ROM', 'Display_Size', 'Battery', 'Front_Cam(MP)', 'Back_Cam(MP)']
X = data[features]
y = data['Discount_Price']


X.fillna(X.mean(), inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'best_fit_model.pkl')




