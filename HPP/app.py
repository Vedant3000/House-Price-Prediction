# app.py

from flask import Flask, render_template, request
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

app = Flask(__name__)

# Load the CatBoost model
cb_model = CatBoostRegressor(iterations=200, depth=7, learning_rate=0.1, l2_leaf_reg=4.0, random_state=42)

# Load the dataset
dataset = pd.read_csv("kc_house_data.csv")

# Identify features (X) and target variable (y)
X = dataset.drop('price', axis=1)
y = dataset['price']

# Train the model on the entire dataset
cb_model.fit(X, y)

# Prediction function
def predict_price(input_features):
    input_array = np.array(input_features).reshape(1, -1)
    predicted_price = cb_model.predict(input_array)
    return predicted_price[0]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = [float(request.form['bedrooms']),
                      float(request.form['bathrooms']),
                      float(request.form['sqft_living']),
                      float(request.form['area']),
                      float(request.form['storeys']),
                      float(request.form['waterfront']),
                      float(request.form['view']),
                      float(request.form['condition']),
                      float(request.form['grade']),
                      float(request.form['sqft_above']),
                      float(request.form['sqft_basement']),
                      float(request.form['yr_built']),
                      float(request.form['yr_renovated']),
                      float(request.form['zipcode']),
                      float(request.form['lat']),
                      float(request.form['long']),
                      float(request.form['sqft_living15']),
                      float(request.form['sqft_lot15'])]

        predicted_price = predict_price(user_input)

        return render_template('index.html', prediction=f'The predicted price is: ${predicted_price:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
