import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("kc_house_data.csv")

dataset.head()

# Identify features (X) and target variable (y)
X = dataset.drop('price', axis=1)
y = dataset['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Initialize the CatBoostRegressor with hyperparameters
cb_model = CatBoostRegressor(
    iterations=200,  # Adjust the number of boosting iterations
    depth=7,         # Modify the depth of the tree
    learning_rate=0.1,  # Adjust the learning rate
    l2_leaf_reg=4.0,    # Try different values for regularization
    random_state=42 # You can set a random seed for reproducibility
)


# Train the model on the training data
cb_model.fit(X_train, y_train)

# Make predictions on the test data
preds = cb_model.predict(X_test)


# Calculate Mean Squared Error (MSE) and R-squared (R2) Score
mse = mean_squared_error(y_test, preds)
cb_r2_score = r2_score(y_test, preds)

print("CatBoost R-squared (R2) Score:", cb_r2_score)
print("Mean Squared Error (MSE):", mse)

def predict_price(model, input_features):
    # Convert user input to a numpy array
    input_array = np.array(input_features).reshape(1, -1)

    # Make predictions
    predicted_price = model.predict(input_array)

    return predicted_price[0]

# Get user input
bedrooms = float(input("Enter the number of bedrooms: "))
bathrooms = float(input("Enter the number of bathrooms: "))
sqft_living = float(input("Enter the square footage of living space: "))
area = float(input("Enter the area: "))
storeys = float(input("Enter the number of storeys: "))
waterfront = float(input("Enter 1 if waterfront, 0 otherwise: "))
view = float(input("Enter the view rating: "))
condition = float(input("Enter the condition rating: "))
grade = float(input("Enter the grade: "))
sqft_above = float(input("Enter the square footage of roof: "))
sqft_basement = float(input("Enter the square footage of the basement: "))
yr_built = float(input("Enter the year built: "))
yr_renovated = float(input("Enter the year renovated: "))
zipcode = float(input("Enter the zipcode: "))
lat = float(input("Enter the latitude: "))
long = float(input("Enter the longitude: "))
sqft_living15 = float(input("Enter the square footage of living space for the nearest 15 neighbors: "))
sqft_lot15 = float(input("Enter the square footage of the lot for the nearest 15 neighbors: "))

# Create a list with user input features
user_input = [bedrooms, bathrooms, sqft_living, area, storeys, waterfront, view, condition, grade, sqft_above,
              sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15]


# Get the predicted price
predicted_price = predict_price(cb_model, user_input)

# Display the predicted price
print(f"The predicted price is: ${predicted_price:,.2f}")