# House-Price-Prediction

# Overview
This repository contains a comprehensive project on House Price Prediction utilizing the powerful CatBoost Regressor, a high-performance algorithm known for its accuracy and efficiency in handling tabular data. The model provides superior predictions with minimal deviation between the actual and predicted house prices compared to other alternatives like linear regression.

To enhance user experience, a Flask web application is included, allowing users to interactively input features and receive instant predictions for house prices. This user-friendly interface makes it easier to explore the model’s predictions in real time.

# Project Highlights
- CatBoost Regressor:
CatBoost provides exceptional accuracy in predicting house prices with a highly optimized gradient boosting algorithm.
The model demonstrates minimal deflection between the actual house prices and the predicted values, outperforming traditional linear regression models.
- Flask Web Application:
A sleek and interactive Flask web app allows users to input house features (e.g., square footage, location, etc.) and obtain instant price predictions.
This web interface connects seamlessly to the underlying CatBoost model, making predictions accessible for non-technical users.

# Key Features
High Accuracy Predictions: Achieves better accuracy and reduced prediction error compared to other models.
User-friendly Flask App: Allows easy input and output for house price prediction.
Least Deflection: CatBoost’s ability to minimize the difference between actual and predicted prices makes it an ideal choice for this task.

# Requirements
To ensure the project runs smoothly, install the following packages listed in the requirements.txt file. These include:

- catboost
- flask
- pandas
- numpy
- scikit-learn

# How to Run

To use this project, follow these steps:
``` 
pip install -r requirements.txt 
```

Run the Flask App: Once the dependencies are installed, run the following command to start the Flask web app:
``` 
python app.py 
```

Access the Web App: The Flask app will run on your local server. Open a web browser and go to:
``` 
http://localhost:5000 
```

# Model Explanation
- The CatBoost Regressor model is trained on historical housing data and can predict the price of a house based on various features.
- By using gradient boosting and handling both numerical and categorical features effectively, CatBoost significantly reduces overfitting and improves generalization, resulting in accurate predictions even on unseen data.

#Results
The CatBoost model has been shown to outperform other regression models like linear regression, providing the lowest error and most reliable predictions.
The Flask app makes it simple for users to interact with the model and explore different scenarios for house pricing.




