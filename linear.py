import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Drop rows with NaN values in the training dataset
train_data_cleaned = train_data.dropna()

# Splitting the cleaned training data into features (X) and target (y)
X_train_cleaned = train_data_cleaned[['x']]
y_train_cleaned = train_data_cleaned['y']

# Splitting the test data into features (X) and target (y)
X_test = test_data[['x']]
y_test = test_data['y']

# Creating and training the model
model = LinearRegression()
model.fit(X_train_cleaned, y_train_cleaned)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the MSE and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the MSE, R² score, and predicted values
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plotting actual vs predicted values
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual vs Predicted Values for Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

