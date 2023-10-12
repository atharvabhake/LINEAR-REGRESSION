# LINEAR-REGRESSION
simple_linear_regression.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#the input data (salary) and the target variable (software company performance)
# Input data (salary)
X = np.array([10, 20, 30, 40, 50, 60]).reshape(-1, 1)

# Target variable (software company performance)
y = np.array([15, 25, 35, 45, 55, 65])
#the LinearRegression model
# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)
#the predict method 
# Make predictions
y_pred = model.predict(X)

#let's plot the chart and graph using the matplotlib library:
# Plot the data points
plt.scatter(X, y, color='blue', label='Actual')

# Plot the regression line
plt.plot(X, y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Salary')
plt.ylabel('Software Company Performance')
plt.title('Linear Regression')

# Add legend
plt.legend()

# Show the plot
plt.show()
