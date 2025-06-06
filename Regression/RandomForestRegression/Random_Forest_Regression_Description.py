# Importing necessary libraries
import numpy as np  # For numerical operations and handling arrays
import matplotlib.pyplot as plt  # For plotting graphs
import pandas as pd  # For data manipulation and analysis

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Reads the CSV file into a DataFrame
X = dataset.iloc[:, 1:-1].values  # Selects the independent variable (position level) as a 2D array
y = dataset.iloc[:, -1].values  # Selects the dependent variable (salary)

# Importing the RandomForestRegressor from sklearn
from sklearn.ensemble import RandomForestRegressor

# Creating and training the model
# n_estimators = number of trees in the forest
# random_state = ensures the result is reproducible
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)  # Fits the model to the whole dataset

# Predicting a new result with a position level of 6.5
# The model will return the predicted salary for this level
regressor.predict([[6.5]])

# Creating a high-resolution grid for smoother curve visualization
X_grid = np.arange(min(X), max(X), 0.01)  # Creates a range of values from min to max with a step of 0.01
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshapes it into a 2D array for prediction

# Visualizing the Random Forest Regression results
plt.scatter(X, y, color='red')  # Plots the actual data points (position levels vs salaries)
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Plots the predicted curve
plt.title('Truth or Bluff (Random Forest Regression)')  # Title of the graph
plt.xlabel('Position level')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()  # Displays the plot
