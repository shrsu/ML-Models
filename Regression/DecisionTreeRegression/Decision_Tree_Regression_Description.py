# Importing necessary libraries
import numpy as np                 # For numerical computations
import matplotlib.pyplot as plt    # For data visualization
import pandas as pd                # For handling data in tabular form

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')  # Load the CSV file into a DataFrame
X = dataset.iloc[:, 1:-1].values   # Select all rows and the second column (Position level) as features
y = dataset.iloc[:, -1].values     # Select all rows and the last column (Salary) as the target

# Training the Decision Tree Regression model on the dataset
from sklearn.tree import DecisionTreeRegressor  # Import the decision tree regressor
regressor = DecisionTreeRegressor(random_state=0)  # Create a regressor object with fixed randomness for reproducibility
regressor.fit(X, y)  # Train the model on the entire dataset

# Predicting a new result (e.g., salary for position level 6.5)
regressor.predict([[6.5]])  # Predict salary for position level 6.5

# Visualizing the Decision Tree Regression results with higher resolution
X_grid = np.arange(min(X), max(X), 0.01)  # Create a grid of values from min to max position level with a step of 0.01
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape the array to the correct shape for prediction

plt.scatter(X, y, color='red')  # Plot the original data points (position level vs salary) in red
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Plot the model's predictions in blue for each X_grid value
plt.title('Truth or Bluff (Decision Tree Regression)')  # Title of the plot
plt.xlabel('Position level')  # Label for the x-axis
plt.ylabel('Salary')  # Label for the y-axis
plt.show()  # Display the plot
