import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
# Assuming 'Position_Salaries.csv' contains salary data based on position levels
dataset = pd.read_csv('Position_Salaries.csv')

# Extract independent variable (Position level) and dependent variable (Salary)
X = dataset.iloc[:, 1:-1].values  # Selecting the second column (Position level)
y = dataset.iloc[:, -1].values    # Selecting the last column (Salary)

# Print extracted data for verification
print(X)
print(y)

# Reshape y to be a 2D array for feature scaling
y = y.reshape(len(y), 1)
print(y)

# Feature scaling - Standardizing both X and y values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  # Fit and transform X
y = sc_y.fit_transform(y)  # Fit and transform y

# Print scaled values for verification
print(X)
print(y)

# Training the SVR model on the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')  # Using Radial Basis Function (RBF) kernel
regressor.fit(X, y.ravel())  # Training the model (y.ravel() converts to 1D array)

# Predicting the salary for level 6.5 after inverse transformation
predicted_salary = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
print(f"Predicted Salary for level 6.5: {predicted_salary}")

# Visualizing the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Actual data points
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')  # SVR predictions
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results with higher resolution
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)  # Creating finer steps for smooth curve
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshaping to a 2D array
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Actual data points
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')  # SVR predictions
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
