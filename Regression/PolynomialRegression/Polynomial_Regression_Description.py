"""
## Importing the libraries
"""

# Importing necessary libraries
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs
import pandas as pd  # For handling datasets

"""## Importing the dataset"""

# Loading the dataset from a CSV file
dataset = pd.read_csv('Position_Salaries.csv')

# Extracting the independent variable (Position Level) as a matrix
X = dataset.iloc[:, 1:-1].values  # Selecting the second column as X (ignoring the first column 'Position')

# Extracting the dependent variable (Salary) as a vector
y = dataset.iloc[:, -1].values  # Selecting the last column as y

"""## Training the Linear Regression model on the whole dataset"""

# Importing and training a simple Linear Regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # Fitting the model to the dataset

"""## Training the Polynomial Regression model on the whole dataset"""

# Importing PolynomialFeatures to create polynomial terms
from sklearn.preprocessing import PolynomialFeatures

# Creating polynomial features up to degree 4
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)  # Transforming X into polynomial features

# Training a new Linear Regression model on the transformed dataset
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

"""## Visualising the Linear Regression results"""

# Scatter plot of original data points
plt.scatter(X, y, color='red')

# Plotting the predictions made by the simple linear regression model
plt.plot(X, lin_reg.predict(X), color='blue')

# Titles and labels
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""## Visualising the Polynomial Regression results"""

# Scatter plot of original data points
plt.scatter(X, y, color='red')

# Plotting the polynomial regression model's predictions
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')

# Titles and labels
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""## Visualising the Polynomial Regression results (for higher resolution and smoother curve)"""

# Creating a high-resolution X grid for a smoother curve
X_grid = np.arange(min(X), max(X), 0.1)  # Generating values from min to max with step 0.1
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshaping it into a 2D array

# Scatter plot of original data points
plt.scatter(X, y, color='red')

# Plotting the polynomial regression model's predictions on high-resolution grid
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')

# Titles and labels
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""## Predicting a new result with Linear Regression"""

# Predicting the salary for position level 6.5 using the Linear Regression model
lin_reg.predict([[6.5]])

"""## Predicting a new result with Polynomial Regression"""

# Predicting the salary for position level 6.5 using the Polynomial Regression model
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
