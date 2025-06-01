import os

# Importing necessary libraries
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
import pandas as pd  # Pandas for handling datasets
# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression

# Get the absolute path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Salary_Data.csv')

# Loading the dataset
# The dataset contains two columns: Years of Experience and Salary
dataset = pd.read_csv(csv_path)

# Extracting the independent variable (Years of Experience)
X = dataset.iloc[:, :-1].values

# Extracting the dependent variable (Salary)
y = dataset.iloc[:, -1].values

# test_size=1/3 means 1/3rd of the data will be used for testing
# random_state=0 ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Creating a Linear Regression model
regressor = LinearRegression()

# Training the model using training data
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)  # Predicting salaries for test data

# Visualizing the Training set results
plt.scatter(X_train, y_train, color='red')  # Scatter plot of actual training data
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Regression line
plt.title('Salary vs Experience (Training set)')  # Chart title
plt.xlabel('Years of Experience')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()  # Displaying the plot

# Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')  # Scatter plot of actual test data
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Regression line (same as training)
plt.title('Salary vs Experience (Test set)')  # Chart title
plt.xlabel('Years of Experience')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()  # Displaying the plot
