# Importing the necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization (not used in this script)
import pandas as pd  # For data manipulation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
# The dataset contains information about 50 startups, including their R&D Spend, Administration, Marketing Spend, State, and Profit

# Load the dataset from a CSV file
dataset = pd.read_csv('50_Startups.csv')

# Separating the independent variables (features) and dependent variable (target)
X = dataset.iloc[:, :-1].values  # Selecting all columns except the last one as features
y = dataset.iloc[:, -1].values   # Selecting the last column (Profit) as the target variable

# Display the independent variables before encoding categorical data
print(X)

# Encoding categorical data (State column) to convert categorical text data into numerical values

# Applying OneHotEncoding to the 4th column (index 3) which contains categorical data (State)
# 'remainder=passthrough' ensures that other numerical columns remain unchanged
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # Converting the transformed dataset into a NumPy array

# Display the transformed independent variables after encoding
print(X)

# Splitting the dataset into training and test sets

# 80% of the data is used for training, and 20% is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set

# Creating a Linear Regression model instance
regressor = LinearRegression()

# Fitting (training) the model using the training dataset
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)  # Making predictions on the test dataset

# Formatting the output to display results in a readable format
np.set_printoptions(precision=2)  # Setting precision to 2 decimal places for better readability

# Displaying the predicted and actual values side by side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))