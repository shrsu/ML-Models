# Importing required libraries
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data manipulation and reading CSV files

# Loading the dataset from a CSV file
dataset = pd.read_csv('Social_Network_Ads.csv')  # Reads the dataset into a DataFrame
X = dataset.iloc[:, :-1].values  # Selecting all rows and all columns except the last one as features (Age, EstimatedSalary)
y = dataset.iloc[:, -1].values   # Selecting the last column as the target variable (Purchased or not)

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split  # Importing train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# 75% of data used for training, 25% for testing, random_state ensures reproducibility

# Displaying the training and test data (for verification)
print(X_train)  # Print training features
print(y_train)  # Print training labels
print(X_test)   # Print test features
print(y_test)   # Print test labels

# Feature Scaling - standardizing features to improve K-NN performance
from sklearn.preprocessing import StandardScaler  # Importing scaler
sc = StandardScaler()  # Creating scaler object
X_train = sc.fit_transform(X_train)  # Fit and transform training data
X_test = sc.transform(X_test)        # Only transform test data using the same scaler

# Displaying the scaled features
print(X_train)  # Print scaled training features
print(X_test)   # Print scaled test features

# Training the K-Nearest Neighbors classifier on the training set
from sklearn.neighbors import KNeighborsClassifier  # Import K-NN classifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# Using 5 neighbors, Minkowski distance with p=2 (Euclidean distance)
classifier.fit(X_train, y_train)  # Fit the classifier on the training data

# Predicting a single new result
print(classifier.predict(sc.transform([[30,87000]])))
# Predict if a user aged 30 with a salary of 87000 will purchase the product

# Predicting the results on the test set
y_pred = classifier.predict(X_test)  # Predicting labels for test set
# Printing predicted vs actual labels side-by-side
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Creating the Confusion Matrix and calculating accuracy
from sklearn.metrics import confusion_matrix, accuracy_score  # Import metrics
cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
print(cm)  # Print confusion matrix
accuracy_score(y_test, y_pred)  # Calculate and return accuracy

# Visualising the Training set results
from matplotlib.colors import ListedColormap  # For custom color maps in plots
X_set, y_set = sc.inverse_transform(X_train), y_train  # Inverse scaling for plotting
# Creating a mesh grid of feature values (Age and Salary)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10,
                               stop = X_set[:, 0].max() + 10,
                               step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 1000,
                               stop = X_set[:, 1].max() + 1000,
                               step = 0.5))
# Predicting the class for each point in the mesh grid
plt.contourf(X1, X2,
             classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))
# Setting plot limits
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plotting the training data points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(['#FA8072', '#1E90FF'])(i),
                label = j)
plt.title('K-NN (Training set)')  # Title of the plot
plt.xlabel('Age')  # X-axis label
plt.ylabel('Estimated Salary')  # Y-axis label
plt.legend()  # Show legend
plt.show()  # Display plot

# Visualising the Test set results
X_set, y_set = sc.inverse_transform(X_test), y_test  # Inverse scaling for test set
# Creating a mesh grid for the test set
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step=0.5),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step=0.5))
# Predicting the class for each grid point
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
# Plotting decision boundary
plt.contourf(X1, X2, Z, alpha=0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']) )
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Defining color palette for plot
colors = ['#FA8072', '#1E90FF']
# Plotting the actual test points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                color=colors[i],
                label=j)
plt.title('K-NN (Test set)')  # Title of the plot
plt.xlabel('Age')  # X-axis label
plt.ylabel('Estimated Salary')  # Y-axis label
plt.legend()  # Show legend
plt.show()  # Display the plot