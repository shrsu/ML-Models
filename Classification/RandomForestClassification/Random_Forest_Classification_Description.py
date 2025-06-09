# Importing essential libraries
import numpy as np                          # For numerical operations and arrays
import matplotlib.pyplot as plt             # For plotting graphs
import pandas as pd                         # For data manipulation and analysis

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Load CSV data into a DataFrame
X = dataset.iloc[:, :-1].values                 # Select all rows and all columns except the last as features
y = dataset.iloc[:, -1].values                  # Select all rows and only the last column as target labels

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 0)     # 75% training and 25% test data

# Displaying training and test sets
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling - standardizing the feature values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                             # Create a StandardScaler object
X_train = sc.fit_transform(X_train)               # Fit and transform training features
X_test = sc.transform(X_test)                     # Only transform test features (use same scaler)

# Displaying scaled feature values
print(X_train)
print(X_test)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators = 10,                            # Number of trees in the forest
    criterion = 'entropy',                        # Splitting criterion: 'entropy' for Information Gain
    random_state = 0)                             # Set random seed for reproducibility
classifier.fit(X_train, y_train)                  # Train the model on the training data

# Predicting a single new result (e.g., Age = 30, Salary = 87000)
print(classifier.predict(sc.transform([[30, 87000]])))  # Transform input and make prediction

# Predicting the Test set results
y_pred = classifier.predict(X_test)               # Predict the target for test features
# Display predictions alongside actual labels
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis = 1))

# Making the Confusion Matrix to evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)             # Compute confusion matrix
print(cm)                                         # Print the confusion matrix
accuracy_score(y_test, y_pred)                    # Compute and return accuracy of the model

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train  # Inverse transform to get original values
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),   # Age grid
    np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25)  # Salary grid
)
# Predict class for every point in the grid and reshape for contour plot
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))  # Draw decision boundary
plt.xlim(X1.min(), X1.max())                         # Set x-axis limit
plt.ylim(X2.min(), X2.max())                         # Set y-axis limit
# Scatter plot of training points with true labels
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(['#FA8072', '#1E90FF'])(i), label = j)
plt.title('Random Forest Classifier (Training set)')  # Title
plt.xlabel('Age')                                     # X-axis label
plt.ylabel('Estimated Salary')                        # Y-axis label
plt.legend()                                          # Show legend
plt.show()                                            # Show plot

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test   # Inverse transform test set for plotting
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),  # Grid for Age
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)   # Grid for Salary
)
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)  # Predictions
plt.contourf(X1, X2, Z, alpha=0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))  # Decision boundary
plt.xlim(X1.min(), X1.max())                            # Set x-axis limit
plt.ylim(X2.min(), X2.max())                            # Set y-axis limit
colors = ['#FA8072', '#1E90FF']                         # Define colors
# Scatter plot of test points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=colors[i], label=j)
plt.title('Random Forest Classifier (Test set)')        # Title
plt.xlabel('Age')                                       # X-axis label
plt.ylabel('Estimated Salary')                          # Y-axis label
plt.legend()                                            # Show legend
plt.show()                                              # Show plot
