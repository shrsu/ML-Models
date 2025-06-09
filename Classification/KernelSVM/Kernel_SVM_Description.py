# Importing necessary libraries
import numpy as np                     # For numerical operations
import matplotlib.pyplot as plt        # For plotting graphs
import pandas as pd                    # For data manipulation

# Loading the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Read CSV file into a DataFrame
X = dataset.iloc[:, :-1].values       # Select all rows and all columns except the last as features
y = dataset.iloc[:, -1].values        # Select the last column as the target variable

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)  # 25% test data, 75% training data

# Feature scaling (standardizing the features)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                 # Create a scaler object
X_train = sc.fit_transform(X_train)   # Fit and transform training data
X_test = sc.transform(X_test)         # Transform test data using the same scaler

# Training the Kernel SVM model using RBF (Gaussian) kernel
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)  # SVC = Support Vector Classifier
classifier.fit(X_train, y_train)     # Train the classifier on the training data

# Predicting a single new result (e.g., a person aged 30 earning 87,000)
print(classifier.predict(sc.transform([[30, 87000]])))  # Input must be scaled

# Predicting the test set results
y_pred = classifier.predict(X_test)  # Predict class labels for the test set
# Concatenating predicted and actual labels side by side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Evaluating the model with a confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  # Create confusion matrix
print(cm)                              # Print confusion matrix
accuracy_score(y_test, y_pred)         # Calculate and return accuracy

# Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train  # Undo scaling for visualization
# Create a grid of points for plotting decision boundary
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
)
# Predict the classifier output for every point in the grid
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
# Plot the decision boundary using contour plot
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())         # Set x-axis limits
plt.ylim(X2.min(), X2.max())         # Set y-axis limits
# Scatter plot of training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
plt.title('Kernel SVM (Training set)')  # Title of the plot
plt.xlabel('Age')                       # X-axis label
plt.ylabel('Estimated Salary')          # Y-axis label
plt.legend()                            # Show legend
plt.show()                              # Display plot

# Visualizing the test set results (same as above with test data)
X_set, y_set = sc.inverse_transform(X_test), y_test  # Undo scaling for visualization
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)
)
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
colors = ['#FA8072', '#1E90FF']       # Colors for plotting classes
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=colors[i], label=j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
