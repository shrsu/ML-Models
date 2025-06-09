# Importing required libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data manipulation and analysis

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Load the dataset
X = dataset.iloc[:, :-1].values  # Select all columns except the last as features (Age, Estimated Salary)
y = dataset.iloc[:, -1].values  # Select the last column as the target (Purchased or not)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  # Import function for train/test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # Split with 75% training and 25% testing

# Printing training and test sets for verification
print(X_train)  # Display training features
print(y_train)  # Display training labels
print(X_test)   # Display test features
print(y_test)   # Display test labels

# Feature Scaling - Standardizing the input features
from sklearn.preprocessing import StandardScaler  # Import scaler
sc = StandardScaler()  # Create scaler object
X_train = sc.fit_transform(X_train)  # Fit to training data and transform
X_test = sc.transform(X_test)  # Transform test data using the same scaler

# Print scaled features for verification
print(X_train)
print(X_test)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier  # Import the Decision Tree Classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)  # Use entropy as the splitting criterion
classifier.fit(X_train, y_train)  # Fit the model on the training data

# Predicting a new result with custom input [Age=30, Salary=87000]
print(classifier.predict(sc.transform([[30, 87000]])))  # Transform input and predict class (0 or 1)

# Predicting the Test set results
y_pred = classifier.predict(X_test)  # Predict on the test set
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))  # Compare predicted vs actual

# Making the Confusion Matrix and calculating Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score  # Import evaluation metrics
cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
print(cm)  # Print confusion matrix
accuracy_score(y_test, y_pred)  # Compute and return accuracy score

# Visualising the Training set results
from matplotlib.colors import ListedColormap  # For setting custom colors in plot
X_set, y_set = sc.inverse_transform(X_train), y_train  # Inverse scaling to original values for visualization
X1, X2 = np.meshgrid(  # Create a grid of values for plotting decision boundaries
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
)
# Plot decision boundary using contour plot
plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(['#FA8072', '#1E90FF'])  # Red for class 0, Blue for class 1
)
plt.xlim(X1.min(), X1.max())  # Set x-axis limit
plt.ylim(X2.min(), X2.max())  # Set y-axis limit
# Scatter plot of actual training data points
for i, j in enumerate(np.unique(y_set)):  # Loop over classes
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)  # Plot points for each class
plt.title('Decision Tree Classifier (Training set)')  # Title of the plot
plt.xlabel('Age')  # Label for x-axis
plt.ylabel('Estimated Salary')  # Label for y-axis
plt.legend()  # Show legend
plt.show()  # Display plot

# Visualising the Test set results
X_set, y_set = sc.inverse_transform(X_test), y_test  # Inverse scaling for test set
X1, X2 = np.meshgrid(  # Create grid for test set decision boundary
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)
)
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)  # Predict on grid
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))  # Plot decision boundary
plt.xlim(X1.min(), X1.max())  # Set x-axis limit
plt.ylim(X2.min(), X2.max())  # Set y-axis limit
colors = ['#FA8072', '#1E90FF']  # Define class colors
# Plot test set points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=colors[i], label=j)
plt.title('Decision Tree Classifier (Test set)')  # Title
plt.xlabel('Age')  # X-axis label
plt.ylabel('Estimated Salary')  # Y-axis label
plt.legend()  # Show legend
plt.show()  # Display plot
