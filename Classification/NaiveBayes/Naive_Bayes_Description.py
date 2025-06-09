# Importing necessary libraries
import numpy as np                 # For numerical operations
import matplotlib.pyplot as plt   # For plotting and visualization
import pandas as pd               # For data handling and analysis

# Load the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Reads CSV file into a DataFrame
X = dataset.iloc[:, :-1].values                   # Extracts all columns except the last one as features (Age, Salary)
y = dataset.iloc[:, -1].values                    # Extracts the last column as the target variable (Purchased)

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)         # Splits data: 75% training, 25% testing

# Feature Scaling to standardize feature values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                             # Create scaler instance
X_train = sc.fit_transform(X_train)               # Fit and transform training data
X_test = sc.transform(X_test)                     # Transform test data using the same scaling

# Training the Naive Bayes classifier on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()                         # Create Gaussian Naive Bayes classifier
classifier.fit(X_train, y_train)                  # Train the model with the training data

# Predicting a single new result (e.g., Age = 30, Salary = 87000)
print(classifier.predict(sc.transform([[30, 87000]])))  # Transform input and predict output class

# Predicting the Test set results
y_pred = classifier.predict(X_test)               # Predicting on test data
# Combine predictions with actual values side-by-side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), 
                      y_test.reshape(len(y_test), 1)), axis=1))

# Evaluate the model using a Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)             # Create confusion matrix
print(cm)                                         # Print confusion matrix
accuracy_score(y_test, y_pred)                    # Calculate and return accuracy

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train    # Inverse transform to original scale
# Create grid of feature values for plotting decision boundary
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
)
# Predicting results over grid points and reshaping for plotting
plt.contourf(X1, X2, 
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF'])
)
plt.xlim(X1.min(), X1.max())                      # Set x-axis limits
plt.ylim(X2.min(), X2.max())                      # Set y-axis limits
# Plot actual training points with color based on class
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
plt.title('Naive Bayes (Training set)')           # Plot title
plt.xlabel('Age')                                 # X-axis label
plt.ylabel('Estimated Salary')                    # Y-axis label
plt.legend()                                      # Add legend
plt.show()                                        # Display plot

# Visualising the Test set results
X_set, y_set = sc.inverse_transform(X_test), y_test      # Inverse transform test set
# Create grid for plotting decision boundary
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)
)
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)  # Predict grid
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))               # Decision boundary
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
colors = ['#FA8072', '#1E90FF']                 # Define custom colors
# Plot actual test points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=colors[i], label=j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
